"""Build the cache used by train.py and evaluate.py from the raw Person-in-WiFi 3D release.

Input   <data-root>/{train,test}_data/{csi/*.mat, keypoint/*.npy, {train,test}_data_list.txt}
Output  one .npz with, per split, raw |CSI| and wrapped phase [N,9,30,20] (float16, no normalisation),
        keypoints [N,3,14,3] in metres (zero padded), a person mask [N,3] and the sample names.
The paper's own denoising (learnable DWT, PhaseFi) runs inside the model on these raw values.
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np

LINKS, SUBCARRIERS, PACKETS, JOINTS, MAX_PEOPLE = 9, 30, 20, 14, 3


def csi_sanitization(csi_rx):
    """The phase sanitisation of the dataset authors' preprocessing (not used by the default --phase-mode raw).
    The three antennas of one receiver are chained through conj products, and one slope and one offset are
    fitted jointly over all 3 x 30 subcarriers, so the phase differences between antennas are preserved."""
    one, two, three = csi_rx[0], csi_rx[1], csi_rx[2]
    antennas, subcarriers, packets = 3, SUBCARRIERS, one.shape[1]
    fi = 312.5 * 2
    ph = np.zeros((antennas, subcarriers, packets), dtype=np.float32)
    ai, bi = np.tile(2 * np.pi * fi * np.arange(subcarriers), antennas), np.ones(antennas * subcarriers)
    a_dot, b_dot, c_dot = np.dot(ai, ai), np.dot(ai, bi), np.dot(bi, bi)
    temp = np.tile(np.arange(subcarriers), antennas).reshape(antennas, subcarriers)
    for t in range(packets):
        ph[0, :, t] = np.unwrap(np.angle(one[:, t]))
        ph[1, :, t] = np.unwrap(ph[0, :, t] + np.angle(two[:, t] * np.conj(one[:, t])))
        ph[2, :, t] = np.unwrap(ph[1, :, t] + np.angle(three[:, t] * np.conj(two[:, t])))
        ci = np.concatenate((ph[0, :, t], ph[1, :, t], ph[2, :, t]))
        d_dot, e_dot = np.dot(ai, ci), np.dot(bi, ci)
        rho = (b_dot * e_dot - c_dot * d_dot) / (a_dot * c_dot - b_dot ** 2)
        beta = (b_dot * d_dot - a_dot * e_dot) / (a_dot * c_dot - b_dot ** 2)
        ph[:, :, t] = ph[:, :, t] + 2 * np.pi * fi * temp * rho + beta
    return np.angle(np.stack([np.abs(one) * np.exp(1j * ph[0]), np.abs(two) * np.exp(1j * ph[1]), np.abs(three) * np.exp(1j * ph[2])]))


PHASE_MODE = "raw"


def load_sample(args):
    import h5py
    csi_path, kp_path = args
    with h5py.File(csi_path, "r") as f:
        arr = f["csi_out"]["real"][:] + 1j * f["csi_out"]["imag"][:]
    arr = arr.transpose(3, 2, 1, 0)  # official loader order -> 3 x 3 x 30 x 20
    amp = np.nan_to_num(np.abs(arr).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).reshape(LINKS, SUBCARRIERS, PACKETS)
    if PHASE_MODE == "sanitized":  # arr is [rx, tx, subcarrier, packet]
        ph = np.nan_to_num(np.stack([csi_sanitization(arr[rx]) for rx in range(arr.shape[0])]).astype(np.float32), nan=0.0).reshape(LINKS, SUBCARRIERS, PACKETS)
    else:
        ph = np.nan_to_num(np.angle(arr).astype(np.float32), nan=0.0).reshape(LINKS, SUBCARRIERS, PACKETS)
    kp = np.load(kp_path).astype(np.float32)
    if kp.ndim != 3 or kp.shape[1:] != (JOINTS, 3):
        raise ValueError(f"unexpected keypoints {kp.shape} in {kp_path}")
    n = min(len(kp), MAX_PEOPLE)
    padded, mask = np.zeros((MAX_PEOPLE, JOINTS, 3), np.float32), np.zeros(MAX_PEOPLE, np.float32)
    padded[:n], mask[:n] = kp[:n], 1.0
    return amp, ph, padded, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--phase-mode", choices=["raw", "sanitized"], default="raw", help="sanitized = the dataset authors' cross-antenna sanitisation")
    a = ap.parse_args()
    global PHASE_MODE
    PHASE_MODE = a.phase_mode
    out = {}
    with Pool(a.workers) as pool:
        for split in ("train", "test"):
            root = a.data_root / f"{split}_data"
            names = [l.split()[0] for l in (root / f"{split}_data_list.txt").read_text().splitlines() if l.strip()]
            jobs = [(root / "csi" / f"{n}.mat", root / "keypoint" / f"{n}.npy") for n in names]
            res = pool.map(load_sample, jobs, chunksize=64)
            amp = np.stack([r[0] for r in res])
            assert np.isfinite(amp).all() and amp.max() < 6e4, "amplitude does not fit float16"
            out[f"amp_{split}"], out[f"phase_{split}"] = amp.astype(np.float16), np.stack([r[1] for r in res]).astype(np.float16)
            out[f"y_{split}"], out[f"mask_{split}"], out[f"names_{split}"] = np.stack([r[2] for r in res]), np.stack([r[3] for r in res]), np.array(names)
            counts = np.bincount(out[f"mask_{split}"].sum(1).astype(int), minlength=4)
            print(split, amp.shape, "people per frame", counts[1:].tolist(), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
