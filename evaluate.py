"""Evaluate a checkpoint of train.py on the official test split of Person-in-WiFi 3D, with the three protocols.

    python evaluate.py --ckpt outputs/wifimamba/link_b128.pt --cache outputs/piw3d_cache.npz

1) the predicted count (head of the model), the top-count confidences, Hungarian, matched persons only (no labels choose)
2) the greedy protocol of the official code (the labels choose among the K predictions), a ceiling
3) the true count, the top-n confidences, Hungarian
The persistent memory runs over the frames in recording order, 128 streams, as in the trainer.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import train as e  # noqa: E402
from wifimamba.metrics import compute_similarity_transform, full_metrics, greedy_match  # noqa: E402
from wifimamba.model import FaithfulConfig, WiFiMamba  # noqa: E402


def fmt(v):
    return "     nan" if v is None else f"{v:8.2f}"


def greedy_present(P, y, mask):
    """The greedy protocol for the person counts that exist in the frames, an empty group gives NaN."""
    out, counts = {}, mask.sum(1).astype(int)
    for n in (1, 2, 3):
        mm = []
        for i in np.flatnonzero(counts == n):
            gt, pr = greedy_match(y[i, :n], P[i])
            for g, p in zip(gt * 1000.0, pr * 1000.0):
                mm.append((np.linalg.norm(p - g, axis=-1).mean(), np.linalg.norm(compute_similarity_transform(p, g) - g, axis=-1).mean(),
                           np.abs(p[:, 0] - g[:, 0]).mean(), np.abs(p[:, 1] - g[:, 1]).mean(), np.abs(p[:, 2] - g[:, 2]).mean()))
        mm = np.array(mm)
        keys = ("mpjpe_mm", "pa_mpjpe_mm", "mpjdle_h_mm", "mpjdle_v_mm", "mpjdle_d_mm")
        out[f"n{n}"] = dict(zip(keys, np.round(mm.mean(0), 1).tolist())) if len(mm) else dict.fromkeys(keys, float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--split", choices=["test", "train"], default="test", help="train = the error on frames the model was fitted on (a subset of --runs runs), to separate under-fitting from generalisation")
    ap.add_argument("--runs", type=int, default=250)
    ap.add_argument("--persons", type=int, default=0, help="only the frames with this number of persons, for a checkpoint trained with --persons")
    a = ap.parse_args()
    dev = torch.device("cuda")
    e.NO_MEMORY, e.ORDERED, e.PHASE_SCALE, e.MATCH_SCALE, e.MATCH_PHYSICAL = False, False, 1.0, 42.0, False
    ck = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    cfg = FaithfulConfig(**{k: v for k, v in ck["cfg"].items() if k in FaithfulConfig.__dataclass_fields__})
    model = WiFiMamba(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    cx = np.load(a.cache)
    sp = a.split
    pick = np.flatnonzero(cx[f"mask_{sp}"].sum(1).astype(int) == a.persons) if a.persons else slice(None)
    y, mask, names = cx[f"y_{sp}"][pick].astype("float32"), cx[f"mask_{sp}"][pick].astype("float32"), cx[f"names_{sp}"][pick]
    data = dict(amp=torch.from_numpy(np.ascontiguousarray(cx[f"amp_{sp}"][pick])).to(dev), phase=torch.from_numpy(np.ascontiguousarray(cx[f"phase_{sp}"][pick])).to(dev))
    runs, _ = e.build_runs(names)
    sel = None
    if sp == "train":  # a fixed random subset of the runs, the state is carried inside a run as in training
        runs = [runs[i] for i in np.sort(np.random.default_rng(0).choice(len(runs), min(a.runs, len(runs)), replace=False))]
        sel = np.concatenate(runs)
    mean, std = torch.from_numpy(ck["mean"].astype("float32")).to(dev), torch.from_numpy(ck["std"].astype("float32")).to(dev)
    P, L, C = e.predict(model, data, runs, 128, dev, mean, std, len(y))
    if sel is not None:
        P, L, C, y, mask = P[sel], L[sel], C[sel], y[sel], mask[sel]
    print(f"checkpoint {a.ckpt} epoch {ck['epoch']} split {sp} frames {len(y)} queries {P.shape[1]} parameters {sum(p.numel() for p in model.parameters()):,}")
    out = {"epoch": int(ck["epoch"])}
    rows = (("1 predicted count (no labels)", full_metrics(P, L, C, y, mask)), ("3 true count, top-n by confidence", full_metrics(P, L, mask.sum(1).astype(int), y, mask)))
    g = greedy_present(P, y, mask)
    keys = ("mpjpe_mm", "pa_mpjpe_mm", "mpjdle_h_mm", "mpjdle_v_mm", "mpjdle_d_mm", "pck30", "pck20", "pck10", "count_acc")
    for name, r in rows[:1]:
        print(f"--- {name}")
        for k in keys:
            print(f"  {k:12s}", "  ".join(fmt(r[f'n{n}'].get(k)) for n in (1, 2, 3)))
        out["no_oracle"] = {f"n{n}": {k: r[f"n{n}"].get(k) for k in keys} for n in (1, 2, 3)}
    print("--- 2 greedy protocol of the official code (labels choose among the predictions, a ceiling)")
    for k in ("mpjpe_mm", "pa_mpjpe_mm", "mpjdle_h_mm", "mpjdle_v_mm", "mpjdle_d_mm"):
        print(f"  {k:12s}", "  ".join(fmt(g[f'n{n}'][k]) for n in (1, 2, 3)))
    out["greedy"] = {f"n{n}": g[f"n{n}"] for n in (1, 2, 3)}
    r = rows[1][1]
    print(f"--- {rows[1][0]}")
    print("  mpjpe_mm    ", "  ".join(fmt(r[f'n{n}']['mpjpe_mm']) for n in (1, 2, 3)))
    out["true_count"] = {f"n{n}": r[f"n{n}"]["mpjpe_mm"] for n in (1, 2, 3)}
    print("paper WiFi-Mamba MPJPE 76.75 89.08 104.81 | PA 62.85 67.55 66.31")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
