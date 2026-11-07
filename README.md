# WiFi-Mamba reproduction on Person-in-WiFi 3D

An unofficial PyTorch implementation of WiFi-Mamba ("Interleaved Selective State Space Models for WiFi-Based 3D Multi-Person Pose Estimation") written from the equations of the paper, with training and evaluation on the Person-in-WiFi 3D dataset.
The official code was not available to us, so every choice the paper leaves open is a guess. These choices are listed in `FaithfulConfig` (`wifimamba/model.py`) and in the docstring at the top of that file.

## Layout

```
wifimamba/model.py    the model (learnable Haar DWT, PhaseFi, DS3M with cross-stream coupling, fusion and neck Mamba blocks, SSA decoder with persistent memory)
wifimamba/scan.py     the selective scan, a Triton kernel with its backward and a PyTorch fallback
wifimamba/metrics.py  MPJPE, PA-MPJPE, MPJDLE, PCK, and the matching rules of the protocols
build_cache.py        raw Person-in-WiFi 3D release -> one .npz cache (|CSI|, wrapped phase, keypoints in metres)
train.py              training with the persistent memory carried over consecutive frames, validation on held-out runs of the train split
evaluate.py           scores a checkpoint on the official test split with three protocols
```

## Requirements

Python 3.10 or later, a CUDA GPU, and the packages of `requirements.txt`. Triton ships with the Linux CUDA wheels of PyTorch. Without it the scan falls back to a slow PyTorch loop.

```bash
pip install -r requirements.txt
```

## Data

Download the Person-in-WiFi 3D release from the [project page](https://aiotgroup.github.io/Person-in-WiFi-3D/). The expected layout is

```
<data-root>/train_data/{csi/*.mat, keypoint/*.npy, train_data_list.txt}
<data-root>/test_data/{csi/*.mat, keypoint/*.npy, test_data_list.txt}
```

The split is the released one (`train_data_list.txt`, `test_data_list.txt`).

```bash
python build_cache.py --data-root <data-root> --out outputs/piw3d_cache.npz
```

The cache keeps the raw amplitude and the wrapped phase. The denoising of the paper (DWT, PhaseFi) runs inside the model.

## Training

The configuration below uses 180 link tokens (one per link and packet), unshared layers, the standard Mamba block for the S6, the paper preset for the rest, 128 parallel streams, and seed 3.

```bash
python train.py --tag link_b128 --preset paper --tokens link --s6 mamba --unwrap-phase --graph \
    --streams 128 --lr 3e-4 --epochs 50 --patience 15 --seed 3 \
    --cache outputs/piw3d_cache.npz --out-dir outputs/wifimamba
```

The checkpoint is selected on held-out runs of the train split (`--val-frac`, with a guard of `--guard` frames around them), never on the test split. `--graph` captures the training step in a CUDA graph. Without it the same step runs eagerly.

## Evaluation

```bash
python evaluate.py --ckpt outputs/wifimamba/link_b128.pt --cache outputs/piw3d_cache.npz --json outputs/wifimamba/link_b128_eval.json
```

Three protocols are reported per number of people (1, 2, 3).

1. Predicted count. The count head decides how many queries are kept (the top confidences), the Hungarian algorithm only pairs the kept queries with the ground truth.
2. Greedy. The matching of the official Person-in-WiFi 3D code, every ground-truth person takes the nearest prediction. The labels choose among the predictions, so this is an upper bound.
3. True count. The true number of people decides how many queries are kept, then as in 1.

All errors are in millimetres. The persistent memory runs over the test frames in recording order, reset at the start of every run of consecutive frames.
