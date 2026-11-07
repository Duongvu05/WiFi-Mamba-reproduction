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

The baseline follows the recipe of the paper. Batch 16 (16 parallel streams of consecutive frames), AdamW with lr 1e-4, 20 packet tokens so that the SSMs scan the packets in time, the matching cost of Eq. (21) on physical coordinates, and the joint loss of Eq. (23) summing |dx| + |dy| + |dz| per joint before the mean over the joints and the matched persons, with the poses regressed in metres.

```bash
python train.py --tag paper_b16 --preset paper --tokens packet --streams 16 --lr 1e-4 \
    --match-space physical --joint-reduce sum --target-norm none --graph \
    --epochs 50 --patience 15 --seed 3 --cache outputs/piw3d_cache.npz --out-dir outputs/wifimamba
```

`--preset paper` sets the conv S6 of Eq. (6) to (8), unshared layers, the DWT with 2 levels on the raw amplitude and no residual around DS3M. `--s6 mamba` swaps in the standard Mamba block and `--tokens link` uses 180 tokens, one per link and packet. With `--tokens link` the SSMs scan the 180 tokens link-major, so their state crosses the boundary between two links. The baseline keeps the persistent memory of the paper (Eq. 18 to 20). Every SSA layer of the decoder has 16 learned memory tokens and a memory SSM whose state is carried from frame t-1 to frame t inside a run of consecutive frames and reset at the start of a run, in training and at test time. The paper does not say whether the gradient flows back through earlier frames. Here the carried state is detached after every frame, so the gradient stays inside the current frame. `--no-memory` resets the state at every frame (an ablation), in the eager and in the `--graph` training step.

At the end of every epoch `<out-dir>/<tag>_last.pt` keeps the weights, the AdamW moments, the scheduler, the epoch, the selection state and the random generators. To continue an interrupted run, repeat the same command with `--resume <out-dir>/<tag>_last.pt`.

The checkpoint is selected on held-out runs of the train split (`--val-frac`, with a guard of `--guard` frames around them), never on the test split. `--graph` captures the training step in a CUDA graph. Without it the same step runs eagerly.

## Evaluation

```bash
python evaluate.py --ckpt outputs/wifimamba/paper_b16.pt --cache outputs/piw3d_cache.npz --json outputs/wifimamba/paper_b16_eval.json
```

Three protocols are reported per number of people (1, 2, 3).

1. Predicted count. The count head decides how many queries are kept (the top confidences), the Hungarian algorithm only pairs the kept queries with the ground truth.
2. Greedy. The matching of the official Person-in-WiFi 3D code, every ground-truth person takes the nearest prediction. The labels choose among the predictions, so this is an upper bound.
3. True count. The true number of people decides how many queries are kept, then as in 1.

Pose errors (MPJPE, PA-MPJPE, MPJDLE, PCK) are averaged over the matched pairs only, in millimetres. PA-MPJPE uses the least squares similarity transform (Umeyama scale). PCK@30, @20 and @10 count the joints closer than 0.3, 0.2 and 0.1 times the distance between the two elbows of the ground truth (joints 4 and 6), check this normaliser before comparing with other papers.
Counting and detection are reported apart from the pose error. `count_acc` is the share of frames whose kept number equals the true one. `pair_recall` and `pair_precision` count every Hungarian pair, however far. `recall` and `precision` count a pair as a true positive only when its MPJPE is at most 500 mm (`tp_mm` of `full_metrics`).
The checkpoint configuration is read back by `evaluate.py`, so a model trained with `--no-memory` is also evaluated without memory. The persistent memory runs over the test frames in recording order, reset at the start of every run of consecutive frames.
