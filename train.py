"""Train and evaluate WiFi-Mamba (wifimamba/model.py) on Person-in-WiFi 3D.

Paper recipe: AdamW, lr 1e-4, weight decay 1e-2, batch 16, 50 epochs, loss
  L = lj*Ljoint + lb*Lbone + lc*Lconf(focal) + ln*Lcount  with Hungarian matching on L1 - confidence.
Persistent SSM memory: frames are fed in recording order through `batch` parallel streams. The memory
state is carried from frame t-1 to frame t, reset at the start of every run, and detached after each step.

Protocol: predicted count, top-`count` confidences kept, Hungarian only to score.
Validation holds out whole runs of consecutive frames and removes training frames within `guard`
frames of them, so validation resembles the test split (a random frame split leaks neighbouring frames).
"""
import argparse
import json
import math
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from wifimamba.model import FaithfulConfig, WiFiMamba  # noqa: E402
from wifimamba.metrics import BONES, full_metrics, metrics  # noqa: E402


def build_runs(names):
    """Contiguous runs of frames in one recording, ordered by frame number -> list of index arrays."""
    seqs = {}
    for i, n in enumerate(names):
        p = str(n).split("_")
        seqs.setdefault("_".join(p[:-1]), []).append((int(p[-1]), i))
    runs, seq_of = [], []
    for s, items in seqs.items():
        items.sort()
        cur = [items[0]]
        for a, b in zip(items, items[1:]):
            if b[0] == a[0] + 1:
                cur.append(b)
            else:
                runs.append(np.array([i for _, i in cur]))
                seq_of.append(s)
                cur = [b]
        runs.append(np.array([i for _, i in cur]))
        seq_of.append(s)
    return runs, seq_of


def plan_streams(runs, n_streams, rng=None):
    """Distribute runs over parallel streams -> idx [S, L] (-1 pad) and reset [S, L] (run starts)."""
    order = rng.permutation(len(runs)) if rng is not None else np.arange(len(runs))
    streams, lens = [[] for _ in range(n_streams)], np.zeros(n_streams, dtype=int)
    for r in order:
        s = int(np.argmin(lens))
        streams[s].append(runs[r])
        lens[s] += len(runs[r])
    streams = [streams[i] for i in np.argsort(-lens, kind="stable")]
    lens = np.sort(lens)[::-1]
    length = int(lens.max())
    idx, reset = -np.ones((n_streams, length), dtype=np.int64), np.zeros((n_streams, length), dtype=bool)
    for s, rs in enumerate(streams):
        pos = 0
        for r in rs:
            idx[s, pos:pos + len(r)] = r
            reset[s, pos] = True
            pos += len(r)
    return idx, reset


NO_MEMORY = False
PHASE_SCALE = 1.0
ORDERED = False
MATCH_PHYSICAL = False  # True matches in metres (Eq. 21), False in z-scored coordinates
JOINT_SCALE = 1.0  # 3 = the L1 of Eq. 23 sums the three coordinates of a joint, 1 = their mean
MATCH_SCALE = 1.0  # 42 = sum over joints and coordinates as in Eq. 21, 1 = mean


def segment_plans(runs, seg_len, batch, rng):
    """Cut every run into consecutive segments of at most seg_len frames, shuffle them across the whole
    training set and yield [batch, seg_len] plans. The persistent state is carried inside a segment and
    reset at its first frame, so batches are shuffled like ordinary training while the memory is still used."""
    segs = [r[i:i + seg_len] for r in runs for i in range(0, len(r), seg_len)]
    order = rng.permutation(len(segs))
    for b0 in range(0, len(order), batch):
        chunk = sorted((segs[i] for i in order[b0:b0 + batch]), key=len, reverse=True)
        idx = -np.ones((len(chunk), seg_len), dtype=np.int64)
        reset = np.zeros((len(chunk), seg_len), dtype=bool)
        for j, sg in enumerate(chunk):
            idx[j, :len(sg)] = sg
            reset[j, 0] = True
        yield idx, reset


def run_stream(model, data, idx, reset, device, train_step=None):
    """Iterate a stream plan (streams sorted by length, so the active ones are a prefix)."""
    n = idx.shape[0]
    state = model.init_state(n, device)
    idx_t, reset_t = torch.as_tensor(idx, device=device), torch.as_tensor(reset, device=device)
    n_active = (idx >= 0).sum(0)
    for s in range(idx.shape[1]):
        a = int(n_active[s])
        if a == 0:
            break
        frames = idx_t[:a, s]
        keep = (~reset_t[:a, s]).float()[:, None, None] * (0.0 if NO_MEMORY else 1.0)
        h_in = [h[:a] * keep for h in state]
        phase = data["phase"][frames].float()
        pose, conf, count, h_out = model(data["amp"][frames].float(), phase * PHASE_SCALE if PHASE_SCALE != 1.0 else phase, h_in)
        for l, h in enumerate(h_out):
            state[l][:a] = h.detach()
        yield pose, conf, count, frames


def focal(prob, target, alpha=0.25, gamma=2.0, valid=None):
    prob = prob.clamp(1e-6, 1 - 1e-6)
    pt = torch.where(target > 0.5, prob, 1 - prob)
    at = torch.where(target > 0.5, torch.full_like(prob, alpha), torch.full_like(prob, 1 - alpha))
    loss = -at * (1 - pt) ** gamma * torch.log(pt)
    if valid is None:
        return loss.mean()
    return (loss * valid[:, None]).sum() / (valid.sum().clamp(min=1.0) * prob.shape[1])


_PERMS, _BONES = {}, {}


def _perms(k, n, device):
    if (k, n) not in _PERMS:
        _PERMS[(k, n)] = torch.tensor(list(permutations(range(k), n)), device=device)
    return _PERMS[(k, n)]


def _bone_idx(device):
    if device not in _BONES:
        _BONES[device] = (torch.tensor([b[0] for b in BONES], device=device), torch.tensor([b[1] for b in BONES], device=device))
    return _BONES[device]


def wm_loss(pose, conf_logit, count_logit, y, mask, mean, std, w, valid=None):
    """Hungarian matching on (L1 pose - confidence) (Eq. 21), by enumerating assignments, then Eq. 22-24.
    No host synchronisation: every person-count group is computed for all rows and weighted by a mask."""
    b, k = pose.shape[:2]
    conf = torch.sigmoid(conf_logit)
    counts = mask.sum(1).long()
    j1, j2 = _bone_idx(pose.device)
    ar = torch.arange(b, device=pose.device)[:, None]
    target_conf = torch.zeros_like(conf)
    joint = bone = pose.new_zeros(())
    matched = (counts.clamp(max=3)).sum().clamp(min=1).float()
    for n in range(1, min(k, 3) + 1):  # a person count above the number of queries does not exist in a model of one scenario
        v = (counts == n).float()
        perms = _perms(n, n, pose.device) if ORDERED else _perms(k, n, pose.device)
        cp = (pose[:, :, None] - y[:, None, :n]).abs().mean((3, 4))  # [B,K,n]
        if MATCH_PHYSICAL:
            cm = ((pose * std + mean)[:, :, None] - (y[:, None, :n] * std + mean)).abs().mean((3, 4))  # [B,K,n] in metres
        else:
            cm = cp
        total = sum((cm * MATCH_SCALE - conf[:, :, None])[:, perms[:, i], i] for i in range(n))  # [B,P]
        ch = perms[total.argmin(1)]  # [B,n]
        pm = cp.permute(0, 2, 1).gather(2, ch[:, :, None]).squeeze(-1)  # [B,n]
        joint = joint + (pm.sum(1) * v).sum() * JOINT_SCALE
        pr = pose[ar, ch] * std + mean  # [B,n,14,3] metres
        gt = y[:, :n] * std + mean
        bl = lambda q: (q[:, :, j1] - q[:, :, j2]).norm(dim=-1)
        bone = bone + ((bl(pr) - bl(gt)).abs().mean(-1).sum(1) * v).sum()
        target_conf = target_conf + (ch[:, :, None] == torch.arange(k, device=pose.device)).float().sum(1) * v[:, None]
    lc = focal(conf, target_conf.clamp(max=1.0), valid=valid)
    ce = F.cross_entropy(count_logit, counts.clamp(0, 3), reduction="none")
    ln = ce.mean() if valid is None else (ce * valid).sum() / valid.sum().clamp(min=1.0)
    joint, bone = joint / matched, bone / matched
    return w[0] * joint + w[1] * bone + w[2] * lc + w[3] * ln, dict(joint=joint.detach(), bone=bone.detach(), conf=lc.detach(), count=ln.detach())


class GraphStep:
    """One whole training step (gather, forward, loss, backward, clipping, AdamW, state update) captured in a
    CUDA graph and replayed. The batch is fixed at `batch` rows. A row with valid = 0 only pads the batch and
    contributes nothing to the loss, so tails of epochs are handled without changing the shapes.
    `eager()` runs the identical code without the graph, for testing."""

    def __init__(self, model, opt, data, yt, mt, mean, std, w, batch, clip=1.0):
        dev = next(model.parameters()).device
        self.model, self.opt, self.batch = model, opt, batch
        self.idx = torch.zeros(batch, dtype=torch.long, device=dev)
        self.keep = torch.zeros(batch, 1, 1, device=dev)
        self.valid = torch.zeros(batch, device=dev)
        self.skipped = torch.zeros((), device=dev)
        self.state = model.init_state(batch, dev)
        params = [q for q in model.parameters()]

        def body():
            frames = self.idx
            phase = data["phase"][frames].float()
            pose, conf, count, h_out = model(data["amp"][frames].float(), phase * PHASE_SCALE if PHASE_SCALE != 1.0 else phase,
                                             [h * self.keep for h in self.state])
            loss, _ = wm_loss(pose, conf, count, yt[frames], mt[frames] * self.valid[:, None], mean, std, w, valid=self.valid)
            loss.backward()
            bad = (~torch.isfinite(loss.detach())).float()
            self.skipped += bad  # a step with a non-finite loss is skipped: zero gradients, zero recurrent state
            for q in params:
                if q.grad is not None:
                    q.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            for h, new in zip(self.state, h_out):
                h.copy_(torch.nan_to_num(new.detach(), nan=0.0, posinf=0.0, neginf=0.0))
            return torch.nan_to_num(loss.detach(), nan=0.0, posinf=0.0, neginf=0.0)

        self.body = body
        lr = opt.param_groups[0]["lr"]
        saved_lr, saved = lr.clone(), [q.detach().clone() for q in params]
        lr.zero_()  # warm-up and capture leave the weights unchanged
        self.valid.fill_(1.0)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                body()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph):
            self.out = body()
        with torch.no_grad():  # restore weights, optimiser state and recurrent state in place
            for q, v in zip(params, saved):
                q.copy_(v)
            for st in opt.state.values():
                for v in st.values():
                    v.zero_()
            lr.copy_(saved_lr)
        self.skipped.zero_()
        self.reset_state()

    def reset_state(self):
        for h in self.state:
            h.zero_()

    def _load(self, col, reset_col):
        self.idx.copy_(col.clamp(min=0))
        self.valid.copy_((col >= 0).float())
        self.keep.copy_((~reset_col).float()[:, None, None] * (0.0 if NO_MEMORY else 1.0))  # --no-memory zeroes the input state at every frame, as run_stream does

    def replay(self, col, reset_col):
        self._load(col, reset_col)
        self.graph.replay()
        return self.out

    def eager(self, col, reset_col):
        self._load(col, reset_col)
        self.opt.zero_grad(set_to_none=True)
        return self.body()


@torch.no_grad()
def predict(model, data, runs, n_streams, device, mean, std, n_frames):
    model.eval()
    k = model.cfg.num_queries
    P, L, C = np.zeros((n_frames, k, 14, 3), "float32"), np.zeros((n_frames, k), "float32"), np.ones(n_frames, dtype=int)
    idx, reset = plan_streams(runs, n_streams)
    for pose, conf, count, frames in run_stream(model, data, idx, reset, device):
        f = frames.cpu().numpy()
        P[f] = (pose * std + mean).cpu().numpy()
        L[f] = (-torch.arange(k, device=conf.device).float().expand_as(conf) if ORDERED else conf).cpu().numpy()
        C[f] = count.argmax(-1).clamp(0, k).cpu().numpy()
    return P, L, C


def selection_score(m):
    """Mean over the person counts present in the validation frames of (summed matched MPJPE + 1000 per missed or extra person) per person."""
    return float(np.mean([((v["mpjpe_mm"] or 0.) * v["matched"] + 1000. * (v["missed"] + v["extra"])) / (v["frames"] * n)
                          for n, v in ((n, m[f"n{n}"]) for n in (1, 2, 3)) if v["frames"]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="wm_faithful")
    ap.add_argument("--streams", type=int, default=16, help="paper batch size 16")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--weights", type=float, nargs=4, default=[5.0, 1.0, 1.0, 0.5])
    ap.add_argument("--queries", type=int, default=3)
    ap.add_argument("--no-share", action="store_true")
    ap.add_argument("--share", action="store_true", help="force the shared weights even with --preset paper")
    ap.add_argument("--segment", type=int, default=0, help="train on shuffled segments of this length instead of long streams")
    ap.add_argument("--match-cost", choices=["sum", "mean"], default="sum", help="Eq. 21 sums the L1 over joints and coordinates")
    ap.add_argument("--unwrap-phase", action="store_true")
    ap.add_argument("--s6", choices=["mamba", "bare", "conv"], default=None, help="bare = the selective SSM of Eq. 6-8 only, conv = Eq. 6-8 with a causal convolution on the input")
    ap.add_argument("--neck", type=int, default=3)
    ap.add_argument("--decoder-layers", type=int, default=3)
    ap.add_argument("--dwt-init-threshold", type=float, default=0.693)
    ap.add_argument("--match-space", choices=["normalized", "physical"], default="normalized")
    ap.add_argument("--target-norm", choices=["joint_axis", "global", "none"], default="joint_axis", help="the space of the regressed poses, joint_axis = z-score per joint and axis (the old runs), global = one scalar std, none = metres")
    ap.add_argument("--joint-reduce", choices=["mean", "sum"], default="mean", help="sum = the L1 of Eq. 23 sums the 3 coordinates of a joint")
    ap.add_argument("--dt-rank", choices=["full", "mamba"], default=None, help="rank of Linear_Delta, mamba = ceil(d / 16)")
    ap.add_argument("--no-phase-regression", action="store_true", help="skip the per-link PhaseFi linear fit")
    ap.add_argument("--graph", action="store_true", help="capture the whole training step in a CUDA graph")
    ap.add_argument("--dt-init", choices=["mamba", "unit"], default="mamba")
    ap.add_argument("--preset", choices=["none", "paper"], default="none", help="paper = our reading of the paper (conv S6, unshared, DWT with 2 levels on the raw amplitude, no residual in DS3M), --s6 and --tokens override it")
    ap.add_argument("--block-kind", choices=["", "mamba", "conv", "bare"], default="", help="kind of the fusion, neck and decoder blocks, empty = the kind of --s6. mamba = the standard Mamba block with the projections and the gate")
    ap.add_argument("--expand", type=int, default=1)
    ap.add_argument("--persons", type=int, default=0, help="train and test only on the frames with this number of persons (a model per scenario as the curves of Fig. 3 suggest), the number of queries is then this number")
    ap.add_argument("--ds3m-residual", action="store_true", help="the input of DS3M is added to its output (the paper preset removes it)")
    ap.add_argument("--amp-only", action="store_true", help="no phase stream and no coupling, a diagnostic")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient norm clipping, 0 = none (the paper states none)")
    ap.add_argument("--const-lr", action="store_true", help="a fixed learning rate, no cosine decay (the paper states none)")
    ap.add_argument("--subset", type=int, default=0, help="fit only the first N frames of the fit runs, an overfitting test, the error on these frames is printed at the end")
    ap.add_argument("--tokens", choices=["packet", "link", "link9"], default=None, help="link = 180 tokens (link-major, 30 subcarriers each) instead of 20 packet tokens")
    ap.add_argument("--phasefi-concat", action="store_true", help="the PhaseFi line is fitted on the 60 values [amplitude, phase] concatenated (k = 1 .. 60 of Appendix C.3) and removed from the phase part")
    ap.add_argument("--phase", choices=["raw", "none"], default="raw", help="none feeds zeros (diagnostic)")
    ap.add_argument("--no-memory", action="store_true", help="reset the persistent state every frame (diagnostic)")
    ap.add_argument("--ordered", action="store_true", help="match only the first n slots and read the first count slots (diagnostic)")
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--no-val", action="store_true", help="no validation split, all the train frames are fitted, every epoch is saved and the last epoch is the result")
    ap.add_argument("--guard", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0, help="debug, limit steps per epoch")
    ap.add_argument("--cache", type=Path, default=ROOT / "outputs/piw3d_cache.npz", help="from build_cache.py")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/wifimamba")
    a = ap.parse_args()
    global NO_MEMORY, PHASE_SCALE, ORDERED, MATCH_SCALE, MATCH_PHYSICAL
    global JOINT_SCALE
    JOINT_SCALE = 3.0 if a.joint_reduce == "sum" else 1.0
    MATCH_PHYSICAL = a.match_space == "physical"
    MATCH_SCALE = 42.0 if a.match_cost == "sum" else 1.0
    NO_MEMORY, ORDERED = a.no_memory, a.ordered
    PHASE_SCALE = 0.0 if a.phase == "none" else 1.0
    torch.backends.cuda.matmul.allow_tf32 = True
    out, dev = a.out_dir, torch.device("cuda")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    cx = np.load(a.cache)
    sel = {sp: (np.flatnonzero(cx[f"mask_{sp}"].sum(1).astype(int) == a.persons) if a.persons else slice(None)) for sp in ("train", "test")}
    if a.persons:
        a.queries = a.persons
        print(f"only the frames with {a.persons} person(s), {len(np.arange(len(cx['mask_train']))[sel['train']])} train and {len(np.arange(len(cx['mask_test']))[sel['test']])} test frames, {a.queries} queries", flush=True)
    raw = {"y_train": cx["y_train"][sel["train"]], "mask_train": cx["mask_train"][sel["train"]], "y_test": cx["y_test"][sel["test"]], "mask_test": cx["mask_test"][sel["test"]],
           "train_names": cx["names_train"][sel["train"]], "test_names": cx["names_test"][sel["test"]]}
    y, mask, y_te, mask_te = raw["y_train"].astype("float32"), raw["mask_train"].astype("float32"), raw["y_test"].astype("float32"), raw["mask_test"].astype("float32")
    G = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)
    data = {"train": dict(amp=G(cx["amp_train"][sel["train"]]), phase=G(cx["phase_train"][sel["train"]])), "test": dict(amp=G(cx["amp_test"][sel["test"]]), phase=G(cx["phase_test"][sel["test"]]))}

    runs, seq_of = build_runs(raw["train_names"])
    test_runs, _ = build_runs(raw["test_names"])
    nval = 0 if a.no_val else max(1, int(a.val_frac * len(runs)))
    order = rng.permutation(len(runs))
    val_ids = set(order[:nval].tolist())
    val_runs = [runs[i] for i in sorted(val_ids)]
    frame_no = {n: int(str(n).split("_")[-1]) for n in raw["train_names"]}
    guarded = set()
    for i in val_ids:
        s = seq_of[i]
        lo, hi = frame_no[raw["train_names"][runs[i][0]]] - a.guard, frame_no[raw["train_names"][runs[i][-1]]] + a.guard
        for j, r in enumerate(runs):
            if j not in val_ids and seq_of[j] == s:
                nums = np.array([frame_no[raw["train_names"][x]] for x in r])
                keep = (nums >= lo) & (nums <= hi)
                if keep.any():
                    guarded.add(j)
    fit_runs = [r for j, r in enumerate(runs) if j not in val_ids and j not in guarded]
    print(f"runs total={len(runs)} val={len(val_runs)} ({sum(map(len, val_runs))} frames) guard-dropped={len(guarded)} fit={len(fit_runs)} ({sum(map(len, fit_runs))} frames)", flush=True)
    if a.subset:  # whole runs until N frames, the last one cut
        kept, total = [], 0
        for r in fit_runs:
            kept.append(r[:a.subset - total]); total += len(kept[-1])
            if total >= a.subset:
                break
        fit_runs = kept
        print(f"overfitting subset {sum(map(len, fit_runs))} frames in {len(fit_runs)} runs", flush=True)
    fit_frames = np.concatenate(fit_runs)
    vals = y[fit_frames][mask[fit_frames] > 0]
    mean_np, std_np = vals.mean(0), vals.std(0).clip(1e-6)
    if a.target_norm == "global":  # one scalar scale, the L1 of Eq. 21 and 23 stays isotropic over the axes and the joints
        std_np = np.full_like(std_np, vals.std())
    elif a.target_norm == "none":  # raw metres
        mean_np, std_np = np.zeros_like(mean_np), np.ones_like(std_np)
    mean, std = G(mean_np.astype("float32")), G(std_np.astype("float32"))
    yt, mt = G((y - mean_np) / std_np), G(mask)
    val_frames = np.concatenate(val_runs) if val_runs else np.zeros(0, dtype=int)

    paper = dict(s6_kind="conv", share_weights=False, token_layout="packet", ds3m_residual=False, amp_minmax=False, amp_scale=1 / 32, dwt_levels=2, dwt_threshold_init=0.693, dt_init="mamba") if a.preset == "paper" else {}
    cfg = FaithfulConfig(num_queries=a.queries, share_weights=not a.no_share, dt_init=a.dt_init, phase_unwrap=a.unwrap_phase, phase_regression=not a.no_phase_regression, dwt_threshold_init=a.dwt_init_threshold, token_layout=a.tokens or "packet", phasefi_concat=a.phasefi_concat, s6_kind=a.s6 or "mamba", num_neck=a.neck, num_decoder=a.decoder_layers)
    for k_, v_ in paper.items():
        setattr(cfg, k_, v_)
    if a.tokens is not None:  # an explicit --tokens wins over the layout of the preset
        cfg.token_layout = a.tokens
    if a.dt_rank is not None:
        cfg.dt_rank = a.dt_rank
    if a.s6 is not None:  # an explicit --s6 wins over the preset
        cfg.s6_kind = a.s6
    if a.share:  # shared weights over the preset (the paper reports 2.14M parameters, the unshared preset has about 3M)
        cfg.share_weights = True
    cfg.block_kind, cfg.expand = a.block_kind, a.expand
    if a.ds3m_residual:
        cfg.ds3m_residual = True
    if a.amp_only:
        cfg.phase_stream = False
    model = WiFiMamba(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    if a.graph:  # capturable AdamW with a tensor learning rate that is set in place every epoch
        opt = torch.optim.AdamW(model.parameters(), lr=torch.tensor(a.lr, device=dev), weight_decay=a.wd, capturable=True)
        sched = None
        gs = GraphStep(model, opt, data["train"], yt, mt, mean, std, a.weights, a.streams, clip=a.clip)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-6)
    print(f"params={n_params:,} share={cfg.share_weights} streams={a.streams} lr={a.lr}", flush=True)
    best, bad, t0, ckpt = float("inf"), 0, time.time(), out / f"{a.tag}.pt"
    for ep in range(1, a.epochs + 1):
        model.train()
        plans = segment_plans(fit_runs, a.segment, a.streams, rng) if a.segment else [plan_streams(fit_runs, a.streams, rng)]
        tot, steps = 0., 0
        if a.graph:
            opt.param_groups[0]["lr"].fill_(a.lr if a.const_lr else 1e-6 + 0.5 * (a.lr - 1e-6) * (1 + math.cos(math.pi * (ep - 1) / a.epochs)))
            tot = torch.zeros((), device=dev)
            gs.reset_state()
            for idx, reset in plans:
                r, length = idx.shape
                if r < a.streams:  # pad the last batch of segments to the fixed batch size
                    idx = np.vstack([idx, -np.ones((a.streams - r, length), dtype=idx.dtype)])
                    reset = np.vstack([reset, np.zeros((a.streams - r, length), dtype=bool)])
                idx_t, reset_t = torch.as_tensor(idx, device=dev), torch.as_tensor(reset, device=dev)
                for t in range(length):
                    tot += gs.replay(idx_t[:, t], reset_t[:, t])
                    steps += 1
                    if a.max_steps and steps >= a.max_steps:
                        break
            if float(gs.skipped) > 0:
                print(f"  warning: {int(gs.skipped)} steps with a non-finite loss were skipped so far", flush=True)
        else:
            for idx, reset in plans:
                for pose, conf, count, frames in run_stream(model, data["train"], idx, reset, dev):
                    loss, parts = wm_loss(pose, conf, count, yt[frames], mt[frames], mean, std, a.weights)
                    if steps % 200 == 0 and not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss {parts}")
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if a.clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                    opt.step()
                    tot += loss.detach()
                    steps += 1
                    if a.max_steps and steps >= a.max_steps:
                        break
            if not a.const_lr:
                sched.step()
        if a.no_val:
            torch.save(dict(model=model.state_dict(), mean=mean_np, std=std_np, epoch=ep, cfg=cfg.__dict__, args=vars(a) | {"out_dir": str(a.out_dir)}), ckpt)
            print(f"epoch={ep} loss={float(tot) / steps:.4f} sec={time.time() - t0:.0f}", flush=True)
            continue
        P, L, C = predict(model, data["train"], val_runs, 128, dev, mean, std, len(y))
        vm = metrics(P[val_frames], L[val_frames], C[val_frames], y[val_frames], mask[val_frames])
        score = selection_score(vm)
        if score < best - 1e-5:
            best, bad = score, 0
            torch.save(dict(model=model.state_dict(), mean=mean_np, std=std_np, epoch=ep, cfg=cfg.__dict__, args=vars(a) | {"out_dir": str(a.out_dir)}, validation=vm), ckpt)
        else:
            bad += 1
        val_mm = [None if vm[f"n{n}"]["mpjpe_mm"] is None else round(vm[f"n{n}"]["mpjpe_mm"], 1) for n in (1, 2, 3)]
        print(f"epoch={ep} loss={float(tot) / steps:.4f} val_mpjpe={val_mm} sel={score:.2f} best={best:.2f} bad={bad} sec={time.time() - t0:.0f}", flush=True)
        if bad >= a.patience:
            break
    saved = torch.load(ckpt, weights_only=False)
    model.load_state_dict(saved["model"])
    if a.subset:  # the error on the fitted frames themselves
        Pf, Lf, Cf = predict(model, data["train"], fit_runs, 128, dev, mean, std, len(y))
        rf = full_metrics(Pf[fit_frames], Lf[fit_frames], mask[fit_frames].sum(1).astype(int), y[fit_frames], mask[fit_frames])
        print("FIT", a.tag, {n: rf[n].get("mpjpe_mm") for n in ("n1", "n2", "n3")}, flush=True)
    P, L, C = predict(model, data["test"], test_runs, 128, dev, mean, std, len(y_te))
    res = full_metrics(P, L, C, y_te, mask_te)
    res.update(params=n_params, best_epoch=saved["epoch"], seconds=time.time() - t0, config=cfg.__dict__)
    (out / f"{a.tag}_test.json").write_text(json.dumps(res, indent=2))
    print("RESULT", a.tag, {n: (round(res[n]["mpjpe_mm"], 2) if res[n]["mpjpe_mm"] is not None else None) for n in ("n1", "n2", "n3")}, flush=True)
    for n in ("n1", "n2", "n3"):
        print(n, {k: res[n].get(k) for k in ("mpjpe_mm", "pa_mpjpe_mm", "mpjdle_h_mm", "mpjdle_v_mm", "mpjdle_d_mm", "pck30", "pck20", "pck10", "count_acc")}, flush=True)


if __name__ == "__main__":
    main()
