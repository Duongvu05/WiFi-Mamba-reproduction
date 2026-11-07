"""WiFi-Mamba (Interleaved Selective State Space Models for WiFi-Based 3D Multi-Person Pose
Estimation) following the paper's equations. The selective scan runs on a Triton kernel (wifimamba/scan.py),
with a plain PyTorch fallback.

Pipeline (paper Fig. 2)
  Preprocessing  learnable Haar-DWT amplitude denoising (Eq. 10) and PhaseFi phase correction
                 plus a learnable residual (Eq. 11), both along the subcarrier axis.
  Backbone       DS3M layers. Two S6 streams with cross-stream state coupling applied at every
                 time step (Eq. 12-15), fused by W_f, then Fusion Mamba blocks.
  Neck           causal SSM layers over the packet axis.
  Head           pose queries -> L x [Query SSM -> cross-attention (Selective State Attention,
                 Eq. 16-17) -> persistent-memory SSM carried across frames (Eq. 18-20)].
  Outputs        3D poses, confidence scores, person count.

Choices the paper leaves open are listed in `FaithfulConfig`.

Known deviations from the paper text
  * Eq. 9 (|CSI| and atan2) is applied when the cache is built (build_cache.py), not in the model.
  * Eq. 10 uses L Haar levels with one learnable threshold per level (the paper preset uses L = 2). The paper gives no L.
  * Eq. 11 regresses over the 30 subcarriers of each link. The appendix writes the sum over 60, which does not
    match this dataset (30 per link). `phase_unwrap=True` unwraps along subcarriers before the fit.
  * Eq. 12-15 use the per-step mean over the SSM state axis (not the full [D, N] state), a low-rank projection for
    W, a channel-wise gate, and the coupling only modulates the stream outputs (no feedback into the recurrence).
    With zero-initialised W_m the modulation starts at 1 + lambda * sigmoid(0) = 1.5, which is what Eq. 15 gives.
  * Eq. 16 uses a residual Mamba block for the query SSM, and the memory gate (Eq. 20) uses the cross-attended query.
  * Mamba blocks keep the standard causal depthwise convolution, SiLU gate and skip term that the paper does not describe.
  * Same-type layers share weights by default (2.2M parameters, close to the paper's 2.14M). Unshared is 5.2M.
  * DS3M adds the residual input before fusion. The paper fuses the modulated outputs directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scan import scan


@dataclass
class FaithfulConfig:
    d_model: int = 256            # paper
    d_state: int = 16             # paper
    num_ds3m: int = 2             # paper
    num_fusion: int = 2           # paper ("2 Mamba blocks")
    num_neck: int = 3             # paper ("three causal SSM layers")
    num_decoder: int = 3          # paper ("3 SSA layers")
    num_memory: int = 16          # paper (M = 16)
    heads: int = 8                # paper
    coupling: float = 1.0         # paper (lambda = 1.0)
    num_queries: int = 3          # open: K, set to the dataset maximum
    expand: int = 1               # open: Mamba expansion factor
    share_weights: bool = True    # open: same-type layers share weights (matches ~2.1M parameters)
    links: int = 9
    subcarriers: int = 30
    packets: int = 20
    keypoints: int = 14
    coupling_bottleneck: int = 32  # open: width of the "learnable projection matrices"
    block_kind: str = ""            # kind of the Mamba blocks of the fusion, the neck and the decoder, "" = the same as s6_kind. The paper draws the S6 of DS3M behind a linear projection and calls the other blocks Mamba blocks
    s6_kind: str = "mamba"          # "mamba" = full Mamba block (in/out projections, conv, gate), "bare" = Eq. 6-8 only, "conv" = the input representation of Eq. 8 (a causal depthwise convolution after the linear projection, no gate, no output projection)
    dwt_threshold_init: float = 0.693  # tau = softplus(theta). 0.693 (theta = 0) zeroes every detail coefficient of min-max amplitude and never learns
    phase_regression: bool = True  # False skips the per-link linear fit (use with a cross-antenna sanitised cache)
    phase_unwrap: bool = False    # open: unwrap phase along subcarriers before the PhaseFi regression
    dt_rank: str = "full"         # open: rank of Linear_Delta in Eq. 8, "full" = d x d, "mamba" = ceil(d / 16) as the Mamba default (with unshared layers this fits the parameter steps of Table 6)
    dt_init: str = "mamba"        # open: "mamba" (step 0.001-0.1, built for long sequences) or "unit" (step ~0.7)
    dwt_levels: int = 1           # open: L of Eq. 10 (the paper gives no value), one threshold tau_l = softplus(theta_l) per level
    amp_minmax: bool = True       # True = per-sample min-max to [0, 1] before the DWT (the old runs), False = the DWT acts on the raw |CSI| as in Fig. 4 of the paper
    amp_scale: float = 1.0        # constant factor of the raw amplitude after the DWT, so that the projection sees O(1) values (the raw |CSI| of the cache is 0 to 64)
    ds3m_residual: bool = True    # False = z = W_f [y~A; y~phi] of the paper, the modulated outputs go to the fusion without the residual input
    outputs: int = 3              # coordinates per joint (3 for Person-in-WiFi 3D, 2 for Wi-Pose)
    phase_stream: bool = True     # False = amplitude only (Wi-Pose), no phase stream and no coupling
    phasefi_concat: bool = False  # open: Appendix C.3 sums the PhaseFi fit over k = 1 .. 60, True = the line is fitted on the 60 values [amplitude (30), phase (30)] concatenated and removed from the phase part, False = the fit on the 30 phase values of the link (Eq. 11 as written)
    token_layout: str = "packet"  # open: "link9" = 9 tokens, one per link with the 20 x 30 values of the link (the SSM scans 9 steps), "packet" = one token per packet with the 9 x 30 values (the SSM scans 20 steps), "link" = one token per (link, packet) with the 30 subcarriers, link-major (the SSM scans 180 steps, as the 180 tokens of Person-in-WiFi 3D)


class S6(nn.Module):
    """Selective SSM (Mamba block)."""

    def __init__(self, d_model, d_state=16, expand=1, dt_init="mamba"):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        di = self.d_inner
        self.in_proj = nn.Linear(d_model, 2 * di)
        self.conv = nn.Conv1d(di, di, 3, padding=2, groups=di)
        self.x_proj = nn.Linear(di, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, di)
        dt = torch.exp(torch.rand(di) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)) if dt_init == "mamba" else torch.zeros(di))  # inverse softplus, or softplus(0)=0.69
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(di, 1))
        self.Dp = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d_model)

    def forward(self, x, h0=None):
        """x [B,T,D] -> (out [B,T,D], mean-state [B,T,d_inner], last state [B,d_inner,N])."""
        t = x.shape[1]
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        xc = F.silu(self.conv(xi.transpose(1, 2))[..., :t]).transpose(1, 2)
        dt_in, bt, ct = torch.split(self.x_proj(xc), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt_in))
        y, hm, hlast = scan(delta, xc, -torch.exp(self.A_log), bt, ct, h0)
        y = (y + xc * self.Dp) * F.silu(z)
        return self.out_proj(y), hm, hlast


class BareS6(nn.Module):
    """The selective SSM exactly as Eq. 6-8 describe it. B_t, C_t and Delta_t are linear in the input, the scan runs
    on the input itself (no input or output projection, no convolution, no gate). About a third of the parameters of S6."""

    def __init__(self, d_model, d_state=16, expand=1, dt_init="mamba"):
        super().__init__()
        self.d_inner, self.d_state = d_model, d_state
        self.lin_b = nn.Linear(d_model, d_state, bias=False)
        self.lin_c = nn.Linear(d_model, d_state, bias=False)
        r = math.ceil(d_model / 16) if _DT_RANK == "mamba" else None
        self.lin_dt_down = nn.Linear(d_model, r, bias=False) if r else nn.Identity()  # the low rank Delta of Mamba, Delta = softplus(W_up W_down x + b)
        self.lin_dt = nn.Linear(r or d_model, d_model)
        dt = torch.exp(torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.lin_dt.bias.copy_(dt + torch.log(-torch.expm1(-dt)) if dt_init == "mamba" else torch.zeros(d_model))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_model, 1))
        self.Dp = nn.Parameter(torch.ones(d_model))

    def forward(self, x, h0=None):
        y, hm, hlast = scan(F.softplus(self.lin_dt(self.lin_dt_down(x))), x, -torch.exp(self.A_log), self.lin_b(x), self.lin_c(x), h0)
        return y + x * self.Dp, hm, hlast


class ConvS6(BareS6):
    """Eq. 6-8 with the input representation of the paper text, "x_t obtained after linear projection and convolution": the linear projection is the stream projection in front of the block,
    here a causal depthwise convolution (kernel 4, the Mamba default) and SiLU, then B_t, C_t, Delta_t from x_t. No gate and no output projection, so the block has about the size of BareS6."""

    def __init__(self, d_model, d_state=16, expand=1, dt_init="mamba", d_conv=4):
        super().__init__(d_model, d_state, expand, dt_init)
        self.conv = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)

    def forward(self, x, h0=None):
        t = x.shape[1]
        xc = F.silu(self.conv(x.transpose(1, 2))[..., :t]).transpose(1, 2)
        y, hm, hlast = scan(F.softplus(self.lin_dt(self.lin_dt_down(xc))), xc, -torch.exp(self.A_log), self.lin_b(xc), self.lin_c(xc), h0)
        return y + xc * self.Dp, hm, hlast


_DT_RANK = "full"  # set from FaithfulConfig.dt_rank by WiFiMamba before its layers are built (BareS6 and ConvS6 read it)


def make_s6(d_model, d_state, expand, dt_init, kind):
    return {"bare": BareS6, "conv": ConvS6}.get(kind, S6)(d_model, d_state, expand, dt_init)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=1, dt_init="mamba", kind="mamba"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s6 = make_s6(d_model, d_state, expand, dt_init, kind)

    def forward(self, x, h0=None):
        y, _, h = self.s6(self.norm(x), h0)
        return x + y, h


def _low_rank(d, r, d_out=None):
    return nn.Sequential(nn.Linear(d, r, bias=False), nn.Linear(r, d_out or d, bias=False))


class DS3M(nn.Module):
    """Dual-stream S3M layer with per-step cross-stream state coupling (Eq. 12-15)."""

    def __init__(self, cfg):
        super().__init__()
        d, r = cfg.d_model, cfg.coupling_bottleneck
        self.lam = cfg.coupling
        self.residual = cfg.ds3m_residual
        self.norm_a, self.norm_p = nn.LayerNorm(d), nn.LayerNorm(d)
        self.s6_a, self.s6_p = (make_s6(d, cfg.d_state, cfg.expand, cfg.dt_init, cfg.s6_kind) for _ in range(2))
        di = self.s6_a.d_inner  # the width of the mean state, d_model for the bare and conv S6, expand * d_model for the full block
        self.w_p2a, self.w_a2p = _low_rank(di, r), _low_rank(di, r)   # W_{phi->A}, W_{A->phi}
        self.w_mod_a, self.w_mod_p = _low_rank(di, r, d), _low_rank(di, r, d)  # W_m^A, W_m^phi, the modulation has the width of the stream output
        self.w_g = nn.Linear(2 * di, di)                               # W_g
        for m in (self.w_p2a, self.w_a2p, self.w_mod_a, self.w_mod_p):
            nn.init.zeros_(m[-1].weight)  # coupling projections are zero-initialised (paper)

    def forward(self, xa, xp):
        ya, ha, _ = self.s6_a(self.norm_a(xa))
        yp, hp, _ = self.s6_p(self.norm_p(xp))
        g = torch.sigmoid(self.w_g(torch.cat([ha, hp], -1)))              # Eq. 14, per step
        ha_c = ha + self.lam * g * self.w_p2a(hp)                          # Eq. 12
        hp_c = hp + self.lam * (1 - g) * self.w_a2p(ha)                    # Eq. 13
        ya = ya * (1 + self.lam * torch.sigmoid(self.w_mod_a(ha_c)))       # Eq. 15
        yp = yp * (1 + self.lam * torch.sigmoid(self.w_mod_p(hp_c)))
        return (xa + ya, xp + yp) if self.residual else (ya, yp)


class AmpS3M(nn.Module):
    """The amplitude stream of DS3M alone (an amplitude only dataset such as Wi-Pose), no phase stream and no coupling."""

    def __init__(self, cfg):
        super().__init__()
        self.residual = cfg.ds3m_residual
        self.norm = nn.LayerNorm(cfg.d_model)
        self.s6 = make_s6(cfg.d_model, cfg.d_state, cfg.expand, cfg.dt_init, cfg.s6_kind)

    def forward(self, xa, xp=None):
        ya, _, _ = self.s6(self.norm(xa))
        return (xa + ya if self.residual else ya), None


class LearnableHaarDenoise(nn.Module):
    """Haar DWT with L levels, soft-thresholded detail coefficients with one learnable threshold tau_l = softplus(theta_l) per level, inverse DWT (Eq. 10, Appendix C.2).
    A level with an odd length repeats the last value before the pair split and drops it after the inverse."""

    def __init__(self, threshold_init=0.693, levels=1):
        super().__init__()
        self.levels = levels
        self.theta = nn.Parameter(torch.full((levels,), math.log(math.expm1(threshold_init))))  # tau = softplus(theta)

    def forward(self, x):  # [..., L]
        r, c, his, lens = 0.7071067811865476, x, [], []
        tau = F.softplus(self.theta)
        for l in range(self.levels):
            lens.append(c.shape[-1])
            if c.shape[-1] % 2:
                c = torch.cat([c, c[..., -1:]], -1)
            e, o = c[..., 0::2], c[..., 1::2]
            hi = (o - e) * r
            his.append(torch.sign(hi) * F.relu(hi.abs() - tau[l]))
            c = (e + o) * r
        for l in reversed(range(self.levels)):
            c = torch.stack([(c - his[l]) * r, (c + his[l]) * r], -1).flatten(-2)[..., :lens[l]]
        return c


def unwrap(ph):
    """Unwrap phase along the last axis."""
    d = ph[..., 1:] - ph[..., :-1]
    d = d - 2 * math.pi * torch.round(d / (2 * math.pi))
    return torch.cat([ph[..., :1], ph[..., :1] + d.cumsum(-1)], -1)


class PhaseFi(nn.Module):
    """Remove the linear CFO/SFO trend across subcarriers by least squares (Eq. 11), then refine."""

    def __init__(self, k, hidden=32, regression=True, concat=False):
        super().__init__()
        self.regression, self.concat, self.k = regression, concat, k
        n = 2 * k if concat else k  # the fit runs over k = 1 .. 60 on [amplitude, phase] when concat, as the sum of Eq. 30
        self.register_buffer("idx", torch.arange(n, dtype=torch.float32) - (n - 1) / 2)
        self.refine = nn.Sequential(nn.Linear(k, hidden), nn.ReLU(), nn.Linear(hidden, k))
        nn.init.zeros_(self.refine[-1].weight)  # the residual refinement starts as the identity
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, ph, amp=None):  # [..., K], amp [..., K] when concat
        if not self.regression:
            return ph + self.refine(ph)
        if self.concat:  # least squares of the line alpha k + beta over the 60 values [amplitude, phase], removed from the phase part
            v = torch.cat([amp, ph], -1)
            mean = v.mean(-1, keepdim=True)
            alpha = ((v - mean) * self.idx).sum(-1, keepdim=True) / (self.idx ** 2).sum()
            c = (v - (alpha * self.idx + mean))[..., self.k:]
            return c + self.refine(c)
        mean = ph.mean(-1, keepdim=True)
        alpha = ((ph - mean) * self.idx).sum(-1, keepdim=True) / (self.idx ** 2).sum()
        c = ph - (alpha * self.idx + mean)
        return c + self.refine(c)


class SSALayer(nn.Module):
    """Query SSM -> cross-attention (Eq. 16-17) -> persistent memory SSM with gate (Eq. 18-20)."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.query_ssm = MambaBlock(d, cfg.d_state, cfg.expand, cfg.dt_init, cfg.block_kind or cfg.s6_kind)
        self.attn = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.memory = nn.Parameter(torch.randn(cfg.num_memory, d) * 0.02)
        self.mem_ssm = MambaBlock(d, cfg.d_state, cfg.expand, cfg.dt_init, cfg.block_kind or cfg.s6_kind)
        self.gate = nn.Linear(2 * d, d)
        self.num_memory = cfg.num_memory

    def forward(self, q, enc, h_prev):
        qs, _ = self.query_ssm(q)                                    # Eq. 16
        q = self.norm(q + self.attn(qs, enc, enc, need_weights=False)[0])  # Eq. 17
        b, k, _ = q.shape
        phi, h_new = self.mem_ssm(torch.cat([self.memory.expand(b, -1, -1), q], 1), h_prev)  # Eq. 18-19
        g = torch.sigmoid(self.gate(torch.cat([q, phi[:, : self.num_memory].mean(1, keepdim=True).expand(-1, k, -1)], -1)))
        return g * phi[:, self.num_memory:] + (1 - g) * q, h_new     # Eq. 20


class WiFiMamba(nn.Module):
    def __init__(self, cfg: FaithfulConfig | None = None):
        super().__init__()
        cfg = cfg or FaithfulConfig()
        self.cfg = cfg
        global _DT_RANK
        _DT_RANK = cfg.dt_rank
        d = cfg.d_model
        self.link_tokens = cfg.token_layout == "link"
        self.link9 = cfg.token_layout == "link9"  # 9 tokens, one per link, the 20 packets x 30 subcarriers of the link flattened (the SSM scans the 9 links)
        in_dim = cfg.subcarriers if self.link_tokens else (cfg.packets * cfg.subcarriers if self.link9 else cfg.links * cfg.subcarriers)
        self.denoise = LearnableHaarDenoise(cfg.dwt_threshold_init, cfg.dwt_levels)
        self.phasefi = PhaseFi(cfg.subcarriers, regression=cfg.phase_regression, concat=cfg.phasefi_concat)
        self.amp_proj = nn.Linear(in_dim, d)
        self.phase_proj = nn.Linear(in_dim, d) if cfg.phase_stream else None
        if self.link9:  # the projection is shared by the 9 tokens, the link is given by an embedding
            self.link_embed = nn.Parameter(torch.randn(cfg.links, 1, d) * 0.02)
        if self.link_tokens:  # the projection is shared by the 180 tokens, so the link and the packet are given by embeddings
            self.link_embed = nn.Parameter(torch.randn(cfg.links, 1, d) * 0.02)
            self.time_embed = nn.Parameter(torch.randn(1, cfg.packets, d) * 0.02)

        def stack(make, n):
            return nn.ModuleList([make() for _ in range(1 if cfg.share_weights else n)])

        self.ds3m = stack(lambda: DS3M(cfg) if cfg.phase_stream else AmpS3M(cfg), cfg.num_ds3m)
        self.fuse_proj = nn.Linear((2 if cfg.phase_stream else 1) * d, d)         # W_f
        self.fusion = stack(lambda: MambaBlock(d, cfg.d_state, cfg.expand, cfg.dt_init, cfg.block_kind or cfg.s6_kind), cfg.num_fusion)
        self.neck = stack(lambda: MambaBlock(d, cfg.d_state, cfg.expand, cfg.dt_init, cfg.block_kind or cfg.s6_kind), cfg.num_neck)
        self.query_embed = nn.Parameter(torch.randn(1, cfg.num_queries, d) * 0.02)
        self.decoder = stack(lambda: SSALayer(cfg), cfg.num_decoder)
        self.pose_head = nn.Linear(d, cfg.keypoints * cfg.outputs)
        self.conf_head = nn.Linear(d, 1)
        self.count_head = nn.Linear(d, 4)  # 0..3 persons

    def preprocess(self, amp, phase):
        """amp, phase [B,L,S,T] raw |CSI| and wrapped phase -> token sequences. phase is None for an amplitude only dataset (the second sequence is then None)."""
        if self.cfg.amp_minmax:
            lo = amp.amin((1, 2, 3), keepdim=True)
            amp = (amp - lo) / (amp.amax((1, 2, 3), keepdim=True) - lo + 1e-8)   # per-sample min-max (official feeder)
        amp_raw = amp.transpose(2, 3) * self.cfg.amp_scale  # [B, L, T, S] the amplitude before the DWT, the second half of the 60 values of the concatenated PhaseFi fit
        amp = self.denoise(amp.transpose(2, 3)).transpose(2, 3) * self.cfg.amp_scale  # along subcarriers
        b, l, s, t = amp.shape
        if self.link9:  # [B, L, T * S], one token per link
            tok = lambda z: z.permute(0, 1, 3, 2).reshape(b, l, t * s)  # noqa: E731
        else:
            tok = (lambda z: z.permute(0, 1, 3, 2).reshape(b, l * t, s)) if self.link_tokens else (lambda z: z.permute(0, 3, 1, 2).reshape(b, t, l * s))  # link-major [B, L * T, S] or [B, T, L * S]
        if phase is None:
            return tok(amp), None
        phase = phase.transpose(2, 3)
        phase = self.phasefi(unwrap(phase) if self.cfg.phase_unwrap else phase, amp_raw if self.cfg.phasefi_concat else None).transpose(2, 3)
        return tok(amp), tok(phase)

    def encode(self, amp, phase):
        xa, xp = self.preprocess(amp, phase if self.cfg.phase_stream else None)
        xa = self.amp_proj(xa)
        xp = None if xp is None else self.phase_proj(xp)
        if self.link9:
            pos = self.link_embed.reshape(1, -1, xa.shape[-1])
            xa, xp = xa + pos, None if xp is None else xp + pos
        if self.link_tokens:
            pos = (self.link_embed + self.time_embed).reshape(1, -1, xa.shape[-1])
            xa, xp = xa + pos, None if xp is None else xp + pos
        for i in range(self.cfg.num_ds3m):
            xa, xp = self.ds3m[i % len(self.ds3m)](xa, xp)
        z = self.fuse_proj(torch.cat([xa, xp], -1) if xp is not None else xa)
        for i in range(self.cfg.num_fusion):
            z, _ = self.fusion[i % len(self.fusion)](z)
        for i in range(self.cfg.num_neck):
            z, _ = self.neck[i % len(self.neck)](z)
        return z

    def init_state(self, batch, device):
        di = self.cfg.d_model * self.cfg.expand
        return [torch.zeros(batch, di, self.cfg.d_state, device=device) for _ in range(self.cfg.num_decoder)]

    def forward(self, amp, phase, state=None):
        """Returns pose [B,K,J,3], confidence logits [B,K], count logits [B,4] and the new
        persistent-memory states (one per decoder layer) for the next frame."""
        b = amp.shape[0]
        enc = self.encode(amp, phase)
        state = state if state is not None else self.init_state(b, amp.device)
        q = self.query_embed.expand(b, -1, -1)
        new_state = []
        for i in range(self.cfg.num_decoder):
            q, h = self.decoder[i % len(self.decoder)](q, enc, state[i])
            new_state.append(h)
        pose = self.pose_head(q).view(b, -1, self.cfg.keypoints, self.cfg.outputs)
        return pose, self.conf_head(q).squeeze(-1), self.count_head(q.mean(1)), new_state
