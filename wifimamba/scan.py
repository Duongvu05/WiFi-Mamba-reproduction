"""Selective scan of the S6 blocks.

scan(delta, x, A, Bt, Ct, h0) returns (y, hm, hlast)
    delta, x [B,T,D]  A [D,N]  Bt, Ct [B,T,N]  h0 [B,D,N] or None
    y   [B,T,D]   sum_n C_t[n] h_t[n]
    hm  [B,T,D]   mean over the state axis of h_t (used by the cross-stream coupling of DS3M)
    hlast [B,D,N] final state
The recurrence is h_t = exp(delta_t A) h_{t-1} + delta_t x_t B_t.

On a CUDA tensor with Triton installed the scan runs on the Triton kernel below (forward and backward). A program owns a block of
channels of one sequence and keeps the state in registers. The forward saves the states h_t [B,T,D,N], the backward walks backwards with
the gradient of the state in registers. B and C are shared by the channels, their gradients are summed over the channel blocks by atomic
adds, so the gradients of Bt and Ct are not bit-for-bit reproducible. Otherwise the plain PyTorch loop `scan_reference` is used (slow).
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:

    @triton.jit
    def _fwd(delta, x, A, Bt, Ct, h0, y, hs, T, D, N: tl.constexpr, BD: tl.constexpr, HAS_H0: tl.constexpr):
        b = tl.program_id(0)
        offs_d = tl.program_id(1) * BD + tl.arange(0, BD)
        offs_n = tl.arange(0, N)
        md = offs_d < D
        A_t = tl.load(A + offs_d[:, None] * N + offs_n[None, :], mask=md[:, None], other=0.0)
        if HAS_H0:
            h = tl.load(h0 + (b * D + offs_d[:, None]) * N + offs_n[None, :], mask=md[:, None], other=0.0)
        else:
            h = tl.zeros([BD, N], dtype=tl.float32)
        for t in range(T):
            base = (b * T + t).to(tl.int64)
            dt_t = tl.load(delta + base * D + offs_d, mask=md, other=0.0)
            x_t = tl.load(x + base * D + offs_d, mask=md, other=0.0)
            B_t = tl.load(Bt + base * N + offs_n)
            C_t = tl.load(Ct + base * N + offs_n)
            h = tl.exp(dt_t[:, None] * A_t) * h + (dt_t * x_t)[:, None] * B_t[None, :]
            tl.store(y + base * D + offs_d, tl.sum(h * C_t[None, :], axis=1), mask=md)
            tl.store(hs + (base * D + offs_d[:, None]) * N + offs_n[None, :], h, mask=md[:, None])

    @triton.jit
    def _bwd(delta, x, A, Bt, Ct, h0, hs, gy, ghm, ghl, gdelta, gx, gA, gB, gC, gh0, T, D, N: tl.constexpr, BD: tl.constexpr,
             HAS_H0: tl.constexpr, HAS_GHL: tl.constexpr):
        b = tl.program_id(0)
        offs_d = tl.program_id(1) * BD + tl.arange(0, BD)
        offs_n = tl.arange(0, N)
        md = offs_d < D
        A_t = tl.load(A + offs_d[:, None] * N + offs_n[None, :], mask=md[:, None], other=0.0)
        if HAS_GHL:
            dh = tl.load(ghl + (b * D + offs_d[:, None]) * N + offs_n[None, :], mask=md[:, None], other=0.0)
        else:
            dh = tl.zeros([BD, N], dtype=tl.float32)
        gA_acc = tl.zeros([BD, N], dtype=tl.float32)
        for i in range(T):
            t = T - 1 - i
            base = (b * T + t).to(tl.int64)
            dt_t = tl.load(delta + base * D + offs_d, mask=md, other=0.0)
            x_t = tl.load(x + base * D + offs_d, mask=md, other=0.0)
            gy_t = tl.load(gy + base * D + offs_d, mask=md, other=0.0)
            ghm_t = tl.load(ghm + base * D + offs_d, mask=md, other=0.0)
            B_t = tl.load(Bt + base * N + offs_n)
            C_t = tl.load(Ct + base * N + offs_n)
            h_t = tl.load(hs + (base * D + offs_d[:, None]) * N + offs_n[None, :], mask=md[:, None], other=0.0)
            h_prev = tl.load(hs + ((base - 1) * D + offs_d[:, None]) * N + offs_n[None, :], mask=md[:, None] & (t > 0), other=0.0)
            if HAS_H0:
                h_prev += tl.load(h0 + (b * D + offs_d[:, None]) * N + offs_n[None, :], mask=md[:, None] & (t == 0), other=0.0)
            dh += gy_t[:, None] * C_t[None, :] + (ghm_t / N)[:, None]  # the gradient of h_t, the part that comes from the later steps is already inside
            a = tl.exp(dt_t[:, None] * A_t)
            dB = tl.sum(dh * B_t[None, :], axis=1)
            tl.store(gx + base * D + offs_d, dt_t * dB, mask=md)
            tl.store(gdelta + base * D + offs_d, tl.sum(dh * h_prev * a * A_t, axis=1) + dB * x_t, mask=md)
            gA_acc += dh * h_prev * a * dt_t[:, None]
            tl.atomic_add(gB + base * N + offs_n, tl.sum(dh * (dt_t * x_t)[:, None], axis=0))
            tl.atomic_add(gC + base * N + offs_n, tl.sum(gy_t[:, None] * h_t, axis=0))
            dh = a * dh
        tl.atomic_add(gA + offs_d[:, None] * N + offs_n[None, :], gA_acc, mask=md[:, None])
        if HAS_H0:
            tl.store(gh0 + (b * D + offs_d[:, None]) * N + offs_n[None, :], dh, mask=md[:, None])


_BD = 32


class _TritonScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, delta, x, A, Bt, Ct, h0):
        delta, x, A, Bt, Ct = [z.contiguous().float() for z in (delta, x, A, Bt, Ct)]
        h0c = h0.contiguous().float() if h0 is not None else None
        b, t, d = x.shape
        n = A.shape[1]
        y = torch.empty_like(x)
        hs = torch.empty(b, t, d, n, device=x.device)
        _fwd[(b, triton.cdiv(d, _BD))](delta, x, A, Bt, Ct, h0c if h0c is not None else x, y, hs, t, d, N=n, BD=_BD, HAS_H0=h0c is not None, num_warps=4)
        ctx.save_for_backward(delta, x, A, Bt, Ct, hs, *([h0c] if h0c is not None else []))
        ctx.has_h0 = h0c is not None
        return y, hs.mean(-1), hs[:, -1].contiguous()

    @staticmethod
    def backward(ctx, gy, ghm, ghlast):
        delta, x, A, Bt, Ct, hs, *rest = ctx.saved_tensors
        h0 = rest[0] if ctx.has_h0 else None
        b, t, d = x.shape
        n = A.shape[1]
        z = lambda g: torch.zeros_like(x) if g is None else g.contiguous().float()
        gy, ghm = z(gy), z(ghm)
        ghl = None if ghlast is None else ghlast.contiguous().float()
        gdelta, gx = torch.empty_like(x), torch.empty_like(x)
        gA, gB, gC = torch.zeros_like(A), torch.zeros_like(Bt), torch.zeros_like(Ct)
        gh0 = torch.empty(b, d, n, device=x.device) if ctx.has_h0 else None
        _bwd[(b, triton.cdiv(d, _BD))](delta, x, A, Bt, Ct, h0 if h0 is not None else x, hs, gy, ghm, ghl if ghl is not None else x, gdelta, gx, gA, gB, gC,
                                       gh0 if gh0 is not None else x, t, d, N=n, BD=_BD, HAS_H0=h0 is not None, HAS_GHL=ghl is not None, num_warps=4)
        return gdelta, gx, gA, gB, gC, gh0


def scan_reference(delta, x, A, Bt, Ct, h0=None):
    """The same recurrence as a PyTorch loop over time (autograd does the backward)."""
    b, t, d = x.shape
    n = A.shape[1]
    h = x.new_zeros(b, d, n) if h0 is None else h0
    ys, hms = [], []
    for i in range(t):
        h = torch.exp(delta[:, i].unsqueeze(-1) * A) * h + (delta[:, i] * x[:, i]).unsqueeze(-1) * Bt[:, i].unsqueeze(1)
        ys.append((h * Ct[:, i].unsqueeze(1)).sum(-1))
        hms.append(h.mean(-1))
    return torch.stack(ys, 1), torch.stack(hms, 1), h


def scan(delta, x, A, Bt, Ct, h0=None):
    if x.is_cuda and triton is not None:
        return _TritonScan.apply(delta, x, A, Bt, Ct, h0)
    return scan_reference(delta, x, A, Bt, Ct, h0)
