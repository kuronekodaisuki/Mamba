"""Mamba2 student LM for Arria 10, pure PyTorch, with hardware-aware QAT.

Pure PyTorch (no mamba_ssm kernels) so that the forward pass can reproduce what
the FPGA computes:
  * linear layers : online Hadamard on the input (FWHT, adds only), weights
                    pre-rotated, W4 per-group, A8 per-token with a power-of-two
                    scale (a shift in hardware)
  * conv1d        : 8-bit per-channel weights
  * softplus/exp  : FastMamba-style NAU approximation ("published"/"improved"),
                    straight-through gradient
  * embedding/head: kept in DDR4, 8-bit by default
The SSM recurrence itself is computed in float (chunked SSD form); the RTL keeps
the state in 24-bit fixed point, whose error is checked with model/ssm_golden.py.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StudentConfig:
    vocab_size: int = 50288
    d_model: int = 256
    n_layer: int = 12
    d_state: int = 64
    headdim: int = 64
    expand: int = 2
    d_conv: int = 4
    chunk: int = 64
    # quantization-aware training (stage 2)
    quant: bool = False
    w_bits: int = 4
    w_group: int = 128
    a_bits: int = 8
    conv_bits: int = 8
    emb_bits: int = 8
    hadamard: bool = True
    nau: str = "off"            # "off" | "published" | "improved"

    @property
    def d_inner(self):
        return self.expand * self.d_model

    @property
    def nheads(self):
        return self.d_inner // self.headdim

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------
# fake quantization (straight-through estimator)
# ----------------------------------------------------------------------------
def ste(x, xq):
    return x + (xq - x).detach()


def fq_weight(w, bits, group):
    """Symmetric per-group quantization along the input dimension."""
    shape = w.shape
    w32 = w.float().reshape(shape[0], -1, group)
    qmax = 2 ** (bits - 1) - 1
    s = w32.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    q = (w32 / s).round().clamp(-qmax - 1, qmax) * s
    return ste(w, q.reshape(shape).to(w.dtype))


def fq_act(x, bits):
    """Per-token symmetric quantization with a power-of-two scale."""
    x32 = x.float()
    qmax = 2 ** (bits - 1) - 1
    s = x32.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    s = torch.exp2(torch.ceil(torch.log2(s)))
    q = (x32 / s).round().clamp(-qmax - 1, qmax) * s
    return ste(x, q.to(x.dtype))


def fwht(x):
    """Normalized fast Walsh-Hadamard transform along the last dim (power of 2)."""
    n = x.shape[-1]
    assert n & (n - 1) == 0, "Hadamard size must be a power of two"
    shape = x.shape
    y = x.reshape(-1, n)
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2)
        h *= 2
    return y.reshape(shape) / math.sqrt(n)


# ----------------------------------------------------------------------------
# FastMamba-style nonlinear approximation (same coefficients as rtl/nau.sv)
# ----------------------------------------------------------------------------
_C0 = [65504, 71432, 77898, 84948, 92636, 101021, 110164, 120135]
_C1 = [47452, 51747, 56430, 61538, 67107, 73181, 79805, 87028]
_LOG2E_Q = 23637 / 2 ** 14


def _nau_exp(x, improved):
    """x <= 0. Float emulation of the NAU exp (approximation error, not LSBs)."""
    c0 = torch.tensor(_C0, device=x.device, dtype=torch.float32) / 65536
    c1 = torch.tensor(_C1, device=x.device, dtype=torch.float32) / 65536
    t = x.float() * _LOG2E_Q
    u = torch.floor(t)
    f = t - u
    k = torch.clamp(torch.floor(f * 8), 0, 7).long()
    y = (c0[k] + c1[k] * (f - k.float() / 8)) * torch.exp2(u)
    if improved:
        xf = x.float()
        y = torch.where(xf > -1 / 16, 1 + xf + 0.5 * xf * xf, y)
    return y


def hw_exp(x, mode):
    if mode == "off":
        return torch.exp(x)
    return ste(torch.exp(x), _nau_exp(x.clamp(max=0), mode == "improved").to(x.dtype))


def hw_softplus(x, mode):
    if mode == "off":
        return F.softplus(x)
    y = _nau_exp(-x.abs(), mode == "improved")
    if mode == "improved":
        y = y - 0.5 * y * y
    return ste(F.softplus(x), (y + x.float().clamp(min=0)).to(x.dtype))


# ----------------------------------------------------------------------------
# SSM: chunked SSD (equals the recurrence h_t = exp(dtA_t) h + x_t B_t^T)
# ----------------------------------------------------------------------------
def _segsum(a):
    """a (..., L) -> (..., L, L) with sum a[j+1..i] for i >= j, -inf above."""
    L = a.shape[-1]
    cs = torch.cumsum(a, dim=-1)
    s = cs[..., :, None] - cs[..., None, :]
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=a.device))
    return s.masked_fill(~mask, float("-inf"))


def ssd_chunked(x, dtA, B, C, chunk):
    """x (b,l,h,p) already multiplied by dt, dtA (b,l,h), B/C (b,l,n)."""
    b, l, h, p = x.shape
    n = B.shape[-1]
    nc = l // chunk
    X = x.reshape(b, nc, chunk, h, p)
    A = dtA.reshape(b, nc, chunk, h).permute(0, 3, 1, 2)          # b h c l
    Bc = B.reshape(b, nc, chunk, n)
    Cc = C.reshape(b, nc, chunk, n)
    Acs = torch.cumsum(A, dim=-1)
    Lm = torch.exp(_segsum(A))                                      # b h c l s
    # Every einsum below has two operands, so no large (b,c,l,s,h,p)-sized
    # intermediate is created (a 3-4 operand einsum may contract in that order).
    CB = torch.einsum("bcln,bcsn->bcls", Cc, Bc)                    # b c l s
    M = Lm * CB[:, None]                                            # b h c l s
    y_diag = torch.einsum("bhcls,bcshp->bclhp", M, X)
    decay = torch.exp(Acs[..., -1:] - Acs)                          # b h c l
    Xd = X * decay.permute(0, 2, 3, 1)[..., None]                   # b c l h p
    states = torch.einsum("bcln,bclhp->bchpn", Bc, Xd)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    dchunk = torch.exp(_segsum(F.pad(Acs[..., -1], (1, 0))))        # b h c+1 c+1
    new_states = torch.einsum("bhzc,bchpn->bzhpn", dchunk, states)
    y_off = torch.einsum("bcln,bchpn->bclhp", Cc, new_states[:, :-1])
    y_off = y_off * torch.exp(Acs).permute(0, 2, 3, 1)[..., None]
    return (y_diag + y_off).reshape(b, l, h, p)


# ----------------------------------------------------------------------------
# layers
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class QLinear(nn.Linear):
    def __init__(self, d_in, d_out, cfg):
        super().__init__(d_in, d_out, bias=False)
        self.cfg = cfg

    def forward(self, x):
        c = self.cfg
        if not c.quant:
            return F.linear(x, self.weight)
        w = self.weight
        if c.hadamard:                       # y = (W H)(H x), H symmetric, H H = I
            x, w = fwht(x), fwht(w)
        return F.linear(fq_act(x, c.a_bits), fq_weight(w, c.w_bits, c.w_group))


class Mamba2Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        di, N, H = cfg.d_inner, cfg.d_state, cfg.nheads
        self.conv_dim = di + 2 * N
        self.in_proj = QLinear(cfg.d_model, 2 * di + 2 * N + H, cfg)
        self.conv = nn.Conv1d(self.conv_dim, self.conv_dim, cfg.d_conv,
                              groups=self.conv_dim, padding=cfg.d_conv - 1, bias=True)
        dt = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        dt = dt.clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # softplus^-1(dt)
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(H))
        self.norm_w = nn.Parameter(torch.ones(di))
        self.out_proj = QLinear(di, cfg.d_model, cfg)

    def forward(self, u):
        c = self.cfg
        b, l, _ = u.shape
        di, N, H, P = c.d_inner, c.d_state, c.nheads, c.headdim

        z, xBC, dt = torch.split(self.in_proj(u), [di, di + 2 * N, H], dim=-1)

        w = self.conv.weight
        if c.quant:
            w = fq_weight(w.reshape(self.conv_dim, c.d_conv), c.conv_bits, c.d_conv)
            w = w.reshape(self.conv_dim, 1, c.d_conv)
        xBC = F.conv1d(xBC.transpose(1, 2), w, self.conv.bias,
                       padding=c.d_conv - 1, groups=self.conv_dim)[..., :l].transpose(1, 2)
        xBC = F.silu(xBC)
        x, B, C = torch.split(xBC, [di, N, N], dim=-1)

        dt = hw_softplus(dt + self.dt_bias, c.nau)                  # (b,l,H)
        dtA = dt * (-torch.exp(self.A_log.float()))
        if c.nau != "off":                                          # product of NAU exp values
            dtA = ste(dtA, torch.log(hw_exp(dtA, c.nau).float().clamp(min=1e-30)))

        x = x.reshape(b, l, H, P)
        pad = (-l) % c.chunk
        xs, dA, Bs, Cs = x * dt[..., None], dtA, B, C
        if pad:
            xs = F.pad(xs, (0, 0, 0, 0, 0, pad))
            dA = F.pad(dA, (0, 0, 0, pad))
            Bs = F.pad(Bs, (0, 0, 0, pad))
            Cs = F.pad(Cs, (0, 0, 0, pad))
        with torch.autocast(device_type=u.device.type, enabled=False):   # keep SSD in fp32
            y = ssd_chunked(xs.float(), dA.float(), Bs.float(), Cs.float(), c.chunk)[:, :l]
        y = (y + x.float() * self.D.float()[:, None]).to(u.dtype).reshape(b, l, di)

        y = y * F.silu(z)                                           # gated RMSNorm
        y32 = y.float()
        y = (y32 * torch.rsqrt(y32.pow(2).mean(-1, keepdim=True) + 1e-5)).to(u.dtype) * self.norm_w
        return self.out_proj(y)


class StudentLM(nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.norms = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(cfg.n_layer)])
        self.layers = nn.ModuleList([Mamba2Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        self._init()

    def _init(self):
        nn.init.normal_(self.emb.weight, std=0.02)
        for layer in self.layers:
            nn.init.normal_(layer.in_proj.weight, std=0.02)
            nn.init.normal_(layer.out_proj.weight, std=0.02 / math.sqrt(2 * self.cfg.n_layer))

    def emb_weight(self):
        w = self.emb.weight
        return fq_weight(w, self.cfg.emb_bits, 128) if self.cfg.quant else w

    def forward(self, ids):
        W = self.emb_weight()
        h = F.embedding(ids, W)
        for norm, layer in zip(self.norms, self.layers):
            h = h + layer(norm(h))
        h = self.norm_f(h)
        if self.cfg.quant:
            h = fq_act(h, self.cfg.a_bits)
        return F.linear(h, W)                                       # tied LM head
