#!/usr/bin/env python3
"""Bit-exact golden model of the fixed-point vector path (no float at run time).

Formats (integers, value = int * 2^-F):
  residual h, gated y, norm input   : W_R = 32, F_R = 16
  in_proj output (z, xBC, dt)       : W_LI = 24, F_LI = 12
  out_proj output                   : W_LO = 32, F_LO = 16
  conv output, SiLU output          : W_LI, F_LI
  norm weights                      : W_NW = 18, F_NW = 12
  sigmoid table output              : Q.16 (unsigned, <= 2^16)
  rsqrt table input / output        : m in [1,4) as Q2.14, r in (0.5,1] as Q.16

Functions (all operate on int64 numpy arrays):
  lin_fixed(acc, s16, k, F, W)  linear-layer row results, exact sum then one rounding
  norm_q8(v, w)                 RMSNorm + per-token int8 quantization -> (q, k)
  silu_fx(x)                    SiLU on Q.12
  conv_fx(x, wq, s16, bq)       depthwise causal conv1d (4 taps) on Q.12
  gate_fx(y, z)                 y (SSM output, Q.16) * SiLU(z) -> Q.16
  embed_fx(q, s16)              embedding row -> residual format

  python3 model/vec_golden.py          # accuracy report of the approximations
"""
import numpy as np

W_R, F_R = 32, 16
W_LI, F_LI = 24, 12
W_LO, F_LO = 32, 16
W_NW, F_NW = 18, 12
EPS = 1e-5


def sat(v, w):
    return np.clip(v, -(1 << (w - 1)), (1 << (w - 1)) - 1)


def rsr(v, s):
    """Arithmetic right shift with round-half-up; s may be an array, s >= 1."""
    s = np.asarray(s, np.int64)
    return (v + (np.int64(1) << (s - 1))) >> s


def shift_round(v, s):
    """v * 2^s rounded half-up (s < 0) or exact (s >= 0); s int array or scalar."""
    v = np.asarray(v, np.int64)
    s = np.broadcast_to(np.asarray(s, np.int64), v.shape)
    left = v << np.maximum(s, 0)
    right = rsr(v, np.maximum(-s, 1))
    return np.where(s >= 0, left, right)


def fp16_fields(s16):
    b = np.asarray(s16, np.float16).view(np.uint16).astype(np.int64)
    sign, E, F = (b >> 15) & 1, (b >> 10) & 31, b & 1023
    m = np.where(E == 0, F, F | 1024) * np.where(sign == 1, -1, 1)
    sh = np.where(E == 0, 0, E - 1)          # s = m * 2^(sh - 24)
    return m, sh


def to_fixed(x, w, f):
    return sat(np.round(np.asarray(x, np.float64) * (1 << f)).astype(np.int64), w)


# ----------------------------------------------------------------------------
# piecewise-linear tables (chord shifted by half its max error, like the NAU)
# ----------------------------------------------------------------------------
def _pwl(fn, lo, hi, nseg, f_out):
    c0, c1 = [], []
    w = (hi - lo) / nseg
    for i in range(nseg):
        a, b = lo + i * w, lo + (i + 1) * w
        slope = (fn(b) - fn(a)) / w
        t = np.linspace(a, b, 1001)
        err = fn(a) + slope * (t - a) - fn(t)
        off = (err.max() + err.min()) / 2
        c0.append(int(round((fn(a) - off) * (1 << f_out))))
        c1.append(int(round(slope * w * (1 << f_out))))   # increment over one segment
    return np.array(c0, np.int64), np.array(c1, np.int64)


SIG_SEGS = 128                   # 128 segments on [0, 16), width 1/8
SIG_FRAC = F_LI - 3              # fraction bits inside a segment (9)
SIG_C0, SIG_C1 = _pwl(lambda t: 1 / (1 + np.exp(-t)), 0.0, 16.0, SIG_SEGS, 16)

RSQ_SEGS = 96                    # m in [1, 4), width 1/32
RSQ_FRAC = 14 - 5                # Q2.14 input, 9 fraction bits inside a segment
RSQ_C0, RSQ_C1 = _pwl(lambda t: 1 / np.sqrt(t), 1.0, 4.0, RSQ_SEGS, 16)


def sigmoid_q16(x):
    """x Q.12 -> sigmoid Q.16 (0..65536)."""
    a = np.abs(x)
    idx = np.minimum(a >> SIG_FRAC, SIG_SEGS - 1)
    fr = a & ((1 << SIG_FRAC) - 1)
    s = SIG_C0[idx] + rsr(SIG_C1[idx] * fr, SIG_FRAC)
    s = np.where(a >= (16 << F_LI), 1 << 16, s)
    return np.where(x < 0, (1 << 16) - s, s)


def silu_fx(x):
    """x Q.12 (W_LI) -> SiLU(x) Q.12 (W_LI)."""
    x = np.asarray(x, np.int64)
    return sat(rsr(x * sigmoid_q16(x), 16), W_LI)


def rsqrt_q16(m):
    """m Q2.14 in [2^14, 2^16) -> 1/sqrt(m) Q.16."""
    idx = (m >> RSQ_FRAC) - 32
    fr = m & ((1 << RSQ_FRAC) - 1)
    return RSQ_C0[idx] + rsr(RSQ_C1[idx] * fr, RSQ_FRAC)


# ----------------------------------------------------------------------------
# linear layer output: exact sum of acc_g * s_g, then one rounding to Q.F
# ----------------------------------------------------------------------------
def lin_fixed(acc, s16, k, F, W):
    """acc (..., O, G) int, s16 (O, G) fp16, k (...,) int -> (..., O) Q.F."""
    m, sh = fp16_fields(s16)
    total = (np.asarray(acc, np.int64) * m << sh).sum(-1)       # units of 2^-24
    k = np.asarray(k, np.int64)[..., None]
    return sat(shift_round(total, k - 24 + F), W)


# ----------------------------------------------------------------------------
# RMSNorm + int8 quantization with a power-of-two scale (per token)
# ----------------------------------------------------------------------------
STATS = {}


def _count(key, n):
    STATS[key] = STATS.get(key, 0) + int(n)


def _pow2_scale_exp(a):
    """Smallest t with a <= 127 * 2^t (a > 0 int); -64 for a == 0."""
    if a == 0:
        return -64
    t = a.bit_length() - 7

    def fits(tt):
        return a <= (127 << tt) if tt >= 0 else (a << -tt) <= 127
    while not fits(t):
        t += 1
    while fits(t - 1):
        t -= 1
    return t


def norm_q8(v, w):
    """v (T, n) Q.16, w (n,) Q.12 -> q (T, n) int8 values, k (T,) with x = q * 2^k.

    ms  = mean(v^2) + eps          exact, Q.32
        = m * 4^j,  m in [1, 4)   (m as Q2.14, rounded)
    r   = 1/sqrt(ms * 2^-32) = R * 2^-16 * 2^(16 - j),  R = rsqrt_q16(m)
    u   = v * w                    Q.16
    P   = u * R                    x = P * 2^(-16 - j)
    q   = round(P / 2^t),  t smallest with max|P| <= 127 * 2^t,   k = t - 16 - j
    """
    v = np.asarray(v, np.int64)
    T, n = v.shape
    lg = int(np.log2(n))
    assert 1 << lg == n
    if np.abs(v).max(initial=0) < (1 << 26):                   # sum of squares fits int64
        ss = (v * v).sum(-1)
    else:
        _count("norm: python-int sum of squares", 1)
        ss = np.array([sum(int(e) * int(e) for e in row) for row in v], dtype=object)
    eps_q = int(round(EPS * 2.0 ** (2 * F_R)))
    ms = [(int(x) >> lg) + eps_q for x in ss]                    # Q.32, exact
    j = np.array([(x.bit_length() - 1) >> 1 for x in ms], np.int64)
    m = np.array([((x << (14 - 2 * jj)) if 2 * jj <= 14 else
                   ((x + (1 << (2 * jj - 15))) >> (2 * jj - 14))) for x, jj in zip(ms, j)],
                 np.int64)                                       # Q2.14, round half-up
    m = np.minimum(m, (1 << 16) - 1)
    R = rsqrt_q16(m)
    u = sat(rsr(v * np.asarray(w, np.int64), F_NW), W_R)        # Q.16
    P = u * R[:, None]
    t = np.array([_pow2_scale_exp(int(a)) for a in np.abs(P).max(-1)], np.int64)
    q = sat(shift_round(P, -t[:, None]), 8)
    k = t - F_R - j
    return q, k


# ----------------------------------------------------------------------------
# conv1d (depthwise, causal, 4 taps) + bias, Q.12 in / out
# ----------------------------------------------------------------------------
def conv_fx(x, wq, s16, bq):
    """x (T, C) Q.12, wq (C, 4) int8, s16 (C,) fp16, bq (C,) Q.12 -> (T, C) Q.12."""
    x = np.asarray(x, np.int64)
    T, C = x.shape
    K = wq.shape[1]
    xp = np.concatenate([np.zeros((K - 1, C), np.int64), x], 0)
    acc = sum(np.asarray(wq[:, k], np.int64) * xp[k:k + T] for k in range(K))
    m, sh = fp16_fields(s16)
    y = shift_round(acc * m, sh - 24) + np.asarray(bq, np.int64)
    return sat(y, W_LI)


def gate_fx(y16, z12):
    """SSM output y (Q.16) * SiLU(z) (Q.12) -> Q.16 (W_R)."""
    return sat(rsr(np.asarray(y16, np.int64) * silu_fx(z12), F_LI), W_R)


def embed_fx(q, s16):
    """embedding int8 (T, d), fp16 group scales (T, G) -> residual Q.16."""
    T, d = q.shape
    G = s16.shape[1]
    m, sh = fp16_fields(s16)
    m = np.repeat(m, d // G, axis=1)
    sh = np.repeat(sh, d // G, axis=1)
    return sat(shift_round(np.asarray(q, np.int64) * m, sh - 24 + F_R), W_R)


# ----------------------------------------------------------------------------
def _report():
    x = np.linspace(-12, 12, 20001)
    xf = to_fixed(x, W_LI, F_LI)
    s = silu_fx(xf) / (1 << F_LI)
    ref = x / (1 + np.exp(-x))
    print(f"SiLU      max abs err {np.abs(s - ref).max():.2e}  (on [-12, 12])")
    m = np.arange(1 << 14, 1 << 16)
    r = rsqrt_q16(m) / 65536
    print(f"rsqrt     max rel err {np.abs(r * np.sqrt(m / 16384) - 1).max():.2e}  (m in [1, 4))")
    rng = np.random.default_rng(0)
    for scale in (1e-2, 1.0, 50.0):
        v = rng.standard_normal((64, 256)) * scale
        v[:, 3] *= 20
        w = rng.uniform(0.2, 3.0, 256)
        q, k = norm_q8(to_fixed(v, W_R, F_R), to_fixed(w, W_NW, F_NW))
        xr = v / np.sqrt((v * v).mean(-1, keepdims=True) + EPS) * w
        sa = np.exp2(np.ceil(np.log2(np.abs(xr).max(-1, keepdims=True) / 127)))
        qr = np.clip(np.round(xr / sa), -128, 127)
        same_k = np.mean(np.exp2(k) == sa[:, 0])
        print(f"norm_q8   scale {scale:6.2f}: int8 values differ from float in "
              f"{np.mean(q != qr):.2%} (by 1 LSB max {np.abs(q - qr).max()}), same scale {same_k:.0%}")


if __name__ == "__main__":
    _report()
