#!/usr/bin/env python3
"""Golden model and test vectors for the linear-layer datapath.

rtl/lin_dot.sv   : int8 activations x int4 weights, K lanes (exact integer)
rtl/lin_scale.sv : rescale one output row from its group sums (lin_rescale below)

  python3 model/lin_golden.py --out vec            # 1000 dot vectors, K = 128
  python3 model/lin_golden.py --scale --out vec    # rescale vectors
  python3 model/lin_golden.py --unit --out vec     # lin_unit, random 300 x 2 groups
  python3 model/lin_golden.py --unit --npz ../export/weights.npz --layer 0 --proj in --out vec
                                                   # lin_unit with trained weights

Files (one vector per line, hex, lane k in the low bits):
  vec/lin_w.hex  K int4   (K*4 bits, same packing as student/export.py hex/L??_*_w.hex)
  vec/lin_a.hex  K int8   (K*8 bits)
  vec/lin_y.hex  expected sum, two's complement in W_O = 12 + log2(K) bits
"""
import argparse
import os

import numpy as np


# ----------------------------------------------------------------------------
# Row rescale: y = 2^k * sum_g acc_g * s_g with ONE rounding to float32
#   s_g fp16 = m_g * 2^(E_g - 25)   (normal),   m_g = 1024 + frac
#            = frac * 2^-24          (subnormal)
#   so s_g = m_g * 2^(sh_g - 24) with sh_g = max(E_g - 1, 0): every term is an
#   integer m_g * acc_g << sh_g in units of 2^-24, summed exactly, then rounded
#   (round-to-nearest-even) to float32 and scaled by 2^(k - 24).
# Results whose unrounded exponent is below the float32 normal range flush to +0.
# ----------------------------------------------------------------------------
def fp16_fields(s16):
    b = np.asarray(s16, np.float16).view(np.uint16).astype(np.int64)
    sign, E, F = (b >> 15) & 1, (b >> 10) & 31, b & 1023
    assert (E < 31).all(), "inf / nan scale"
    m = np.where(E == 0, F, F | 1024) * np.where(sign == 1, -1, 1)
    sh = np.where(E == 0, 0, E - 1)
    return m, sh


def int_to_f32(total, e2):
    """Round integer total * 2^e2 (e2 int array or scalar) to float32 (RNE, FTZ)."""
    total = np.asarray(total, np.int64)
    assert (np.abs(total) < 2 ** 53).all()
    f = total.astype(np.float64)                     # exact
    _, ex = np.frexp(np.abs(f))                      # |total| in [2^(ex-1), 2^ex)
    biased = (ex - 1) + e2 + 127
    y = (f * np.exp2(np.asarray(e2, np.float64))).astype(np.float32)   # one rounding
    return np.where((total == 0) | (biased <= 0), np.float32(0), y).astype(np.float32)


def lin_rescale(acc, s16, k):
    """acc (..., O, G) int group sums, s16 (O, G) fp16, k (...,) or (..., 1) int."""
    m, sh = fp16_fields(s16)
    total = (np.asarray(acc, np.int64) * m << sh).sum(-1)
    k = np.asarray(k, np.int64)
    if k.ndim == total.ndim and k.shape[-1] == 1:
        k = k
    elif k.ndim == total.ndim - 1:
        k = k[..., None]
    return int_to_f32(total, k - 24)


def pack(v, bits):
    m = (1 << bits) - 1
    x = 0
    for k, e in enumerate(v):
        x |= (int(e) & m) << (bits * k)
    return f"{x:0{(len(v) * bits + 3) // 4}x}"


def scale_vectors(out, n_rows, seed):
    """vec/lsc_in.hex : one beat per line, 45 bits
                        {first[44], last[43], k[42:35], scale[34:19], acc[18:0]}
       vec/lsc_y.hex  : one float32 per row"""
    rng = np.random.default_rng(seed)
    beats, ys = [], []
    for r in range(n_rows):
        G = int(rng.choice([1, 2, 4]))
        acc = rng.integers(-131072, 131073, G)
        if r < 4:
            acc[:] = [131072, -131072, 0, 1][r]
        s = np.exp(rng.uniform(np.log(1e-4), np.log(2.0), G)).astype(np.float16)
        if r == 5:
            s[:] = np.float16(6e-8)                  # fp16 subnormal
        if r == 6:
            s[0] = -s[0]
        k = int(rng.integers(-20, 4))
        y = lin_rescale(acc[None, :], s[None, :], np.array([k]))[0]
        for g in range(G):
            word = ((int(g == 0) << 44) | (int(g == G - 1) << 43) | ((k & 0xFF) << 35)
                    | (int(s[g].view(np.uint16)) << 19) | (int(acc[g]) & 0x7FFFF))
            beats.append(f"{word:012x}")
        ys.append(f"{int(np.float32(y).view(np.uint32)):08x}")
    with open(f"{out}/lsc_in.hex", "w") as f:
        f.writelines(b + "\n" for b in beats)
    with open(f"{out}/lsc_y.hex", "w") as f:
        f.writelines(y + "\n" for y in ys)
    print(f"{n_rows} rows, {len(beats)} beats -> {out}/lsc_*.hex")
    return len(beats)


def unit_vectors(out, seed, rows=300, groups=2, npz=None, layer=0, proj="in", K=128):
    """Vectors for rtl/lin_unit.sv.
       vec/lu_w.hex   rows*G words, K int4 each (export hex format)
       vec/lu_s.hex   rows*G fp16 scales
       vec/lu_a.hex   G activation groups, K int8 each
       vec/lu_cfg.hex rows, G, k (two's complement, 8 bits)
       vec/lu_y.hex   rows float32 results"""
    rng = np.random.default_rng(seed)
    if npz:
        z = np.load(npz)
        W = z[f"L{layer}.{proj}_w"].astype(np.int64)            # (O, din)
        S = z[f"L{layer}.{proj}_s"].astype(np.float16)          # (O, G)
        rows, din = W.shape
        groups = din // K
        src = f"{npz} layer {layer} {proj}_proj"
    else:
        din = groups * K
        W = rng.integers(-8, 8, (rows, din))
        S = np.exp(rng.uniform(np.log(1e-3), np.log(0.1), (rows, groups))).astype(np.float16)
        src = "random"
    A = rng.integers(-128, 128, din)
    k = int(rng.integers(-12, -2))
    acc = (W * A).reshape(rows, groups, K).sum(-1)               # exact group sums
    Y = lin_rescale(acc[None], S, np.array([k]))[0]

    with open(f"{out}/lu_w.hex", "w") as f:
        f.writelines(pack(W[r, g * K:(g + 1) * K], 4) + "\n"
                     for r in range(rows) for g in range(groups))
    with open(f"{out}/lu_s.hex", "w") as f:
        f.writelines(f"{int(v):04x}\n" for v in S.view(np.uint16).reshape(-1))
    with open(f"{out}/lu_a.hex", "w") as f:
        f.writelines(pack(A[g * K:(g + 1) * K], 8) + "\n" for g in range(groups))
    with open(f"{out}/lu_cfg.hex", "w") as f:
        f.write(f"{rows:x}\n{groups:x}\n{k & 0xFF:02x}\n")
    with open(f"{out}/lu_y.hex", "w") as f:
        f.writelines(f"{int(v):08x}\n" for v in Y.astype(np.float32).view(np.uint32))
    print(f"lin_unit vectors ({src}): {rows} rows x {groups} groups, k={k} -> {out}/lu_*.hex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="vec")
    ap.add_argument("--scale", action="store_true", help="vectors for rtl/lin_scale.sv")
    ap.add_argument("--unit", action="store_true", help="vectors for rtl/lin_unit.sv")
    ap.add_argument("--npz", default=None, help="exported weights.npz (else random weights)")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--proj", choices=["in", "out"], default="in")
    a = ap.parse_args()
    if a.unit:
        os.makedirs(a.out, exist_ok=True)
        unit_vectors(a.out, a.seed, npz=a.npz, layer=a.layer, proj=a.proj, K=a.K)
        return
    if a.scale:
        os.makedirs(a.out, exist_ok=True)
        scale_vectors(a.out, a.n, a.seed)
        return
    K = a.K
    w_o = 12 + int(np.log2(K))
    rng = np.random.default_rng(a.seed)

    W = rng.integers(-8, 8, (a.n, K))
    A = rng.integers(-128, 128, (a.n, K))
    # corner cases: largest positive / negative sums, alternating signs, zeros
    corners = [(-8, -128), (7, 127), (-8, 127), (7, -128), (0, 0)]
    for i, (w, x) in enumerate(corners):
        W[i], A[i] = w, x
    W[len(corners)] = np.where(np.arange(K) % 2, 7, -8)
    A[len(corners)] = np.where(np.arange(K) % 2, -128, 127)
    Y = (W * A).sum(1)
    assert np.abs(Y).max() < 2 ** (w_o - 1)

    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/lin_w.hex", "w") as f:
        f.writelines(pack(r, 4) + "\n" for r in W)
    with open(f"{a.out}/lin_a.hex", "w") as f:
        f.writelines(pack(r, 8) + "\n" for r in A)
    with open(f"{a.out}/lin_y.hex", "w") as f:
        f.writelines(f"{int(y) & ((1 << w_o) - 1):0{(w_o + 3) // 4}x}\n" for y in Y)
    print(f"{a.n} vectors, K={K}, |sum| max {np.abs(Y).max()} -> {a.out}/lin_*.hex")


if __name__ == "__main__":
    main()
