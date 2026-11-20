#!/usr/bin/env python3
"""Test vectors for rtl/lin_dot.sv (int8 activations x int4 weights, K lanes).

  python3 model/lin_golden.py --out vec            # 1000 vectors, K = 128

Files (one vector per line, hex, lane k in the low bits):
  vec/lin_w.hex  K int4   (K*4 bits, same packing as student/export.py hex/L??_*_w.hex)
  vec/lin_a.hex  K int8   (K*8 bits)
  vec/lin_y.hex  expected sum, two's complement in W_O = 12 + log2(K) bits
"""
import argparse
import os

import numpy as np


def pack(v, bits):
    m = (1 << bits) - 1
    x = 0
    for k, e in enumerate(v):
        x |= (int(e) & m) << (bits * k)
    return f"{x:0{(len(v) * bits + 3) // 4}x}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="vec")
    a = ap.parse_args()
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
