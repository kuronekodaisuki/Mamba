#!/usr/bin/env python3
"""Bit-accurate golden model of ssm_scan_unit.sv and test-vector generator.

Usage:
  python3 model/ssm_golden.py --out vec                     # small config for tb
  python3 model/ssm_golden.py --N 128 --P 64 --heads 24 --T 256 --no-write
"""
import argparse
import os
import numpy as np

# Fixed-point formats: must match the ssm_scan_unit.sv parameters
W_H, F_H = 24, 16
W_A, F_A = 18, 16
W_X, F_X = 16, 10
F_XD = 14              # dt*x has its own fraction (dt can be ~1e-3)
W_B, F_B = 16, 12
W_D, F_D = 16, 12
W_Y, F_Y = 24, 16
SH_A = F_A
SH_B = F_XD + F_B - F_H
SH_C = F_H + F_B - F_Y
SH_D = F_D + F_X - F_Y


STATS = {"max_h": 0}     # largest |state| (in LSBs) seen by run_fixed


def set_state_format(w_h, f_h):
    """Change the state format (W_H, F_H) and the shifts that depend on it."""
    global W_H, F_H, SH_B, SH_C
    W_H, F_H = w_h, f_h
    SH_B = F_XD + F_B - F_H
    SH_C = F_H + F_B - F_Y
    assert SH_B >= 1 and SH_C >= 1, "unsupported state format"


def sat(v, w):
    return np.clip(v, -(1 << (w - 1)), (1 << (w - 1)) - 1)


def rsr(v, s):
    """Arithmetic right shift with round-half-up (matches rshift_round)."""
    return (v + (1 << (s - 1))) >> s


def quant(x, w, f):
    return sat(np.round(x * (1 << f)).astype(np.int64), w)


def make_inputs(rng, N, P, H, T):
    # Ranges follow the Mamba2 default initialisation
    A = -rng.uniform(1.0, 16.0, H)                                  # A in [-16, -1]
    dt = np.exp(rng.uniform(np.log(1e-3), np.log(1e-1), (T, H)))    # dt in [1e-3, 1e-1]
    x = rng.standard_normal((T, H, P))
    B = rng.standard_normal((T, N))
    C = rng.standard_normal((T, N))
    D = rng.uniform(0.5, 1.5, H)
    return dict(dA=np.exp(dt * A), dt=dt, x=x, B=B, C=C, D=D)


def run_float(v):
    T, H, P = v["x"].shape
    N = v["B"].shape[1]
    h = np.zeros((H, P, N))
    y = np.zeros((T, H, P))
    for t in range(T):
        xdt = v["x"][t] * v["dt"][t][:, None]
        h = v["dA"][t][:, None, None] * h + xdt[:, :, None] * v["B"][t]
        y[t] = (h * v["C"][t]).sum(-1) + v["D"][:, None] * v["x"][t]
    return y


def quantize(v):
    return dict(
        dA=quant(v["dA"], W_A, F_A),
        xdt=quant(v["x"] * v["dt"][:, :, None], W_X, F_XD),
        x=quant(v["x"], W_X, F_X),
        B=quant(v["B"], W_B, F_B),
        C=quant(v["C"], W_B, F_B),
        D=quant(v["D"], W_D, F_D),
    )


def run_fixed(q):
    """Mirrors the RTL arithmetic exactly (int64 is wide enough)."""
    T, H, P = q["x"].shape
    N = q["B"].shape[1]
    h = np.zeros((H, P, N), np.int64)       # first token: previous state = 0
    y = np.zeros((T, H, P), np.int64)
    n_sat = 0
    for t in range(T):
        ta = rsr(q["dA"][t][:, None, None] * h, SH_A)
        tb = rsr(q["xdt"][t][:, :, None] * q["B"][t], SH_B)
        raw = ta + tb
        h = sat(raw, W_H)
        n_sat += int(np.count_nonzero(raw != h))
        STATS["max_h"] = max(STATS["max_h"], int(np.abs(raw).max()))
        sc = (h * q["C"][t]).sum(-1)
        y[t] = sat(rsr(sc, SH_C) + rsr(q["D"][:, None] * q["x"][t], SH_D), W_Y)
    return y, n_sat


def write_hex(path, arr, w):
    mask = (1 << w) - 1
    with open(path, "w") as f:
        for val in np.asarray(arr).reshape(-1):
            f.write(f"{int(val) & mask:0{(w + 3) // 4}x}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--T", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="vec")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--wh", type=int, default=W_H, help="state width W_H")
    ap.add_argument("--fh", type=int, default=F_H, help="state fraction F_H")
    a = ap.parse_args()
    set_state_format(a.wh, a.fh)

    rng = np.random.default_rng(a.seed)
    v = make_inputs(rng, a.N, a.P, a.heads, a.T)
    q = quantize(v)
    y_fix, n_sat = run_fixed(q)
    y_ref = run_float(v)

    err = y_fix / (1 << F_Y) - y_ref
    print(f"config N={a.N} P={a.P} heads={a.heads} T={a.T}")
    print(f"  max |y|            : {np.abs(y_ref).max():.4f}")
    print(f"  max abs error      : {np.abs(err).max():.3e}")
    print(f"  RMS error / RMS(y) : {np.sqrt((err**2).mean() / (y_ref**2).mean()):.3e}")
    print(f"  saturated states   : {n_sat}")

    if not a.no_write:
        os.makedirs(a.out, exist_ok=True)
        write_hex(f"{a.out}/B.hex",   q["B"],   W_B)   # [t][n]
        write_hex(f"{a.out}/C.hex",   q["C"],   W_B)   # [t][n]
        write_hex(f"{a.out}/dA.hex",  q["dA"],  W_A)   # [t][h]
        write_hex(f"{a.out}/D.hex",   q["D"],   W_D)   # [h]
        write_hex(f"{a.out}/xdt.hex", q["xdt"], W_X)   # [t][h][p]
        write_hex(f"{a.out}/x.hex",   q["x"],   W_X)   # [t][h][p]
        write_hex(f"{a.out}/y.hex",   y_fix,    W_Y)   # [t][h][p]
        print(f"  vectors written to {a.out}/")


if __name__ == "__main__":
    main()
