#!/usr/bin/env python3
"""Hardware reference model: runs the exported student using only the exported
integers and the FPGA arithmetic. Pure numpy (no torch).

  linear layers : int8 activations (per-token power-of-two scale) x int4 weights,
                  integer accumulation per 128-element group, then fp16 group
                  scale and activation scale (FP32 DSP)
  conv1d        : int8 per-channel weights, fp32 bias
  softplus/exp  : bit-exact NAU          (model/nau_golden.py == rtl/nau.sv)
  SSM recurrence: bit-exact fixed point  (model/ssm_golden.py == rtl/ssm_scan_unit.sv)
  RMSNorm, SiLU, gating, residual : fp32 (FP32 DSP)

mode="float" replaces the NAU and SSM by float math on the same weights, to
separate fixed-point error from quantization error.

Self test (no checkpoint needed):
  python3 hwref.py --selftest
"""
import argparse
import json
import os
import sys

import numpy as np


def _import_golden(golden_dir=None):
    here = os.path.dirname(os.path.abspath(__file__))
    for d in ([golden_dir] if golden_dir else []) + [os.path.join(here, "..", "model"), here]:
        if d and os.path.exists(os.path.join(d, "ssm_golden.py")):
            sys.path.insert(0, os.path.abspath(d))
            break
    try:
        import nau_golden as ng
        import ssm_golden as sg
    except ImportError as e:
        raise ImportError("ssm_golden.py / nau_golden.py not found: copy them next to "
                          "hwref.py or pass --golden-dir") from e
    assert (sg.W_A, sg.F_A) == (ng.W_DA, ng.F_DA), "dA format mismatch between golden models"
    return ng, sg


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def sat(v, w):
    return np.clip(v, -(1 << (w - 1)), (1 << (w - 1)) - 1)


def rsr(v, s):
    return (v + (1 << (s - 1))) >> s


def to_fixed(x, w, f):
    """float -> saturated fixed point; returns (int64 array, number clipped)."""
    r = np.round(np.asarray(x, np.float64) * (1 << f)).astype(np.int64)
    q = sat(r, w)
    return q, int(np.count_nonzero(q != r))


def fwht(x):
    n = x.shape[-1]
    shape = x.shape
    y = x.reshape(-1, n)
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = np.stack((a + b, a - b), axis=2)
        h *= 2
    return (y.reshape(shape) / np.sqrt(n)).astype(x.dtype)


def quant_act(x, bits):
    """Per-token symmetric quantization with power-of-two scale (as fq_act)."""
    x = x.astype(np.float32)
    qmax = 2 ** (bits - 1) - 1
    s = np.maximum(np.abs(x).max(-1, keepdims=True), np.float32(1e-8)) / np.float32(qmax)
    s = np.exp2(np.ceil(np.log2(s))).astype(np.float32)
    q = np.clip(np.round(x / s), -qmax - 1, qmax)
    return q.astype(np.float32), s


def silu(x):
    return x / (1 + np.exp(-x))


def rmsnorm(x, w, eps=1e-5):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


# ----------------------------------------------------------------------------
# vectorized bit-exact NAU (identical to nau_golden scalar functions)
# ----------------------------------------------------------------------------
class NauVec:
    def __init__(self, ng):
        self.g = ng
        self.C0 = np.array(ng.C0, np.int64)
        self.C1 = np.array(ng.C1, np.int64)

    def exp_core(self, neg):
        g = self.g
        neg = np.asarray(neg, np.int64)
        t = neg * g.LOG2E_Q
        s = -(t >> g.F_T)
        f16 = (t & ((1 << g.F_T) - 1)) >> (g.F_T - g.F_C)
        k, r = f16 >> g.R_BITS, f16 & ((1 << g.R_BITS) - 1)
        y = self.C0[k] + rsr(self.C1[k] * r, g.F_C)
        yo = y << (g.F_O - g.F_C)
        sc = np.clip(s, 0, g.F_O + 3)
        half = np.where(sc > 0, np.left_shift(np.int64(1), np.maximum(sc - 1, 0)), 0)
        out = np.where(s > g.F_O + 2, 0, (yo + half) >> sc)
        if g.EXP_NEAR0:
            near = neg > -(1 << (g.F_I - 4))
            out = np.where(near, (1 << g.F_O) + neg + rsr(neg * neg, g.F_I + 1), out)
        return out

    def nau(self, softplus, x):
        g = self.g
        x = np.asarray(x, np.int64)
        if not softplus:
            return sat(self.exp_core(np.minimum(x, 0)), g.W_O)
        y = self.exp_core(-np.abs(x))
        if g.SP_ORDER2:
            y = y - rsr(y * y, g.F_O + 1)
        return sat(y + np.maximum(x, 0), g.W_O)

    def discretize(self, dt_raw, dt_bias, A):
        g = self.g
        z = sat((dt_raw + dt_bias) << (g.F_I - g.F_RAW), g.W_I)
        delta = self.nau(True, z)
        da_in = sat(rsr(delta * A, g.F_O + g.F_AA - g.F_I), g.W_I)
        e = self.nau(False, da_in)
        return delta, sat(rsr(e, g.F_O - g.F_DA), g.W_DA)


# ----------------------------------------------------------------------------
# reference model
# ----------------------------------------------------------------------------
class HwRef:
    def __init__(self, npz_path, mode="fixed", golden_dir=None, state_fmt=None, vec="float"):
        z = np.load(npz_path)
        self.p = {k: z[k] for k in z.files}
        self.cfg = json.loads(str(self.p.pop("cfg_json")))
        self.mode = mode
        self.vec = vec                               # "float" or "fixed" (model/vec_golden.py)
        self.ng, self.sg = _import_golden(golden_dir)
        nau_mode = self.cfg.get("nau", "improved")
        self.ng.EXP_NEAR0 = self.ng.SP_ORDER2 = int(nau_mode != "published")
        self.nv = NauVec(self.ng)
        c = self.cfg
        self.di = c["expand"] * c["d_model"]
        self.H = self.di // c["headdim"]
        import lin_golden                           # same folder as the other golden models
        import vec_golden
        self.lg = lin_golden
        self.vg = vec_golden
        if vec == "fixed":
            assert mode == "fixed", "vec='fixed' needs mode='fixed'"
            assert not self.cfg["hadamard"], "fixed vector path assumes no Hadamard rotation"
        self.state_fmt = state_fmt or (self.sg.W_H, self.sg.F_H)
        self.stats = {}
        self.dump = None                             # set to {} to record forward_fixed tensors

    def _count(self, key, n):
        self.stats[key] = self.stats.get(key, 0) + n

    def _deq_w(self, name):
        q = self.p[name + "_w"].astype(np.float32)
        s = self.p[name + "_s"].astype(np.float32)
        return q, s

    def _rescaled_dot(self, xa, sa, q, s16):
        """Exact group dot products, then rtl/lin_scale.sv: one rounding to fp32."""
        G = s16.shape[1]
        grp = q.shape[1] // G
        acc = np.stack([np.rint(xa[:, g * grp:(g + 1) * grp] @ q[:, g * grp:(g + 1) * grp].T)
                        for g in range(G)], axis=-1).astype(np.int64)      # (T, O, G)
        k = np.rint(np.log2(sa[:, 0])).astype(np.int64)                    # sa = 2^k
        return self.lg.lin_rescale(acc, s16, k)

    def qlinear(self, x, name):
        c = self.cfg
        q = self.p[name + "_w"].astype(np.float32)  # (O, din) int4 values
        s16 = self.p[name + "_s"]                   # (O, G) fp16
        if c["hadamard"]:
            x = fwht(x)
        xa, sa = quant_act(x, c["a_bits"])
        return self._rescaled_dot(xa, sa, q, s16)

    def conv(self, xbc, i):
        cs = self.p[f"L{i}.conv_s"].astype(np.float32).reshape(-1, 1)        # per channel
        w = self.p[f"L{i}.conv_w"].astype(np.float32) * cs
        b = self.p[f"L{i}.conv_b"].astype(np.float32)
        K = w.shape[1]
        xp = np.concatenate([np.zeros((K - 1, xbc.shape[1]), np.float32), xbc], 0)
        T = xbc.shape[0]
        out = b + sum(w[:, k] * xp[k:k + T] for k in range(K))
        return out.astype(np.float32)

    def ssm(self, x, B, C, dt_raw, i):
        """x (T,H,P), B/C (T,N), dt_raw (T,H) float -> y (T,H,P) float."""
        ng, sg, nv = self.ng, self.sg, self.nv
        dt_bias = self.p[f"L{i}.dt_bias"].astype(np.int64)
        A = self.p[f"L{i}.A"].astype(np.int64)
        D = self.p[f"L{i}.D"].astype(np.int64)
        if self.mode == "float":
            dt = np.logaddexp(0, dt_raw + dt_bias / 2 ** ng.F_RAW)
            dA = np.exp(dt * (A / 2 ** ng.F_AA))
            h = np.zeros((self.H, x.shape[2], B.shape[1]))
            y = np.zeros(x.shape)
            for t in range(x.shape[0]):
                h = dA[t][:, None, None] * h + (x[t] * dt[t][:, None])[:, :, None] * B[t]
                y[t] = (h * C[t]).sum(-1) + (D / 2 ** sg.F_D)[:, None] * x[t]
            return y.astype(np.float32)

        raw_q, n1 = to_fixed(dt_raw, ng.W_RAW, ng.F_RAW)
        x_q, n2 = to_fixed(x, sg.W_X, sg.F_X)
        B_q, n3 = to_fixed(B, sg.W_B, sg.F_B)
        C_q, n4 = to_fixed(C, sg.W_B, sg.F_B)
        self._count("clip dt_raw", n1)
        self._count("clip x", n2)
        self._count("clip B/C", n3 + n4)
        return (self.ssm_int(raw_q, x_q, B_q, C_q, i) / 2 ** sg.F_Y).astype(np.float32)

    def ssm_int(self, raw_q, x_q, B_q, C_q, i):
        """Integer SSM: dt_raw Q.F_RAW (T,H), x Q.F_X (T,H,P), B/C Q.F_B (T,N) -> y Q.F_Y."""
        ng, sg, nv = self.ng, self.sg, self.nv
        dt_bias = self.p[f"L{i}.dt_bias"].astype(np.int64)
        A = self.p[f"L{i}.A"].astype(np.int64)
        D = self.p[f"L{i}.D"].astype(np.int64)
        delta, dA = nv.discretize(raw_q, dt_bias, A)
        xdt = sat(rsr(delta[:, :, None] * x_q, ng.F_O + sg.F_X - sg.F_XD), sg.W_X)
        self._count("clip dt*x", int(np.count_nonzero(
            xdt != rsr(delta[:, :, None] * x_q, ng.F_O + sg.F_X - sg.F_XD))))
        sg.set_state_format(*self.state_fmt)
        sg.STATS["max_h"] = 0
        y, n_sat = sg.run_fixed(dict(dA=dA, xdt=xdt, x=x_q, B=B_q, C=C_q, D=D))
        self._count("saturated SSM state", n_sat)
        self.stats["max |state|"] = max(self.stats.get("max |state|", 0.0),
                                        round(sg.STATS["max_h"] / 2 ** sg.F_H, 3))
        return y

    def block(self, u, i):
        c, di, H = self.cfg, self.di, self.H
        N, P = c["d_state"], c["headdim"]
        zxbcdt = self.qlinear(u, f"L{i}.in")
        z, xbc, dt_raw = np.split(zxbcdt, [di, 2 * di + 2 * N], axis=1)
        xbc = silu(self.conv(xbc, i))
        x, B, C = np.split(xbc, [di, di + N], axis=1)
        T = u.shape[0]
        y = self.ssm(x.reshape(T, H, P), B, C, dt_raw, i).reshape(T, di)
        y = rmsnorm(y * silu(z), self.p[f"L{i}.norm_w"])
        return self.qlinear(y, f"L{i}.out")

    # ------------------------------------------------------------------------
    # fixed-point vector path (model/vec_golden.py): no float between layers
    # ------------------------------------------------------------------------
    def _maxabs(self, key, v, f):
        self.stats[key] = max(self.stats.get(key, 0.0), round(float(np.abs(v).max()) / 2 ** f, 3))

    def _wfix(self, name, w, f):
        q, n = to_fixed(self.p[name], w, f)
        self._count(f"clip {name.split('.')[-1]}", n)
        return q

    def lin_fx(self, q, k, name, F, W):
        """int8 activations q (T, din) with scale 2^k -> rows in Q.F (W bits)."""
        wq = self.p[name + "_w"].astype(np.float32)
        s16 = self.p[name + "_s"]
        G = s16.shape[1]
        grp = wq.shape[1] // G
        qf = q.astype(np.float32)
        acc = np.stack([np.rint(qf[:, g * grp:(g + 1) * grp] @ wq[:, g * grp:(g + 1) * grp].T)
                        for g in range(G)], axis=-1).astype(np.int64)
        return self.vg.lin_fixed(acc, s16, k, F, W)

    def forward_fixed(self, ids):
        vg, ng, sg, c = self.vg, self.ng, self.sg, self.cfg
        di, H = self.di, self.H
        N, P = c["d_state"], c["headdim"]
        T = len(ids)
        h = vg.embed_fx(self.p["emb_w"][ids].astype(np.int64), self.p["emb_s"][ids])
        d = self.dump
        for i in range(c["n_layer"]):
            if d is not None:
                d[(i, "h_in")] = h.copy()
            q, k = vg.norm_q8(h, self._wfix(f"L{i}.norm", vg.W_NW, vg.F_NW))
            zx = self.lin_fx(q, k, f"L{i}.in", vg.F_LI, vg.W_LI)
            self._maxabs("max |in_proj out|", zx, vg.F_LI)
            z, xbc, dt12 = np.split(zx, [di, 2 * di + 2 * N], axis=1)
            cv = vg.conv_fx(xbc, self.p[f"L{i}.conv_w"], self.p[f"L{i}.conv_s"],
                            self._wfix(f"L{i}.conv_b", vg.W_LI, vg.F_LI))
            self._maxabs("max |conv out|", cv, vg.F_LI)
            xbc = vg.silu_fx(cv)
            x12, B12, C12 = np.split(xbc, [di, di + N], axis=1)
            # Q.12 -> SSM input formats (round, saturate, count clipping)
            def conv_to(v, w, f, key):
                r = vg.shift_round(v, f - vg.F_LI)
                self._count(key, int(np.count_nonzero(r != sat(r, w))))
                return sat(r, w)
            raw_q = conv_to(dt12, ng.W_RAW, ng.F_RAW, "clip dt_raw")
            x_q = conv_to(x12, sg.W_X, sg.F_X, "clip x")
            B_q = conv_to(B12, sg.W_B, sg.F_B, "clip B/C")
            C_q = conv_to(C12, sg.W_B, sg.F_B, "clip B/C")
            y = self.ssm_int(raw_q, x_q.reshape(T, H, P), B_q, C_q, i).reshape(T, di)
            g = vg.gate_fx(y, z)
            self._maxabs("max |gated y|", g, vg.F_R)
            q2, k2 = vg.norm_q8(g, self._wfix(f"L{i}.norm_w", vg.W_NW, vg.F_NW))
            out = self.lin_fx(q2, k2, f"L{i}.out", vg.F_LO, vg.W_LO)
            h = sat(h + out, vg.W_R)
            if d is not None:
                d.update({(i, "q_in"): q, (i, "k_in"): k, (i, "zx"): zx, (i, "xbc"): xbc,
                          (i, "raw_q"): raw_q, (i, "x_q"): x_q, (i, "B_q"): B_q, (i, "C_q"): C_q,
                          (i, "y"): y, (i, "g"): g, (i, "q_g"): q2, (i, "k_g"): k2,
                          (i, "out"): out, (i, "h_out"): h.copy()})
            self._maxabs("max |residual|", h, vg.F_R)
        q, k = vg.norm_q8(h, self._wfix("norm_f", vg.W_NW, vg.F_NW))
        emb = self.p["emb_w"].astype(np.float32)
        es16 = self.p["emb_s"]
        sa = np.exp2(k.astype(np.float64))[:, None]
        return np.concatenate([self._rescaled_dot(q[t:t + 64].astype(np.float32), sa[t:t + 64],
                                                  emb, es16)
                               for t in range(0, T, 64)], axis=0)

    def forward(self, ids):
        if self.vec == "fixed":
            return self.forward_fixed(np.asarray(ids))
        c = self.cfg
        emb = self.p["emb_w"].astype(np.float32)
        es16 = self.p["emb_s"]
        es = es16.astype(np.float32)
        grp = emb.shape[1] // es.shape[1]
        h = (emb[ids] * np.repeat(es[ids], grp, axis=1)).astype(np.float32)
        for i in range(c["n_layer"]):
            h = h + self.block(rmsnorm(h, self.p[f"L{i}.norm"]), i)
        h = rmsnorm(h, self.p["norm_f"])
        xa, sa = quant_act(h, c["a_bits"])
        # tied head: int8 x int8 group sums, same rescale as the linear layers
        return np.concatenate([self._rescaled_dot(xa[t:t + 64], sa[t:t + 64], emb, es16)
                               for t in range(0, len(ids), 64)], axis=0)


# ----------------------------------------------------------------------------
# self test with random weights (no torch, no checkpoint)
# ----------------------------------------------------------------------------
def _np_qweight(w, bits, group):
    O = w.shape[0]
    w3 = w.reshape(O, -1, group).astype(np.float32)
    qmax = 2 ** (bits - 1) - 1
    s = np.maximum(np.abs(w3).max(-1, keepdims=True), 1e-8) / qmax
    q = np.clip(np.round(w3 / s), -qmax - 1, qmax)
    return q.reshape(w.shape).astype(np.int8), s[..., 0].astype(np.float16)


def _random_npz(path, rng, golden_dir):
    ng, sg = _import_golden(golden_dir)
    cfg = dict(vocab_size=1000, d_model=128, n_layer=2, d_state=32, headdim=32, expand=2,
               d_conv=4, w_bits=4, w_group=128, a_bits=8, conv_bits=8, emb_bits=8,
               hadamard=False, nau="improved")
    d, di, N, H = 128, 256, 32, 8
    p = {"cfg_json": json.dumps(cfg)}
    p["emb_w"], p["emb_s"] = _np_qweight(rng.standard_normal((1000, d)) * 0.5, 8, 128)
    p["norm_f"] = np.ones(d, np.float32)
    for i in range(2):
        p[f"L{i}.norm"] = np.ones(d, np.float32)
        p[f"L{i}.in_w"], p[f"L{i}.in_s"] = _np_qweight(
            rng.standard_normal((2 * di + 2 * N + H, d)) / np.sqrt(d), 4, 128)
        cw, cs = _np_qweight(rng.standard_normal((di + 2 * N, 4)) * 0.5, 8, 4)
        p[f"L{i}.conv_w"], p[f"L{i}.conv_s"] = cw, cs.reshape(-1)
        p[f"L{i}.conv_b"] = np.zeros(di + 2 * N, np.float32)
        dt = np.exp(rng.uniform(np.log(1e-3), np.log(1e-1), H))
        p[f"L{i}.dt_bias"] = to_fixed(dt + np.log(-np.expm1(-dt)), ng.W_RAW, ng.F_RAW)[0].astype(np.int16)
        p[f"L{i}.A"] = to_fixed(-rng.uniform(1, 16, H), ng.W_AA, ng.F_AA)[0].astype(np.int16)
        p[f"L{i}.D"] = to_fixed(np.ones(H), sg.W_D, sg.F_D)[0].astype(np.int16)
        p[f"L{i}.norm_w"] = np.ones(di, np.float32)
        p[f"L{i}.out_w"], p[f"L{i}.out_s"] = _np_qweight(
            rng.standard_normal((d, di)) / np.sqrt(di), 4, 128)
    np.savez(path, **p)


def selftest(golden_dir=None, tmpdir="."):
    ng, sg = _import_golden(golden_dir)
    rng = np.random.default_rng(0)
    # 1) vectorized NAU / discretize == scalar golden
    raw = np.concatenate([rng.integers(-32768, 32767, 3000), np.arange(-3000, 3000, 7)])
    bias = rng.integers(-8000, 2000, raw.size)
    A = -rng.integers(1, 32767, raw.size)
    xs = np.concatenate([rng.integers(-(1 << 23), 1 << 23, 3000), np.arange(-20000, 20000, 13)])
    for opt in (0, 1):
        ng.EXP_NEAR0 = ng.SP_ORDER2 = opt
        nv = NauVec(ng)
        bad = 0
        for sp in (False, True):
            v = nv.nau(sp, xs)
            bad += sum(int(v[j]) != ng.nau(sp, int(xs[j])) for j in range(xs.size))
        d, a = nv.discretize(raw, bias, A)
        for j in range(raw.size):
            gd, ga = ng.discretize(int(raw[j]), int(bias[j]), int(A[j]))
            bad += (int(d[j]) != gd) + (int(a[j]) != ga)
        print(f"NauVec vs nau_golden (options={opt}): mismatches {bad}")
    # 2) end-to-end: fixed vs float on a random tiny model
    path = os.path.join(tmpdir, "hwref_selftest.npz")
    _random_npz(path, rng, golden_dir)
    ids = rng.integers(0, 1000, 64)
    lf = HwRef(path, "fixed", golden_dir)
    l_fix = lf.forward(ids)
    l_flt = HwRef(path, "float", golden_dir).forward(ids)
    rel = np.linalg.norm(l_fix - l_flt) / np.linalg.norm(l_flt)
    top1 = (l_fix.argmax(-1) == l_flt.argmax(-1)).mean()
    print(f"random model, fixed vs float SSM: rel logit diff {rel:.2e}, top-1 agreement {top1:.1%}")
    print("fixed-point events:", lf.stats)
    lv = HwRef(path, "fixed", golden_dir, vec="fixed")
    l_vec = lv.forward(ids)
    rel = np.linalg.norm(l_vec - l_fix) / np.linalg.norm(l_fix)
    top1 = (l_vec.argmax(-1) == l_fix.argmax(-1)).mean()
    print(f"random model, fixed vector path vs float vector path: rel logit diff {rel:.2e}, "
          f"top-1 agreement {top1:.1%}")
    print("fixed vector path events:", lv.stats)
    os.remove(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--make-test-model", default=None, metavar="DIR",
                    help="write a small random model (weights.npz + validation tokens) to DIR")
    ap.add_argument("--golden-dir", default=None)
    ap.add_argument("--tmpdir", default=".")
    a = ap.parse_args()
    if a.selftest:
        selftest(a.golden_dir, a.tmpdir)
    if a.make_test_model:
        os.makedirs(a.make_test_model, exist_ok=True)
        rng = np.random.default_rng(0)
        _random_npz(os.path.join(a.make_test_model, "weights.npz"), rng, a.golden_dir)
        np.save(os.path.join(a.make_test_model, "val_tokens_16x512.npy"),
                rng.integers(0, 1000, (16, 513)).astype(np.int32))
        print(f"test model (d_model 128, 2 layers, d_state 32, headdim 32) -> {a.make_test_model}")
