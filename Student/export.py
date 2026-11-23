#!/usr/bin/env python3
"""Export a QAT checkpoint to FPGA formats and verify it with the hardware
reference model (hwref.py).

  python3 export.py --ckpt ../ckpt/qat_nohad.pt --out ../export --verify 16

Output (../export):
  weights.npz      all exported integers / scales (input of hwref.py)
  manifest.json    config, fixed-point formats, file list, sizes, clip counts
  hex/             $readmemh files for on-chip memories (one value or one
                   packed word per line, two's complement, little-endian lanes)
    L00_in_w.hex     int4, one line per (row, group): 128 nibbles = 512 bits,
                     element k of the group in bits [4k+3:4k]
    L00_in_s.hex     fp16 scale per (row, group)
    L00_out_w.hex / L00_out_s.hex   same for out_proj
    L00_conv_w.hex   int8, one line per channel: tap k in bits [8k+7:8k]
                     (tap 3 multiplies the current token)
    L00_conv_s.hex   fp16 per channel ;  L00_conv_b.hex  fp32 per channel
    L00_dt_bias.hex  int16 Q5.10 ; L00_A.hex int16 Q5.10 ; L00_D.hex int16 Q3.12
    L00_norm.hex, L00_norm_w.hex, norm_f.hex   fp32
  ddr/             binary images for DDR4
    emb_int8.bin     vocab x d_model int8, row-major
    emb_scale.bin    vocab x (d_model/128) fp16
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

from model import StudentConfig, StudentLM, fwht
from hwref import HwRef, _import_golden, to_fixed


def q_weight(w, bits, group):
    """Same arithmetic as model.fq_weight; returns int8 values and fp32 scales."""
    O = w.shape[0]
    w32 = w.detach().float().cpu().reshape(O, -1, group)
    qmax = 2 ** (bits - 1) - 1
    s = w32.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
    q = (w32 / s).round().clamp(-qmax - 1, qmax)
    return q.reshape(w.shape).to(torch.int8).numpy(), s[..., 0].numpy()


# ---------------------------------------------------------------------------
# hex writers
# ---------------------------------------------------------------------------
_HEX = np.array(list("0123456789abcdef"))


def hex_packed(path, q, lane_bits):
    """One line per row of q; lane j of the row in bits [lane_bits*(j+1)-1 : lane_bits*j]."""
    digits = lane_bits // 4
    u = (q.astype(np.int64) & ((1 << lane_bits) - 1))
    with open(path, "w") as f:
        for row in u:
            f.write("".join(f"{int(v):0{digits}x}" for v in row[::-1]) + "\n")


def hex_int4_groups(path, q, group):
    O, din = q.shape
    nib = (q.astype(np.int64) & 0xF).reshape(O * (din // group), group)[:, ::-1]
    with open(path, "w") as f:
        for row in _HEX[nib]:
            f.write("".join(row) + "\n")


def hex_words(path, v, bits):
    m = (1 << bits) - 1
    with open(path, "w") as f:
        for x in np.asarray(v).reshape(-1):
            f.write(f"{int(x) & m:0{(bits + 3) // 4}x}\n")


def hex_fp(path, v, dtype):
    u = np.asarray(v, dtype).reshape(-1).view(np.uint16 if dtype == np.float16 else np.uint32)
    w = 4 if dtype == np.float16 else 8
    with open(path, "w") as f:
        for x in u:
            f.write(f"{int(x):0{w}x}\n")


# ---------------------------------------------------------------------------
def export(ckpt_path, out_dir, golden_dir):
    ng, sg = _import_golden(golden_dir)
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = StudentConfig(**ck["cfg"])
    if not cfg.quant:
        print("warning: checkpoint was not trained with QAT (cfg.quant = False)")
    sd = ck["model"]
    di, N, H = cfg.d_inner, cfg.d_state, cfg.nheads
    os.makedirs(os.path.join(out_dir, "hex"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "ddr"), exist_ok=True)
    hx = lambda name: os.path.join(out_dir, "hex", name)

    p = {"cfg_json": json.dumps(cfg.to_dict())}
    clips, bits_onchip = {}, 0

    def fixed(name, x, w, f):
        q, n = to_fixed(x, w, f)
        if n:
            clips[name] = n
        return q.astype(np.int16)

    # embedding / tied head -> DDR4
    eq, es = q_weight(sd["emb.weight"], cfg.emb_bits, 128)
    p["emb_w"], p["emb_s"] = eq, es.astype(np.float16)
    eq.tofile(os.path.join(out_dir, "ddr", "emb_int8.bin"))
    p["emb_s"].tofile(os.path.join(out_dir, "ddr", "emb_scale.bin"))
    p["norm_f"] = sd["norm_f.weight"].float().numpy()
    hex_fp(hx("norm_f.hex"), p["norm_f"], np.float32)

    for i in range(cfg.n_layer):
        L = f"L{i:02d}_"
        pre = f"layers.{i}."
        for name, key in (("in", "in_proj"), ("out", "out_proj")):
            w = sd[pre + key + ".weight"].float()
            if cfg.hadamard:
                w = fwht(w)
            q, s = q_weight(w, cfg.w_bits, cfg.w_group)
            p[f"L{i}.{name}_w"], p[f"L{i}.{name}_s"] = q, s.astype(np.float16)
            hex_int4_groups(hx(L + f"{name}_w.hex"), q, cfg.w_group)
            hex_fp(hx(L + f"{name}_s.hex"), s, np.float16)
            bits_onchip += q.size * cfg.w_bits + s.size * 16

        cw = sd[pre + "conv.weight"].float().reshape(di + 2 * N, cfg.d_conv)
        cq, cs = q_weight(cw, cfg.conv_bits, cfg.d_conv)
        p[f"L{i}.conv_w"], p[f"L{i}.conv_s"] = cq, cs.reshape(-1).astype(np.float16)
        p[f"L{i}.conv_b"] = sd[pre + "conv.bias"].float().numpy()
        hex_packed(hx(L + "conv_w.hex"), cq, 8)
        hex_fp(hx(L + "conv_s.hex"), p[f"L{i}.conv_s"], np.float16)
        hex_fp(hx(L + "conv_b.hex"), p[f"L{i}.conv_b"], np.float32)
        bits_onchip += cq.size * 8 + cs.size * 16 + cq.shape[0] * 32

        p[f"L{i}.dt_bias"] = fixed(f"L{i}.dt_bias", sd[pre + "dt_bias"].float().numpy(), ng.W_RAW, ng.F_RAW)
        p[f"L{i}.A"] = fixed(f"L{i}.A", -np.exp(sd[pre + "A_log"].float().numpy()), ng.W_AA, ng.F_AA)
        p[f"L{i}.D"] = fixed(f"L{i}.D", sd[pre + "D"].float().numpy(), sg.W_D, sg.F_D)
        hex_words(hx(L + "dt_bias.hex"), p[f"L{i}.dt_bias"], 16)
        hex_words(hx(L + "A.hex"), p[f"L{i}.A"], 16)
        hex_words(hx(L + "D.hex"), p[f"L{i}.D"], 16)

        p[f"L{i}.norm"] = sd[f"norms.{i}.weight"].float().numpy()
        p[f"L{i}.norm_w"] = sd[pre + "norm_w"].float().numpy()
        hex_fp(hx(L + "norm.hex"), p[f"L{i}.norm"], np.float32)
        hex_fp(hx(L + "norm_w.hex"), p[f"L{i}.norm_w"], np.float32)
        bits_onchip += (3 * H + di + cfg.d_model) * 32

    np.savez(os.path.join(out_dir, "weights.npz"), **p)
    manifest = {
        "checkpoint": os.path.abspath(ckpt_path),
        "config": cfg.to_dict(),
        "formats": {
            "linear weight": f"int{cfg.w_bits}, symmetric, one fp16 scale per {cfg.w_group} inputs",
            "linear activation": f"int{cfg.a_bits} per token, power-of-two scale",
            "hadamard": cfg.hadamard,
            "conv weight": f"int{cfg.conv_bits} per channel, fp16 scale, fp32 bias",
            "dt_raw, dt_bias, A": f"int{ng.W_RAW} Q.{ng.F_RAW}",
            "x": f"int{sg.W_X} Q.{sg.F_X}", "dt*x": f"int{sg.W_X} Q.{sg.F_XD}",
            "B, C": f"int{sg.W_B} Q.{sg.F_B}", "D": f"int{sg.W_D} Q.{sg.F_D}",
            "delta (NAU out)": f"int{ng.W_O} Q.{ng.F_O}", "dA": f"int{sg.W_A} Q.{sg.F_A}",
            "SSM state": f"int{sg.W_H} Q.{sg.F_H}", "SSM y": f"int{sg.W_Y} Q.{sg.F_Y}",
            "NAU options": "improved" if cfg.nau != "published" else "published",
            "embedding / head": f"int{cfg.emb_bits}, fp16 scale per 128 (DDR4)",
            "norms, SiLU, gating": "fp32",
        },
        "clipped parameters": clips,
        "on-chip weight bits": bits_onchip,
        "on-chip weight Mb": round(bits_onchip / 1e6, 2),
        "ddr bytes": int(eq.size + p["emb_s"].nbytes),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"exported to {out_dir}: on-chip weights {bits_onchip/1e6:.2f} Mb "
          f"({bits_onchip/(2713*20*1024):.0%} of M20K), DDR4 {manifest['ddr bytes']/1e6:.1f} MB")
    if clips:
        print("warning: clipped parameters", clips)
    return cfg, ck


# ---------------------------------------------------------------------------
def load_val_rows(out_dir, n_seq, seq):
    """Validation token blocks, cached in out_dir so later runs need no network."""
    cache = os.path.join(out_dir, f"val_tokens_{n_seq}x{seq}.npy")
    if os.path.exists(cache):
        print(f"using cached validation tokens {cache}")
        return np.load(cache).tolist()
    import train                                   # token stream (datasets, transformers)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    args = argparse.Namespace(dataset="Salesforce/wikitext", dataset_config="wikitext-103-raw-v1",
                              text_field="text", seq=seq)
    stream = train.token_stream(args, tok, "validation")
    rows = [next(stream) for _ in range(n_seq)]
    np.save(cache, np.array(rows, dtype=np.int32))
    return rows


def verify(cfg, ck, out_dir, n_seq, seq, golden_dir, with_float, state_fmts, vecs):
    rows = load_val_rows(out_dir, n_seq, seq)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = StudentLM(cfg).to(dev).eval()
    model.load_state_dict(ck["model"])
    npz = os.path.join(out_dir, "weights.npz")
    refs = {f"W{w}F{f} vec-{v}": HwRef(npz, "fixed", golden_dir, (w, f), vec=v)
            for w, f in state_fmts for v in vecs}
    if with_float:
        refs["float"] = HwRef(npz, "float", golden_dir)

    nll = {k: 0.0 for k in ["torch", *refs]}
    agree = {k: 0 for k in refs}
    n_tok = 0
    for j, r in enumerate(rows):
        ids = np.array(r[:-1])
        tgt = np.array(r[1:])
        with torch.no_grad():
            lt = model(torch.tensor(ids[None], device=dev)).float()[0].cpu().numpy()

        def add(key, logits):
            l = logits[:, : lt.shape[1]].astype(np.float64)
            m = l.max(-1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(l - m).sum(-1))
            nll[key] += float((lse - l[np.arange(len(tgt)), tgt]).sum())

        add("torch", lt)
        for k, ref in refs.items():
            lr = ref.forward(ids)
            add(k, lr)
            agree[k] += int((lr.argmax(-1) == lt.argmax(-1)).sum())
        n_tok += len(tgt)
        print(f"  sequence {j + 1}/{n_seq}", end="\r", flush=True)

    print(f"\nverification on {n_seq} x {seq} validation tokens")
    print(f"  torch QAT model (fp32)        : perplexity {math.exp(nll['torch'] / n_tok):.2f}")
    for k in refs:
        print(f"  hwref {k:20s}: perplexity {math.exp(nll[k] / n_tok):.2f}"
              f"   top-1 agreement with torch {agree[k] / n_tok:.1%}")
    for k, ref in refs.items():
        if k != "float":
            print(f"  {k} fixed-point events:", ref.stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="../export")
    ap.add_argument("--golden-dir", default=None, help="folder with ssm_golden.py / nau_golden.py")
    ap.add_argument("--verify", type=int, default=0, help="number of validation sequences")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--float-ref", action="store_true", help="also run hwref with float SSM/NAU")
    ap.add_argument("--vec", default="float,fixed",
                    help="vector path(s) between layers: float (fp32) and/or fixed (vec_golden)")
    ap.add_argument("--state-fmt", default="20:16",
                    help="SSM state formats W_H:F_H to compare, e.g. 24:16,20:16,20:14")
    a = ap.parse_args()
    fmts = [tuple(int(v) for v in f.split(":")) for f in a.state_fmt.split(",")]
    cfg, ck = export(a.ckpt, a.out, a.golden_dir)
    if a.verify:
        verify(cfg, ck, a.out, a.verify, a.seq, a.golden_dir, a.float_ref, fmts, a.vec.split(","))
