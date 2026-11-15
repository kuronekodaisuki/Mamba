#!/usr/bin/env python3
"""Teacher loader: Mamba2 checkpoints from the Hugging Face Hub.

impl = "torch"    : loads the mamba_ssm-format weights into the pure-PyTorch
                    StudentLM (no CUDA extensions; works on Windows)
impl = "mamba_ssm": uses mamba_ssm.MambaLMHeadModel (Linux, fast kernels)
impl = "auto"     : mamba_ssm if it can be imported, otherwise torch

Sanity check (prints teacher perplexity on a short text, and the difference
between the two implementations when both are available):
  python teacher.py --check
"""
import argparse
import json
import re

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from model import StudentConfig, StudentLM


class _MambaSsmWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).logits


def _map_key(k):
    k = k.replace("backbone.embedding.", "emb.").replace("backbone.norm_f.", "norm_f.")
    m = re.match(r"backbone\.layers\.(\d+)\.(.*)", k)
    if not m:
        return k
    i, rest = m.groups()
    if rest == "norm.weight":
        return f"norms.{i}.weight"
    rest = rest.replace("mixer.", "")
    rest = {"conv1d.weight": "conv.weight", "conv1d.bias": "conv.bias",
            "norm.weight": "norm_w"}.get(rest, rest)
    return f"layers.{i}.{rest}"


def load_torch_teacher(name, device):
    cj = json.load(open(hf_hub_download(name, "config.json")))
    ssm = cj.get("ssm_cfg", {}) or {}
    if ssm.get("layer") != "Mamba2":
        raise ValueError(f"{name} is not a Mamba2 checkpoint")
    if ssm.get("ngroups", 1) != 1:
        raise ValueError("only ngroups = 1 is supported")
    sd = torch.load(hf_hub_download(name, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    emb = sd["backbone.embedding.weight"]
    head = sd.pop("lm_head.weight", None)
    if head is not None and not torch.equal(head, emb):
        raise ValueError("untied LM head is not supported")
    cfg = StudentConfig(vocab_size=emb.shape[0], d_model=cj["d_model"], n_layer=cj["n_layer"],
                        d_state=ssm.get("d_state", 128), headdim=ssm.get("headdim", 64),
                        expand=ssm.get("expand", 2), d_conv=ssm.get("d_conv", 4))
    model = StudentLM(cfg)
    new = {_map_key(k): v.float() for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(new, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"weight mapping failed\n missing: {missing}\n unexpected: {unexpected}")
    return model.to(device).eval().requires_grad_(False), cfg.vocab_size


def load_mamba_ssm_teacher(name, device):
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    m = MambaLMHeadModel.from_pretrained(name, device=device, dtype=torch.bfloat16)
    m.eval().requires_grad_(False)
    return _MambaSsmWrap(m), m.lm_head.weight.shape[0]


def load_teacher(name, impl, device):
    if impl == "auto":
        try:
            import mamba_ssm  # noqa: F401
            impl = "mamba_ssm"
        except Exception:
            impl = "torch"
    print(f"teacher: {name} ({impl} implementation)")
    if impl == "mamba_ssm":
        return load_mamba_ssm_teacher(name, device)
    return load_torch_teacher(name, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="state-spaces/mamba2-130m")
    ap.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    import torch.nn.functional as F

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    text = ("The Arria 10 is a family of mid-range FPGAs manufactured on a 20 nm process. "
            "It integrates hardened floating-point DSP blocks, high-speed transceivers and "
            "PCI Express hard IP, and is widely used for networking, signal processing and "
            "acceleration of machine learning workloads in data centers.")
    ids = torch.tensor([tok(text)["input_ids"]], device=dev)
    t, _ = load_torch_teacher(a.teacher, dev)
    with torch.no_grad():
        lt = t(ids).float()
    ppl = torch.exp(F.cross_entropy(lt[0, :-1], ids[0, 1:])).item()
    print(f"torch teacher perplexity on sample text: {ppl:.2f} (expect roughly 10-40)")
    try:
        m, _ = load_mamba_ssm_teacher(a.teacher, dev)
        with torch.no_grad():
            lm = m(ids).float()
        print(f"max |logit diff| torch vs mamba_ssm: {(lt - lm).abs().max().item():.3f} "
              "(bf16 reference, small values expected)")
    except Exception as e:
        print(f"mamba_ssm comparison skipped ({type(e).__name__})")
