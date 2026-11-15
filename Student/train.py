#!/usr/bin/env python3
"""Distill Mamba2-130M (FP16 teacher) into an on-chip-sized student, then QAT.

Stage 1 (float distillation):
  python3 student/train.py --stage distill --out ckpt/distill.pt
Stage 2 (quantization-aware distillation with the FPGA arithmetic):
  python3 student/train.py --stage qat --init ckpt/distill.pt --nau improved --out ckpt/qat.pt

Requirements (CUDA GPU): torch, transformers, datasets, huggingface_hub
mamba-ssm is optional (Linux only); without it the teacher runs on the pure-PyTorch
implementation in model.py (see teacher.py).
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from model import StudentConfig, StudentLM
from teacher import load_teacher


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["distill", "qat"], required=True)
    ap.add_argument("--teacher", default="state-spaces/mamba2-130m")
    ap.add_argument("--teacher-impl", choices=["auto", "torch", "mamba_ssm"], default="auto")
    ap.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--init", default=None, help="student checkpoint to start from")
    ap.add_argument("--out", required=True)
    # student shape (fits Arria 10 M20K, see student/budget.py)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=12)
    ap.add_argument("--d-state", type=int, default=64)
    # quantization (stage qat)
    ap.add_argument("--w-bits", type=int, default=4)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--emb-bits", type=int, default=8)
    ap.add_argument("--no-hadamard", action="store_true")
    ap.add_argument("--nau", choices=["off", "published", "improved"], default="improved")
    # optimisation
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.9, help="weight of the KD loss")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=50)
    return ap.parse_args()


def token_stream(args, tok, split):
    """Endless stream of packed (seq+1)-token blocks."""
    ds = load_dataset(args.dataset, args.dataset_config, split=split, streaming=True)
    buf = []
    while True:
        for ex in ds:
            text = ex[args.text_field]
            if not text or not text.strip():
                continue
            buf.extend(tok(text)["input_ids"] + [tok.eos_token_id])
            while len(buf) >= args.seq + 1:
                yield buf[: args.seq + 1]
                buf = buf[args.seq + 1:]
        if split != "train":
            return


def batches(stream, bs, device):
    while True:
        rows = [next(stream) for _ in range(bs)]
        t = torch.tensor(rows, dtype=torch.long, device=device)
        yield t[:, :-1], t[:, 1:]


def kd_loss(s_logits, t_logits, labels, temp, alpha, vocab):
    s = s_logits[..., :vocab].float()
    t = t_logits[..., :vocab].float()
    kd = F.kl_div(F.log_softmax(s / temp, -1), F.log_softmax(t / temp, -1),
                  log_target=True, reduction="batchmean") * temp * temp / s.shape[1]
    ce = F.cross_entropy(s.reshape(-1, vocab), labels.reshape(-1))
    return alpha * kd + (1 - alpha) * ce, kd.detach(), ce.detach()


@torch.no_grad()
def evaluate(model, args, tok, device, n_batches):
    model.eval()
    it = batches(token_stream(args, tok, "validation"), 8, device)
    tot, n = 0.0, 0
    try:
        for _ in range(n_batches):
            x, y = next(it)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            tot += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   y.reshape(-1), reduction="sum").item()
            n += y.numel()
    except (StopIteration, RuntimeError):
        pass
    model.train()
    return math.exp(tot / max(n, 1))


def main():
    args = get_args()
    device = "cuda"
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    teacher, vocab = load_teacher(args.teacher, args.teacher_impl, device)

    if args.init:
        ck = torch.load(args.init, map_location="cpu")
        cfg = StudentConfig(**ck["cfg"])
    else:
        ck = None
        cfg = StudentConfig(vocab_size=vocab, d_model=args.d_model,
                            n_layer=args.n_layer, d_state=args.d_state)
    if args.stage == "qat":
        cfg.quant, cfg.w_bits, cfg.a_bits = True, args.w_bits, args.a_bits
        cfg.emb_bits, cfg.hadamard, cfg.nau = args.emb_bits, not args.no_hadamard, args.nau

    model = StudentLM(cfg).to(device)
    if ck is not None:
        model.load_state_dict(ck["model"])
    n_par = sum(p.numel() for p in model.parameters())
    n_emb = model.emb.weight.numel()
    print(f"student: {cfg}\n  body {(n_par - n_emb)/1e6:.2f}M params, embedding {n_emb/1e6:.2f}M")

    lr = args.lr or (1e-3 if args.stage == "distill" else 1e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / args.warmup)
                                              * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    it = batches(token_stream(args, tok, "train"), args.batch, device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, y = next(it)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t_logits = teacher(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_logits = model(x)
        loss, kd, ce = kd_loss(s_logits, t_logits, y, args.temp, args.alpha, vocab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f"step {step:6d}  loss {loss.item():.4f}  kd {kd.item():.4f}  ce {ce.item():.4f}"
                  f"  lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            ppl = evaluate(model, args, tok, device, args.eval_batches)
            print(f"== step {step}: validation perplexity {ppl:.2f}", flush=True)
            torch.save({"cfg": cfg.to_dict(), "model": model.state_dict(), "step": step}, args.out)

    print("teacher validation perplexity:",
          f"{evaluate(teacher, args, tok, device, args.eval_batches):.2f}")


if __name__ == "__main__":
    main()
