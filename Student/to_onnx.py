#!/usr/bin/env python3
"""Export the student to ONNX for viewing in Netron.

  pip install onnx netron
  python3 to_onnx.py --ckpt ../ckpt/distill.pt --layers 2 --functions --out ../export/student_view.onnx
  netron ../export/student_view.onnx

Options
  --float      export without fake quantization even for a QAT checkpoint
  --layers N   keep only the first N layers (the 12-layer graph is long)
  --functions  wrap every Mamba2 block in an ONNX function: Netron shows one
               node per block and you can open it to see the inside
The SSM is the chunked (SSD) form used for training, so it appears as Einsum,
CumSum and Exp nodes rather than as the recurrence the FPGA computes.
"""
import argparse
import os

import torch

from model import Mamba2Block, StudentConfig, StudentLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="../export/student.onnx")
    ap.add_argument("--seq", type=int, default=64, help="tokens of the example input")
    ap.add_argument("--layers", type=int, default=0, help="0 = all layers")
    ap.add_argument("--float", action="store_true")
    ap.add_argument("--functions", action="store_true")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    cfg = StudentConfig(**ck["cfg"])
    sd = ck["model"]
    if a.float:
        cfg.quant = False
    if a.layers:
        cfg.n_layer = a.layers
        sd = {k: v for k, v in sd.items()
              if not (k.startswith(("layers.", "norms.")) and int(k.split(".")[1]) >= a.layers)}
    model = StudentLM(cfg).eval()
    model.load_state_dict(sd)

    ids = torch.randint(0, cfg.vocab_size, (1, a.seq))
    kw = dict(input_names=["ids"], output_names=["logits"], opset_version=17,
              do_constant_folding=True)
    if a.functions:
        kw["export_modules_as_functions"] = {Mamba2Block}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with torch.no_grad():
        try:
            torch.onnx.export(model, (ids,), a.out, dynamo=False, **kw)
        except TypeError:                      # older torch without the dynamo argument
            torch.onnx.export(model, (ids,), a.out, **kw)
    print(f"wrote {a.out}  ({cfg.n_layer} layers, quant={cfg.quant}, "
          f"{os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
