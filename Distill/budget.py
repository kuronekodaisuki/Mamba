#!/usr/bin/env python3
"""On-chip memory budget of Mamba2 student models on Arria 10 GX 1150.

Mamba2 block (mamba_ssm, ngroups=1, expand=2, d_conv=4):
  in_proj  : d_model -> 2*d_inner + 2*d_state + nheads   (z, x, B, C, dt)
  conv1d   : (d_inner + 2*d_state) channels x 4 taps (+bias)
  dt_bias, A_log, D : nheads ; gated RMSNorm : d_inner ; out_proj : d_inner -> d_model
Embedding / tied LM head (vocab x d_model) are kept in DDR4: a token needs only
one embedding row, and the head is read once per generated token.
"""
M20K_BLOCKS, M20K_BITS = 2713, 20 * 1024
ONCHIP = M20K_BLOCKS * M20K_BITS                  # 55.6e6 bits
TARGET = 0.70                                     # leave room for packing/FIFOs
VOCAB = 50288                                     # Mamba2 (GPT-NeoX tokenizer, padded)

W_BITS, GROUP, SCALE_BITS = 4, 128, 16            # linear weights, per-group scale
CONV_BITS, VEC_BITS = 8, 16
STATE_BITS, CONVST_BITS, ACT_BITS = 24, 8, 16


def budget(d_model, n_layer, d_state=64, headdim=64, expand=2):
    di = expand * d_model
    nh = di // headdim
    conv_dim = di + 2 * d_state
    lin = d_model * (2 * di + 2 * d_state + nh) + di * d_model
    small = conv_dim + 3 * nh + di + d_model            # conv bias, dt_bias/A_log/D, norms
    conv = conv_dim * 4
    p_layer = lin + conv + small
    w_bits = n_layer * (lin * (W_BITS + SCALE_BITS / GROUP)
                        + conv * CONV_BITS + small * VEC_BITS)
    state = n_layer * di * d_state * STATE_BITS
    convst = n_layer * conv_dim * 3 * CONVST_BITS
    act = n_layer * 2 * (2 * di + 2 * d_state + nh) * ACT_BITS   # double-buffered
    total = w_bits + state + convst + act
    return dict(
        cfg=f"d={d_model:4d} L={n_layer:2d} N={d_state:3d} heads={nh:2d}",
        body_M=n_layer * p_layer / 1e6,
        emb_MB=VOCAB * d_model * 4 / 8 / 1e6,
        w_Mb=w_bits / 1e6, st_Mb=(state + convst + act) / 1e6,
        use=total / ONCHIP,
        mac_body=n_layer * (lin + conv) / 1e6, mac_head=VOCAB * d_model / 1e6,
    )


if __name__ == "__main__":
    print(f"M20K total {ONCHIP/1e6:.1f} Mb, target <= {TARGET:.0%}\n")
    print(f"{'config':32s} {'body':>6s} {'weights':>8s} {'state+buf':>9s} {'on-chip':>8s}"
          f" {'emb/head':>9s} {'MAC body':>9s} {'MAC head':>9s}")
    print(f"{'':32s} {'Mparam':>6s} {'Mb':>8s} {'Mb':>9s} {'use':>8s} {'DDR4 MB':>9s}"
          f" {'M/token':>9s} {'M/token':>9s}")
    for d, L, N in [(256, 8, 64), (256, 12, 64), (256, 16, 64), (256, 24, 64),
                    (256, 12, 128), (512, 4, 64), (512, 6, 64)]:
        b = budget(d, L, N)
        flag = "  ok" if b["use"] <= TARGET else ("  tight" if b["use"] <= 1 else "  NO")
        print(f"{b['cfg']:32s} {b['body_M']:6.2f} {b['w_Mb']:8.1f} {b['st_Mb']:9.1f}"
              f" {b['use']:7.0%}{flag:6s} {b['emb_MB']:7.1f} {b['mac_body']:9.2f} {b['mac_head']:9.2f}")
