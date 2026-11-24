# Mamba2 student model on Arria 10 GX (LightMamba / FastMamba style accelerator)

This file is the hand-off from an earlier chat session. Read it before touching anything.

## Goal

Run a distilled, quantized Mamba2 student (teacher: state-spaces/mamba2-130m) fully on-chip
on the Intel Arria 10 GX FPGA Development Kit (10AX115S2F45I1SG), token by token, and later
compare throughput with a GPU. The owner has done this before for a CNN (MiniSeg, int8,
1 px/clk on Arria II) and knows SystemVerilog.

## Machine / tools

- Ubuntu, RTX 3080 Ti 12 GB (training / export), Python env with torch (+ mamba_ssm installed).
- Quartus Prime Pro 25.3 at `~/altera_pro/25.3/quartus` (bin is `$B` in the owner's shell).
- Questa (vlog / vsim) for simulation.

### Quartus workarounds on this machine (all required)

- Run every Quartus tool as `env -u LD_LIBRARY_PATH quartus_xxx ...`, outside conda.
- Run the stages separately (`quartus_sh -t build_core.tcl`, `quartus_syn`, `quartus_fit`,
  `quartus_sta`); the combined flow crashed.
- Synthesis with parallel helpers hangs (futex wait): use `quartus_syn --parallel=1`.
- `AUTO_SHIFT_REGISTER_RECOGNITION OFF` (in build_core.tcl).
- After an interrupted run: `rm -rf qdb tmp-clearbox dni`.
- Fit takes ~7-10 min, synthesis a few minutes.
- `maxfan` is ignored by Quartus Pro; use explicit duplicated registers with `(* dont_merge *)`.
- Never write a multiplier with an operand wider than 18 bits that Quartus could map onto the
  18x19 mode AY port (Error 15649): split into partial products of <= 18 bits.
- Memories declared inside generate loops became register + mux: always instantiate `sdp_ram`.

## Folder layout (owner's machine)

```
Mamba/
  ckpt/                 checkpoints (distill.pt, qat.pt, qat_nohad.pt, ...)
  export/               export of the trained student (weights.npz, mem/, dump/, manifest.json)
  testmodel/            small random model (2 layers, d_model 128) for quick RTL tests
  Student/              python: model.py teacher.py train.py budget.py export.py hwref.py to_onnx.py
                        (+ golden models ssm_golden.py nau_golden.py lin_golden.py vec_golden.py,
                         hwref.py finds them in ../model, next to itself, or --golden-dir)
    syn/                ALL .sv files (rtl + testbenches + synthesis wrapper) live here, flat
      vec/mem/          memory images  (copied from export/mem or testmodel/mem)
      vec/dump/         per-layer reference dumps (copied from export/dump or testmodel/dump)
```

`build_core.tcl` reads RTL from the current folder (syn/) only. (It used to prefer `../rtl/`,
which held stale copies: that was why the previous runs ignored the RTL edits.)

## Model and number formats

Student: d_model 256, 12 layers, d_inner 512, d_state N 64, headdim P 64 (8 heads/layer),
conv 4 taps, W4 (int4 weights, fp16 scale per 128-group) x A8 (int8 activations, per-token
power-of-two scale 2^k). Embedding / LM head (50288 x 256) stays off-chip (DDR4, not done yet).
Perplexity: teacher 27.77, distilled 44.82, QAT without Hadamard 45.63; on the exported
512-token val set: torch 35.21, hwref fixed-point SSM 35.11, full fixed-point vector path 35.06.

Fixed point (bit exact between RTL and `hwref.py --vec fixed` / `vec_golden.py`):
residual h Q.16 32b; in_proj out Q.12 24b; out_proj out Q.16 32b; norm weight Q.12 18b;
SSM state 20b Q.16; x Q.10 16b; dt*x frac 14; B, C Q.12 16b; dA Q.16 18b; delta Q.18 24b;
y Q.16 24b; dt_raw / dt_bias / A Q.10 16b; D Q.12.
lin_scale: exact integer sum of acc_g * fp16 mantissa << sh (units 2^-24), one round-half-up
shift by k-24+F_OUT, saturate. norm_q8: exact mean square, rsqrt PWL (96 segments),
power-of-two int8 scale. SiLU: 128-segment sigmoid PWL on [0,16). NAU (exp / softplus):
FastMamba 8-segment 2^x PWL with the "improved" options EXP_NEAR0 = SP_ORDER2 = 1.

**Rule: any change of arithmetic in RTL must be mirrored in vec_golden.py / hwref.py and
re-verified. Pure pipelining changes must leave tb_mamba_core bit exact.**

## RTL modules (Student/syn)

| file | role | latency |
|---|---|---|
| fixed_point.sv | rshift_round, saturate | comb |
| adder_tree.sv | pipelined adder tree | log2 N |
| sdp_ram.sv | 1W1R RAM template, optional IN_REG / OUT_REG | 1+IN+OUT |
| sdp_ram_wide.sv | banked (8192 deep) x sliced (40 b) RAM with 2-level address register tree | 5 |
| nau.sv | exp / softplus | 10 |
| ssm_discretize.sv | delta = softplus(dt+bias), dA = exp(delta*A); also pipe_delay | 24 |
| ssm_scan_unit.sv | h = dA*h + xdt*B, y = C.h + D*x, state for all 96 heads in M20K | 10+log2 N |
| lin_dot.sv | 128-lane int8 x int4 dot | 2+log2 K |
| lin_scale.sv | row rescale (see above) | 11 |
| lin_unit.sv | act buffer + weight address gen + lin_dot + lin_scale (RL = memory latency) | |
| norm_q8.sv | RMSNorm + int8 quantization (FSM S_C1..S_C9 incl. S_C7B/S_C8B/S_C8C) | |
| silu_unit.sv / gate_unit.sv | SiLU; g = y*SiLU(z) | 7 / 9 |
| conv_unit.sv | depthwise 4-tap conv + SiLU, history per (layer, channel) | |
| mamba_core.sv | layer FSM: norm -> in_proj -> conv -> discretize -> SSM -> gate -> gated norm -> out_proj -> residual, all 12 layers of one token; host write port cfg_* | |
| core_top.sv | synthesis wrapper: serial shift chain -> cfg port, XOR-reduced output | |
| top.sdc | 250 MHz (4.0 ns), I/O false paths | |
| build_core.tcl / run_core.sh / worst.tcl | project setup / full flow / worst paths | |

mamba_core memories: linw 40128 x 512 + lins 40128 x 16 (sdp_ram_wide, lin_unit RL = 5),
normw, convp, ssmp, h, z, x. Host port cfg_sel: 0 linw, 1 lins, 2 normw, 3 convp, 4 ssmp, 5 h.

## Verification (all PASS before the last RTL edits)

Unit tests: tb_nau, tb_ssm_discretize, tb_ssm_scan_unit, tb_lin_dot, tb_lin_scale,
tb_lin_unit, tb_norm_q8, tb_conv_unit, tb_silu_gate.

Full core, small model:
```
python3 hwref.py --make-test-model ../testmodel
python3 export.py --out ../testmodel --mem --dump 6
cp -r ../testmodel/mem/* syn/vec/mem/ ; cp -r ../testmodel/dump/* syn/vec/dump/
```
Full core, trained student (the important one):
```
python3 export.py --out ../export --mem --dump 6
cp ../export/mem/* syn/vec/mem/ ; cp ../export/dump/* syn/vec/dump/
cd syn
vlog -sv +define+FULL fixed_point.sv adder_tree.sv sdp_ram.sv sdp_ram_wide.sv nau.sv \
  ssm_discretize.sv ssm_scan_unit.sv lin_dot.sv lin_scale.sv lin_unit.sv norm_q8.sv \
  silu_unit.sv gate_unit.sv conv_unit.sv mamba_core.sv tb_mamba_core.sv
vsim -c tb_mamba_core -do "run -all; quit"
```
Expected: `PASS: 6 tokens x 12 layers match the fixed-point reference` (~4 min).
The testbench stops with $fatal if vec/mem or vec/dump do not match the compiled config.
Result so far: 18432 residual, 83520 in_proj, 46080 conv, 36864 SSM y, 36864 gate values,
144 norm exponents, 0 mismatches. ~60,600 cycles per token (~243 us at 250 MHz, ~4,100 tok/s).

(The edits listed here earlier were simulated on 2026-11-24: PASS; see Status below.)

## Synthesis status (core_top, 250 MHz target)

Previous state (before 2026-11-24, stale ../rtl was synthesized; kept for history):
Resources: ALM 16.6k (4 %), M20K 2213 / 2713 (82 %), DSP 185 (12 %).
Timing: setup slack -2.108 ns (~164 MHz), TNS -1808.

Worst paths: `u_scan|state_mem ... ram_block~reg1 -> hprev[..] (1 LUT) -> u_scan|mult_NN~DATAOUTA0~reg0`
i.e. M20K read -> LUT -> DSP multiplier in one cycle, although the RTL has
rd_s2 -> rd_s2b -> op_h registers in between. Quartus moved/absorbed them (the node name
`hprev` is Quartus-generated; it is not in the source). Also `u_core|dA_q_rtl_rtl_0` (the
8-entry dA_q array inferred as RAM) -> scan multipliers.

### Status 2026-11-24: timing closed at 250 MHz

- Setup slack **+0.195 ns**, TNS 0; hold +0.019 ns (slow / fast 900 mV -40C corners).
  ALM 17.1k (4 %), registers 40.8k, M20K 2082 / 2713 (77 %), DSP 183 (12 %).
  Worst path now: norm_q8 u*w multiplier -> u_d (retimed, ~RTM).
- tb_mamba_core (+define+FULL, trained student): PASS, 0 mismatches, 60,968 cycles/token
  (~244 us at 250 MHz, ~4,100 tok/s). All unit TBs PASS.
- Root cause of the earlier "no change" runs: the qsf read the stale `../rtl/`.
- Fixes (pure pipelining; numbers unchanged):
  - conv_unit: RL=2 (convp OUT_REG), first-token clear before the align register,
    2-stage tap sum (s01/s23 -> acc2, preserve), operand stage c2b before acc*m
    (Quartus had packed acc2 into the DSP input register). Latency RL + 10 + 7.
  - norm_q8: RL=3 (normw OUT_REG + nw_q register in mamba_core), state S_C6B
    (rc1*fr product registered before R), u_buf OUT_REG + pass-2 stage o1b.
  - silu_unit: round / saturate split (latency 6 -> 7); gate_unit y delay 7 (latency 9).
  - mamba_core: cfg_* / tok_go registered, delta_q ramstyle logic + operand stage q1b,
    lbase = layer*NH registered.
  - ssm_scan_unit: per-memory dont_merge write registers (WB_LAT 8).
  - sdp_ram_wide: bank select bs3 duplicated per slice, mux per slice.
- Lesson: M20K output (uTco ~1.9 ns) must never feed a DSP multiplier in the same cycle
  (DSP output-register uTsu ~2.1 ns); and a register in front of a multiplier may be
  packed into the DSP input register, so the logic before it must be short.
- Quartus on this machine (run_core.sh handles both):
  - Tools sometimes die at load time with `DDM_T::verify_token(token) : Cannot identify
    the client` (any stage, even `--version`; for a while it failed every time). Running
    with the python venv / conda removed from the environment made it work again;
    run_core.sh strips them and retries every stage once.
  - quartus_syn --parallel=1 still hung once (futex wait, ~1 % CPU, log stops in the
    altsyncram parameter list): run_core.sh kills it after 30 min and retries.
    Kill a hung run by PID, not `pkill -f run_core` (matches the calling shell too).

## PCIe host interface (2026-11-24, not yet tried on the board)

- `ip/pcie_hip.ip`: Arria 10 hard IP, Avalon-MM (no DMA), Gen2 x8, 128-bit, 250 MHz,
  BAR0 32-bit non-prefetchable 64 KB (Avalon-MM without DMA has no Gen3 x8 mode).
  Made with ip-deploy (wrala_hwtcl=7, cg_impl_cra_av_slave_port_hwtcl=0,
  avmm_addr_width_hwtcl=64) + `qsys-generate pcie_hip.ip --synthesis=VERILOG`.
- `host_if.sv`: BAR0 (rxm_bar0, bursts) -> mamba_core cfg port / control / OUT buffer.
  Map in the file header: ID, STATUS, CTRL (go / first), CFG_PTR {sel, addr},
  CAP_LAYER, CYCLES, CFG_WIN (64 B, commits on byte 63, addr auto-increments),
  OUT (D x 32-bit residual of CAP_LAYER). Register reads are sampled one cycle late
  so they see the effect of earlier writes (PCIe ordering).
- `tb_host_if.sv` (+define+FULL): loads everything through the BAR (bursts and narrow
  writes), runs 6 tokens, reads OUT with burst reads: PASS, 60,758 cycles/token.
- `pcie_top.sv` / `pcie_top.sdc` / `build_pcie.tcl` / `run_pcie.sh` (project lpcie):
  the core runs on coreclkout_hip (no CDC). Pins from the Intel A10 dev kit example
  design (refclk AL37, PERST# BC30, npor T12, 8 lanes, LEDs K26 L28 K25 BA27 assumed
  active low: heartbeat / link / busy / done).
  Result: setup +0.259 ns at 250 MHz, hold +0.017; ALM 21.8k, M20K 2101, DSP 183;
  output_files/lpcie.sof.
- Timing fixes needed in the PCIe top (core_top had closed): norm_q8 state S_C7A
  (umax -> um_q operand register before umax*R; enum now 5 bits) and pass-2 stage
  o3b (rounding constant before the variable shift).
- Host: `Student/host/mamba_host.py` (root; mmap of sysfs resource0, 8-byte ctypes
  stores in order) loads vec/mem, runs the dumped tokens, checks the last layer.
  Board: B550M Phantom Gaming 4, slot next to the GPU (probably x4 electrical ->
  Gen2 x4, enough). Configure by JTAG, then warm reboot so the BIOS enumerates it.
- Quartus DDM crash: known bug of Quartus Pro 23.3-25.3.1 (Altera KB 349714, time
  from the DDM client id off by one hour since 2026). Patch for 25.3 (0.27 per the KB)
  / fixed in 26.1. Until it is installed, qrun.sh (sourced by run_core.sh and
  run_pcie.sh) retries a stage on the start-up crash and kills a stage whose child
  died (serv_req_info.txt with ddm_set_lassessor; the parent otherwise hangs in a futex).
  Do not run the two flows at the same time (shared qdb / dni). Never `pkill -f` a
  pattern that also matches the calling shell.

## Next steps after timing closes

1. Bring-up of pcie_top on the board (see the PCIe section).
2. Embedding + LM head in DDR4 (tied, 50288 x 256).
3. Throughput: matmuls are ~70 % of the 5,050 cycles/layer; overlap stages or raise lin_dot
   parallelism; pipeline tokens/layers.
4. GPU comparison with the same distilled / quantized model.

## Working style

- The owner prefers Japanese explanations, concise.
- Before any RTL change, predict which reference values are affected; after it, re-run the
  relevant unit TB and tb_mamba_core.
- Read the actual Quartus reports (fit.rpt RAM summary, sta worst paths) instead of guessing.
