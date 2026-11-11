# Mamba2 SSM scan unit for Arria 10 (step 1 of a LightMamba-style accelerator)

```
rtl/fixed_point.sv      rshift_round / saturate helpers
rtl/adder_tree.sv       pipelined adder tree
rtl/ssm_scan_unit.sv    selective-scan state update (1 row/clk, N lanes)
model/ssm_golden.py     bit-accurate golden model + vector generator
tb/tb_ssm_scan_unit.sv  self-checking testbench
```

## Simulate (Questa-Intel FPGA Edition)

```
python3 model/ssm_golden.py --out vec        # N=8 P=4 heads=2 T=6
vlog -sv rtl/fixed_point.sv rtl/adder_tree.sv rtl/ssm_scan_unit.sv tb/tb_ssm_scan_unit.sv
vsim -c tb_ssm_scan_unit -do "run -all; quit"
```

Expected: `PASS: 48 outputs match the golden model`.
If you change N / P / heads / T, change both the generator arguments and the
localparams at the top of the testbench.

## Accuracy vs float (full Mamba2-130M head config, 512 tokens)

```
python3 model/ssm_golden.py --N 128 --P 64 --heads 24 --T 512 --no-write
```

## FastMamba-style SSM front end (step 2)

```
rtl/nau.sv              Nonlinear Approximation Unit (exp / softplus, 7-cycle pipeline)
rtl/ssm_discretize.sv   delta = softplus(dt_raw + dt_bias), dA = exp(delta*A)
model/nau_golden.py     bit-accurate model, coefficient table, accuracy report
tb/tb_nau.sv, tb/tb_ssm_discretize.sv
```

```
python3 model/nau_golden.py --out vec                 # as published in FastMamba
vlog -sv rtl/fixed_point.sv rtl/nau.sv rtl/ssm_discretize.sv tb/tb_nau.sv tb/tb_ssm_discretize.sv
vsim -c tb_nau -do "run -all; quit"
vsim -c tb_ssm_discretize -do "run -all; quit"

python3 model/nau_golden.py --improved --out vec      # EXP_NEAR0 = SP_ORDER2 = 1
vlog -sv +define+IMPROVED ...                         # same file list
```
