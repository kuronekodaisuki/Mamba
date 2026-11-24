# Quartus Prime Pro: create the project for core_top (mamba_core, student configuration)
#   quartus_sh -t build_core.tcl      (project setup only; run_core.sh runs the stages)
load_package project

set proj lmamba_core
project_new $proj -overwrite

set_global_assignment -name FAMILY "Arria 10"
set_global_assignment -name DEVICE 10AX115S2F45I1SG
set_global_assignment -name TOP_LEVEL_ENTITY core_top
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files

# RTL files are always taken from this folder (a stale ../rtl was silently used before)
set rtl ""
foreach f {
  fixed_point.sv adder_tree.sv sdp_ram.sv sdp_ram_wide.sv nau.sv ssm_discretize.sv
  ssm_scan_unit.sv lin_dot.sv lin_scale.sv lin_unit.sv norm_q8.sv silu_unit.sv
  gate_unit.sv conv_unit.sv mamba_core.sv
} {
  set_global_assignment -name SYSTEMVERILOG_FILE $rtl$f
}
set_global_assignment -name SYSTEMVERILOG_FILE core_top.sv
set_global_assignment -name SDC_FILE top.sdc
set_global_assignment -name OPTIMIZATION_MODE "SUPERIOR PERFORMANCE"
# workarounds found on this machine (synthesis hang / slow shift-register mapping)
set_global_assignment -name AUTO_SHIFT_REGISTER_RECOGNITION OFF

project_close
