# Quartus Prime Pro: create the project for core_top (mamba_core, student configuration)
#   quartus_sh -t build_core.tcl      (project setup only; run_core.sh runs the stages)
load_package project

set proj lmamba_core
project_new $proj -overwrite

set_global_assignment -name FAMILY "Arria 10"
set_global_assignment -name DEVICE 10AX115S2F45I1SG
set_global_assignment -name TOP_LEVEL_ENTITY core_top
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files

foreach f {
  ../rtl/fixed_point.sv
  ../rtl/adder_tree.sv
  ../rtl/sdp_ram.sv
  ../rtl/sdp_ram_wide.sv
  ../rtl/nau.sv
  ../rtl/ssm_discretize.sv
  ../rtl/ssm_scan_unit.sv
  ../rtl/lin_dot.sv
  ../rtl/lin_scale.sv
  ../rtl/lin_unit.sv
  ../rtl/norm_q8.sv
  ../rtl/silu_unit.sv
  ../rtl/gate_unit.sv
  ../rtl/conv_unit.sv
  ../rtl/mamba_core.sv
  core_top.sv
} {
  set_global_assignment -name SYSTEMVERILOG_FILE $f
}
set_global_assignment -name SDC_FILE top.sdc
set_global_assignment -name OPTIMIZATION_MODE "SUPERIOR PERFORMANCE"
# workarounds found on this machine (synthesis hang / slow shift-register mapping)
set_global_assignment -name AUTO_SHIFT_REGISTER_RECOGNITION OFF

project_close
