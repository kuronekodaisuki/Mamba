# Quartus Prime Pro: synthesize + fit + timing of syn_top on the Arria 10 GX dev kit device
#   cd syn
#   quartus_sh -t build.tcl
# Results: output_files/lmamba_syn.fit.summary  (ALM / DSP / M20K)
#          output_files/lmamba_syn.sta.rpt      ("Fmax Summary")
load_package flow

set proj lmamba_syn
project_new $proj -overwrite

set_global_assignment -name FAMILY "Arria 10"
# Arria 10 GX FPGA Development Kit (check the part number printed on your board)
set_global_assignment -name DEVICE 10AX115S2F45I1SG
set_global_assignment -name TOP_LEVEL_ENTITY syn_top
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files

foreach f {
  ../rtl/fixed_point.sv
  ../rtl/adder_tree.sv
  ../rtl/ssm_scan_unit.sv
  ../rtl/nau.sv
  ../rtl/ssm_discretize.sv
  syn_top.sv
} {
  set_global_assignment -name SYSTEMVERILOG_FILE $f
}
set_global_assignment -name SDC_FILE top.sdc
set_global_assignment -name OPTIMIZATION_MODE "SUPERIOR PERFORMANCE"

execute_flow -compile
project_close
