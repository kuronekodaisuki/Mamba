# Quartus Prime Pro: project for the board top pcie_top (PCIe hard IP + host_if + mamba_core)
#   quartus_sh -t build_pcie.tcl      (project setup only; run_pcie.sh runs the stages)
# Pins: Arria 10 GX FPGA Development Kit, taken from the Intel PCIe example design
# (ip/altera/altera_pcie/altera_pcie_a10_ed/example_design/a10-revd-devkit-pinout-qsys-script.tcl)
load_package project

set proj lpcie
project_new $proj -overwrite

set_global_assignment -name FAMILY "Arria 10"
set_global_assignment -name DEVICE 10AX115S2F45I1SG
set_global_assignment -name TOP_LEVEL_ENTITY pcie_top
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files

foreach f {
  fixed_point.sv adder_tree.sv sdp_ram.sv sdp_ram_wide.sv nau.sv ssm_discretize.sv
  ssm_scan_unit.sv lin_dot.sv lin_scale.sv lin_unit.sv norm_q8.sv silu_unit.sv
  gate_unit.sv conv_unit.sv mamba_core.sv host_if.sv pcie_top.sv
} {
  set_global_assignment -name SYSTEMVERILOG_FILE $f
}
set_global_assignment -name IP_FILE ip/pcie_hip.ip
set_global_assignment -name SDC_FILE pcie_top.sdc
set_global_assignment -name OPTIMIZATION_MODE "SUPERIOR PERFORMANCE"
set_global_assignment -name AUTO_SHIFT_REGISTER_RECOGNITION OFF

# ---- pins --------------------------------------------------------------------------------
set_location_assignment PIN_AL37 -to pcie_refclk
set_location_assignment PIN_AL38 -to "pcie_refclk(n)"
set_instance_assignment -name IO_STANDARD HCSL -to pcie_refclk
set_instance_assignment -name XCVR_IO_PIN_TERMINATION 100_OHMS -to pcie_refclk

set_location_assignment PIN_BC30 -to pcie_perstn
set_instance_assignment -name IO_STANDARD "1.8 V" -to pcie_perstn
set_location_assignment PIN_T12 -to npor_n
set_instance_assignment -name IO_STANDARD "1.8 V" -to npor_n

set rx  {PIN_AT40 PIN_AP40 PIN_AN42 PIN_AM40 PIN_AL42 PIN_AK40 PIN_AJ42 PIN_AH40}
set rxn {PIN_AT39 PIN_AP39 PIN_AN41 PIN_AM39 PIN_AL41 PIN_AK39 PIN_AJ41 PIN_AH39}
set tx  {PIN_BB44 PIN_BA42 PIN_AY44 PIN_AW42 PIN_AV44 PIN_AU42 PIN_AT44 PIN_AR42}
set txn {PIN_BB43 PIN_BA41 PIN_AY43 PIN_AW41 PIN_AV43 PIN_AU41 PIN_AT43 PIN_AR41}
for {set i 0} {$i < 8} {incr i} {
  set_location_assignment [lindex $rx $i]  -to "pcie_rx\[$i\]"
  set_location_assignment [lindex $rxn $i] -to "pcie_rx\[$i\](n)"
  set_location_assignment [lindex $tx $i]  -to "pcie_tx\[$i\]"
  set_location_assignment [lindex $txn $i] -to "pcie_tx\[$i\](n)"
  set_instance_assignment -name IO_STANDARD CML -to "pcie_rx\[$i\]"
  set_instance_assignment -name IO_STANDARD "HSSI DIFFERENTIAL I/O" -to "pcie_tx\[$i\]"
  set_instance_assignment -name XCVR_VCCR_VCCT_VOLTAGE 1_0V -to "pcie_rx\[$i\]"
  set_instance_assignment -name XCVR_VCCR_VCCT_VOLTAGE 1_0V -to "pcie_tx\[$i\]"
}

# user LEDs (active low, 1.8 V): heartbeat, link, busy, done
set leds {PIN_K26 PIN_L28 PIN_K25 PIN_BA27}
for {set i 0} {$i < 4} {incr i} {
  set_location_assignment [lindex $leds $i] -to "led_n\[$i\]"
  set_instance_assignment -name IO_STANDARD "1.8 V" -to "led_n\[$i\]"
}

project_close
