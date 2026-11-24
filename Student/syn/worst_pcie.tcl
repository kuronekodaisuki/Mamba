# quartus_sta -t worst.tcl : list the worst setup paths of lpcie
project_open lpcie
create_timing_netlist
read_sdc
update_timing_netlist
report_timing -setup -npaths 40 -detail summary -file worst_paths_pcie.txt
report_timing -setup -npaths 3 -detail full_path -file worst_detail_pcie.txt
delete_timing_netlist
project_close
