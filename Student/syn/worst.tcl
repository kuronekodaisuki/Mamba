# quartus_sta -t worst.tcl : list the worst setup paths of lmamba_core
project_open lmamba_core
create_timing_netlist
read_sdc
update_timing_netlist
report_timing -setup -npaths 40 -detail summary -file worst_paths.txt
report_timing -setup -npaths 3 -detail full_path -file worst_detail.txt
delete_timing_netlist
project_close
