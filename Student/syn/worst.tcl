project_open lmamba_core
create_timing_netlist
read_sdc
update_timing_netlist
report_timing -setup -npaths 20 -detail path_only -file worst_paths.txt
project_close