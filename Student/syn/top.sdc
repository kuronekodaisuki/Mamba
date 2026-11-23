# 300 MHz target: the timing report then shows how far above / below it we are
create_clock -name clk -period 4.000 [get_ports clk]
derive_clock_uncertainty

# measurement wrapper: I/O timing is not meaningful
set_false_path -from [get_ports {rst_n sin ctl[*]}]
set_false_path -to   [get_ports sout]
