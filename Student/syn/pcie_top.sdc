# pcie_top: the hard IP SDC (ip/pcie_hip) runs derive_pll_clocks -create_base_clocks
# and derive_clock_uncertainty; coreclkout_hip (250 MHz) is derived from the refclk.
create_clock -name pcie_refclk -period 10.000 [get_ports pcie_refclk]

# asynchronous inputs / slow outputs
set_false_path -from [get_ports {pcie_perstn npor_n}]
set_false_path -to   [get_ports {led_n[*]}]
