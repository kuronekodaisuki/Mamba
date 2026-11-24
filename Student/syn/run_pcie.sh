#!/bin/bash
# Synthesis, fit and timing of pcie_top (board top) as separate steps (the combined flow
# crashed on this machine). Run from syn/. Quartus environment and the retry on
# the start-up crash: qrun.sh. Do not run run_core.sh and run_pcie.sh at the
# same time (both clear qdb / dni in this folder).
set -e
cd "$(dirname "$0")"
. ./qrun.sh
P=lpcie
rm -rf qdb tmp-clearbox dni
run quartus_sh -t build_pcie.tcl
# synthesis normally takes 1-3 min; it sometimes hangs in a futex wait (even with
# --parallel=1), so it is killed after 30 min
run timeout 1800 quartus_syn --parallel=1 $P
run quartus_fit $P
run quartus_asm $P                  # .sof for JTAG configuration
run quartus_sta $P
run quartus_sta -t worst_pcie.tcl
echo "see output_files/$P.syn.summary, $P.fit.summary, $P.sta.summary, worst_paths_pcie.txt, $P.sof"
