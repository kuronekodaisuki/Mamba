#!/bin/bash
# Synthesis, fit and timing of core_top as separate steps (the combined flow
# crashed on this machine). Run from syn/ with Quartus Pro 25.3 on PATH,
# outside conda.
set -e
Q="env -u LD_LIBRARY_PATH"
P=lmamba_core
rm -rf qdb tmp-clearbox dni
$Q quartus_sh -t build_core.tcl
$Q quartus_syn --parallel=1 $P      # parallel synthesis hung in the helper processes
$Q quartus_fit $P
$Q quartus_sta $P
$Q quartus_sta -t worst.tcl
echo "see output_files/$P.syn.summary, $P.fit.summary, $P.sta.summary, worst_paths.txt"
