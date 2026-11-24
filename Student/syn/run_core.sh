#!/bin/bash
# Synthesis, fit and timing of core_top as separate steps (the combined flow
# crashed on this machine). Run from syn/ with Quartus Pro 25.3 on PATH,
# outside conda.
set -e
# Quartus Pro 25.3 install (change if yours is elsewhere)
QROOT=${QROOT:-$HOME/altera_pro/25.3/quartus}
export PATH=$QROOT/bin:$PATH
# no python venv / conda in the Quartus environment (an active venv was suspected
# in the "Cannot identify the client" start-up crashes)
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v conda | paste -sd:)
[ -n "$VIRTUAL_ENV" ] && PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "^$VIRTUAL_ENV/bin" | paste -sd:)
export PATH
Q="env -u LD_LIBRARY_PATH -u VIRTUAL_ENV -u VIRTUAL_ENV_PROMPT -u PYTHONHOME -u PYTHONPATH"
P=lmamba_core
# Quartus Pro 25.3 here sometimes dies right at start-up with
# "DDM_T::verify_token ... Cannot identify the client" (seen on quartus_sh and
# quartus_fit, random stage); running the same stage again works. Retry once.
run() { $Q "$@" || { echo "*** $1 failed, retrying once"; sleep 10; $Q "$@"; }; }
rm -rf qdb tmp-clearbox dni
run quartus_sh -t build_core.tcl
# synthesis normally takes 1-3 min; it sometimes hangs in a futex wait (even with
# --parallel=1), so it is killed after 30 min and retried once
run timeout 1800 quartus_syn --parallel=1 $P
run quartus_fit $P
run quartus_sta $P
run quartus_sta -t worst.tcl
echo "see output_files/$P.syn.summary, $P.fit.summary, $P.sta.summary, worst_paths.txt"
