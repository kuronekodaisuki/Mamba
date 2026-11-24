# Common Quartus environment / runner for run_core.sh and run_pcie.sh (source it).
#
# Quartus Prime Pro 23.3-25.3.1 bug (Altera KB 349714 "Why do I unexpectedly observe
# intermittent DDM errors?"): the time decoded from the DDM client id is off by one
# hour, so executables die at load time with
#   DDM_T::verify_token(token) : Cannot identify the client
# in windows of minutes (since 2026). Fixed by the 25.3 patch (0.27 per the KB) / 26.1.
# When a child process (e.g. the clearbox generator inside quartus_syn) dies that
# way, the parent waits forever in a futex: the only trace is serv_req_info.txt.
# `run` therefore watches for that file too, kills the stage and retries
# (every 60 s, up to 15 times). Other failures are returned as they are.
QROOT=${QROOT:-$HOME/altera_pro/25.3/quartus}
PATH=$QROOT/bin:$QROOT/../qsys/bin:$(echo "$PATH" | tr ':' '\n' | grep -v conda | paste -sd:)
[ -n "$VIRTUAL_ENV" ] && PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "^$VIRTUAL_ENV/bin" | paste -sd:)
export PATH
Q="env -u LD_LIBRARY_PATH -u VIRTUAL_ENV -u VIRTUAL_ENV_PROMPT -u PYTHONHOME -u PYTHONPATH"

killtree() {             # signal $1 to all descendants of pid $2 (not $2 itself)
  local c
  for c in $(pgrep -P "$2"); do killtree "$1" "$c"; kill -"$1" "$c" 2>/dev/null; done
}

ddm_crash_since() {      # a DDM start-up crash report newer than marker $1
  find . -maxdepth 4 -name serv_req_info.txt -newer "$1" 2>/dev/null \
    | xargs -r grep -l ddm_set_lassessor 2>/dev/null | grep -q .
}

run() {
  local i log mark pid rc hit
  log=$(mktemp); mark=$(mktemp)
  for i in $(seq 15); do
    touch "$mark"
    ( set -o pipefail; $Q "$@" 2>&1 | tee "$log" ) &   # waiting for it waits for tee too
    pid=$!
    hit=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep 10
      if [ $hit -eq 0 ] && ddm_crash_since "$mark"; then hit=1; continue; fi
      if [ $hit -eq 1 ]; then                    # still running 10 s after a child crash
        echo "*** $1: child process died with the DDM error, killing the stage"
        killtree TERM "$pid"; sleep 5; killtree KILL "$pid"
        break
      fi
    done
    rc=0; wait "$pid" || rc=$?               # (set -e safe)
    if [ $rc -eq 0 ] && [ $hit -eq 0 ]; then rm -f "$log" "$mark"; return 0; fi
    if [ $hit -eq 0 ] && ! grep -q verify_token "$log"; then rm -f "$log" "$mark"; return $rc; fi
    echo "*** $1: DDM crash (attempt $i), retrying in 60 s"
    sleep 60
  done
  rm -f "$log" "$mark"; return 1
}
