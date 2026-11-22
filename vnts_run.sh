#!/usr/bin/env bash
# vnts_run.sh — Run any VNTS benchmark / training script with hard CPU caps.
#
# Usage:
#   ./vnts_run.sh python3 benchmarks/run_orma_depth_scaling_benchmark.py
#   ./vnts_run.sh pytest tests/
#
# Hard limits applied:
#   - nice 19       : lowest scheduling priority (OS yields CPU to everything else first)
#   - OMP/MKL/BLAS  : 1 thread → no multi-core BLAS explosions
#
# Optional cpulimit (if installed) can be enabled by setting VNTS_USE_CPULIMIT=1.
# Default is disabled because cpulimit's --foreground mode may conflict with
# interactive terminals and pytest output buffering.
#
# To raise thread count (carefully):
#   VNTS_NUM_THREADS=2 ./vnts_run.sh python3 benchmarks/...

set -euo pipefail

THREADS="${VNTS_NUM_THREADS:-1}"
USE_CPULIMIT="${VNTS_USE_CPULIMIT:-0}"
CPU_PCT="${VNTS_CPU_PCT:-40}"

export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export BLIS_NUM_THREADS="${THREADS}"
export VECLIB_MAXIMUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export VNTS_NUM_THREADS="${THREADS}"

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <command> [args...]" >&2
    exit 1
fi

echo "[vnts_run] Threads=${THREADS} | nice=19 | cpulimit=${USE_CPULIMIT} (${CPU_PCT}%)"
echo "[vnts_run] Command: $*"

if [ "${USE_CPULIMIT}" = "1" ] && command -v cpulimit &>/dev/null; then
    exec nice -n 19 cpulimit -l "${CPU_PCT}" -f -q -- "$@"
else
    exec nice -n 19 "$@"
fi
