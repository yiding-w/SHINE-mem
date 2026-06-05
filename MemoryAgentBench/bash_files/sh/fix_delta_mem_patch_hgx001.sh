#!/usr/bin/env bash
# One-liner fix on hgx001 when benchmark_compare rejects --no-skip-shine.
set -euo pipefail
SHINE_ROOT="${SHINE_ROOT:-/ceph/home/muhan01/wyd/SHINE-mem}"
DELTA_MEM_ROOT="${DELTA_MEM_ROOT:-${SHINE_ROOT}/third_party/delta-Mem}"
MAB_ROOT="${MAB_ROOT:-${SHINE_ROOT}/MemoryAgentBench}"
bash "${MAB_ROOT}/deltamem_patches/apply_to_delta_mem.sh" "${DELTA_MEM_ROOT}"
echo "OK. Retry: bash ${MAB_ROOT}/bash_files/sh/run_delta_mem_hgx001.sh compare-mab"
