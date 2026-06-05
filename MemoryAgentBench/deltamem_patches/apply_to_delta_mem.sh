#!/usr/bin/env bash
# Apply SHINE patches onto a fresh declare-lab/delta-Mem clone.
set -euo pipefail

PATCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
DELTA_MEM_ROOT="${1:-${DELTA_MEM_ROOT:-}}"

if [[ -z "${DELTA_MEM_ROOT}" ]]; then
  echo "Usage: DELTA_MEM_ROOT=/path/to/delta-Mem bash apply_to_delta_mem.sh" >&2
  exit 1
fi

if [[ ! -d "${DELTA_MEM_ROOT}/deltamem" ]]; then
  echo "Invalid delta-Mem root: ${DELTA_MEM_ROOT}" >&2
  exit 1
fi

echo "Applying SHINE patches to ${DELTA_MEM_ROOT}"

mkdir -p "${DELTA_MEM_ROOT}/deltamem/eval"
cp "${PATCH_ROOT}/deltamem/eval/shine_memory_agent_bench.py" \
  "${DELTA_MEM_ROOT}/deltamem/eval/shine_memory_agent_bench.py"

cd "${DELTA_MEM_ROOT}"
if [[ -f "${PATCH_ROOT}/benchmark_compare_shine.patch" ]]; then
  if git apply --check "${PATCH_ROOT}/benchmark_compare_shine.patch" 2>/dev/null; then
    git apply "${PATCH_ROOT}/benchmark_compare_shine.patch"
    echo "Applied benchmark_compare_shine.patch"
  else
    echo "benchmark_compare_shine.patch already applied or conflicts; skipping patch step"
  fi
fi

# Older patch used store_true → no --no-skip-shine; upgrade in place if needed.
BC_FILE="${DELTA_MEM_ROOT}/deltamem/eval/benchmark_compare.py"
if grep -q 'skip-shine.*store_true' "${BC_FILE}" 2>/dev/null; then
  sed -i 's/--skip-shine", action="store_true", default=True/--skip-shine", action=argparse.BooleanOptionalAction, default=True/' "${BC_FILE}"
  echo "Upgraded --skip-shine to BooleanOptionalAction (--no-skip-shine supported)"
fi

if [[ -f "${PATCH_ROOT}/scripts/run_shine_mab_qwen3_8b.sh" ]]; then
  mkdir -p "${DELTA_MEM_ROOT}/scripts"
  cp "${PATCH_ROOT}/scripts/run_shine_mab_qwen3_8b.sh" "${DELTA_MEM_ROOT}/scripts/"
  chmod +x "${DELTA_MEM_ROOT}/scripts/run_shine_mab_qwen3_8b.sh"
fi

echo "SHINE patches applied."
