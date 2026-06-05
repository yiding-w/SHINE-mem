#!/usr/bin/env bash
# One-time setup on hgx001: δ-mem venv + SHINE deps + directory layout.
set -euo pipefail

SHINE_ROOT="${SHINE_ROOT:-/ceph/home/muhan01/wyd/SHINE-mem}"
DELTA_MEM_ROOT="${DELTA_MEM_ROOT:-${SHINE_ROOT}/third_party/delta-Mem}"
MAB_ROOT="${MAB_ROOT:-${SHINE_ROOT}/MemoryAgentBench}"

echo "SHINE_ROOT=${SHINE_ROOT}"
echo "DELTA_MEM_ROOT=${DELTA_MEM_ROOT}"
echo "MAB_ROOT=${MAB_ROOT}"

if [[ ! -d "${SHINE_ROOT}" ]]; then
  echo "SHINE_ROOT not found: ${SHINE_ROOT}" >&2
  exit 1
fi

mkdir -p "${SHINE_ROOT}/third_party"
if [[ ! -d "${DELTA_MEM_ROOT}/.git" ]]; then
  git clone --depth 1 https://github.com/declare-lab/delta-Mem.git "${DELTA_MEM_ROOT}"
fi

# Overlay SHINE support into the δ-mem clone (benchmark_compare + shine_memory_agent_bench.py).
bash "${MAB_ROOT}/deltamem_patches/apply_to_delta_mem.sh" "${DELTA_MEM_ROOT}"

cd "${DELTA_MEM_ROOT}"

# Prefer uv env from upstream; fall back to plain venv if uv missing.
if command -v uv >/dev/null 2>&1; then
  INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}" bash scripts/setup_uv_env.sh
  PYTHON_BIN="${DELTA_MEM_ROOT}/.venv/bin/python"
else
  echo "uv not found; creating plain venv"
  python3 -m venv .venv
  PYTHON_BIN="${DELTA_MEM_ROOT}/.venv/bin/python"
  "${PYTHON_BIN}" -m pip install -U pip setuptools wheel
  DS_BUILD_OPS=0 "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

# SHINE + MAB lightweight deps into the same venv.
"${PYTHON_BIN}" -m pip install -r "${MAB_ROOT}/requirements-shine-mab.txt"
"${PYTHON_BIN}" -m pip install hydra-core omegaconf scikit-learn nltk rouge_score editdistance

# Verify imports.
export PYTHONPATH="${DELTA_MEM_ROOT}:${SHINE_ROOT}:${MAB_ROOT}"
"${PYTHON_BIN}" - <<'PY'
import torch
from deltamem.eval import benchmark_compare
print("delta-mem OK", torch.__version__, "cuda=", torch.cuda.is_available())
PY

cat <<EOF

Setup complete.

Activate for runs:
  export SHINE_ROOT=${SHINE_ROOT}
  export DELTA_MEM_ROOT=${DELTA_MEM_ROOT}
  export MAB_ROOT=${MAB_ROOT}
  export PYTHONPATH=\${DELTA_MEM_ROOT}:\${SHINE_ROOT}:\${MAB_ROOT}
  export PYTHON_BIN=${PYTHON_BIN}

Then:
  bash ${MAB_ROOT}/bash_files/sh/run_delta_mem_hgx001.sh smoke

EOF
