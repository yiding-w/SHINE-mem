#!/usr/bin/env bash
# One-time setup: clone doc-to-lora, install D2L + δ-mem eval paths for d2l-mab runs.
#
# Use the DEDICATED doc-to-lora conda env (transformers ~4.51.3), NOT delta-Mem .venv.
#
#   conda activate doc-to-lora   # or your D2L env name
#   export SHINE_ROOT=/path/to/SHINE-mem
#   bash MemoryAgentBench/bash_files/sh/setup_doc_to_lora_mab.sh
#
# Then:
#   export PYTHON_BIN=$(which python)
#   export D2L_ROOT=${SHINE_ROOT}/../doc-to-lora
#   bash MemoryAgentBench/bash_files/sh/run_delta_mem_hgx001.sh d2l-mab
set -euo pipefail

SHINE_ROOT="${SHINE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
DELTA_MEM_ROOT="${DELTA_MEM_ROOT:-${SHINE_ROOT}/third_party/delta-Mem}"
MAB_ROOT="${MAB_ROOT:-${SHINE_ROOT}/MemoryAgentBench}"
D2L_ROOT="${D2L_ROOT:-${SHINE_ROOT}/../doc-to-lora}"
D2L_CHECKPOINT_DIR="${D2L_CHECKPOINT_DIR:-/ceph/home/muhan01/huggingfacemodels/doc-to-lora/qwen_4b_d2l/checkpoint-20000}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

echo "SHINE_ROOT=${SHINE_ROOT}"
echo "D2L_ROOT=${D2L_ROOT}"
echo "DELTA_MEM_ROOT=${DELTA_MEM_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Set PYTHON_BIN to your doc-to-lora conda python." >&2
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "conda env: ${CONDA_PREFIX}"
fi

mkdir -p "$(dirname "${D2L_ROOT}")"
if [[ ! -d "${D2L_ROOT}/.git" ]]; then
  echo "Cloning SakanaAI/doc-to-lora → ${D2L_ROOT}"
  git clone --depth 1 https://github.com/SakanaAI/doc-to-lora.git "${D2L_ROOT}"
else
  echo "doc-to-lora already at ${D2L_ROOT}"
fi

echo "Installing doc-to-lora (editable)..."
"${PYTHON_BIN}" -m pip install -e "${D2L_ROOT}"

echo "Installing δ-mem MAB eval deps (no transformers pin)..."
"${PYTHON_BIN}" -m pip install -r "${MAB_ROOT}/requirements-d2l-mab.txt"

if [[ ! -d "${DELTA_MEM_ROOT}/deltamem" ]]; then
  echo "Missing ${DELTA_MEM_ROOT}. Run setup_delta_mem_hgx001.sh first." >&2
  exit 1
fi

bash "${MAB_ROOT}/deltamem_patches/apply_to_delta_mem.sh" "${DELTA_MEM_ROOT}"

site="$("${PYTHON_BIN}" -c "import site; print(site.getsitepackages()[0])")"
cat > "${site}/shine_d2l_mab_paths.pth" <<EOF
${DELTA_MEM_ROOT}
${SHINE_ROOT}
${MAB_ROOT}
${D2L_ROOT}
EOF
echo "Wrote ${site}/shine_d2l_mab_paths.pth"

mkdir -p "${D2L_CHECKPOINT_DIR}"
CKPT="${D2L_CHECKPOINT_DIR}/pytorch_model.bin"
if [[ ! -f "${CKPT}" ]]; then
  echo "Downloading D2L checkpoint (qwen_4b_d2l checkpoint-20000)..."
  "${PYTHON_BIN}" -m pip install -q huggingface_hub
  "${PYTHON_BIN}" <<PY
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(
    repo_id="SakanaAI/doc-to-lora",
    filename="qwen_4b_d2l/checkpoint-20000/pytorch_model.bin",
    local_dir=os.path.dirname("${D2L_CHECKPOINT_DIR}"),
)
dest = "${CKPT}"
if os.path.abspath(path) != os.path.abspath(dest):
    shutil.copy2(path, dest)
print("checkpoint:", dest)
PY
else
  echo "D2L checkpoint present: ${CKPT}"
fi

echo "Preflight imports..."
export PYTHONPATH="${DELTA_MEM_ROOT}:${SHINE_ROOT}:${MAB_ROOT}:${D2L_ROOT}"
"${PYTHON_BIN}" -c "
import torch, transformers
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from deltamem.eval import benchmark_compare
from methods.doc_to_lora_runner import DocToLoraRunner
print('preflight OK', torch.__version__, transformers.__version__)
"

cat <<EOF

Setup complete.

Run full MAB (δ-mem protocol, same as base/SHINE):
  conda activate doc-to-lora
  export SHINE_ROOT=${SHINE_ROOT}
  export D2L_ROOT=${D2L_ROOT}
  export PYTHON_BIN=\$(which python)
  export CUDA_VISIBLE_DEVICES=0
  bash ${MAB_ROOT}/bash_files/sh/run_delta_mem_hgx001.sh d2l-mab

Output: \${OUTPUT_ROOT:-${SHINE_ROOT}/outputs/delta_mem_qwen3_4b_full}/d2l_model/memory_agent_bench.json
Note: D2L uses Qwen3-4B backbone — compare fairly on protocol, not model size.

EOF
