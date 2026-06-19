#!/usr/bin/env bash
# One-time setup: clone doc-to-lora, install D2L + δ-mem eval paths for d2l-mab runs.
#
# IMPORTANT (cluster): /home often has a per-user quota. Full `pip install -e doc-to-lora`
# pulls vllm/deepspeed/jupyter (~200 packages) and can hit "Disk quota exceeded".
# This script defaults to editable install WITH --no-deps plus minimal MAB eval deps only.
#
# Use the DEDICATED doc-to-lora conda env (transformers ~4.51.3), NOT delta-Mem .venv.
# Put large paths on /ceph when home is tight:
#   export PIP_CACHE_DIR=/ceph/home/$USER/.cache/pip
#   export HF_HOME=/ceph/home/$USER/huggingfacemodels
#   export D2L_ROOT=/ceph/home/$USER/doc-to-lora
#
#   conda activate doc-to-lora
#   export SHINE_ROOT=/path/to/SHINE-mem
#   bash MemoryAgentBench/bash_files/sh/setup_doc_to_lora_mab.sh
#
# If doc-to-lora is already installed in the env:
#   SKIP_D2L_PIP=1 bash MemoryAgentBench/bash_files/sh/setup_doc_to_lora_mab.sh
set -euo pipefail

SHINE_ROOT="${SHINE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)}"
DELTA_MEM_ROOT="${DELTA_MEM_ROOT:-${SHINE_ROOT}/third_party/delta-Mem}"
MAB_ROOT="${MAB_ROOT:-${SHINE_ROOT}/MemoryAgentBench}"
D2L_ROOT="${D2L_ROOT:-${SHINE_ROOT}/../doc-to-lora}"
D2L_CHECKPOINT_DIR="${D2L_CHECKPOINT_DIR:-/ceph/home/muhan01/huggingfacemodels/doc-to-lora/qwen_4b_d2l/checkpoint-20000}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SKIP_D2L_PIP="${SKIP_D2L_PIP:-0}"
D2L_PIP_NO_DEPS="${D2L_PIP_NO_DEPS:-1}"

# Keep pip/tmp off NFS /home when possible (override before running if needed).
if [[ -z "${PIP_CACHE_DIR:-}" ]] && [[ -d "/ceph/home/${USER:-}" ]]; then
  export PIP_CACHE_DIR="/ceph/home/${USER}/.cache/pip"
fi
if [[ -z "${TMPDIR:-}" ]] && [[ -d "/ceph/home/${USER:-}" ]]; then
  export TMPDIR="/ceph/home/${USER}/.cache/tmp"
fi
mkdir -p "${PIP_CACHE_DIR:-/tmp}" "${TMPDIR:-/tmp}" 2>/dev/null || true

echo "SHINE_ROOT=${SHINE_ROOT}"
echo "D2L_ROOT=${D2L_ROOT}"
echo "DELTA_MEM_ROOT=${DELTA_MEM_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "PIP_CACHE_DIR=${PIP_CACHE_DIR:-<default>}"
echo "SKIP_D2L_PIP=${SKIP_D2L_PIP} D2L_PIP_NO_DEPS=${D2L_PIP_NO_DEPS}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Set PYTHON_BIN to your doc-to-lora conda python." >&2
  exit 1
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "conda env: ${CONDA_PREFIX}"
fi

d2l_import_ok() {
  PYTHONPATH="${D2L_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -c "
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
print('ctx_to_lora OK')
" 2>/dev/null
}

mkdir -p "$(dirname "${D2L_ROOT}")"
if [[ ! -d "${D2L_ROOT}/.git" ]]; then
  echo "Cloning SakanaAI/doc-to-lora → ${D2L_ROOT}"
  git clone --depth 1 https://github.com/SakanaAI/doc-to-lora.git "${D2L_ROOT}"
else
  echo "doc-to-lora already at ${D2L_ROOT}"
fi

if [[ "${SKIP_D2L_PIP}" == "1" ]]; then
  echo "SKIP_D2L_PIP=1 — skipping pip installs (env must already have ctx_to_lora + torch)."
elif d2l_import_ok; then
  echo "ctx_to_lora already importable — skipping pip (set FORCE_D2L_PIP=1 to reinstall)."
elif [[ "${FORCE_D2L_PIP:-0}" == "1" && "${D2L_PIP_NO_DEPS}" != "1" ]]; then
  echo "WARNING: FORCE_D2L_PIP=1 with full deps — may download vllm/deepspeed (~5GB+)." >&2
  echo "         Prefer: D2L_PIP_NO_DEPS=1 (default) or use official doc-to-lora env." >&2
  "${PYTHON_BIN}" -m pip install -e "${D2L_ROOT}"
else
  echo "Installing doc-to-lora editable (--no-deps; avoids vllm/deepspeed/jupyter bulk)..."
  "${PYTHON_BIN}" -m pip install --no-deps -e "${D2L_ROOT}"
  echo "Installing minimal runtime + δ-mem MAB eval deps..."
  "${PYTHON_BIN}" -m pip install -r "${MAB_ROOT}/requirements-d2l-runtime.txt"
  "${PYTHON_BIN}" -m pip install -r "${MAB_ROOT}/requirements-d2l-mab.txt"
fi

if ! d2l_import_ok; then
  echo "ERROR: ctx_to_lora still not importable." >&2
  echo "  Option A: create env per SakanaAI/doc-to-lora README on /ceph (not /home)" >&2
  echo "  Option B: SKIP_D2L_PIP=1 if you already installed doc-to-lora elsewhere" >&2
  echo "  Option C: pip install missing packages manually (see requirements-d2l-runtime.txt)" >&2
  exit 1
fi

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
  "${PYTHON_BIN}" -m pip install -q huggingface_hub 2>/dev/null || true
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

If pip failed with quota: put cache on /ceph and retry:
  export PIP_CACHE_DIR=/ceph/home/\$USER/.cache/pip TMPDIR=/ceph/home/\$USER/.cache/tmp
  export D2L_ROOT=/ceph/home/\$USER/doc-to-lora
  bash ${MAB_ROOT}/bash_files/sh/setup_doc_to_lora_mab.sh

EOF
