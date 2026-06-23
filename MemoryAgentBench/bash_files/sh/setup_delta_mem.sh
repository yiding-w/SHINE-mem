#!/usr/bin/env bash
# One-time setup: δ-mem venv + SHINE deps (any server; configure via server_env.sh).
#
#   cp bash_files/sh/server_env.example.sh bash_files/sh/server_env.sh   # edit paths
#   bash bash_files/sh/setup_delta_mem.sh
#
# hgx001 shortcut (unchanged paths):
#   bash bash_files/sh/setup_delta_mem_hgx001.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/source_server_env.sh"

DELTA_MEM_ROOT="${DELTA_MEM_ROOT:-${SHINE_ROOT}/third_party/delta-Mem}"
MAB_ROOT="${MAB_ROOT:-${SHINE_ROOT}/MemoryAgentBench}"
TORCH_INDEX="${TORCH_INDEX:-cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
RECREATE_VENV="${RECREATE_VENV:-0}"

echo "SHINE_ROOT=${SHINE_ROOT}"
echo "DELTA_MEM_ROOT=${DELTA_MEM_ROOT}"
echo "TORCH=${TORCH_VERSION}+${TORCH_INDEX} transformers==4.57.1"
echo "GPUs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} NUM_GPUS=${NUM_GPUS:-?}"

if [[ ! -d "${SHINE_ROOT}" ]]; then
  echo "SHINE_ROOT not found: ${SHINE_ROOT}" >&2
  exit 1
fi

mkdir -p "${SHINE_ROOT}/third_party"
if [[ ! -d "${DELTA_MEM_ROOT}/.git" ]]; then
  git clone --depth 1 https://github.com/declare-lab/delta-Mem.git "${DELTA_MEM_ROOT}"
fi

bash "${MAB_ROOT}/deltamem_patches/apply_to_delta_mem.sh" "${DELTA_MEM_ROOT}"
bash "${MAB_ROOT}/bash_files/sh/download_mab_recsys_entity2id.sh" || true
bash "${MAB_ROOT}/bash_files/sh/generate_server_agent_config.sh"

LM_EVAL_ROOT="${LM_EVAL_ROOT:-${SHINE_ROOT}/third_party/lm-evaluation-harness}"
if [[ ! -d "${LM_EVAL_ROOT}/.git" ]]; then
  echo "Cloning lm-evaluation-harness -> ${LM_EVAL_ROOT}"
  mkdir -p "${SHINE_ROOT}/third_party"
  git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git "${LM_EVAL_ROOT}"
fi

install_torch() {
  local py="$1"
  "${py}" -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==0.20.1" \
    "torchaudio==2.5.1" \
    --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"
}

install_venv_paths() {
  local py="$1"
  local site
  site="$("${py}" -c "import site; print(site.getsitepackages()[0])")"
  cat > "${site}/shine_delta_mem_paths.pth" <<EOF
${DELTA_MEM_ROOT}
${SHINE_ROOT}
${MAB_ROOT}
EOF
  echo "Wrote ${site}/shine_delta_mem_paths.pth"
}

install_eval_deps() {
  local py="$1"
  "${py}" -m pip install -r "${MAB_ROOT}/requirements-delta-eval.txt"
  "${py}" -m pip install -r "${MAB_ROOT}/requirements-shine-mab.txt"
  "${py}" -m pip install hydra-core omegaconf scikit-learn nltk rouge_score editdistance
  "${py}" -m pip install "transformers==4.57.1" "numpy<2"
  install_torch "${py}"
}

if [[ "${USE_EXISTING_PYTHON:-0}" == "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
  echo "USE_EXISTING_PYTHON=1 -> ${PYTHON_BIN}"
  install_eval_deps "${PYTHON_BIN}"
  install_venv_paths "${PYTHON_BIN}"
else
  cd "${DELTA_MEM_ROOT}"
  if [[ "${RECREATE_VENV}" == "1" && -d .venv ]]; then
    rm -rf .venv
  fi
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
  fi
  PYTHON_BIN="${DELTA_MEM_ROOT}/.venv/bin/python"
  "${PYTHON_BIN}" -m pip install -U pip setuptools wheel
  install_torch "${PYTHON_BIN}"
  install_eval_deps "${PYTHON_BIN}"
  install_venv_paths "${PYTHON_BIN}"

  if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
    "${PYTHON_BIN}" -m pip install --no-build-isolation flash-attn || true
  else
    echo "Skipping flash-attn (eval uses sdpa)."
  fi
fi

export PYTHONPATH="${DELTA_MEM_ROOT}:${SHINE_ROOT}:${MAB_ROOT}"
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import torch
import transformers
from deltamem.eval import benchmark_compare

print("delta-mem import OK")
print("  torch:", torch.__version__, "| cuda:", getattr(torch.version, "cuda", "?"))
print("  transformers:", transformers.__version__)
print("  cuda available:", torch.cuda.is_available())
print("  device count:", torch.cuda.device_count())
if importlib.util.find_spec("flash_attn") is None:
    print("  flash_attn: not installed (OK)")
PY

cat <<EOF

Setup complete.

  source ${SCRIPT_DIR}/source_server_env.sh
  bash ${SCRIPT_DIR}/download_eval_assets.sh    # models + datasets (first time)
  bash ${SCRIPT_DIR}/run_delta_mem.sh base-mab  # frozen Qwen3-8B, full MAB only

EOF
