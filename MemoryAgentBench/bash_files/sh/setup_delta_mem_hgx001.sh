#!/usr/bin/env bash
# One-time setup on hgx001: δ-mem venv + SHINE deps + directory layout.
#
# hgx001：驱动 CUDA 12.1（12010）→ torch 2.5.1+cu121 + transformers 4.57.1（勿用 transformers 5.x）
# 若已有可用的 conda MABench：USE_EXISTING_PYTHON=1 PYTHON_BIN=$(which python) bash ...
set -euo pipefail

SHINE_ROOT="${SHINE_ROOT:-/ceph/home/muhan01/wyd/SHINE-mem}"
DELTA_MEM_ROOT="${DELTA_MEM_ROOT:-${SHINE_ROOT}/third_party/delta-Mem}"
MAB_ROOT="${MAB_ROOT:-${SHINE_ROOT}/MemoryAgentBench}"
TORCH_INDEX="${TORCH_INDEX:-cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
RECREATE_VENV="${RECREATE_VENV:-0}"

echo "SHINE_ROOT=${SHINE_ROOT}"
echo "DELTA_MEM_ROOT=${DELTA_MEM_ROOT}"
echo "TORCH=${TORCH_VERSION}+${TORCH_INDEX} transformers==4.57.1"

if [[ ! -d "${SHINE_ROOT}" ]]; then
  echo "SHINE_ROOT not found: ${SHINE_ROOT}" >&2
  exit 1
fi

mkdir -p "${SHINE_ROOT}/third_party"
if [[ ! -d "${DELTA_MEM_ROOT}/.git" ]]; then
  git clone --depth 1 https://github.com/declare-lab/delta-Mem.git "${DELTA_MEM_ROOT}"
fi

bash "${MAB_ROOT}/deltamem_patches/apply_to_delta_mem.sh" "${DELTA_MEM_ROOT}"

# recsys_redial_full needs entity2id.json for Recall@k scoring
bash "${MAB_ROOT}/bash_files/sh/download_mab_recsys_entity2id.sh"

LM_EVAL_ROOT="${LM_EVAL_ROOT:-${SHINE_ROOT}/third_party/lm-evaluation-harness}"
if [[ ! -d "${LM_EVAL_ROOT}/.git" ]]; then
  echo "Cloning lm-evaluation-harness → ${LM_EVAL_ROOT}"
  mkdir -p "${SHINE_ROOT}/third_party"
  git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git "${LM_EVAL_ROOT}"
fi

install_torch() {
  local py="$1"
  # 固定版本，避免 pip 反复解析下载多个 torch wheel（你之前 Ctrl-C 会留下半装环境）
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
  # 钉死版本，防止被其它包升级
  "${py}" -m pip install "transformers==4.57.1" "numpy<2"
  install_torch "${py}"
}

if [[ "${USE_EXISTING_PYTHON:-0}" == "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
  echo "USE_EXISTING_PYTHON=1 → ${PYTHON_BIN}"
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
if importlib.util.find_spec("flash_attn") is None:
    print("  flash_attn: not installed (OK)")
PY

cat <<EOF

Setup complete. Always use this Python for runs (不要只靠 conda 的 python):

  export PYTHON_BIN=${PYTHON_BIN}
  export SHINE_ROOT=${SHINE_ROOT}
  export DELTA_MEM_ROOT=${DELTA_MEM_ROOT}
  export MAB_ROOT=${MAB_ROOT}
  source ${MAB_ROOT}/bash_files/sh/env_delta_mem.sh

  bash ${MAB_ROOT}/bash_files/sh/run_delta_mem_hgx001.sh compare-mab

若之前装到一半 Ctrl-C 过：RECREATE_VENV=1 TORCH_INDEX=cu121 bash ${MAB_ROOT}/bash_files/sh/setup_delta_mem_hgx001.sh

EOF
