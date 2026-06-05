#!/usr/bin/env bash
# One-time setup on hgx001: δ-mem venv + SHINE deps + directory layout.
#
# hgx001 常见：驱动 CUDA 12.1（报错 12010）→ 默认用 cu121 PyTorch，不装 flash-attn。
# 若已有可用的 conda（如 MABench），可复用：
#   USE_EXISTING_PYTHON=1 bash setup_delta_mem_hgx001.sh
set -euo pipefail

SHINE_ROOT="${SHINE_ROOT:-/ceph/home/muhan01/wyd/SHINE-mem}"
DELTA_MEM_ROOT="${DELTA_MEM_ROOT:-${SHINE_ROOT}/third_party/delta-Mem}"
MAB_ROOT="${MAB_ROOT:-${SHINE_ROOT}/MemoryAgentBench}"
# cu121 matches driver CUDA 12.1 (nvidia error 12010); override with TORCH_INDEX=cu118 if needed
TORCH_INDEX="${TORCH_INDEX:-cu121}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
RECREATE_VENV="${RECREATE_VENV:-0}"

echo "SHINE_ROOT=${SHINE_ROOT}"
echo "DELTA_MEM_ROOT=${DELTA_MEM_ROOT}"
echo "MAB_ROOT=${MAB_ROOT}"
echo "TORCH_INDEX=${TORCH_INDEX} INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN}"

if [[ ! -d "${SHINE_ROOT}" ]]; then
  echo "SHINE_ROOT not found: ${SHINE_ROOT}" >&2
  exit 1
fi

mkdir -p "${SHINE_ROOT}/third_party"
if [[ ! -d "${DELTA_MEM_ROOT}/.git" ]]; then
  git clone --depth 1 https://github.com/declare-lab/delta-Mem.git "${DELTA_MEM_ROOT}"
fi

bash "${MAB_ROOT}/deltamem_patches/apply_to_delta_mem.sh" "${DELTA_MEM_ROOT}"

# IFEval 需要 lm-evaluation-harness（与 δ-mem README 一致，放在 third_party/ 下）
LM_EVAL_ROOT="${LM_EVAL_ROOT:-${SHINE_ROOT}/third_party/lm-evaluation-harness}"
if [[ ! -d "${LM_EVAL_ROOT}/.git" ]]; then
  echo "Cloning lm-evaluation-harness → ${LM_EVAL_ROOT}"
  mkdir -p "${SHINE_ROOT}/third_party"
  git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git "${LM_EVAL_ROOT}"
fi

if [[ "${USE_EXISTING_PYTHON:-0}" == "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
  echo "USE_EXISTING_PYTHON=1 → ${PYTHON_BIN}"
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

  # 1) PyTorch 先按集群驱动装（避免 requirements 拉到 cu124）
  "${PYTHON_BIN}" -m pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

  # 2) δ-mem 其余依赖（评测不需要 flash-attn / 完整 deepspeed 训练栈）
  DS_BUILD_OPS=0 "${PYTHON_BIN}" -m pip install -r requirements.txt

  # 3) 把 torch 钉回与驱动匹配的版本（requirements 可能升级 torch）
  "${PYTHON_BIN}" -m pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" --force-reinstall

  if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
    "${PYTHON_BIN}" -m pip install --no-build-isolation flash-attn || {
      echo "flash-attn install failed; continue without it (eval uses default attn)." >&2
    }
  else
    echo "Skipping flash-attn (INSTALL_FLASH_ATTN=0). benchmark_compare attn defaults to None."
  fi
fi

# SHINE + MAB lightweight deps
"${PYTHON_BIN}" -m pip install -r "${MAB_ROOT}/requirements-shine-mab.txt"
"${PYTHON_BIN}" -m pip install hydra-core omegaconf scikit-learn nltk rouge_score editdistance pyyaml

export PYTHONPATH="${DELTA_MEM_ROOT}:${SHINE_ROOT}:${MAB_ROOT}"
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import torch
from deltamem.eval import benchmark_compare

cuda_ok = torch.cuda.is_available()
print("delta-mem import OK")
print("  torch:", torch.__version__, "| torch.cuda:", getattr(torch.version, "cuda", "?"))
print("  cuda available:", cuda_ok)
if cuda_ok:
    print("  device:", torch.cuda.get_device_name(0))
else:
    print("  WARNING: CUDA unavailable. If driver error 12010, reinstall with TORCH_INDEX=cu121:")
    print("    TORCH_INDEX=cu121 RECREATE_VENV=1 bash setup_delta_mem_hgx001.sh")
if importlib.util.find_spec("flash_attn") is None:
    print("  flash_attn: not installed (OK for MAB eval)")
PY

cat <<EOF

Setup complete.

  export SHINE_ROOT=${SHINE_ROOT}
  export DELTA_MEM_ROOT=${DELTA_MEM_ROOT}
  export MAB_ROOT=${MAB_ROOT}
  export PYTHONPATH=\${DELTA_MEM_ROOT}:\${SHINE_ROOT}:\${MAB_ROOT}
  export PYTHON_BIN=${PYTHON_BIN}

Then (δ-mem 官方全套，非 smoke):
  bash ${MAB_ROOT}/bash_files/sh/run_delta_mem_hgx001.sh base

EOF
