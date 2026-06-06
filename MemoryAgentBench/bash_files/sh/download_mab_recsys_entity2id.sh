#!/usr/bin/env bash
# Download entity2id.json required for MemoryAgentBench recsys_redial scoring.
# Source: https://huggingface.co/datasets/ai-hyz/MemoryAgentBench (root file entity2id.json)
set -euo pipefail

SHINE_ROOT="${SHINE_ROOT:-/ceph/home/muhan01/wyd/SHINE-mem}"
MAB_ROOT="${MAB_ROOT:-${SHINE_ROOT}/MemoryAgentBench}"
DEST="${MAB_ROOT}/processed_data/Recsys_Redial/entity2id.json"
HF_URL="https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/entity2id.json"

mkdir -p "$(dirname "${DEST}")"

if [[ -f "${DEST}" ]]; then
  echo "recsys entity2id OK: ${DEST}"
  exit 0
fi

echo "Downloading entity2id.json → ${DEST}"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL -o "${DEST}" "${HF_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -q -O "${DEST}" "${HF_URL}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  "${PYTHON_BIN}" - <<PY
import shutil
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="ai-hyz/MemoryAgentBench", filename="entity2id.json")
shutil.copy(path, "${DEST}")
print("Downloaded via huggingface_hub")
PY
fi

if [[ ! -s "${DEST}" ]]; then
  echo "Download failed or empty file: ${DEST}" >&2
  exit 1
fi

echo "recsys entity2id ready: ${DEST} ($(wc -c < "${DEST}") bytes)"
