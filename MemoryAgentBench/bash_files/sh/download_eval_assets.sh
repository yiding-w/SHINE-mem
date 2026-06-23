#!/usr/bin/env bash
# Prefetch models + benchmark datasets for δ-mem eval on a fresh server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/source_server_env.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"
mkdir -p "$(dirname "${BASE_MODEL}")" "$(dirname "${SHINE_MODEL_ROOT}")"

bash "${SCRIPT_DIR}/generate_server_agent_config.sh"

echo "=== HF cache: ${HF_HOME}"
echo "=== BASE_MODEL: ${BASE_MODEL}"
echo "=== SHINE_MODEL_ROOT: ${SHINE_MODEL_ROOT}"
echo "=== HF_ENDPOINT: ${HF_ENDPOINT}"

export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE HF_ENDPOINT

_hf_download() {
  local repo="$1"
  local dest="$2"
  shift 2
  if [[ -d "${dest}" ]]; then
    if [[ -f "${dest}/config.json" ]] || find "${dest}" -name 'metanetwork.pth' -print -quit | grep -q .; then
      echo "Skip existing model dir: ${dest}"
      return 0
    fi
  fi
  mkdir -p "${dest}"
  echo "Downloading ${repo} -> ${dest}"
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_ENDPOINT="${HF_ENDPOINT}" huggingface-cli download "${repo}" --local-dir "${dest}" "$@"
    return 0
  fi
  HF_ENDPOINT="${HF_ENDPOINT}" "${PYTHON_BIN}" - <<PY
import os
from huggingface_hub import snapshot_download

endpoint = os.environ.get("HF_ENDPOINT", "").strip()
if endpoint:
    os.environ["HF_ENDPOINT"] = endpoint
snapshot_download(repo_id="${repo}", local_dir="${dest}")
PY
}

# --- Models ---
_hf_download "Qwen/Qwen3-8B" "${BASE_MODEL}"

# SHINE checkpoint (same as README hf download)
_hf_download "Yewei-Liu/SHINE-ift_mqa_1qa" "${SHINE_MODEL_ROOT}"

# --- MAB recsys mapping ---
bash "${SCRIPT_DIR}/download_mab_recsys_entity2id.sh" || true

# --- Benchmark datasets (warm HF cache) ---
echo "=== Prefetching benchmark datasets into ${HF_DATASETS_CACHE}"
HF_ENDPOINT="${HF_ENDPOINT}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
endpoint = os.environ.get("HF_ENDPOINT", "").strip()
if endpoint:
    os.environ["HF_ENDPOINT"] = endpoint

from datasets import load_dataset
from huggingface_hub import hf_hub_download

cache = Path(os.environ["HF_DATASETS_CACHE"])
hub = Path(os.environ["HF_HUB_CACHE"])
cache.mkdir(parents=True, exist_ok=True)
hub.mkdir(parents=True, exist_ok=True)

print("hotpotqa/hotpot_qa ...")
load_dataset("hotpotqa/hotpot_qa", name="distractor", split="validation", cache_dir=str(cache))

print("Idavidrein/gpqa gpqa_diamond ...")
load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", cache_dir=str(cache))

print("google/IFEval ...")
load_dataset("google/IFEval", split="train", cache_dir=str(cache))

splits = [
    "Accurate_Retrieval",
    "Test_Time_Learning",
    "Long_Range_Understanding",
    "Conflict_Resolution",
]
for split in splits:
    parquet = f"data/{split}-00000-of-00001.parquet"
    print(f"ai-hyz/MemoryAgentBench {parquet} ...")
    path = hf_hub_download(
        repo_id="ai-hyz/MemoryAgentBench",
        filename=parquet,
        repo_type="dataset",
    )
    print(f"  cached: {path}")

print("entity2id.json ...")
hf_hub_download(repo_id="ai-hyz/MemoryAgentBench", filename="entity2id.json", repo_type="dataset")

print("Dataset prefetch done.")
PY

# --- LoCoMo data (ships with delta-Mem clone) ---
DELTA_MEM_ROOT="${DELTA_MEM_ROOT:-${SHINE_ROOT}/third_party/delta-Mem}"
LOCOMO="${DELTA_MEM_ROOT}/data/locomo10.json"
if [[ ! -f "${LOCOMO}" ]]; then
  echo "WARN: ${LOCOMO} missing. Clone delta-Mem: bash ${SCRIPT_DIR}/setup_delta_mem.sh" >&2
else
  echo "LoCoMo data OK: ${LOCOMO}"
fi

echo ""
echo "=== Asset check ==="
for path in "${BASE_MODEL}" "${SHINE_MODEL_ROOT}"; do
  if [[ -d "${path}" ]]; then
    echo "OK dir: ${path}"
  else
    echo "MISSING: ${path}" >&2
  fi
done

ENTITY2ID="${MAB_ROOT}/processed_data/Recsys_Redial/entity2id.json"
if [[ -f "${ENTITY2ID}" ]]; then
  echo "OK entity2id: ${ENTITY2ID} ($(wc -c < "${ENTITY2ID}") bytes)"
else
  echo "WARN: missing ${ENTITY2ID} — scp from old server or set ENTITY2ID_SRC" >&2
fi

echo ""
echo "Download step finished. Next:"
echo "  source ${SCRIPT_DIR}/source_server_env.sh"
echo "  bash ${SCRIPT_DIR}/run_delta_mem.sh base-mab"
