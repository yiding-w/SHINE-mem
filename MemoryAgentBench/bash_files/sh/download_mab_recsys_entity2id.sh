#!/usr/bin/env bash
# entity2id.json for MemoryAgentBench recsys_redial scoring.
# Source: ai-hyz/MemoryAgentBench (root file entity2id.json), ~1.7MB.
#
# hgx001 often cannot reach huggingface.co; this script tries:
#   1) existing DEST
#   2) ENTITY2ID_SRC copy
#   3) local HF hub cache
#   4) hf-mirror.com / huggingface.co / HF_ENDPOINT mirror
#
# Exit 0 if file ready; exit 1 only when REQUIRE_RECSYS_ENTITY2ID=1 and still missing.
set -euo pipefail

SHINE_ROOT="${SHINE_ROOT:-/ceph/home/muhan01/wyd/SHINE-mem}"
MAB_ROOT="${MAB_ROOT:-${SHINE_ROOT}/MemoryAgentBench}"
DEST="${MAB_ROOT}/processed_data/Recsys_Redial/entity2id.json"
MIN_BYTES="${ENTITY2ID_MIN_BYTES:-1000000}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-/ceph/home/muhan01/huggingfacemodels}/hub}"

URLS=(
  "${HF_ENDPOINT%/}/datasets/ai-hyz/MemoryAgentBench/resolve/main/entity2id.json"
  "https://hf-mirror.com/datasets/ai-hyz/MemoryAgentBench/resolve/main/entity2id.json"
  "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/main/entity2id.json"
)

_is_valid() {
  [[ -f "$1" ]] && [[ "$(wc -c < "$1" | tr -d ' ')" -ge "${MIN_BYTES}" ]]
}

_copy_if_valid() {
  local src="$1"
  if _is_valid "${src}"; then
    mkdir -p "$(dirname "${DEST}")"
    cp -f "${src}" "${DEST}"
    echo "recsys entity2id ready (copied): ${DEST} ($(wc -c < "${DEST}") bytes)"
    return 0
  fi
  return 1
}

mkdir -p "$(dirname "${DEST}")"

if _is_valid "${DEST}"; then
  echo "recsys entity2id OK: ${DEST}"
  exit 0
fi

if [[ -n "${ENTITY2ID_SRC:-}" ]] && _copy_if_valid "${ENTITY2ID_SRC}"; then
  exit 0
fi

# Search HF hub cache (dataset snapshot may already exist from prior MAB runs).
if [[ -d "${HF_HUB_CACHE}" ]]; then
  while IFS= read -r cached; do
    if _copy_if_valid "${cached}"; then
      exit 0
    fi
  done < <(find "${HF_HUB_CACHE}" -path '*MemoryAgentBench*' -name 'entity2id.json' 2>/dev/null | head -20)
fi

_download_url() {
  local url="$1"
  local tmp="${DEST}.tmp.$$"
  echo "Trying: ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --connect-timeout 20 --max-time 300 -o "${tmp}" "${url}" && _is_valid "${tmp}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -T 300 -O "${tmp}" "${url}" && _is_valid "${tmp}"
  else
    return 1
  fi
}

for url in "${URLS[@]}"; do
  tmp="${DEST}.tmp.$$"
  if _download_url "${url}"; then
    mv -f "${tmp}" "${DEST}"
    echo "recsys entity2id ready (downloaded): ${DEST} ($(wc -c < "${DEST}") bytes)"
    exit 0
  fi
  rm -f "${tmp}" 2>/dev/null || true
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Trying huggingface_hub (endpoint=${HF_ENDPOINT})..."
  if HF_ENDPOINT="${HF_ENDPOINT}" "${PYTHON_BIN}" - <<PY
import os
import shutil
from huggingface_hub import hf_hub_download

endpoint = os.environ.get("HF_ENDPOINT", "").strip()
if endpoint:
    os.environ["HF_ENDPOINT"] = endpoint
dest = "${DEST}"
path = hf_hub_download(
    repo_id="ai-hyz/MemoryAgentBench",
    filename="entity2id.json",
    repo_type="dataset",
    endpoint=endpoint or None,
)
shutil.copy(path, dest)
size = os.path.getsize(dest)
if size < int("${MIN_BYTES}"):
    raise SystemExit(f"file too small: {size}")
print(f"copied from hub cache/download: {dest} ({size} bytes)")
PY
  then
    exit 0
  fi
fi

echo "WARN: could not fetch entity2id.json (hgx001 may block huggingface.co)." >&2
echo "  Needed at: ${DEST}" >&2
echo "  From your laptop (has network), run:" >&2
echo "    scp MemoryAgentBench/processed_data/Recsys_Redial/entity2id.json \\" >&2
echo "        muhan01@hgx001:/ceph/home/muhan01/wyd/SHINE-mem/MemoryAgentBench/processed_data/Recsys_Redial/" >&2
echo "  Or on server:" >&2
echo "    export ENTITY2ID_SRC=/path/to/entity2id.json" >&2
echo "    bash MemoryAgentBench/bash_files/sh/download_mab_recsys_entity2id.sh" >&2
echo "  Eval can start without it; recsys_redial (~54% of MAB) will crash if missing." >&2

if [[ "${REQUIRE_RECSYS_ENTITY2ID:-0}" == "1" ]]; then
  exit 1
fi
exit 0
