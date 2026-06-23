#!/usr/bin/env bash
# Bootstrap δ-mem + SHINE eval on a fresh Linux GPU server.
#
# Usage:
#   git clone https://github.com/yiding-w/SHINE-mem.git
#   cd SHINE-mem
#   bash MemoryAgentBench/bash_files/sh/bootstrap_new_server.sh
#
# Or set REPO_URL / INSTALL_DIR:
#   INSTALL_DIR=$HOME/work bash MemoryAgentBench/bash_files/sh/bootstrap_new_server.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${REPO_URL:-https://github.com/yiding-w/SHINE-mem.git}"
INSTALL_DIR="${INSTALL_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

if [[ ! -f "${SCRIPT_DIR}/server_env.example.sh" ]]; then
  echo "Run this script from a SHINE-mem clone (MemoryAgentBench/bash_files/sh/)." >&2
  exit 1
fi

ENV_FILE="${SCRIPT_DIR}/server_env.sh"
if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${SCRIPT_DIR}/server_env.example.sh" "${ENV_FILE}"
  echo "Created ${ENV_FILE} — edit SHINE_ROOT, HF_HOME, BASE_MODEL, CUDA_GPU_IDS, then re-run."
  echo "  Default GPUs: CUDA_GPU_IDS=5,6  NUM_GPUS=2"
  exit 0
fi

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/source_server_env.sh"

echo "=== [1/3] setup_delta_mem.sh ==="
bash "${SCRIPT_DIR}/setup_delta_mem.sh"

echo "=== [2/3] download_eval_assets.sh (models + datasets; may take a while) ==="
bash "${SCRIPT_DIR}/download_eval_assets.sh"

echo "=== [3/3] ready to run ==="
cat <<EOF

Bootstrap done.

  # foreground — frozen 8B, full MAB only (recommended first):
  source ${SCRIPT_DIR}/source_server_env.sh
  bash ${SCRIPT_DIR}/run_delta_mem.sh base-mab

  # background:
  bash ${SCRIPT_DIR}/nohup_run_delta_mem.sh base-mab
  tail -f ${OUTPUT_ROOT}/logs/nohup_base-mab.log

  GPUs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  NUM_GPUS=${NUM_GPUS}
  Models: ${BASE_MODEL}
  Output: ${OUTPUT_ROOT}/base_model/memory_agent_bench.json

EOF
