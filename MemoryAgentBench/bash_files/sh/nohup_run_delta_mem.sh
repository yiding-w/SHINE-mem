#!/usr/bin/env bash
# Run eval in background with nohup (survives SSH disconnect).
#
# Usage:
#   bash MemoryAgentBench/bash_files/sh/nohup_run_delta_mem.sh base-mab
#   tail -f outputs/delta_mem_qwen3_8b_full/logs/base_model_memory_agent_bench.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/source_server_env.sh"

MODE="${1:-base-mab}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SHINE_ROOT}/outputs/delta_mem_qwen3_8b_full}"
LOG_ROOT="${OUTPUT_ROOT}/logs"
NOHUP_LOG="${LOG_ROOT}/nohup_${MODE}.log"
PID_FILE="${LOG_ROOT}/nohup_${MODE}.pid"

mkdir -p "${LOG_ROOT}"

nohup bash "${SCRIPT_DIR}/run_delta_mem.sh" "${MODE}" >> "${NOHUP_LOG}" 2>&1 &
echo $! | tee "${PID_FILE}"

echo "Started mode=${MODE} pid=$(cat "${PID_FILE}")"
echo "  nohup log: ${NOHUP_LOG}"
echo "  task log:  ${LOG_ROOT}/ (base_model_*.log or compare_mab.log)"
echo "  tail -f ${NOHUP_LOG}"
