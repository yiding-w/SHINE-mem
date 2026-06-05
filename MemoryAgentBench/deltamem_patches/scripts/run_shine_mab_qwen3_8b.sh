#!/usr/bin/env bash
# δ-mem benchmark_compare with SHINE variant (Qwen3-8B, MemoryAgentBench only).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

SHINE_ROOT="${SHINE_ROOT:-${ROOT_DIR}/..}"
MAB_ROOT="${MEMORY_AGENT_BENCH_ROOT:-${SHINE_ROOT}/MemoryAgentBench}"
SHINE_AGENT_CONFIG="${SHINE_AGENT_CONFIG:-${MAB_ROOT}/configs/agent_conf/SHINE_Agents/SHINE_agent_qwen3_8b.yaml}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/ceph/home/muhan01/huggingfacemodels/Qwen3-8B}"
OUTPUT_JSON="${OUTPUT_JSON:-${SHINE_ROOT}/outputs/delta_mem_qwen3_8b/shine_memory_agent_bench.json}"

HF_HOME="${HF_HOME:-/ceph/home/muhan01/huggingfacemodels}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

export PYTHONPATH="${ROOT_DIR}:${SHINE_ROOT}:${MAB_ROOT}"
export SHINE_ROOT TOKENIZERS_PARALLELISM=false

mkdir -p "$(dirname "${OUTPUT_JSON}")"

EXTRA_SOURCE_FLAGS=()
if [[ -n "${MAB_SOURCES:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SOURCE_FLAGS=(--memory-agent-bench-sources ${MAB_SOURCES})
fi

exec "${PYTHON_BIN}" -m deltamem.eval.benchmark_compare \
  --model-path "${BASE_MODEL_PATH}" \
  --device cuda:0 \
  --dtype bfloat16 \
  --datasets-cache-dir "${HF_DATASETS_CACHE}" \
  --hub-cache-dir "${HF_HUB_CACHE}" \
  --external-memory-agent-bench-root "${MAB_ROOT}" \
  --shine-root "${SHINE_ROOT}" \
  --shine-agent-config "${SHINE_AGENT_CONFIG}" \
  --tasks memory_agent_bench \
  --memory-agent-bench-splits Accurate_Retrieval Test_Time_Learning Long_Range_Understanding Conflict_Resolution \
  "${EXTRA_SOURCE_FLAGS[@]}" \
  --seed 42 \
  --memory-agent-bench-max-new-tokens 4096 \
  --memory-agent-bench-max-context-chars "${MAB_MAX_CONTEXT_CHARS:-120000}" \
  --no-memory-agent-bench-use-official-prompt \
  --eval-do-sample \
  --eval-temperature 0.4 \
  --eval-top-p 0.9 \
  --eval-top-k 10 \
  --skip-base \
  --skip-delta \
  --skip-lora \
  --no-skip-shine \
  --output-json "${OUTPUT_JSON}"
