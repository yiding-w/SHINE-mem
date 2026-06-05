#!/usr/bin/env bash
# δ-mem benchmark_compare: frozen Qwen3-8B baseline + SHINE (same protocol, one JSON).
#
# Usage:
#   bash setup_delta_mem_hgx001.sh     # once: clone δ-mem + apply SHINE patches
#   bash run_delta_mem_hgx001.sh smoke # base only, EventQA
#   bash run_delta_mem_hgx001.sh shine # SHINE only, EventQA
#   bash run_delta_mem_hgx001.sh all   # base + SHINE, full MAB
set -euo pipefail

MODE="${1:-smoke}"
SHINE_ROOT="${SHINE_ROOT:-/ceph/home/muhan01/wyd/SHINE-mem}"
DELTA_MEM_ROOT="${DELTA_MEM_ROOT:-${SHINE_ROOT}/third_party/delta-Mem}"
MAB_ROOT="${MAB_ROOT:-${SHINE_ROOT}/MemoryAgentBench}"
PYTHON_BIN="${PYTHON_BIN:-${DELTA_MEM_ROOT}/.venv/bin/python}"

BASE_MODEL="${BASE_MODEL:-/ceph/home/muhan01/huggingfacemodels/Qwen3-8B}"
HF_HOME="${HF_HOME:-/ceph/home/muhan01/huggingfacemodels}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SHINE_ROOT}/outputs/delta_mem_qwen3_8b}"
SHINE_AGENT_CONFIG="${SHINE_AGENT_CONFIG:-${MAB_ROOT}/configs/agent_conf/SHINE_Agents/SHINE_agent_qwen3_8b.yaml}"
MAB_MAX_CONTEXT_CHARS="${MAB_MAX_CONTEXT_CHARS:-120000}"

export SHINE_ROOT DELTA_MEM_ROOT MAB_ROOT
export PYTHONPATH="${DELTA_MEM_ROOT}:${SHINE_ROOT}:${MAB_ROOT}"
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TOKENIZERS_PARALLELISM=false

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Run: bash ${MAB_ROOT}/bash_files/sh/setup_delta_mem_hgx001.sh" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

case "${MODE}" in
  smoke)
    OUT="${OUTPUT_ROOT}/smoke_compare.json"
    SPLITS=(Accurate_Retrieval)
    SOURCES=(eventqa_full)
    SKIP_BASE=""
    SKIP_SHINE="--skip-shine"
    ;;
  shine)
    OUT="${OUTPUT_ROOT}/smoke_shine.json"
    SPLITS=(Accurate_Retrieval)
    SOURCES=(eventqa_full)
    SKIP_BASE="--skip-base"
    SKIP_SHINE="--no-skip-shine"
    ;;
  mab)
    OUT="${OUTPUT_ROOT}/mab_base.json"
    SPLITS=(Accurate_Retrieval Test_Time_Learning Long_Range_Understanding Conflict_Resolution)
    SOURCES=()
    SKIP_BASE=""
    SKIP_SHINE="--skip-shine"
    ;;
  all)
    OUT="${OUTPUT_ROOT}/mab_compare.json"
    SPLITS=(Accurate_Retrieval Test_Time_Learning Long_Range_Understanding Conflict_Resolution)
    SOURCES=()
    SKIP_BASE=""
    SKIP_SHINE="--no-skip-shine"
    ;;
  *)
    echo "Usage: $0 smoke|shine|mab|all" >&2
    exit 1
    ;;
esac

echo "=== δ-mem benchmark_compare (${MODE}) → ${OUT}"

CMD=(
  "${PYTHON_BIN}" -m deltamem.eval.benchmark_compare
  --model-path "${BASE_MODEL}"
  --device cuda:0
  --dtype bfloat16
  --datasets-cache-dir "${HF_DATASETS_CACHE}"
  --hub-cache-dir "${HF_HUB_CACHE}"
  --external-memory-agent-bench-root "${MAB_ROOT}"
  --shine-root "${SHINE_ROOT}"
  --shine-agent-config "${SHINE_AGENT_CONFIG}"
  --tasks memory_agent_bench
  --memory-agent-bench-splits "${SPLITS[@]}"
  --seed 42
  --memory-agent-bench-max-new-tokens 4096
  --memory-agent-bench-max-context-chars "${MAB_MAX_CONTEXT_CHARS}"
  --no-memory-agent-bench-use-official-prompt
  --eval-do-sample --eval-temperature 0.4 --eval-top-p 0.9 --eval-top-k 10
  --skip-delta --skip-lora
  --output-json "${OUT}"
)

[[ -n "${SKIP_BASE}" ]] && CMD+=("${SKIP_BASE}")
CMD+=("${SKIP_SHINE}")
[[ ${#SOURCES[@]} -gt 0 ]] && CMD+=(--memory-agent-bench-sources "${SOURCES[@]}")

"${CMD[@]}"

echo "Done. Inspect: ${OUT}"
echo "  base  → .base.memory_agent_bench.summary.overall"
echo "  shine → .shine.memory_agent_bench.summary.overall"
