#!/usr/bin/env bash
# δ-mem Qwen3-8B eval driver (any server). Configure via server_env.sh.
#
# Usage (after setup + download):
#   source MemoryAgentBench/bash_files/sh/source_server_env.sh
#   bash MemoryAgentBench/bash_files/sh/run_delta_mem.sh base-mab
#
# Modes:
#   base        full suite (LoCoMo + HotpotQA + GPQA + IFEval + MAB), frozen 8B
#   base-mab    frozen Qwen3-8B × full MAB only
#   compare-mab base + SHINE on full MAB
#   shine-mab   SHINE × full MAB only
#   all         base full suite + SHINE MAB
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/source_server_env.sh"

MODE="${1:-base-mab}"
SHINE_ROOT="${SHINE_ROOT:?SHINE_ROOT not set}"
DELTA_MEM_ROOT="${DELTA_MEM_ROOT:-${SHINE_ROOT}/third_party/delta-Mem}"
MAB_ROOT="${MAB_ROOT:-${SHINE_ROOT}/MemoryAgentBench}"
VENV_PYTHON="${DELTA_MEM_ROOT}/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PYTHON}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}. Run: bash ${SCRIPT_DIR}/setup_delta_mem.sh" >&2
  exit 1
fi

BASE_MODEL="${BASE_MODEL:?BASE_MODEL not set — edit server_env.sh}"
HF_HOME="${HF_HOME:-${SHINE_ROOT}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SHINE_ROOT}/outputs/delta_mem_qwen3_8b_full}"
LOG_ROOT="${OUTPUT_ROOT}/logs"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
MAB_EVAL_BATCH_SIZE="${MAB_EVAL_BATCH_SIZE:-16}"
MAB_MAX_CONTEXT_CHARS="${MAB_MAX_CONTEXT_CHARS:-120000}"
SEED="${SEED:-42}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
NUM_GPUS="${NUM_GPUS:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS}}"

SHINE_AGENT_CONFIG="${SHINE_AGENT_CONFIG:-${MAB_ROOT}/configs/agent_conf/SHINE_Agents/SHINE_agent_server.yaml}"
LOCOMO_DATA_FILE="${LOCOMO_DATA_FILE:-${DELTA_MEM_ROOT}/data/locomo10.json}"

MAB_SPLITS=(Accurate_Retrieval Test_Time_Learning Long_Range_Understanding Conflict_Resolution)

export SHINE_ROOT DELTA_MEM_ROOT MAB_ROOT HF_ENDPOINT
export PYTHONPATH="${DELTA_MEM_ROOT}:${SHINE_ROOT}:${MAB_ROOT}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE

if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
bash "${MAB_ROOT}/bash_files/sh/download_mab_recsys_entity2id.sh" || true

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "BASE_MODEL=${BASE_MODEL}"
if ! PYTHONPATH="${DELTA_MEM_ROOT}:${SHINE_ROOT}:${MAB_ROOT}" "${PYTHON_BIN}" -c "import torch, transformers; from deltamem.eval import benchmark_compare; print('preflight OK', torch.__version__, transformers.__version__, 'gpus=', torch.cuda.device_count())"; then
  echo "Fix env: RECREATE_VENV=1 bash ${SCRIPT_DIR}/setup_delta_mem.sh" >&2
  exit 1
fi

# Respect CUDA_VISIBLE_DEVICES from server_env.sh (e.g. 5,6). Only auto-pick if unset.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ -n "${CUDA_GPU_IDS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_GPU_IDS}"
  elif [[ "${NUM_GPUS}" -ge 8 ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  elif [[ "${NUM_GPUS}" -ge 4 ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
  elif [[ "${NUM_GPUS}" -ge 2 ]]; then
    export CUDA_VISIBLE_DEVICES=0,1
  else
    export CUDA_VISIBLE_DEVICES=0
  fi
fi

_ngpu=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
if [[ "${_ngpu}" -lt "${NPROC_PER_NODE}" ]]; then
  NPROC_PER_NODE="${_ngpu}"
fi
echo "GPUs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} | torchrun nproc=${NPROC_PER_NODE} | ATTN=${ATTN_IMPLEMENTATION}"

OFFLINE_FLAG=()
[[ "${LOCAL_FILES_ONLY}" == "1" ]] && OFFLINE_FLAG=(--local-files-only)

COMMON_BENCHMARK_FLAGS=(
  --model-path "${BASE_MODEL}"
  --device cuda:0
  --dtype bfloat16
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --datasets-cache-dir "${HF_DATASETS_CACHE}"
  --hub-cache-dir "${HF_HUB_CACHE}"
  --external-memory-agent-bench-root "${MAB_ROOT}"
  --seed "${SEED}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --base-inference-backend transformers
  --hotpotqa-max-new-tokens 32
  --hotpotqa-official-decoding
  --gpqa-max-new-tokens 8192
  --gpqa-official-decoding
  --ifeval-max-new-tokens 1500
  --memory-agent-bench-max-new-tokens 4096
  --memory-agent-bench-eval-batch-size "${MAB_EVAL_BATCH_SIZE}"
  --memory-agent-bench-max-context-chars "${MAB_MAX_CONTEXT_CHARS}"
  --memory-agent-bench-splits "${MAB_SPLITS[@]}"
  --no-memory-agent-bench-use-official-prompt
  --eval-do-sample
  --eval-temperature 0.4
  --eval-top-p 0.9
  --eval-top-k 10
  --skip-delta
  --skip-lora
  "${OFFLINE_FLAG[@]}"
)

run_distributed() {
  local master_port="$1"
  shift
  if [[ "${NPROC_PER_NODE}" -le 1 ]]; then
    "${PYTHON_BIN}" "$@"
    return
  fi
  "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_addr 127.0.0.1 \
    --master_port "${master_port}" \
    "$@"
}

run_locomo_base() {
  local out="${OUTPUT_ROOT}/base_model/locomo.json"
  local log="${LOG_ROOT}/base_model_locomo.log"
  mkdir -p "$(dirname "${out}")"
  if [[ -f "${out}" && "${FORCE:-0}" != "1" ]]; then
    echo "Skip existing ${out}"
    return 0
  fi
  echo "=== LoCoMo (base) -> ${out}"
  run_distributed 30071 \
    -m deltamem.eval.locomo_delta \
    --model-path "${BASE_MODEL}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --max-new-tokens 50 \
    --seed "${SEED}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --answer-reserve-tokens 50 \
    --full-history-mode official_prompt \
    --categories 1 2 3 4 \
    --output-json "${out}" \
    --data-file "${LOCOMO_DATA_FILE}" \
    2>&1 | tee "${log}"
}

run_benchmark_task_base() {
  local task="$1"
  local port="$2"
  local out="${OUTPUT_ROOT}/base_model/${task}.json"
  local log="${LOG_ROOT}/base_model_${task}.log"
  mkdir -p "$(dirname "${out}")"
  if [[ -f "${out}" && "${FORCE:-0}" != "1" ]]; then
    echo "Skip existing ${out}"
    return 0
  fi
  echo "=== ${task} (base) -> ${out}"
  run_distributed "${port}" \
    -m deltamem.eval.benchmark_compare \
    "${COMMON_BENCHMARK_FLAGS[@]}" \
    --tasks "${task}" \
    --skip-shine \
    --output-json "${out}" \
    2>&1 | tee "${log}"
}

run_mab_base_only() {
  run_benchmark_task_base memory_agent_bench 30174
}

run_mab_compare() {
  local out="${OUTPUT_ROOT}/compare_mab/base_and_shine.json"
  local log="${LOG_ROOT}/compare_mab.log"
  mkdir -p "$(dirname "${out}")"
  echo "=== memory_agent_bench base + SHINE -> ${out}"
  run_distributed 30180 \
    -m deltamem.eval.benchmark_compare \
    "${COMMON_BENCHMARK_FLAGS[@]}" \
    --tasks memory_agent_bench \
    --shine-root "${SHINE_ROOT}" \
    --shine-agent-config "${SHINE_AGENT_CONFIG}" \
    --no-skip-shine \
    --output-json "${out}" \
    2>&1 | tee "${log}"
}

run_mab_shine_only() {
  local out="${OUTPUT_ROOT}/shine_model/memory_agent_bench.json"
  local log="${LOG_ROOT}/shine_mab.log"
  mkdir -p "$(dirname "${out}")"
  echo "=== memory_agent_bench (SHINE only) -> ${out}"
  run_distributed 30181 \
    -m deltamem.eval.benchmark_compare \
    "${COMMON_BENCHMARK_FLAGS[@]}" \
    --tasks memory_agent_bench \
    --shine-root "${SHINE_ROOT}" \
    --shine-agent-config "${SHINE_AGENT_CONFIG}" \
    --skip-base \
    --no-skip-shine \
    --output-json "${out}" \
    2>&1 | tee "${log}"
}

run_full_base_suite() {
  run_locomo_base
  run_benchmark_task_base hotpotqa 30171
  run_benchmark_task_base gpqa_diamond 30172
  run_benchmark_task_base ifeval 30173
  run_benchmark_task_base memory_agent_bench 30174
}

case "${MODE}" in
  base|full|suite)
    run_full_base_suite
    ;;
  shine-mab|shine)
    run_mab_shine_only
    ;;
  base-mab|mab-base)
    run_mab_base_only
    ;;
  compare-mab|compare)
    run_mab_compare
    ;;
  all)
    run_full_base_suite
    run_mab_shine_only
    ;;
  *)
    echo "Usage: $0 [base|base-mab|shine-mab|compare-mab|all]" >&2
    exit 1
    ;;
esac

cat <<EOF

Done (${MODE}). Results: ${OUTPUT_ROOT}
  base MAB:  ${OUTPUT_ROOT}/base_model/memory_agent_bench.json
  compare:   ${OUTPUT_ROOT}/compare_mab/base_and_shine.json
  logs:      ${LOG_ROOT}/

FORCE=1 to rerun; LOCAL_FILES_ONLY=1 for offline cache only.

EOF
