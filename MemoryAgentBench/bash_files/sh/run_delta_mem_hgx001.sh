#!/usr/bin/env bash
# δ-mem Qwen3-8B 官方全套评测（hgx001 路径），可选加 SHINE。
#
# 与 declare-lab/delta-Mem/scripts/run_qasper_multimodel_write8192_benchmark_suite_qwen3_8b.sh
# 对齐：locomo + hotpotqa + gpqa_diamond + ifeval + memory_agent_bench（全 split、全 source）。
#
# Usage:
#   bash setup_delta_mem_hgx001.sh
#   bash run_delta_mem_hgx001.sh              # 默认：复现 frozen base 全套（exact 协议）
#   bash run_delta_mem_hgx001.sh base         # 同上
#   bash run_delta_mem_hgx001.sh shine-mab    # 仅 SHINE × 完整 MAB
#   bash run_delta_mem_hgx001.sh base-mab     # 仅 frozen Qwen3-8B × 完整 MAB
set -euo pipefail

MODE="${1:-base}"
SHINE_ROOT="${SHINE_ROOT:-/ceph/home/muhan01/wyd/SHINE-mem}"
DELTA_MEM_ROOT="${DELTA_MEM_ROOT:-${SHINE_ROOT}/third_party/delta-Mem}"
MAB_ROOT="${MAB_ROOT:-${SHINE_ROOT}/MemoryAgentBench}"
VENV_PYTHON="${DELTA_MEM_ROOT}/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PYTHON}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}. Run setup_delta_mem_hgx001.sh first." >&2
  exit 1
fi
# 避免 conda 的 python 混入 .venv 的 site-packages（会出 torch/transformers 版本错乱）
if [[ -n "${CONDA_PREFIX:-}" ]] && [[ "${PYTHON_BIN}" != "${VENV_PYTHON}" ]]; then
  echo "WARNING: conda active (${CONDA_PREFIX}) but PYTHON_BIN=${PYTHON_BIN}" >&2
  echo "         Recommend: export PYTHON_BIN=${VENV_PYTHON}" >&2
fi

BASE_MODEL="${BASE_MODEL:-/ceph/home/muhan01/huggingfacemodels/Qwen3-8B}"
HF_HOME="${HF_HOME:-/ceph/home/muhan01/huggingfacemodels}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SHINE_ROOT}/outputs/delta_mem_qwen3_8b_full}"
LOG_ROOT="${OUTPUT_ROOT}/logs"

# 官方套件默认（qwen3_8b benchmark suite）
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
MAB_EVAL_BATCH_SIZE="${MAB_EVAL_BATCH_SIZE:-16}"
MAB_MAX_CONTEXT_CHARS="${MAB_MAX_CONTEXT_CHARS:-120000}"
SEED="${SEED:-42}"
# 无 flash-attn 时用 sdpa；有 flash-attn 可设 flash_attention_2
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
NUM_GPUS="${NUM_GPUS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS}}"

SHINE_AGENT_CONFIG="${SHINE_AGENT_CONFIG:-${MAB_ROOT}/configs/agent_conf/SHINE_Agents/SHINE_agent_qwen3_8b_deltamem.yaml}"
LOCOMO_DATA_FILE="${LOCOMO_DATA_FILE:-${DELTA_MEM_ROOT}/data/locomo10.json}"

MAB_SPLITS=(Accurate_Retrieval Test_Time_Learning Long_Range_Understanding Conflict_Resolution)
BENCHMARK_TASKS=(hotpotqa gpqa_diamond ifeval memory_agent_bench)

export SHINE_ROOT DELTA_MEM_ROOT MAB_ROOT
export PYTHONPATH="${DELTA_MEM_ROOT}:${SHINE_ROOT}:${MAB_ROOT}"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false
export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE

if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
# Non-fatal: hgx001 often cannot reach huggingface.co. Missing file only breaks recsys (~54%).
bash "${MAB_ROOT}/bash_files/sh/download_mab_recsys_entity2id.sh" || true
echo "PYTHON_BIN=${PYTHON_BIN}"
if ! PYTHONPATH="${DELTA_MEM_ROOT}:${SHINE_ROOT}:${MAB_ROOT}" "${PYTHON_BIN}" -c "import torch, transformers; from deltamem.eval import benchmark_compare; print('preflight OK', torch.__version__, transformers.__version__)"; then
  echo "Fix env: RECREATE_VENV=1 TORCH_INDEX=cu121 bash ${MAB_ROOT}/bash_files/sh/setup_delta_mem_hgx001.sh" >&2
  exit 1
fi

# 默认 4 卡（0–3）。申请 4-GPU 作业后调度器通常会设 CUDA_VISIBLE_DEVICES=0,1,2,3
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ "${NUM_GPUS}" -ge 8 ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  elif [[ "${NUM_GPUS}" -ge 4 ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
  else
    export CUDA_VISIBLE_DEVICES=0
  fi
fi
_ngpu=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}')
if [[ "${_ngpu}" -lt "${NPROC_PER_NODE}" ]]; then
  NPROC_PER_NODE="${_ngpu}"
fi
echo "GPUs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} | torchrun nproc=${NPROC_PER_NODE} (target NUM_GPUS=${NUM_GPUS}) | ATTN=${ATTN_IMPLEMENTATION}"
if [[ "${_ngpu}" -lt "${NUM_GPUS}" ]]; then
  echo "NOTE: 仅 ${_ngpu} 张卡可见，期望 ${NUM_GPUS} 张。请用 4-GPU 作业提交，例如: srun --gres=gpu:4 ... 或 export CUDA_VISIBLE_DEVICES=0,1,2,3" >&2
fi

OFFLINE_FLAG=()
[[ "${LOCAL_FILES_ONLY}" == "1" ]] && OFFLINE_FLAG=(--local-files-only)

HOTPOTQA_FLAG=(--hotpotqa-official-decoding)
GPQA_FLAG=(--gpqa-official-decoding)

# δ-mem 官方 benchmark_compare 公共参数（与 qwen3_8b suite 一致）
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
  "${HOTPOTQA_FLAG[@]}"
  --gpqa-max-new-tokens 8192
  "${GPQA_FLAG[@]}"
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
  echo "=== LoCoMo (base) → ${out}"
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
  echo "=== ${task} (base) → ${out}"
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
  echo "=== memory_agent_bench base + SHINE → ${out}"
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
  echo "=== memory_agent_bench (SHINE only) → ${out}"
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
    # SHINE: batch=1 per query; each torchrun rank loads a full metanetwork. Default 1 GPU for memory headroom.
    # LoRA evidence capped by shine_context_max_length (default 8196); query prompt still uses MAB_MAX_CONTEXT_CHARS.
    if [[ -n "${SHINE_NUM_GPUS:-}" ]]; then
      NUM_GPUS="${SHINE_NUM_GPUS}"
      NPROC_PER_NODE="${SHINE_NUM_GPUS}"
    elif [[ "${NUM_GPUS}" -gt 1 && -z "${SHINE_ALLOW_MULTI_GPU:-}" ]]; then
      echo "NOTE: shine-mab using NUM_GPUS=1 (override: SHINE_NUM_GPUS=4 or SHINE_ALLOW_MULTI_GPU=1)" >&2
      NUM_GPUS=1
      NPROC_PER_NODE=1
    fi
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
    echo "  base        δ-mem 官方全套 frozen Qwen3-8B（默认）" >&2
    echo "  base-mab    仅 frozen Qwen3-8B × 完整 MAB（不跑 SHINE）" >&2
    echo "  shine-mab   完整 MAB + SHINE" >&2
    echo "  compare-mab 完整 MAB，base 与 SHINE 同一 JSON" >&2
    echo "  all         base 全套 + SHINE MAB" >&2
    exit 1
    ;;
esac

cat <<EOF

Done (${MODE}). Results: ${OUTPUT_ROOT}
  base LoCoMo:     ${OUTPUT_ROOT}/base_model/locomo.json
  base HotpotQA:   ${OUTPUT_ROOT}/base_model/hotpotqa.json
  base GPQA:       ${OUTPUT_ROOT}/base_model/gpqa_diamond.json
  base IFEval:     ${OUTPUT_ROOT}/base_model/ifeval.json
  base MAB:        ${OUTPUT_ROOT}/base_model/memory_agent_bench.json
    → .base.memory_agent_bench.summary.overall

重跑加 FORCE=1；仅离线缓存加 LOCAL_FILES_ONLY=1

EOF
