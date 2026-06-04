#!/bin/bash
# MemoryAgentBench × Qwen3-8B (HF long-context baseline) × SHINE
# Run on GPU server after editing paths in configs/agent_conf/*/

set -euo pipefail

source ~/.bashrc 2>/dev/null || true
conda activate MABench 2>/dev/null || true

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Repo roots: SHINE code at repo root, MAB at MemoryAgentBench/
MAB_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export SHINE_ROOT="${SHINE_ROOT:-$(cd "${MAB_ROOT}/.." && pwd)}"
root="${MAB_ROOT}"
cd "$root"

export HF_HOME="${HF_HOME:-$root/.cache/huggingface}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

# Which lines in shine_mab_eval.txt to run (inclusive). Default: smoke lines 4-5
START_LINE="${START_LINE:-4}"
END_LINE="${END_LINE:-5}"
STEP="${STEP:-1}"

file_name=shine_mab_eval.txt
NUM_GPUS="${NUM_GPUS:-1}"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-0}"

echo "SHINE_ROOT=$SHINE_ROOT"
echo "MAB root=$root"
echo "Running lines ${START_LINE}..${END_LINE} from bash_files/configs/${file_name}"

for line in $(seq "$START_LINE" "$STEP" "$END_LINE"); do
    cfg=$(sed -n "${line}p" "${root}/bash_files/configs/${file_name}")
    if [[ -z "${cfg// }" ]] || [[ "$cfg" == \#* ]]; then
        continue
    fi
    agent_config=$(echo "$cfg" | awk '{print $1}')
    dataset_config=$(echo "$cfg" | awk '{print $2}')

    if [[ -f "configs/agent_conf/SHINE_Agents/${agent_config}" ]]; then
        agent_cfg_path="configs/agent_conf/SHINE_Agents/${agent_config}"
    elif [[ -f "configs/agent_conf/Local_HF_Agents/${agent_config}" ]]; then
        agent_cfg_path="configs/agent_conf/Local_HF_Agents/${agent_config}"
    else
        echo "Missing agent config: ${agent_config}" >&2
        exit 1
    fi

    echo "................Start: ${agent_config} + ${dataset_config}..........."
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" python main.py \
        --agent_config "${agent_cfg_path}" \
        --dataset_config "configs/data_conf/${dataset_config}" \
        ${FORCE:+--force}
    echo "................End..........."
done

# Examples:
#   START_LINE=4 END_LINE=5 bash bash_files/sh/run_shine_mab.sh   # smoke EventQA
#   START_LINE=7 END_LINE=10 bash bash_files/sh/run_shine_mab.sh  # AR Ruler+EventQA
#   FORCE=1 START_LINE=4 END_LINE=5 bash bash_files/sh/run_shine_mab.sh
