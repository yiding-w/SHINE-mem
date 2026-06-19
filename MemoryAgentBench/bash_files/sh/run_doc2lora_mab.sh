#!/bin/bash
# MemoryAgentBench × doc-to-lora (SakanaAI D2L) × Qwen3-4B hypernetwork
# Runs in the DEDICATED D2L env (built from doc-to-lora/pyproject.toml), NOT MABench,
# to avoid transformers/accelerate version conflicts with the SHINE/Qwen3-8B runs.
#
# Before running:
#   - clone doc-to-lora to $SHINE_ROOT/../doc-to-lora (or set D2L_ROOT)
#   - download checkpoint: huggingface-cli download SakanaAI/doc-to-lora \
#       --local-dir <D2L_ROOT>/trained_d2l --include "qwen_4b_d2l/*"
#   - build/activate the env (conda env D2L, or set CONDA_ENV=...):
#       pip install -e <D2L_ROOT>           # the ctx_to_lora package + its pinned deps
#       # MAB glue deps that agent.py/main.py import at top level (not in D2L pyproject):
#       pip install tiktoken openai langchain-core python-dotenv
#     (letta/mem0/cognee/zep are imported lazily and are NOT needed for the D2L agent.)

set -euo pipefail

source ~/.bashrc 2>/dev/null || true
conda activate "${CONDA_ENV:-D2L}" 2>/dev/null || true

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# Repo roots: SHINE code at repo root, MAB at MemoryAgentBench/
MAB_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export SHINE_ROOT="${SHINE_ROOT:-$(cd "${MAB_ROOT}/.." && pwd)}"
# doc-to-lora repo root (its src/ is importable as ctx_to_lora). Used by the runner
# when agent_config['d2l_root'] is unset.
export D2L_ROOT="${D2L_ROOT:-$(cd "${SHINE_ROOT}/.." && pwd)/doc-to-lora}"
root="${MAB_ROOT}"
cd "$root"

export HF_HOME="${HF_HOME:-$root/.cache/huggingface}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

# Which lines in doc2lora_mab_eval.txt to run (inclusive). Default: smoke line 3 (EventQA).
START_LINE="${START_LINE:-3}"
END_LINE="${END_LINE:-3}"
STEP="${STEP:-1}"

file_name=doc2lora_mab_eval.txt
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-0}"

echo "SHINE_ROOT=$SHINE_ROOT"
echo "D2L_ROOT=$D2L_ROOT"
echo "MAB root=$root"
echo "Running lines ${START_LINE}..${END_LINE} from bash_files/configs/${file_name}"

for line in $(seq "$START_LINE" "$STEP" "$END_LINE"); do
    cfg=$(sed -n "${line}p" "${root}/bash_files/configs/${file_name}")
    if [[ -z "${cfg// }" ]] || [[ "$cfg" == \#* ]]; then
        continue
    fi
    agent_config=$(echo "$cfg" | awk '{print $1}')
    dataset_config=$(echo "$cfg" | awk '{print $2}')

    if [[ -f "configs/agent_conf/DocToLora_Agents/${agent_config}" ]]; then
        agent_cfg_path="configs/agent_conf/DocToLora_Agents/${agent_config}"
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
#   START_LINE=3 END_LINE=3 bash bash_files/sh/run_doc2lora_mab.sh   # smoke EventQA
#   START_LINE=6 END_LINE=7 bash bash_files/sh/run_doc2lora_mab.sh   # AR Ruler+EventQA
#   FORCE=1 START_LINE=3 END_LINE=3 bash bash_files/sh/run_doc2lora_mab.sh
