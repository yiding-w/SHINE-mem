#!/bin/bash
# Phase 2: streaming long-history memory training on Qwen3.5-4B, 2 GPUs.
# TP mode with tensor_parallel_size=1 == pure data-parallel (a full 4B replica per
# GPU; the 4B is too small for PP and its device_map is single-GPU). detach_state=full
# -> FullTPDetachState. CORRECT streaming needs batch_size=1 so one history's segments
# accumulate into the same W position across consecutive steps (reset on repo change).
# With DP=NPROC the data is split into NPROC contiguous halves -> at most ~1 history
# straddles a rank boundary per epoch (negligible). Set NPROC=1 for a fully-clean
# single-stream validation run.
#
# Usage:
#   # 1) synth data first (segmented format):
#   python datagen/generate_memory_seg.py --out data/mem_synth/train.jsonl --num 3000 \
#       --filler-hf arxiv,wiki,dialog --filler-hf-num 40000 \
#       --segments 8,16,32,64 --segment-tokens 1800 --tokenizer ./models/Qwen3.5-4B
#   python datagen/generate_memory_seg.py --out data/mem_synth/val.jsonl --num 200 \
#       --segments 8,16 --segment-tokens 1800 --seed 7 --tokenizer ./models/Qwen3.5-4B
#   # 2) (recommended) verify the data contract:
#   python datagen/check_stream.py --jsonl data/mem_synth/train.jsonl --model ./models/Qwen3.5-4B
#   # 3) train:
#   bash scripts/run_memory_stream_bs1.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # SHINE_V2-main root

# NOTE: run this with your training conda env ALREADY active (e.g. `conda activate MABench`).
# We intentionally do NOT `source ~/.bashrc` here — some .bashrc files auto-attach
# screen/tmux or re-exec, which silently hangs/kills the script under `set -e`.

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# --- data config (configs/data/ is gitignored in V2; create it if missing) ---
DCFG=configs/data/pretrain_annealing/memory_stream.yaml
if [ ! -f "$DCFG" ]; then
  mkdir -p "$(dirname "$DCFG")"
  cat > "$DCFG" <<'YAML'
name: memory_stream
dataset_module: mydatasets.pretrain_annealing.memory_stream
context_seq_length: 2048      # per-segment memory window (the streaming chunk)
conv_seq_length: 1024         # per-segment QA (+ final QA on last segment)
data_path: "data/mem_synth"
train_file: "train.jsonl"
val_file: "val.jsonl"
shuffle: false                # CRITICAL: streaming needs in-order, repo-contiguous
validation_split_num: 256
YAML
  echo "[run] wrote $DCFG"
fi

MODEL=${MODEL:-Qwen3_5-4B}
NPROC=${NPROC:-2}
MASTER_PORT=${MASTER_PORT:-29531}

echo "[run] model=$MODEL nproc=$NPROC  (TP mode, tp_size=1 -> DP=$NPROC, batch_size=1)"

exec torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
  meta_train.py --config-name=main_pretrain_annealing \
    model="$MODEL" \
    m2p_transformer=full_prenorm_gatedlastnorm_4b \
    data=pretrain_annealing/memory_stream \
    detach_state=full \
    parallel.mode=tp \
    parallel.tensor_parallel_size=1 \
    parallel.total_gpus="$NPROC" \
    training.tp_batchsize.batch_size=1 \
    training.resume_from=null \
    training.save_best_only=true \
    "$@"
