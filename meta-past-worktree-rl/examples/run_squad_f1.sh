#!/usr/bin/env bash
# Run RL training on SQuAD v1 (single-passage extractive QA) with F1
# reward. SHINE was pretrained on SQuAD-train, so this is primarily a
# smoke / baseline run — heldout F1 starts high and RL gains are small.
# Useful for stack regression checks and as an apples-to-apples
# infrastructure baseline against MuSiQue / BBH.
#
# Usage:
#   bash examples/run_squad_f1.sh
#   bash examples/run_squad_f1.sh 4   # use 4 GPUs instead of 8
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${1:-8}"
PYBIN="/ceph/home/muhan01/.conda/envs/vllm_serve/bin"
CONFIG="meta_past/config/rl_squad_grpo.yaml"

mkdir -p runs/squad_grpo
echo "[run_squad_f1] launching torchrun with nproc_per_node=${NPROC}"
echo "[run_squad_f1] config=${CONFIG}"
echo

exec "${PYBIN}/torchrun" \
    --nproc_per_node="${NPROC}" --standalone \
    scripts/train.py \
    --config "${CONFIG}"
