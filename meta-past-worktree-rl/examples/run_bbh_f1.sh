#!/usr/bin/env bash
# Run RL training on BIG-Bench Hard (in-parameter few-shot) with F1
# reward. No external services needed.
#
# Each SHINE context = K demonstrations from a single BBH task; the
# Qwen3+LoRA prompt is a held-out query from the SAME task. The model
# must answer using only what the LoRA encoded — see
# meta_past/data/bbh_contexts.py for the construction details.
#
# Usage:
#   bash examples/run_bbh_f1.sh
#   bash examples/run_bbh_f1.sh 4   # use 4 GPUs instead of 8
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${1:-8}"
PYBIN="/ceph/home/muhan01/.conda/envs/vllm_serve/bin"
CONFIG="meta_past/config/rl_bbh_grpo.yaml"

mkdir -p runs/bbh_grpo
echo "[run_bbh_f1] launching torchrun with nproc_per_node=${NPROC}"
echo "[run_bbh_f1] config=${CONFIG}"
echo

exec "${PYBIN}/torchrun" \
    --nproc_per_node="${NPROC}" --standalone \
    scripts/train.py \
    --config "${CONFIG}"
