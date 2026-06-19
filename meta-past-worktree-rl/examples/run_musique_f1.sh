#!/usr/bin/env bash
# Run RL training on MuSiQue (multi-hop QA) with the F1 reward — no
# external services needed.
#
# Usage:
#   bash examples/run_musique_f1.sh
#   bash examples/run_musique_f1.sh 4   # use 4 GPUs instead of 8
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${1:-8}"
PYBIN="/ceph/home/muhan01/.conda/envs/vllm_serve/bin"
CONFIG="meta_past/config/rl_musique_grpo.yaml"

mkdir -p runs/musique_grpo
echo "[run_musique_f1] launching torchrun with nproc_per_node=${NPROC}"
echo "[run_musique_f1] config=${CONFIG}"
echo

exec "${PYBIN}/torchrun" \
    --nproc_per_node="${NPROC}" --standalone \
    scripts/train.py \
    --config "${CONFIG}"
