#!/usr/bin/env bash
# Run after a smoke run completes. Evaluates the final checkpoint on the
# same val[200:250] slice as the baseline and prints a side-by-side diff.
#
# Usage: scripts/post_run_analysis.sh <run_dir>
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <run_dir>" >&2
  exit 1
fi
RUN_DIR="$1"

# Prefer the final checkpoint; fall back to the highest-numbered one.
if [[ -d "$RUN_DIR/checkpoint-final" ]]; then
  CKPT="$RUN_DIR/checkpoint-final"
else
  CKPT=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
fi

if [[ -z "$CKPT" || ! -d "$CKPT" ]]; then
  echo "No checkpoint found under $RUN_DIR" >&2
  exit 1
fi

OUT_JSON="$RUN_DIR/heldout_eval.json"
LABEL="$(basename "$RUN_DIR") $(basename "$CKPT")"

echo "[post-run] summary =============================================="
python scripts/summarize_run.py "$RUN_DIR"
echo
echo "[post-run] evaluating $CKPT on val[200:250] ====================="
CUDA_VISIBLE_DEVICES=0 python scripts/eval.py \
    --ckpt "$CKPT" \
    --context-start 200 --context-count 50 \
    --questions-per-context 2 \
    --max-new-tokens 64 \
    --label "$LABEL" \
    --out "$OUT_JSON"

echo
echo "[post-run] diff vs baseline ======================================"
python scripts/compare_eval.py \
    runs/baselines/shine_ift_mqa_1qa_val_200_250.json \
    "$OUT_JSON"
