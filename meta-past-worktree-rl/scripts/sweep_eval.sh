#!/usr/bin/env bash
# Orchestrator: run the eval harness (full core suite) over three
# SHINE checkpoints back-to-back. Writes one out_dir per checkpoint
# plus a final comparison TSV at runs/eval_sweep/comparison.tsv.
#
# Usage:
#   bash scripts/sweep_eval.sh                  # default: 8 GPUs, full core
#   bash scripts/sweep_eval.sh --smoke          # smoke (50 items each)
#   bash scripts/sweep_eval.sh --nproc 4
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Pass-through flags to run_eval.sh (every arg except --out/--ckpt, which
# we manage per-checkpoint).
PASSTHROUGH=()
while [ $# -gt 0 ]; do
  PASSTHROUGH+=( "$1" )
  shift
done

# (label, ckpt_dir, thinking) tuples — order is the eval order.
# `thinking` must match how the ckpt was trained so the chat template
# at eval matches train. Mismatches degrade SHINE substantially (see
# smoke run notes: forcing enable_thinking=false on a ckpt trained with
# the flag absent makes mmlu_zeroshot/shine drop below the zero-shot
# base-model baseline).
LABELS=(shine_orig musique_best bbh_best)
CKPTS=(
  "${HOME}/huggingfacemodels/SHINE-ift_mqa_1qa"
  "runs/musique_grpo/checkpoint-best"
  "runs/bbh_grpo/checkpoint-best"
)
THINKINGS=(
  "null"     # original SHINE pretrain — no enable_thinking flag in train
  "null"     # rl_musique_grpo.yaml has no enable_thinking line
  "false"    # rl_bbh_grpo.yaml sets enable_thinking: false
)

SWEEP_DIR="runs/eval_sweep"
mkdir -p "${SWEEP_DIR}"
SWEEP_LOG="${SWEEP_DIR}/sweep.log"
SUMMARY_TSV="${SWEEP_DIR}/comparison.tsv"

# Header for the comparison TSV (one row per ckpt × dataset × mode × shots).
echo -e "ckpt\tdataset\tbucket\tmode\tshots\tn_items\tscore_mean\telapsed_s" \
  > "${SUMMARY_TSV}"

echo "[sweep] start $(date -Iseconds)" | tee -a "${SWEEP_LOG}"
echo "[sweep] passthrough: ${PASSTHROUGH[*]:-(none)}" | tee -a "${SWEEP_LOG}"

for i in "${!LABELS[@]}"; do
  LABEL="${LABELS[$i]}"
  CKPT="${CKPTS[$i]}"
  THINKING="${THINKINGS[$i]}"
  OUT="runs/eval_${LABEL}"

  if [ ! -e "${CKPT}" ]; then
    echo "[sweep] SKIP ${LABEL} — ckpt missing: ${CKPT}" | tee -a "${SWEEP_LOG}"
    continue
  fi

  echo "" | tee -a "${SWEEP_LOG}"
  echo "============================================================" | tee -a "${SWEEP_LOG}"
  echo "[sweep] ${LABEL}: ckpt=${CKPT}  thinking=${THINKING}  out=${OUT}" | tee -a "${SWEEP_LOG}"
  echo "[sweep] start $(date -Iseconds)" | tee -a "${SWEEP_LOG}"
  echo "============================================================" | tee -a "${SWEEP_LOG}"

  mkdir -p "${OUT}"
  # The eval may print a SIGSEGV traceback at the very end (vLLM teardown
  # is brittle) AFTER summary.jsonl has been written; we ignore exit code
  # and trust the on-disk summary.
  bash examples/run_eval.sh \
        --ckpt "${CKPT}" \
        --thinking "${THINKING}" \
        --out "${OUT}" \
        "${PASSTHROUGH[@]}" \
        2>&1 | tee -a "${SWEEP_LOG}" || true
  if [ -f "${OUT}/summary.jsonl" ]; then
    echo "[sweep] ${LABEL} OK (summary written) $(date -Iseconds)" | tee -a "${SWEEP_LOG}"
  else
    echo "[sweep] ${LABEL} FAILED (no summary) $(date -Iseconds)" | tee -a "${SWEEP_LOG}"
  fi

  # Append summary.jsonl into the comparison TSV.
  if [ -f "${OUT}/summary.jsonl" ]; then
    /ceph/home/muhan01/.conda/envs/vllm_serve/bin/python - <<EOF >> "${SUMMARY_TSV}"
import json, sys
with open("${OUT}/summary.jsonl") as f:
    for line in f:
        r = json.loads(line)
        print("\t".join([
            "${LABEL}", r["dataset"], r["bucket"], r["mode"],
            str(r.get("shots") or ""),
            str(r["n_items"]), f"{r['score_mean']:.4f}",
            f"{r['elapsed_s']:.1f}",
        ]))
EOF
  fi
done

echo "" | tee -a "${SWEEP_LOG}"
echo "[sweep] end $(date -Iseconds)" | tee -a "${SWEEP_LOG}"
echo "[sweep] comparison TSV: ${SUMMARY_TSV}" | tee -a "${SWEEP_LOG}"
echo "" | tee -a "${SWEEP_LOG}"

# Pretty-print the comparison.
column -t -s $'\t' "${SUMMARY_TSV}" | tee -a "${SWEEP_LOG}"
