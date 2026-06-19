#!/usr/bin/env bash
# Run the SHINE-hypernet eval harness. Default: "core" subset
# (~12 datasets covering all 3 buckets), 8 GPUs (DDP data parallel).
#
# Flags:
#   --smoke              Truncate each dataset to 50 items (fast sanity).
#   --nproc N            Number of GPUs (default: 8). N=1 = single-process.
#   --datasets STR       What to run:
#                          - "core" (default): recommended v1 subset
#                          - "all"           : every registered dataset
#                          - "<a,b,c>"       : comma-separated explicit list
#   --shots STR          K-shot list for bucket-B datasets (default: 1,4,16
#                        in full mode, 4 in smoke mode).
#   --out DIR            Where per-mode JSONL + summary go.
#                        Defaults: runs/eval_core | runs/eval_smoke | runs/eval_all
#   --modes STR          Comma-separated modes (default: shine,icl,zero).
#   --ckpt PATH          SHINE hypernet checkpoint dir
#                        (default: ~/huggingfacemodels/SHINE-ift_mqa_1qa).
#   --thinking VAL       enable_thinking: true (default) / false / null.
#                        Must match how the ckpt was trained.
#   --force_think VAL    true / false (default false). Append "<think>\n" to
#                        every prompt so completions start mid-thinking.
#                        Must match how the ckpt was trained.
#   --limit N            Cap each dataset to N items (overrides --smoke).
#
# Examples:
#   bash examples/run_eval.sh                                  # core, 8 GPUs, full
#   bash examples/run_eval.sh --smoke                          # core, smoke
#   bash examples/run_eval.sh --nproc 4                        # core, 4 GPUs
#   bash examples/run_eval.sh --datasets all                   # every dataset
#   bash examples/run_eval.sh --datasets mmlu,humaneval        # specific list
#   bash examples/run_eval.sh --datasets bbh --shots 1,2,4,8,16  # K-sweep
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
PYBIN="/ceph/home/muhan01/.conda/envs/vllm_serve/bin"

# Dataset presets ---------------------------------------------------------------
# v1 "core" — the curated recommendation from eval_datasets.md.
CORE="squad,boolq,hotpotqa,drop,narrativeqa,pubmedqa,bbh,arc_challenge,gsm8k,babi,humaneval,mmlu_zeroshot"
# Every registered adapter. Kept in sync with meta_past/eval/datasets/__init__.py.
ALL="squad,musique,hotpotqa,2wikimulti,drop,narrativeqa,pubmedqa,boolq,triviaqa,newsqa,\
bbh,mmlu,mmlu_pro,agieval,arc_challenge,openbookqa,commonsenseqa,hellaswag,piqa,gsm8k,\
strategyqa,babi,bigbench_non_hard,natural_instr,truthfulqa_mc,\
gsm8k_zeroshot,mmlu_zeroshot,humaneval,truthfulqa_gen"

# Defaults ----------------------------------------------------------------------
SMOKE=0
NPROC=8
DATASETS_RAW="core"
SHOTS=""
MODES="shine,icl,zero"
OUT=""
CKPT=""
THINKING=""
FORCE_THINK=""
LIMIT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --smoke)        SMOKE=1; shift;;
    --nproc)        NPROC="$2"; shift 2;;
    --datasets)     DATASETS_RAW="$2"; shift 2;;
    --shots)        SHOTS="$2"; shift 2;;
    --modes)        MODES="$2"; shift 2;;
    --out)          OUT="$2"; shift 2;;
    --ckpt)         CKPT="$2"; shift 2;;
    --thinking)     THINKING="$2"; shift 2;;
    --force_think)  FORCE_THINK="$2"; shift 2;;
    --limit)        LIMIT="$2"; shift 2;;
    -h|--help)
      sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0;;
    *)
      echo "unknown flag: $1" >&2
      echo "see: bash examples/run_eval.sh --help" >&2
      exit 1;;
  esac
done

CKPT_ARGS=()
[ -n "${CKPT}" ] && CKPT_ARGS=( --ckpt_dir "${CKPT}" )
THINKING_ARGS=()
[ -n "${THINKING}" ] && THINKING_ARGS=( --enable_thinking "${THINKING}" )
FORCE_THINK_ARGS=()
[ -n "${FORCE_THINK}" ] && FORCE_THINK_ARGS=( --force_think "${FORCE_THINK}" )

# Resolve datasets preset
case "${DATASETS_RAW}" in
  core)  DATASETS="${CORE}";;
  all)   DATASETS="${ALL}";;
  *)     DATASETS="${DATASETS_RAW}";;
esac

# Resolve smoke + limit flags. --limit takes precedence over --smoke.
if [ -n "${LIMIT}" ]; then
  LIMIT_ARGS=( --limit "${LIMIT}" )
elif [ "${SMOKE}" = "1" ]; then
  LIMIT_ARGS=( --limit 50 )
else
  LIMIT_ARGS=()
fi

if [ "${SMOKE}" = "1" ] && [ -z "${LIMIT}" ]; then
  [ -z "${SHOTS}" ] && SHOTS="4"
  [ -z "${OUT}" ] && OUT="runs/eval_smoke"
else
  [ -z "${SHOTS}" ] && SHOTS="1,4,16"
  if [ -z "${OUT}" ]; then
    case "${DATASETS_RAW}" in
      core) OUT="runs/eval_core";;
      all)  OUT="runs/eval_all";;
      *)    OUT="runs/eval_custom";;
    esac
  fi
fi

mkdir -p "${OUT}"
echo "[run_eval] datasets=${DATASETS_RAW}  smoke=${SMOKE}  nproc=${NPROC}"
echo "[run_eval] shots=${SHOTS}  modes=${MODES}  out=${OUT}"
echo

if [ "${NPROC}" -gt 1 ]; then
  exec "${PYBIN}/torchrun" \
      --nproc_per_node="${NPROC}" --standalone \
      scripts/run_eval.py \
      --datasets "${DATASETS}" \
      --modes "${MODES}" \
      --shots "${SHOTS}" \
      --out_dir "${OUT}" \
      "${CKPT_ARGS[@]}" \
      "${THINKING_ARGS[@]}" \
      "${FORCE_THINK_ARGS[@]}" \
      "${LIMIT_ARGS[@]}"
else
  exec "${PYBIN}/python" scripts/run_eval.py \
      --datasets "${DATASETS}" \
      --modes "${MODES}" \
      --shots "${SHOTS}" \
      --out_dir "${OUT}" \
      "${CKPT_ARGS[@]}" \
      "${THINKING_ARGS[@]}" \
      "${FORCE_THINK_ARGS[@]}" \
      "${LIMIT_ARGS[@]}"
fi
