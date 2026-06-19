#!/usr/bin/env bash
# Run RL training on MuSiQue with an LLM-as-judge reward.
#
# Brings up the judge HTTP server in the background, waits for it to be
# healthy, then launches torchrun. Cleans up the judge on exit (Ctrl-C,
# training done, error).
#
# Backends:
#   --backend openai          forward to OpenAI gpt-4o-mini
#                             (needs OPENAI_API_KEY in env, default)
#   --backend openai-compat   forward to a local OpenAI-compatible server
#                             (e.g. a vLLM ``vllm serve`` instance)
#                             requires --judge-base-url and --judge-model
#
# Usage:
#   # OpenAI gpt-4o-mini (default)
#   OPENAI_API_KEY=sk-... bash examples/run_musique_judge.sh
#
#   # Local Qwen3-32B judge (assumes ``vllm serve Qwen3-32B-Instruct`` is
#   # already running at http://127.0.0.1:8000)
#   bash examples/run_musique_judge.sh \
#       --backend openai-compat \
#       --judge-base-url http://127.0.0.1:8000/v1 \
#       --judge-model Qwen3-32B-Instruct
#
#   # Override the GPU count for training
#   bash examples/run_musique_judge.sh --nproc 4
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYBIN="/ceph/home/muhan01/.conda/envs/vllm_serve/bin"
CONFIG="meta_past/config/rl_musique_grpo_judge.yaml"
JUDGE_PORT="8124"
JUDGE_BACKEND="openai"
JUDGE_BASE_URL=""
JUDGE_MODEL="gpt-4o-mini"
NPROC="8"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)        JUDGE_BACKEND="$2"; shift 2 ;;
        --judge-base-url) JUDGE_BASE_URL="$2"; shift 2 ;;
        --judge-model)    JUDGE_MODEL="$2"; shift 2 ;;
        --judge-port)     JUDGE_PORT="$2"; shift 2 ;;
        --nproc)          NPROC="$2"; shift 2 ;;
        --config)         CONFIG="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "${JUDGE_BACKEND}" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: backend=openai requires OPENAI_API_KEY in env." >&2
    exit 1
fi
if [[ "${JUDGE_BACKEND}" == "openai-compat" && -z "${JUDGE_BASE_URL}" ]]; then
    echo "ERROR: backend=openai-compat requires --judge-base-url" >&2
    exit 1
fi

mkdir -p runs/musique_grpo_judge
JUDGE_LOG="runs/musique_grpo_judge/judge_server.log"

echo "[run_musique_judge] starting judge server: backend=${JUDGE_BACKEND} model=${JUDGE_MODEL} port=${JUDGE_PORT}"
echo "[run_musique_judge] judge log: ${JUDGE_LOG}"

JUDGE_ARGS=(--port "${JUDGE_PORT}" --backend "${JUDGE_BACKEND}" --model "${JUDGE_MODEL}")
if [[ -n "${JUDGE_BASE_URL}" ]]; then
    JUDGE_ARGS+=(--base-url "${JUDGE_BASE_URL}")
fi

"${PYBIN}/python" scripts/judge_server.py "${JUDGE_ARGS[@]}" \
    > "${JUDGE_LOG}" 2>&1 &
JUDGE_PID=$!
echo "[run_musique_judge] judge server pid=${JUDGE_PID}"

cleanup() {
    if [[ -n "${JUDGE_PID:-}" ]] && kill -0 "${JUDGE_PID}" 2>/dev/null; then
        echo "[run_musique_judge] shutting down judge server (pid=${JUDGE_PID})"
        kill "${JUDGE_PID}" 2>/dev/null || true
        wait "${JUDGE_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait for /healthz (process alive)
echo -n "[run_musique_judge] waiting for judge /healthz"
for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${JUDGE_PORT}/healthz" >/dev/null 2>&1; then
        echo " ... process up."
        break
    fi
    echo -n "."
    sleep 1
done
if ! curl -sf "http://127.0.0.1:${JUDGE_PORT}/healthz" >/dev/null 2>&1; then
    echo
    echo "ERROR: judge server did not come up in 30s. See ${JUDGE_LOG}." >&2
    exit 1
fi

# Sanity-check the backend with a real /evaluate call. /healthz only
# proves the FastAPI process is alive — a missing/wrong OPENAI_API_KEY
# or unreachable openai-compat URL surfaces ONLY here.
echo "[run_musique_judge] sanity-checking judge backend with a real /evaluate call..."
SMOKE_PAYLOAD='{"question": "What is the capital of France?", "reference": "Paris", "pred": "Paris"}'
SMOKE_OUT=$(
    curl -sS --max-time 60 \
        -X POST "http://127.0.0.1:${JUDGE_PORT}/evaluate" \
        -H "Content-Type: application/json" \
        -d "${SMOKE_PAYLOAD}" \
        -w "\n__HTTP_STATUS__%{http_code}" \
    || true
)
SMOKE_BODY="${SMOKE_OUT%__HTTP_STATUS__*}"
SMOKE_STATUS="${SMOKE_OUT##*__HTTP_STATUS__}"

if [[ "${SMOKE_STATUS}" != "200" ]]; then
    echo "ERROR: judge /evaluate returned HTTP ${SMOKE_STATUS}." >&2
    echo "  body: ${SMOKE_BODY}" >&2
    echo "  judge server log: ${JUDGE_LOG}" >&2
    if [[ "${JUDGE_BACKEND}" == "openai" ]]; then
        echo "  hint: confirm OPENAI_API_KEY is valid and has gpt-4o-mini access." >&2
    else
        echo "  hint: confirm ${JUDGE_BASE_URL} is reachable and serves model ${JUDGE_MODEL}." >&2
    fi
    exit 1
fi
# Response should look like {"result": "True"} for the easy probe above.
if ! echo "${SMOKE_BODY}" | grep -qE '"result"\s*:\s*"True"'; then
    echo "WARNING: judge /evaluate returned 200 but didn't say True for an" >&2
    echo "  obviously-correct probe. Backend is responding but may be off." >&2
    echo "  body: ${SMOKE_BODY}" >&2
    echo "  Continuing anyway — training will proceed with this judge."
else
    echo "[run_musique_judge] judge backend OK (probe returned True)."
fi

echo "[run_musique_judge] launching torchrun with nproc_per_node=${NPROC}"
echo "[run_musique_judge] config=${CONFIG}"
echo

exec "${PYBIN}/torchrun" \
    --nproc_per_node="${NPROC}" --standalone \
    scripts/train.py \
    --config "${CONFIG}"
