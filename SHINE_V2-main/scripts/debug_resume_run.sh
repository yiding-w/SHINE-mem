#!/bin/bash
# =============================================================================
# Debug Resume Verification Script
#
# Verifies resume correctness by comparing:
#   Step 1: Run A — train 20 steps continuously (saves checkpoint at step 10)
#   Step 2: Run B — resume from Run A's step 10 checkpoint, train to step 20
#   Step 3: Compare step 11-20 outputs between Run A and Run B
#
# Usage:
#   bash scripts/debug_resume_run.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

export DEBUG_RESUME=1

echo "=============================================="
echo "  Debug Resume Verification"
echo "=============================================="
echo ""

# Common launch args
COMMON_ARGS="--nodes all --mode pretrain --parallel tp --tp_size 4 --sp_size 2 \
  --data pretrain/trajectory_all_transfer_v2 \
  --detach_state full_reset_threshold_1000.0 \
  --optimizer pretrain/lr1e-4 \
  --m2p_transformer full_prenorm_gatedlastnorm \
  --model Qwen3_6-27B"

# Function: wait for training to finish
wait_for_training() {
    local name="$1"
    local timeout_minutes=60
    local poll_interval=15
    local elapsed=0
    local max_seconds=$((timeout_minutes * 60))

    echo "  Waiting for '$name' to finish (timeout ${timeout_minutes}m)..."

    # Phase 1: Wait for processes to appear (max 120s)
    local appear_timeout=120
    local appear_elapsed=0
    while true; do
        sleep $poll_interval
        appear_elapsed=$((appear_elapsed + poll_interval))
        elapsed=$((elapsed + poll_interval))
        STATUS_OUTPUT=$(./scripts/launch_cluster.sh status --nodes all 2>&1 || true)
        if echo "$STATUS_OUTPUT" | grep -q "RUNNING"; then
            echo "  Processes detected as RUNNING. Waiting for completion..."
            break
        fi
        if [ $appear_elapsed -ge $appear_timeout ]; then
            echo "  WARNING: Processes never appeared as RUNNING after ${appear_timeout}s."
            echo "  Assuming training failed to start or finished instantly."
            return 0
        fi
    done

    # Phase 2: Wait for processes to finish
    while true; do
        sleep $poll_interval
        elapsed=$((elapsed + poll_interval))
        STATUS_OUTPUT=$(./scripts/launch_cluster.sh status --nodes all 2>&1 || true)
        if echo "$STATUS_OUTPUT" | grep -q "RUNNING"; then
            if [ $elapsed -ge $max_seconds ]; then
                echo "  ERROR: Timeout waiting for '$name' (${timeout_minutes}m elapsed)"
                echo "  Stopping stale processes..."
                ./scripts/launch_cluster.sh stop --nodes all 2>/dev/null || true
                exit 1
            fi
            if [ $((elapsed % 300)) -eq 0 ]; then
                echo "  Still running... (${elapsed}s elapsed)"
            fi
        else
            echo "  '$name' finished. (took ~${elapsed}s)"
            break
        fi
    done
}

# =============================================================================
# Step 1: Run A — continuous 20 steps (saves checkpoint at step 10)
# =============================================================================
echo "[Step 1/3] Run A: Training 20 steps continuously..."

rm -rf checkpoint/debug_resume_runA 2>/dev/null || true
rm -rf logs/debug_resume_runA 2>/dev/null || true

./scripts/launch_cluster.sh start $COMMON_ARGS \
  --name debug_resume_runA \
  --training pretrain/debug_resume

wait_for_training "Run A (20 steps)"
echo ""

# Verify step 10 checkpoint exists
CKPT_DIR="checkpoint/debug_resume_runA/pretrain"
if [ ! -d "$CKPT_DIR/step_10" ]; then
    echo "ERROR: Run A did not produce a step_10 checkpoint!"
    exit 1
fi
echo "  Run A step_10 checkpoint confirmed."

# =============================================================================
# Step 2: Run B — resume from Run A's step 10 checkpoint, train to step 20
# =============================================================================
echo "[Step 2/3] Run B: Resuming from Run A's step 10 checkpoint..."

rm -rf checkpoint/debug_resume_runB 2>/dev/null || true
rm -rf logs/debug_resume_runB 2>/dev/null || true

# Copy Run A's step 10 checkpoint to Run B's checkpoint directory
mkdir -p checkpoint/debug_resume_runB/pretrain
cp -r "$CKPT_DIR/step_10" checkpoint/debug_resume_runB/pretrain/step_10

echo "  Copied step_10 checkpoint to debug_resume_runB."

# Launch Run B — resume_from: latest will find step_10 and resume from it
./scripts/launch_cluster.sh start $COMMON_ARGS \
  --name debug_resume_runB \
  --training pretrain/debug_resume

wait_for_training "Run B (resume 10→20)"
echo ""

# =============================================================================
# Step 3: Compare results
# =============================================================================
echo "[Step 3/3] Comparing Run A vs Run B (steps 11-20)..."
echo ""

python3 scripts/debug_resume_compare.py \
  logs/debug_resume_runA/pretrain/node_0_debug_steps.jsonl \
  logs/debug_resume_runB/pretrain/node_0_debug_steps.jsonl \
  --start_step 11

echo ""
echo "Done."
