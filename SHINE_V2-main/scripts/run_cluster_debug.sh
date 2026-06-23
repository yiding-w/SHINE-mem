#!/bin/bash
# PP vs TP Forward Comparison — Single Node Launcher
#
# Usage:
#   ./scripts/run_cluster_debug.sh pp       # Run PP mode (8 GPUs)
#   ./scripts/run_cluster_debug.sh tp       # Run TP mode (4 GPUs)
#   ./scripts/run_cluster_debug.sh compare  # Compare results
#   ./scripts/run_cluster_debug.sh all      # Run PP, then TP, then compare (on same node)
#
# The 'all' mode runs everything sequentially on the same node:
#   1. PP mode with 8 GPUs
#   2. TP mode with 4 GPUs (loads PP checkpoint)
#   3. Compare results

set -e

MODE=${1:-all}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORK_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

cd "$WORK_DIR"

# Unlock memory limit
ulimit -l unlimited 2>/dev/null || prlimit --memlock=unlimited:unlimited 2>/dev/null || true

# Environment
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1
export TORCH_SHOW_CPP_STACKTRACES=1
export SHINE_COMPILE_HN=0
export TORCHELASTIC_ERROR_FILE="/tmp/torch_elastic_error.json"
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "=== PP vs TP Forward Comparison ==="
echo "Mode: $MODE"
echo "Working dir: $WORK_DIR"
echo ""

run_pp() {
    echo "--- Running PP mode (8 GPUs) ---"
    torchrun \
        --nproc_per_node=8 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29503 \
        meta_train_debug.py --mode pp
    echo ""
}

run_tp() {
    echo "--- Running TP mode (4 GPUs) ---"
    torchrun \
        --nproc_per_node=4 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29503 \
        meta_train_debug.py --mode tp
    echo ""
}

run_compare() {
    echo "--- Comparing PP vs TP results ---"
    python3 meta_train_debug.py --mode compare
    echo ""
}

case "$MODE" in
    pp)
        run_pp
        ;;
    tp)
        run_tp
        ;;
    compare)
        run_compare
        ;;
    all)
        # Clean up old results
        rm -rf debug_pp_vs_tp 2>/dev/null || true
        mkdir -p debug_pp_vs_tp

        run_pp
        echo "PP complete. Now running TP..."
        echo ""
        run_tp
        echo "TP complete. Now comparing..."
        echo ""
        run_compare
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 {pp|tp|compare|all}"
        exit 1
        ;;
esac