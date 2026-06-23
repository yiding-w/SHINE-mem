#!/bin/bash
# Multi-node debug test script
# Usage: ./run_debug_test.sh <num_nodes> <node_rank> <master_ip> [tp_size] [test_name]
#
# Launches meta_train_debug.py with torchrun for multi-node distributed testing.
# Mirrors run_training_tp.sh environment setup.
#
# Examples:
#   # Single node, all tests:
#   ./scripts/run_debug_test.sh 1 0 localhost 2 all
#
#   # 4 nodes, checkpoint test only:
#   ./scripts/run_debug_test.sh 4 0 28.49.32.254 2 checkpoint
#
#   # 4 nodes, consistency test:
#   ./scripts/run_debug_test.sh 4 0 28.49.32.254 2 consistency

set -e

NUM_NODES=${1:-1}
NODE_RANK=${2:-0}
MASTER_IP=${3:-localhost}
TP_SIZE=${4:-2}
TEST_NAME=${5:-all}
PARALLEL_MODE=${6:-tp}
SKIP_TESTS=${7:-""}
SKIP_SUB=${8:-""}

# Unlock memory limit for RDMA
if ! ulimit -l unlimited 2>/dev/null; then
    prlimit --memlock=unlimited:unlimited 2>/dev/null || true
fi

# Prevent Python from writing .pyc bytecode cache files
export PYTHONDONTWRITEBYTECODE=1

# Required environment variables
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$NUM_NODES" -gt 1 ]; then
    # === Multi-node NCCL configuration ===
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_4,mlx5_bond_3,mlx5_bond_2,mlx5_bond_7,mlx5_bond_6,mlx5_bond_8,mlx5_bond_5
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_IB_TC=136
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TIMEOUT=22
    export NCCL_IB_RETRY_CNT=13
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_NET_GDR_LEVEL=5
    export NCCL_NET_GDR_READ=1
    export NCCL_BUFFSIZE=8388608
    export NCCL_NTHREADS=512
    export NCCL_MAX_NCHANNELS=32
    export NCCL_MIN_NCHANNELS=32
    export NCCL_P2P_DISABLE=0
    export NCCL_SHM_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_CROSS_NIC=1
    export NCCL_PXN_DISABLE=0
    export NCCL_CHECK_DISABLE=1
    export NCCL_IB_MERGE_VFS=1
    export NCCL_NCHANNELS_PER_NET_PEER=8
    export NCCL_P2P_NET_CHUNKSIZE=524288
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_DEBUG=WARN
    export NCCL_DEBUG_SUBSYS=INIT,NET
else
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_LEVEL=NVL
fi

# Basic validation
if [ "$NUM_NODES" -gt 1 ] && [ "$MASTER_IP" = "localhost" ]; then
    echo "Error: For multi-node testing, specify a real IP address for master_addr"
    exit 1
fi

GPUS_PER_NODE=8

echo "=== Starting Debug Test ==="
echo "Nodes: $NUM_NODES"
echo "This node rank: $NODE_RANK"
echo "Master IP: $MASTER_IP"
echo "TP size: $TP_SIZE"
echo "Test: $TEST_NAME"
echo "Parallel mode: $PARALLEL_MODE"
echo "GPUs per node: $GPUS_PER_NODE"

export HYDRA_FULL_ERROR=1
export TORCH_SHOW_CPP_STACKTRACES=1

# Build skip argument
SKIP_ARG=""
if [ -n "$SKIP_TESTS" ]; then
    SKIP_ARG="--skip $SKIP_TESTS"
fi
SKIP_SUB_ARG=""
if [ -n "$SKIP_SUB" ]; then
    SKIP_SUB_ARG="--skip_sub $SKIP_SUB"
fi

# Run test
exec prlimit --memlock=unlimited:unlimited \
    torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_IP \
    --master_port=29502 \
    meta_train_debug.py --test $TEST_NAME --parallel $PARALLEL_MODE --tp_size $TP_SIZE --cleanup $SKIP_ARG $SKIP_SUB_ARG

echo "=== Debug Test Completed ==="
