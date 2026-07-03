#!/bin/bash
# Multi-node TP (Tensor Parallel) training script
# Usage: ./run_training_tp.sh <num_nodes> <node_rank> <master_ip> [tp_size]
#
# Mirrors run_training.sh but launches with parallel.mode=tp.
# TP groups are formed within each node; DP spans across nodes.

set -e  # Exit on error

# Unlock memory limit for RDMA (ibv_reg_mr requires large pinned memory)
if ! ulimit -l unlimited 2>/dev/null; then
    prlimit --pid=$$ --memlock=unlimited:unlimited 2>/dev/null || true
fi

# Default values
NUM_NODES=${1:-1}
NODE_RANK=${2:-0}
MASTER_IP=${3:-localhost}
TP_SIZE=${4:-2}  # TP=2 DP=4 per node measured ~6x PP on Qwen3.6-27B
SP_SIZE=${5:-1}  # Sequence parallel size (default: 1 = disabled)

# --- Network proxy for external access (wandb, etc.) ---
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

# --- Wandb configuration ---
WANDB_SH="$(cd "$(dirname "$0")/../.." && pwd)/wandb.sh"
if [ -f "$WANDB_SH" ]; then
    source "$WANDB_SH"
else
    echo "Warning: $WANDB_SH not found, wandb may not be configured."
fi
export WANDB_NAME=${WANDB_NAME:-""}
export TRAINING_MODE=${TRAINING_MODE:-"pretrain"}
export EXP_NAME=${EXP_NAME:-""}
export ANNEALING_NAME=${ANNEALING_NAME:-""}
export SFT_NAME=${SFT_NAME:-""}
export DATA_CONFIG=${DATA_CONFIG:-""}
export MODEL_CONFIG=${MODEL_CONFIG:-""}
export TRAINING_CONFIG=${TRAINING_CONFIG:-""}
export OPTIMIZER_CONFIG=${OPTIMIZER_CONFIG:-""}
export M2P_TRANSFORMER_CONFIG=${M2P_TRANSFORMER_CONFIG:-""}
export DEBUG_CONFIG=${DEBUG_CONFIG:-""}
export TOKENIZER_CONFIG=${TOKENIZER_CONFIG:-""}
export DETACH_STATE_CONFIG=${DETACH_STATE_CONFIG:-""}
export LAUNCH_CMD=${LAUNCH_CMD:-""}
export FORCE_OVERWRITE=${FORCE_OVERWRITE:-""}
export EVALUATION_BASELINE=${EVALUATION_BASELINE:-""}
# Force online mode
export WANDB_MODE=online

# Prevent Python from writing .pyc bytecode cache files (NFS corruption)
export PYTHONDONTWRITEBYTECODE=1

# --- Compile cache (tp) ---
# Triton/Inductor caches MUST use local /tmp to avoid CephFS "Stale file handle" errors
# (multiple GPU workers on the same node concurrently read/write cache files).
# We clean old caches on startup to prevent /tmp from filling up.
# TileLang cache stays on shared FS (single-writer, benefits from persistence).
CACHE_BASE="$(cd "$(dirname "$0")/.." && pwd)/cache/compile/tp"
NODE_CACHE_DIR="${CACHE_BASE}/node_${NODE_RANK}"
LOCAL_CACHE_DIR="/tmp/shine_compile_cache_tp_node_${NODE_RANK}"

# Clean old local cache to free /tmp space (will be regenerated on first run)
rm -rf "${LOCAL_CACHE_DIR}"
mkdir -p "${LOCAL_CACHE_DIR}/triton" "${LOCAL_CACHE_DIR}/inductor"
mkdir -p "${NODE_CACHE_DIR}/tilelang"

export TRITON_CACHE_DIR="${LOCAL_CACHE_DIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_CACHE_DIR}/inductor"
export TILELANG_CACHE_DIR="${NODE_CACHE_DIR}/tilelang"

# Suppress TVM/TileLang type-registration warnings
export TVM_LOG_LEVEL=ERROR

# Required environment variables
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$NUM_NODES" -gt 1 ]; then
    # === Multi-node NCCL configuration ===
    # Hardware: 8x 200Gb/s HDR RoCE v2 NICs (mlx5_bond_1~8)
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_4,mlx5_bond_3,mlx5_bond_2,mlx5_bond_7,mlx5_bond_6,mlx5_bond_8,mlx5_bond_5
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_IB_TC=136
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TIMEOUT=22
    export NCCL_IB_RETRY_CNT=13
    export NCCL_IB_CUDA_SUPPORT=1

    # --- GPUDirect RDMA ---
    export NCCL_NET_GDR_LEVEL=5
    export NCCL_NET_GDR_READ=1

    # --- Performance tuning ---
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

    # --- Socket interface for OOB ---
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1

    # --- Debug ---
    export NCCL_DEBUG=WARN
    export NCCL_DEBUG_SUBSYS=INIT,NET
else
    # Single-node: use NVLink/shared memory, no network needed
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_LEVEL=NVL
fi

# Basic validation
if [ "$NUM_NODES" -gt 1 ] && [ "$MASTER_IP" = "localhost" ]; then
    echo "Error: For multi-node training, specify a real IP address for master_addr"
    echo "Usage: ./run_training_tp.sh <num_nodes> <node_rank> <master_ip> [tp_size]"
    exit 1
fi

GPUS_PER_NODE=8
DP_PER_NODE=$((GPUS_PER_NODE / TP_SIZE))
TOTAL_DP=$((DP_PER_NODE * NUM_NODES))

echo "=== Starting TP Training ==="
echo "Nodes: $NUM_NODES"
echo "This node rank: $NODE_RANK"
echo "Master IP: $MASTER_IP"
echo "TP size: $TP_SIZE"
echo "DP per node: $DP_PER_NODE"
echo "Total DP: $TOTAL_DP"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Training mode: $TRAINING_MODE"
echo "Experiment name: $EXP_NAME"
echo "Annealing name: ${ANNEALING_NAME:-N/A}"
echo "SFT name: ${SFT_NAME:-N/A}"

# Resolve actual config selections
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIGS_DIR="$SCRIPT_DIR/../configs"
if [ "$TRAINING_MODE" = "sft" ]; then
    _MAIN_YAML="$CONFIGS_DIR/main_sft.yaml"
elif [ "$TRAINING_MODE" = "pretrain_annealing" ]; then
    _MAIN_YAML="$CONFIGS_DIR/main_pretrain_annealing.yaml"
else
    _MAIN_YAML="$CONFIGS_DIR/main_pretrain.yaml"
fi
_get_yaml_default() {
    local key="$1"
    grep -E "^\s*-\s+${key}:" "$_MAIN_YAML" 2>/dev/null | sed -E 's/^\s*-\s+'"${key}"':\s*([^ #]+).*/\1/' | head -1
}
_ACTUAL_MODEL="${MODEL_CONFIG:-$(_get_yaml_default model)}"
_ACTUAL_M2P="${M2P_TRANSFORMER_CONFIG:-$(_get_yaml_default m2p_transformer)}"
_ACTUAL_TRAINING="${TRAINING_CONFIG:-$(_get_yaml_default training)}"
_ACTUAL_OPTIMIZER="${OPTIMIZER_CONFIG:-$(_get_yaml_default optimizer)}"
_ACTUAL_DATA="${DATA_CONFIG:-$(_get_yaml_default data)}"
_ACTUAL_DEBUG="${DEBUG_CONFIG:-$(_get_yaml_default debug)}"
_ACTUAL_TOKENIZER="${TOKENIZER_CONFIG:-$(_get_yaml_default tokenizer)}"
_ACTUAL_DETACH_STATE="${DETACH_STATE_CONFIG:-$(_get_yaml_default detach_state)}"

echo "Config overrides: model=${_ACTUAL_MODEL} m2p=${_ACTUAL_M2P} training=${_ACTUAL_TRAINING} optimizer=${_ACTUAL_OPTIMIZER} data=${_ACTUAL_DATA} debug=${_ACTUAL_DEBUG} tokenizer=${_ACTUAL_TOKENIZER} detach_state=${_ACTUAL_DETACH_STATE}"
echo "Launch command: ${LAUNCH_CMD:-N/A}"

# Show full Hydra stack traces on error
export HYDRA_FULL_ERROR=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_DISABLE_ADDR2LINE=1

# --- Validate mode-prefixed configs ---
validate_mode_prefix() {
    local config_name="$1"
    local config_value="$2"
    local mode="$3"
    if [ -n "$config_value" ]; then
        if [[ "$config_value" != "${mode}/"* ]]; then
            echo "Error: ${config_name} config '${config_value}' must be prefixed with '${mode}/' for mode '${mode}'."
            exit 1
        fi
    fi
}
validate_mode_prefix "training" "$TRAINING_CONFIG" "$TRAINING_MODE"
validate_mode_prefix "optimizer" "$OPTIMIZER_CONFIG" "$TRAINING_MODE"
validate_mode_prefix "data" "$DATA_CONFIG" "$TRAINING_MODE"

# Build Hydra overrides
HYDRA_OVERRIDES="parallel.mode=tp parallel.tensor_parallel_size=$TP_SIZE parallel.sequence_parallel_size=$SP_SIZE parallel.pipeline_parallel_size=1 parallel.total_gpus=$GPUS_PER_NODE"
if [ -n "$DATA_CONFIG" ]; then
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES data=${DATA_CONFIG}"
fi
if [ -n "$MODEL_CONFIG" ]; then
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES model=${MODEL_CONFIG}"
fi
if [ -n "$TRAINING_CONFIG" ]; then
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES training=${TRAINING_CONFIG}"
fi
if [ -n "$OPTIMIZER_CONFIG" ]; then
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES optimizer=${OPTIMIZER_CONFIG}"
fi
if [ -n "$M2P_TRANSFORMER_CONFIG" ]; then
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES m2p_transformer=${M2P_TRANSFORMER_CONFIG}"
fi
if [ -n "$DEBUG_CONFIG" ]; then
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES debug=${DEBUG_CONFIG}"
fi
if [ -n "$TOKENIZER_CONFIG" ]; then
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES tokenizer=${TOKENIZER_CONFIG}"
fi
if [ -n "$DETACH_STATE_CONFIG" ]; then
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES detach_state=${DETACH_STATE_CONFIG}"
fi

# Select Hydra config based on training mode
if [ "$TRAINING_MODE" = "sft" ]; then
    CONFIG_NAME="main_sft"
elif [ "$TRAINING_MODE" = "pretrain_annealing" ]; then
    CONFIG_NAME="main_pretrain_annealing"
else
    CONFIG_NAME="main_pretrain"
fi

# Run training
exec prlimit --memlock=unlimited:unlimited \
    torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_IP \
    --master_port=29503 \
    meta_train.py --config-name=$CONFIG_NAME $HYDRA_OVERRIDES

echo "=== TP Training Completed ==="
