#!/bin/bash
# Simple torchrun training script
# Usage: ./run_training.sh <num_nodes> <node_rank> <master_ip> [pipeline_stages]

set -e  # Exit on error

# Unlock memory limit for RDMA (ibv_reg_mr requires large pinned memory)
# In Kubernetes containers, bash ulimit cannot exceed hard limit (64KB),
# but prlimit with CAP_SYS_RESOURCE can override it via setrlimit syscall.
if ! ulimit -l unlimited 2>/dev/null; then
    # Fallback: use prlimit to raise memlock for this process
    prlimit --pid=$$ --memlock=unlimited:unlimited 2>/dev/null || true
fi

# Default values
NUM_NODES=${1:-1}
NODE_RANK=${2:-0}
MASTER_IP=${3:-localhost}
PIPELINE_STAGES=${4:-8}

# --- Network proxy for external access (wandb, etc.) ---
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

# --- Wandb configuration ---
# Source wandb credentials (WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY)
WANDB_SH="$(cd "$(dirname "$0")/../.." && pwd)/wandb.sh"
if [ -f "$WANDB_SH" ]; then
    source "$WANDB_SH"
else
    echo "Warning: $WANDB_SH not found, wandb may not be configured."
fi
# WANDB_NAME is passed via environment variable from launch_cluster.sh
export WANDB_NAME=${WANDB_NAME:-""}
# Training mode and experiment identifiers from launch_cluster.sh
export TRAINING_MODE=${TRAINING_MODE:-"pretrain"}
export EXP_NAME=${EXP_NAME:-""}
export ANNEALING_NAME=${ANNEALING_NAME:-""}
export SFT_NAME=${SFT_NAME:-""}
# Config overrides from launch_cluster.sh
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

# Prevent Python from writing .pyc bytecode cache files.
# On shared NFS filesystems, concurrent .pyc writes from multiple nodes
# can cause "marshal data too short" corruption errors.
export PYTHONDONTWRITEBYTECODE=1

# --- Persistent compile cache ---
# Triton/TileLang/Inductor compile caches are stored persistently so that
# kernel recompilation is avoided across restarts.
# --- Compile cache (pp) ---
# Triton/Inductor caches use local /tmp to avoid CephFS "Stale file handle" errors
# (multiple GPU workers on the same node concurrently read/write cache files).
# TileLang cache stays on shared FS (rarely causes issues and benefits from persistence).
CACHE_BASE="$(cd "$(dirname "$0")/.." && pwd)/cache/compile/pp"
NODE_CACHE_DIR="${CACHE_BASE}/node_${NODE_RANK}"
LOCAL_CACHE_DIR="/tmp/shine_compile_cache_pp_node_${NODE_RANK}"

mkdir -p "${LOCAL_CACHE_DIR}/triton" "${LOCAL_CACHE_DIR}/inductor"
mkdir -p "${NODE_CACHE_DIR}/tilelang"

export TRITON_CACHE_DIR="${LOCAL_CACHE_DIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_CACHE_DIR}/inductor"
export TILELANG_CACHE_DIR="${NODE_CACHE_DIR}/tilelang"

# Suppress TVM/TileLang type-registration warnings (harmless, very verbose)
export TVM_LOG_LEVEL=ERROR

# Required environment variables for NCCL
export OMP_NUM_THREADS=1
# Reduce CUDA memory fragmentation by using expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$NUM_NODES" -gt 1 ]; then
    # === Multi-node NCCL configuration ===
    # Hardware: 8x 200Gb/s HDR RoCE v2 NICs (mlx5_bond_1~8)
    # GPU-NIC PIX affinity: GPU0-bond_1, GPU1-bond_4, GPU2-bond_3,
    #   GPU3-bond_2, GPU4-bond_7, GPU5-bond_6, GPU6-bond_8, GPU7-bond_5

    # --- IB/RoCE Transport ---
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
    echo "Usage: ./run_training.sh <num_nodes> <node_rank> <master_ip> [pipeline_stages]"
    exit 1
fi

echo "=== Starting Training ==="
echo "Nodes: $NUM_NODES"
echo "This node rank: $NODE_RANK"
echo "Master IP: $MASTER_IP"
echo "Pipeline stages: $PIPELINE_STAGES"
echo "GPUs per node: 8"
echo "Training mode: $TRAINING_MODE"
echo "Experiment name: $EXP_NAME"
echo "Annealing name: ${ANNEALING_NAME:-N/A}"
echo "SFT name: ${SFT_NAME:-N/A}"

# Resolve actual config selections (override or default from main yaml)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIGS_DIR="$SCRIPT_DIR/../configs"
if [ "$TRAINING_MODE" = "sft" ]; then
    _MAIN_YAML="$CONFIGS_DIR/main_sft.yaml"
elif [ "$TRAINING_MODE" = "pretrain_annealing" ]; then
    _MAIN_YAML="$CONFIGS_DIR/main_pretrain_annealing.yaml"
else
    _MAIN_YAML="$CONFIGS_DIR/main_pretrain.yaml"
fi
# Extract defaults from main yaml (format: "  - <key>: <value>" or "  - <key>")
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
# Show C++ stack traces for PyTorch errors
export TORCH_SHOW_CPP_STACKTRACES=1
# Disable addr2line symbolization for torch.compile internal exceptions
# (Dynamo uses exceptions for control flow; symbolizing them is very slow and noisy)
export TORCH_DISABLE_ADDR2LINE=1

# --- Validate mode-prefixed configs ---
# training, optimizer, data configs must start with "${TRAINING_MODE}/" if specified
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

# Build Hydra overrides from config selections
HYDRA_OVERRIDES="parallel.pipeline_parallel_size=$PIPELINE_STAGES"
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
# Use prlimit to ensure child processes inherit unlimited memlock
# (required for ibv_reg_mr in RDMA/RoCE with GPUDirect)
exec prlimit --memlock=unlimited:unlimited \
    torchrun \
    --nproc_per_node=8 \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_IP \
    --master_port=29502 \
    meta_train.py --config-name=$CONFIG_NAME $HYDRA_OVERRIDES

echo "=== Training Completed ==="