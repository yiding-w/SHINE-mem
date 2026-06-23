export SHINE_ROOT="/home/wangyiding/SHINE-mem"
export HF_HOME="/data/yidingw"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_ENDPOINT="https://hf-mirror.com"

export BASE_MODEL="${HF_HOME}/models/Qwen3-8B"
export SHINE_MODEL_ROOT="${HF_HOME}/models/SHINE-ift_mqa_1qa"

export CUDA_GPU_IDS="5,6"
export NUM_GPUS=2
export NPROC_PER_NODE=2
export CUDA_VISIBLE_DEVICES="${CUDA_GPU_IDS}"

export TORCH_INDEX="cu121"
export TORCH_VERSION="2.5.1"
export DELTA_MEM_ROOT="${SHINE_ROOT}/third_party/delta-Mem"
export MAB_ROOT="${SHINE_ROOT}/MemoryAgentBench"
export OUTPUT_ROOT="${SHINE_ROOT}/outputs/delta_mem_qwen3_8b_full"
export ATTN_IMPLEMENTATION="sdpa"
export SEED="42"
