#!/usr/bin/env bash
# Copy to server_env.sh and edit paths for your machine:
#   cp MemoryAgentBench/bash_files/sh/server_env.example.sh MemoryAgentBench/bash_files/sh/server_env.sh
#
# Then: source MemoryAgentBench/bash_files/sh/server_env.sh

_MAB_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SHINE_ROOT="${SHINE_ROOT:-$(cd "${_MAB_SH_DIR}/../../.." && pwd)}"
export DELTA_MEM_ROOT="${DELTA_MEM_ROOT:-${SHINE_ROOT}/third_party/delta-Mem}"
export MAB_ROOT="${MAB_ROOT:-${SHINE_ROOT}/MemoryAgentBench}"

# HuggingFace cache + model paths (edit for your disk layout)
export HF_HOME="${HF_HOME:-${SHINE_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export BASE_MODEL="${BASE_MODEL:-${HF_HOME}/models/Qwen3-8B}"
export SHINE_MODEL_ROOT="${SHINE_MODEL_ROOT:-${HF_HOME}/models/SHINE-ift_mqa_1qa}"

# GPUs: physical device ids on this node (torch sees them as cuda:0, cuda:1, ...)
export CUDA_GPU_IDS="${CUDA_GPU_IDS:-5,6}"
export NUM_GPUS="${NUM_GPUS:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_GPU_IDS}}"

# PyTorch wheel index (cu121 for driver CUDA 12.1; cu124 for 12.4+)
export TORCH_INDEX="${TORCH_INDEX:-cu121}"
export TORCH_VERSION="${TORCH_VERSION:-2.5.1}"

export PYTHON_BIN="${PYTHON_BIN:-${DELTA_MEM_ROOT}/.venv/bin/python}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SHINE_ROOT}/outputs/delta_mem_qwen3_8b_full}"

export SHINE_AGENT_CONFIG="${SHINE_AGENT_CONFIG:-${MAB_ROOT}/configs/agent_conf/SHINE_Agents/SHINE_agent_server.yaml}"

# Eval defaults (δ-mem Qwen3-8B suite protocol)
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
export MAB_EVAL_BATCH_SIZE="${MAB_EVAL_BATCH_SIZE:-16}"
export MAB_MAX_CONTEXT_CHARS="${MAB_MAX_CONTEXT_CHARS:-120000}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export SEED="${SEED:-42}"
