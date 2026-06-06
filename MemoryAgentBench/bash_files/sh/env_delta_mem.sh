#!/usr/bin/env bash
# Source before manual python commands:  source bash_files/sh/env_delta_mem.sh
SHINE_ROOT="${SHINE_ROOT:-/ceph/home/muhan01/wyd/SHINE-mem}"
export SHINE_ROOT
export DELTA_MEM_ROOT="${DELTA_MEM_ROOT:-${SHINE_ROOT}/third_party/delta-Mem}"
export MAB_ROOT="${MAB_ROOT:-${SHINE_ROOT}/MemoryAgentBench}"
export PYTHON_BIN="${PYTHON_BIN:-${DELTA_MEM_ROOT}/.venv/bin/python}"
export PYTHONPATH="${DELTA_MEM_ROOT}:${SHINE_ROOT}:${MAB_ROOT}"
export HF_HOME="${HF_HOME:-/ceph/home/muhan01/huggingfacemodels}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export NUM_GPUS="${NUM_GPUS:-4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS}}"
