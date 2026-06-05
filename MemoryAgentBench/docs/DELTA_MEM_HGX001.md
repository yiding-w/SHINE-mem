# hgx001：clone δ-mem，在其代码上跑 baseline + SHINE

## 架构

```
git clone declare-lab/delta-Mem
git clone EleutherAI/lm-evaluation-harness   # IFEval 依赖
        ↓
apply_to_delta_mem.sh
        ↓
deltamem.eval.benchmark_compare
  ├── base   … frozen Qwen3-8B
  └── shine  … SHINE（--no-skip-shine）
```

## 一次性安装

```bash
export SHINE_ROOT=/ceph/home/muhan01/wyd/SHINE-mem
cd $SHINE_ROOT/MemoryAgentBench
TORCH_INDEX=cu121 RECREATE_VENV=1 bash bash_files/sh/setup_delta_mem_hgx001.sh
```

默认 **cu121**、**不装 flash-attn**；跑评测时 `ATTN_IMPLEMENTATION=sdpa`（与无 flash-attn 一致）。

### 安装失败排查

| 现象 | 处理 |
|------|------|
| `NVIDIA driver ... too old (12010)` | `TORCH_INDEX=cu121 RECREATE_VENV=1 bash .../setup_delta_mem_hgx001.sh` |
| `No module named 'flash_attn'` | 用最新 setup（已跳过 flash-attn verify） |
| 已有 conda `MABench` | `USE_EXISTING_PYTHON=1 PYTHON_BIN=$(which python) bash .../setup_delta_mem_hgx001.sh` |

## 跑评测 — 与 δ-mem 官方 **完全一致** 的全套

对齐 `run_qasper_multimodel_write8192_benchmark_suite_qwen3_8b.sh`：

| 任务 | 入口 |
|------|------|
| LoCoMo | `deltamem.eval.locomo_delta` |
| HotpotQA | `benchmark_compare --tasks hotpotqa` |
| GPQA Diamond | `benchmark_compare --tasks gpqa_diamond` |
| IFEval | `benchmark_compare --tasks ifeval` |
| MemoryAgentBench | `benchmark_compare --tasks memory_agent_bench`（**4 split、全部 source**） |

协议：`unified prompt`、`max_context_chars=120000`、`T=0.4/top_p=0.9/top_k=10`、HotpotQA/GPQA official decoding、`seed=42`。

```bash
export SHINE_ROOT=/ceph/home/muhan01/wyd/SHINE-mem
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # 官方 8 卡；单卡则 NPROC_PER_NODE=1
export PYTHON_BIN=$SHINE_ROOT/third_party/delta-Mem/.venv/bin/python

# 复现 frozen Qwen3-8B 全套（默认，不是 smoke）
bash bash_files/sh/run_delta_mem_hgx001.sh base

# 完整 MAB：base + SHINE 同一 JSON
bash bash_files/sh/run_delta_mem_hgx001.sh compare-mab

# base 全套 + SHINE 仅 MAB
bash bash_files/sh/run_delta_mem_hgx001.sh all
```

输出目录：`$SHINE_ROOT/outputs/delta_mem_qwen3_8b_full/`

| 文件 | 含义 |
|------|------|
| `base_model/memory_agent_bench.json` | frozen backbone MAB 总分 → `base.memory_agent_bench.summary.overall` |
| `base_model/hotpotqa.json` 等 | 其余基准 |
| `compare_mab/base_and_shine.json` | base 与 SHINE 对比 |

重跑：`FORCE=1 bash ...`；数据已缓存离线：`LOCAL_FILES_ONLY=1 bash ...`

## 环境变量

| 变量 | 默认 |
|------|------|
| `BASE_MODEL` | `/ceph/home/muhan01/huggingfacemodels/Qwen3-8B` |
| `NPROC_PER_NODE` | `8`（单卡自动缩小） |
| `ATTN_IMPLEMENTATION` | `sdpa` |
| `SHINE_AGENT_CONFIG` | `SHINE_agent_qwen3_8b_deltamem.yaml`（`T=0.4` 与官方套件一致） |

## 和 MAB `main.py` 的区别

要比 δ-mem 论文数字，必须用本页 **全套** 流程；`main.py` 对齐的是 MAB 原论文协议。
