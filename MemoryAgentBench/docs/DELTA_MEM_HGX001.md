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
RECREATE_VENV=1 TORCH_INDEX=cu121 bash bash_files/sh/setup_delta_mem_hgx001.sh
```

**不要**用 δ-mem 完整 `requirements.txt`（会装 transformers 5.x，与 torch 2.5+cu121 不兼容 → `float8_e8m0fnu` 报错）。setup 已改用 `requirements-delta-eval.txt`（`transformers==4.57.1`）。

装到一半 Ctrl-C 后务必 `RECREATE_VENV=1` 重装。跑评测前：

```bash
export PYTHON_BIN=/ceph/home/muhan01/wyd/SHINE-mem/third_party/delta-Mem/.venv/bin/python
$PYTHON_BIN -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
# 期望：2.5.1+cu121  4.57.1
```

固定 **torch 2.5.1+cu121** + **transformers 4.57.1**（勿用 δ-mem 全量 `requirements.txt`，会装 transformers 5.x 导致 `float8_e8m0fnu` 报错）。

跑评测**必须**指定 venv Python：

```bash
export PYTHON_BIN=$SHINE_ROOT/third_party/delta-Mem/.venv/bin/python
```

默认 **不装 flash-attn**；`ATTN_IMPLEMENTATION=sdpa`。

### 安装失败排查

| 现象 | 处理 |
|------|------|
| `NVIDIA driver ... too old (12010)` | `TORCH_INDEX=cu121 RECREATE_VENV=1 bash .../setup_delta_mem_hgx001.sh` |
| `No module named 'flash_attn'` | 用最新 setup（已跳过 flash-attn） |
| `torch has no attribute float8_e8m0fnu` | transformers 5.x 与 cu121 torch 不兼容 → `RECREATE_VENV=1` 重跑 setup |
| 已有 conda `MABench` | `USE_EXISTING_PYTHON=1 PYTHON_BIN=$(which python) bash .../setup_delta_mem_hgx001.sh` |
| `-m: command not found` | `git pull` 后重跑；并 `export PYTHON_BIN=.../.venv/bin/python` |
| `entity2id.json` not found（recsys） | hgx001 常连不上 huggingface.co → **scp 手动拷贝**（见下） |
| `curl: Failed to connect to huggingface.co` | 同上；或 `export HF_ENDPOINT=https://hf-mirror.com` 后重试 download 脚本 |

### recsys 依赖（完整 MAB 必装）

`memory_agent_bench` 含 `recsys_redial_full`，评分需要：

`MemoryAgentBench/processed_data/Recsys_Redial/entity2id.json`（约 1.7MB）

**hgx001 无法访问 HuggingFace 时**（最常见），在本机下载后 scp：

```bash
# 本机（有网）
mkdir -p MemoryAgentBench/processed_data/Recsys_Redial
curl -fsSL -o MemoryAgentBench/processed_data/Recsys_Redial/entity2id.json \
  "https://hf-mirror.com/datasets/ai-hyz/MemoryAgentBench/resolve/main/entity2id.json"

scp MemoryAgentBench/processed_data/Recsys_Redial/entity2id.json \
  muhan01@hgx001:/ceph/home/muhan01/wyd/SHINE-mem/MemoryAgentBench/processed_data/Recsys_Redial/
```

服务器上也可试镜像：`HF_ENDPOINT=https://hf-mirror.com bash bash_files/sh/download_mab_recsys_entity2id.sh`

**说明**：旧版 run 脚本不会在开头下载此文件，评测能启动，到 recsys（约 54%）才崩溃；新版会先尝试下载，连不上 HF 时会 **WARN 并继续**（需 scp 才能在 54% 之后不崩）。

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
export CUDA_VISIBLE_DEVICES=0,1,2,3   # 默认 4 卡；改 8 卡: NUM_GPUS=8
export NUM_GPUS=4
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
| `NUM_GPUS` / `NPROC_PER_NODE` | `4`（作业只分到更少卡时自动缩小） |
| `ATTN_IMPLEMENTATION` | `sdpa` |
| `SHINE_AGENT_CONFIG` | `SHINE_agent_qwen3_8b_deltamem.yaml`（`T=0.4` 与官方套件一致） |

## 和 MAB `main.py` 的区别

要比 δ-mem 论文数字，必须用本页 **全套** 流程；`main.py` 对齐的是 MAB 原论文协议。
