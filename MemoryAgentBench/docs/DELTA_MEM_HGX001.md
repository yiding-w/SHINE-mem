# hgx001：clone δ-mem，在其代码上跑 baseline + SHINE

## 架构（按你的要求）

```
git clone declare-lab/delta-Mem     # 官方仓库
        ↓
apply_to_delta_mem.sh             # 打上 SHINE 补丁（在 δ-mem 代码里改）
        ↓
deltamem.eval.benchmark_compare   # 同一入口、同一 JSON
  ├── base   … frozen Qwen3-8B
  ├── delta  … δ-mem adapter（可选）
  └── shine  … SHINE agent（我们加的第四节）
```

**不再**用独立的 `shine_mab_eval.py` 做主路径；它已改为转发到 `benchmark_compare`。

## 补丁内容（在 SHINE-mem 里版本管理）

| 文件 | 作用 |
|------|------|
| `deltamem_patches/deltamem/eval/shine_memory_agent_bench.py` | 新模块：SHINE 跑 MAB |
| `deltamem_patches/benchmark_compare_shine.patch` | 给 `benchmark_compare.py` 加 `--no-skip-shine` |
| `deltamem_patches/apply_to_delta_mem.sh` | clone 后自动覆盖 |

## 一次性安装（hgx001）

```bash
export SHINE_ROOT=/ceph/home/muhan01/wyd/SHINE-mem
cd $SHINE_ROOT/MemoryAgentBench
bash bash_files/sh/setup_delta_mem_hgx001.sh
```

会：`git clone` → `apply_to_delta_mem.sh` → 建 δ-mem `.venv` → 装 SHINE 依赖。

## 跑评测

```bash
export SHINE_ROOT=/ceph/home/muhan01/wyd/SHINE-mem
export CUDA_VISIBLE_DEVICES=0

# δ-mem 官方 frozen baseline（EventQA smoke）
bash bash_files/sh/run_delta_mem_hgx001.sh smoke

# 只跑 SHINE（同一协议）
bash bash_files/sh/run_delta_mem_hgx001.sh shine

# 完整 MAB：baseline + SHINE 进同一个 JSON
bash bash_files/sh/run_delta_mem_hgx001.sh all
```

或直接调用 δ-mem（补丁已打进 clone）：

```bash
cd $SHINE_ROOT/third_party/delta-Mem
export PYTHONPATH=$PWD:$SHINE_ROOT:$SHINE_ROOT/MemoryAgentBench

# baseline only
.venv/bin/python -m deltamem.eval.benchmark_compare \
  --model-path /ceph/home/muhan01/huggingfacemodels/Qwen3-8B \
  --tasks memory_agent_bench \
  --external-memory-agent-bench-root $SHINE_ROOT/MemoryAgentBench \
  --skip-delta --skip-lora --skip-shine \
  --no-memory-agent-bench-use-official-prompt \
  --memory-agent-bench-max-context-chars 120000 \
  --eval-do-sample --eval-temperature 0.4 \
  --output-json $SHINE_ROOT/outputs/base_mab.json

# SHINE only
.venv/bin/python -m deltamem.eval.benchmark_compare \
  --model-path /ceph/home/muhan01/huggingfacemodels/Qwen3-8B \
  --tasks memory_agent_bench \
  --external-memory-agent-bench-root $SHINE_ROOT/MemoryAgentBench \
  --shine-root $SHINE_ROOT \
  --shine-agent-config $SHINE_ROOT/MemoryAgentBench/configs/agent_conf/SHINE_Agents/SHINE_agent_qwen3_8b.yaml \
  --skip-base --skip-delta --skip-lora --no-skip-shine \
  --no-memory-agent-bench-use-official-prompt \
  --memory-agent-bench-max-context-chars 120000 \
  --eval-do-sample --eval-temperature 0.4 \
  --output-json $SHINE_ROOT/outputs/shine_mab.json
```

## 看结果

输出 JSON 结构：

- `base.memory_agent_bench.summary.overall` — δ-mem frozen Qwen3-8B
- `shine.memory_agent_bench.summary.overall` — SHINE

与 δ-mem 论文 Qwen3-8B 套件一致：`unified prompt`、`max_context_chars=120000`、sampling `T=0.4`。

## 和 MAB `main.py` 的关系

| | δ-mem `benchmark_compare` | MAB `main.py` |
|--|---------------------------|---------------|
| 用途 | 对齐 δ-mem 论文 | 对齐 MAB 原论文 / SHINE 日常开发 |
| SHINE 入口 | `--no-skip-shine` | `SHINE_agent_qwen3_8b.yaml` |

两套协议并存；和 δ-mem 表比数字请用本页流程。
