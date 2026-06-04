# MemoryAgentBench × Qwen3-8B × SHINE

在 [MemoryAgentBench](https://arxiv.org/abs/2507.05257) 上评测论文四类能力：**AR / TTL / LRU / SF（Selective Forgetting）**。

> **命名说明**：论文里第四类是 **Selective Forgetting (SF)**，评测集为 **FactConsolidation**（要求遗忘过时事实、采用更新后信息）。官方开源代码与 HuggingFace 把同一 split 标成 **Conflict_Resolution (CR)**，路径仍是 `configs/data_conf/Conflict_Resolution/Factconsolidation_*.yaml`，与 SF 是同一套任务，不是另一项能力。  
> 另：[δ-mem (2605.12357)](https://arxiv.org/abs/2605.12357) 是在 MAB 等基准上评测的**记忆方法**论文，不是 SF 的定义；若要做方法对比需单独实现 δ-mem agent。

## 目录布局（服务器）

克隆本仓库后：

```
SHINE-mem/                 # 仓库根目录（SHINE 训练/推理）
├── test.py, LoraQwen.py, configs/, ...
└── MemoryAgentBench/      # MAB 评测（已接 SHINE / Qwen3-8B local agent）
```

## 1. 环境

```bash
conda create -n MABench python=3.10.16 -y
conda activate MABench

cd MemoryAgentBench   # 在仓库根目录下
pip install torch
pip install -r requirements.txt
pip install "numpy<2" hydra-core omegaconf

# SHINE 依赖（与 README 一致，版本可按集群调整）
pip install transformers==4.57.1 datasets scikit-learn
```

## 2. 模型路径（必改）

编辑以下 YAML 中的路径为你的 ceph 路径：

| 文件 | 字段 |
|------|------|
| `configs/agent_conf/Local_HF_Agents/HF_long_context_agent_qwen3_8b.yaml` | `model_path` → Qwen3-8B |
| `configs/agent_conf/SHINE_Agents/SHINE_agent_qwen3_8b.yaml` | `base_model_path`, `shine_model_root`, `shine_root` |

默认 checkpoint 根目录：

`/ceph/home/muhan01/huggingfacemodels/SHINE-ift_mqa_1qa`

需包含 `metanetwork.pth` 与 `metalora.pth`，常见结构：

- `.../train/checkpoint-epoch-1/`
- 或直接 `.../checkpoint-epoch-1/`

代码会自动选最新的 `checkpoint-*` 目录。

## 3. 跑通 Qwen3-8B 基础测试（长上下文 baseline）

```bash
cd MemoryAgentBench   # 在仓库根目录下
export SHINE_ROOT=/path/to/SHINE-mem
export CUDA_VISIBLE_DEVICES=0

# Smoke：EventQA，yaml 里 max_test_samples=5
python main.py \
  --agent_config configs/agent_conf/Local_HF_Agents/HF_long_context_agent_qwen3_8b.yaml \
  --dataset_config configs/data_conf/Accurate_Retrieval/EventQA/Eventqa_full.yaml
```

结果：`outputs/qwen3-8b-local-longcontext/Accurate_Retrieval/<name_tag>_results.json`

## 4. 跑 SHINE

```bash
python main.py \
  --agent_config configs/agent_conf/SHINE_Agents/SHINE_agent_qwen3_8b.yaml \
  --dataset_config configs/data_conf/Accurate_Retrieval/EventQA/Eventqa_full.yaml
```

SHINE 行为（与论文一致）：

1. **Memorize**：各 chunk 拼成 evidence，一次 `generate_lora_dict`
2. **Query**：仅 user 问题 + 已注入 LoRA，不把长 context 放进 prompt

## 5. 批量 AR / TTL / LRU / SF

`bash_files/configs/shine_mab_eval.txt` 中每行一对 agent + dataset。

```bash
# 仅 smoke（第 4–5 行：HF + SHINE × EventQA）
bash bash_files/sh/run_shine_mab.sh

# AR（Ruler + EventQA，约第 7–10 行）
START_LINE=7 END_LINE=10 bash bash_files/sh/run_shine_mab.sh

# TTL
START_LINE=12 END_LINE=13 bash bash_files/sh/run_shine_mab.sh

# LRU（DetectiveQA）
START_LINE=15 END_LINE=16 bash bash_files/sh/run_shine_mab.sh

# SF (FactConsolidation sh + mh)
START_LINE=18 END_LINE=21 bash bash_files/sh/run_shine_mab.sh

# 全量（去掉注释行后自行设 START/END）
```

重跑加：`FORCE=1 bash bash_files/sh/run_shine_mab.sh`

## 6. 指标说明（MAB 官方）

| 能力 | 代表数据集 | 主指标 |
|------|------------|--------|
| AR | EventQA, Ruler | `substring_exact_match` |
| TTL | ICL_* | `exact_match` |
| LRU | DetectiveQA | `exact_match` |
| SF（代码里写 CR） | FactConsolidation (sh/mh) | `substring_exact_match` |

`InfBench_sum` / `LongMemEval` 需另跑 `llm_based_eval/`，不在此脚本默认列表。

## 7. 长度与生成参数（与 Qwen3-8B 标准测试对齐）

与 `scripts/Qwen3-8B/test.sh` 一致（`ift_mqa_1qa` / msmarco 类评测）：

| 参数 | SHINE 标准 (`test.sh`) | `SHINE_agent_qwen3_8b.yaml` 默认 |
|------|------------------------|----------------------------------|
| evidence 截断 | `test.context_max_length=4500` | `shine_context_max_length: 4500` |
| 问题 prompt | `test.conversation_max_length=300` | `shine_conversation_max_length: 300` |
| 生成长度 | `test.max_new_tokens=128`（`configs/Qwen3-8B.yaml`） | 见下 |

**8192 不够吗？** 对 **MAB 全局长上下文**（如 SF 262k、AR 数十万 token）来说，仅用 8192/4500 都会**截断** memorized context；这是 SHINE 在 **4500 token 上下文上训练** 的固有限制，不是 MAB 配置写错。要与官方 SHINE 数字可比，应使用 **4500**；若强行加大 `shine_context_max_length` 而未按更长 context 重训，收益不确定且易 OOM。

**max_new_tokens**：MAB 各任务答案很短（SF 常为 10，ICL 为 20）。默认 `use_mab_generation_max_length: true`，按各 `dataset_config` 的 `generation_max_length` 生成；若要与 `test.sh` 完全一致，设 `use_mab_generation_max_length: false` 并固定 `max_new_tokens: 128`。

**HF 长上下文 baseline** 仍按 MAB 的 `context_max_length` 截断（可达 131k），与 SHINE 的 4500 evidence 上限是**有意区分**的（in-context vs 参数化记忆）。

`model.lora_r` / `metanetwork` 等见 `configs/Qwen3-8B-MAB.yaml`。

## 8. 故障排查

| 现象 | 处理 |
|------|------|
| 找不到 `metanetwork.pth` | 设置 `shine_checkpoint_dir` 为具体 checkpoint 目录 |
| CUDA OOM | 减小 `shine_context_max_length` 或 `max_test_samples` |
| HF 数据集下载失败 | `export HF_ENDPOINT=https://hf-mirror.com` |
| 恢复中断评测 | 保留 `./agents/.../exp_*` 目录，勿加 `--force` |
