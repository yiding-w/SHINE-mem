# MemoryAgentBench × Qwen3-8B × SHINE

在 [MemoryAgentBench](https://arxiv.org/abs/2507.05257) 上评测论文四类能力：**AR / TTL / LRU / SF（Selective Forgetting）**。

> **命名说明**：论文里第四类是 **Selective Forgetting (SF)**，评测集为 **FactConsolidation**（要求遗忘过时事实、采用更新后信息）。官方开源代码与 HuggingFace 把同一 split 标成 **Conflict_Resolution (CR)**，路径仍是 `configs/data_conf/Conflict_Resolution/Factconsolidation_*.yaml`，与 SF 是同一套任务，不是另一项能力。  
> 另：[δ-mem (2605.12357)](https://arxiv.org/abs/2605.12357) 是在 MAB 等基准上评测的**记忆方法**论文，不是 SF 的定义。  
> **与 δ-mem 论文数字对比**：请直接 clone 他们的代码跑 frozen baseline，见 [DELTA_MEM_HGX001.md](DELTA_MEM_HGX001.md)（`setup_delta_mem_hgx001.sh` + `run_delta_mem_hgx001.sh`）。本文档的 `main.py` 路径对齐的是 **MAB 官方**协议，不是 δ-mem 套件。

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

# 必须在仓库内的 MemoryAgentBench 目录（clone 后路径示例）：
cd /path/to/SHINE-mem/MemoryAgentBench
ls requirements.txt   # 若不存在，先 git pull 最新 yiding-w/SHINE-mem

# 若 pip 报 usercustomize / antlr4（集群 Python 启动脚本问题），先执行：
pip install antlr4-python3-runtime
# 或临时绕过： export PYTHONNOUSERSITE=1

# 仅跑 SHINE / Qwen 长上下文 baseline（推荐）：
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-shine-mab.txt
pip install hydra-core omegaconf scikit-learn transformers==4.57.1

# SHINE_ROOT 必须是仓库根目录（含 LoraQwen.py），不是 MemoryAgentBench 子目录：
export SHINE_ROOT=/ceph/home/muhan01/wyd/SHINE-mem

# 跑完整 MAB（含 RAG、mem0 等，很重）：
# pip install -r requirements.txt
# pip install "numpy<2"
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

SHINE 行为（δ-mem 对齐，`query_include_context: true`，见 `SHINE_agent_qwen3_8b_deltamem.yaml`）：

1. **Memorize**：各 chunk 拼成 evidence，一次 `generate_lora_dict`（SHINE 记忆）
2. **Query**：与 δ-mem **base 相同 prompt**（unified：`MEMORY_CONTEXT_QA_PROMPT_TEMPLATE` 含 context+question），并注入上一步 LoRA

若 `query_include_context: false`，Query 仅问题（旧 SHINE 论文设定，与 δ-mem base 不可直接比）。

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

## 7. 长度与生成参数

### 默认：MAB 全长（当前 `SHINE_agent_qwen3_8b.yaml`）

| 开关 | 行为 |
|------|------|
| `use_mab_context_max_length: true` | evidence 截断 = **该任务** `dataset_config.context_max_length`（如 SF 262k→300000，ICL→131072） |
| `use_mab_conversation_max_length: true` | 问题侧上限 = `shine_conversation_max_length`（默认 **4096**，容纳长 prompt） |
| `use_mab_generation_max_length: true` | 生成 = 该任务 `generation_max_length`（SF=10，Detective=2000，…） |

启动时会打印：`[ShineMABRunner] sub_dataset=... evidence_max_len=...`

**注意**：全长会显著增加显存与时间；hypernetwork 在约 **4500 token** context 上训练，加长是**试验性**外推，OOM 时可改回下一节的短配置。

### 可选：与 `scripts/Qwen3-8B/test.sh` 对齐（4500 / 300 / 128）

```yaml
use_mab_context_max_length: false
use_mab_conversation_max_length: false
shine_context_max_length: 4500
shine_conversation_max_length: 300
use_mab_generation_max_length: false
max_new_tokens: 128
```

`model.lora_r` 等见 `configs/Qwen3-8B-MAB.yaml`。

## 8. 故障排查

### `NVIDIA driver on your system is too old (found version 12010)`

在 `HFLocalLongContextRunner` / `ShineMABRunner` 执行 `.to(cuda)` 加载 Qwen3-8B 时触发：当前 **PyTorch 是按较新的 CUDA（如 cu124）编译的**，而节点驱动只支持更旧的 CUDA（报错里的 `12010` 一般对应 **CUDA 12.1** 级别）。

在计算节点上检查：

```bash
nvidia-smi          # 右上角 CUDA Version / Driver Version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

**处理办法（二选一）：**

1. **换与驱动匹配的 PyTorch**（常见、无需 root）：

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

若 `nvidia-smi` 显示 CUDA 11.8，则用 `cu118`。

2. **升级 GPU 驱动**（需管理员 / 换更新驱动的节点）。

装好后重新跑 `python main.py ...`。集群上也可加载已用旧 CUDA 编译好的 conda 环境，避免混用 `cu124` 的 pip 包。

| 现象 | 处理 |
|------|------|
| 找不到 `metanetwork.pth` | 设置 `shine_checkpoint_dir` 为具体 checkpoint 目录 |
| CUDA OOM | 减小 `shine_context_max_length` 或 `max_test_samples` |
| HF 数据集下载失败 | `export HF_ENDPOINT=https://hf-mirror.com` |
| 恢复中断评测 | 保留 `./agents/.../exp_*` 目录，勿加 `--force` |
