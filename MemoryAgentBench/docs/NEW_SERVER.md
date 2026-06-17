# New GPU server — δ-mem + SHINE eval (generic; not hgx001-specific)

## 脚本一览

| 脚本 | 作用 |
|------|------|
| `server_env.example.sh` | 环境模板（**复制为 `server_env.sh` 并修改**） |
| `source_server_env.sh` | 加载 `server_env.sh` |
| `bootstrap_new_server.sh` | 一键：setup + 下载模型/数据 |
| `setup_delta_mem.sh` | 安装 venv、clone δ-mem、打 SHINE patch |
| `download_eval_assets.sh` | 下载 Qwen3-8B、SHINE ckpt、MAB/基准数据集 |
| `run_delta_mem.sh` | 跑评测（`base-mab` / `compare-mab` / `base` 等） |
| `nohup_run_delta_mem.sh` | 后台 nohup 跑 |
| `download_mab_recsys_entity2id.sh` | recsys 用的 entity2id.json |

hgx001 旧脚本仍可用：`setup_delta_mem_hgx001.sh` / `run_delta_mem_hgx001.sh`。

---

## 快速开始（新服务器，GPU 5、6）

```bash
# 1. 克隆
git clone https://github.com/yiding-w/SHINE-mem.git
cd SHINE-mem

# 2. 配置环境（必做）
cp MemoryAgentBench/bash_files/sh/server_env.example.sh \
   MemoryAgentBench/bash_files/sh/server_env.sh

# 编辑 server_env.sh，至少确认：
#   SHINE_ROOT=.../SHINE-mem
#   HF_HOME=.../大磁盘路径/.cache/huggingface
#   CUDA_GPU_IDS=5,6
#   NUM_GPUS=2
#   HF_ENDPOINT=https://hf-mirror.com   # 国内/内网建议镜像

vim MemoryAgentBench/bash_files/sh/server_env.sh

# 3. 一键安装 + 下载（耗时：模型 ~16GB + 数据集）
bash MemoryAgentBench/bash_files/sh/bootstrap_new_server.sh

# 4. 只跑 frozen Qwen3-8B × 完整 MAB（推荐先跑这个）
source MemoryAgentBench/bash_files/sh/source_server_env.sh
bash MemoryAgentBench/bash_files/sh/run_delta_mem.sh base-mab
```

### 后台跑（断 SSH 不断任务）

```bash
bash MemoryAgentBench/bash_files/sh/nohup_run_delta_mem.sh base-mab
tail -f outputs/delta_mem_qwen3_8b_full/logs/nohup_base-mab.log
# 或
tail -f outputs/delta_mem_qwen3_8b_full/logs/base_model_memory_agent_bench.log
```

---

## `server_env.sh` 关键变量

```bash
export CUDA_GPU_IDS="5,6"      # 物理卡号；进程内映射为 cuda:0, cuda:1
export NUM_GPUS=2
export CUDA_VISIBLE_DEVICES="${CUDA_GPU_IDS}"

export HF_HOME="/data/yourname/huggingface"
export BASE_MODEL="${HF_HOME}/models/Qwen3-8B"
export SHINE_MODEL_ROOT="${HF_HOME}/models/SHINE-ift_mqa_1qa"

export TORCH_INDEX=cu121       # 驱动 12.1 用 cu121；12.4+ 可试 cu124
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 输出位置

| 模式 | 结果 JSON | 日志 |
|------|-----------|------|
| `base-mab` | `outputs/delta_mem_qwen3_8b_full/base_model/memory_agent_bench.json` | `logs/base_model_memory_agent_bench.log` |
| `compare-mab` | `outputs/.../compare_mab/base_and_shine.json` | `logs/compare_mab.log` |
| `base` 全套 | `outputs/.../base_model/{locomo,hotpotqa,...}.json` | `logs/base_model_*.log` |

总分路径：`base.memory_agent_bench.summary.overall`

---

## 无法联网时

1. 在能上网的机器下载后 **rsync/scp** 到 `${HF_HOME}`：
   - `Qwen3-8B/` → `${BASE_MODEL}`
   - `SHINE-ift_mqa_1qa/` → `${SHINE_MODEL_ROOT}`
   - `entity2id.json` → `MemoryAgentBench/processed_data/Recsys_Redial/`

2. 服务器上：
```bash
export LOCAL_FILES_ONLY=1
bash MemoryAgentBench/bash_files/sh/run_delta_mem.sh base-mab
```

---

## 评测模式

```bash
bash .../run_delta_mem.sh base-mab      # 仅 frozen 8B × 完整 MAB
bash .../run_delta_mem.sh compare-mab   # 8B + SHINE 对比
bash .../run_delta_mem.sh base          # δ-mem 官方五基准全套
bash .../run_delta_mem.sh shine-mab     # 仅 SHINE × MAB
```

重跑：`FORCE=1 bash .../run_delta_mem.sh base-mab`
