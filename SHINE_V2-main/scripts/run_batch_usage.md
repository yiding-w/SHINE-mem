# run_batch.sh 使用指南

基于目录队列的批量训练运行器，支持运行中动态增删任务。

## 快速开始

```bash
# 1. 初始化队列目录
./scripts/run_batch.sh init

# 2. 导入任务（从命令列表文件）
./scripts/run_batch.sh import ./scripts/run_command_list.sh

# 3. 启动批量运行（后台 nohup）
./scripts/run_batch.sh start --nodes all --poll-interval 60

# 4. 查看状态
./scripts/run_batch.sh status
```

---

## 队列结构

```
logs/.batch_queue/
├── pending/    ← 待运行（可随时增删/调序）
├── running/    ← 正在运行（自动管理，勿手动修改）
├── done/       ← 已完成（自动管理）
└── failed/     ← 失败的（自动管理）
```

每个任务是一个 `.job` 文件，文件名格式为 `001_name.job`，内容为要执行的命令（一行）。

---

## 所有命令

| 命令 | 说明 |
|------|------|
| `init` | 初始化队列目录 |
| `start [options]` | 启动批量运行 |
| `stop` | 停止运行，把正在跑的任务放回 pending 队首 |
| `status` | 查看当前运行状态（PID、耗时、队列计数等） |
| `list [filter]` | 查看队列内容，filter 可选：`pending` `running` `done` `failed` `all` |
| `add <cmd> [--priority N]` | 添加任务到 pending 队列 |
| `remove <N或文件名>` | 从 pending 删除任务 |
| `reorder <N或文件名> <新位置>` | 调整 pending 中任务的顺序 |
| `import <file>` | 从命令列表文件批量导入任务 |
| `clear [target]` | 清空队列，target 可选：`pending` `done` `failed` `all` |

---

## start 选项

```bash
./scripts/run_batch.sh start [options]
```

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--nodes <spec>` | `all` | 节点规格（传给 launch_cluster.sh） |
| `--poll-interval <秒>` | `60` | 训练状态轮询间隔 |
| `--cooldown <秒>` | `30` | 任务结束后等待时间 |
| `--gpu-idle-threshold <%>` | `10` | GPU 利用率低于此值视为空闲 |
| `--foreground` | - | 前台运行（不 nohup） |

---

## 运行中动态管理队列

**核心机制**：所有对 pending 队列的操作（add/remove/reorder）都会通过 `flock` 获取锁，此时主循环被阻塞，确保不会出现竞争条件。

### 添加任务

```bash
# 追加到队尾
./scripts/run_batch.sh add './scripts/launch_cluster.sh start --nodes all --mode pretrain --name my_exp'

# 插入到指定位置（如第 1 位）
./scripts/run_batch.sh add './scripts/launch_cluster.sh start --nodes all --name urgent_exp' --priority 1
```

### 删除任务

```bash
# 按编号删除（编号通过 list 查看）
./scripts/run_batch.sh remove 2

# 按文件名删除（支持部分匹配）
./scripts/run_batch.sh remove my_exp
```

### 调整顺序

```bash
# 把第 3 个任务移到第 1 位
./scripts/run_batch.sh reorder 3 1
```

### 查看队列

```bash
# 查看所有队列
./scripts/run_batch.sh list

# 只看待运行
./scripts/run_batch.sh list pending

# 只看已完成
./scripts/run_batch.sh list done
```

---

## 任务类型自动检测

脚本会根据命令内容自动判断任务类型：

| 类型 | 判断条件 | 执行方式 |
|------|----------|----------|
| GPU 训练 | 命令包含 `launch_cluster.sh start` | 异步启动，轮询等待训练结束，cooldown |
| 预处理/其他 | 其他所有命令 | 同步执行，等待退出码 |

---

## 停止与恢复

```bash
# 停止所有（会 kill 进程并把 running 的任务放回 pending 队首）
./scripts/run_batch.sh stop

# 重新启动（会从 pending 队列继续）
./scripts/run_batch.sh start --nodes all
```

---

## 监控日志

```bash
# 查看实时日志
tail -f logs/batch_run_*.log

# 查看状态摘要
./scripts/run_batch.sh status
```

---

## 清理

```bash
# 清空已完成队列
./scripts/run_batch.sh clear done

# 清空失败队列
./scripts/run_batch.sh clear failed

# 清空所有（pending + done + failed）
./scripts/run_batch.sh clear all
```

---

## 典型工作流示例

```bash
# 初始化 + 导入
./scripts/run_batch.sh init
./scripts/run_batch.sh import ./scripts/run_command_list.sh

# 手动再加几个任务
./scripts/run_batch.sh add 'python mydatasets/pretrain/trajectory_all_transfer.py --preprocess --model_path ./models/Qwen3.6-27B/'
./scripts/run_batch.sh add './scripts/launch_cluster.sh start --nodes all --mode pretrain --parallel tp --tp_size 4 --name exp_A --data pretrain/trajectory_all_transfer --detach_state full --training pretrain/savevram --optimizer pretrain/lr1e-4 --m2p_transformer full_prenorm_gatedlastnorm --model Qwen3_6-27B'

# 确认队列
./scripts/run_batch.sh list pending

# 启动
./scripts/run_batch.sh start --nodes all --poll-interval 60 --cooldown 30

# 运行中发现需要加一个紧急任务
./scripts/run_batch.sh add './scripts/launch_cluster.sh start --nodes all --name urgent' --priority 1

# 运行中删除不需要的任务
./scripts/run_batch.sh remove 3

# 查看进度
./scripts/run_batch.sh status
```
