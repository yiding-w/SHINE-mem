# Launch pretrain training (use all 4 nodes)
./scripts/launch_cluster.sh start --nodes all --mode pretrain --name "<experiment_name>"

# Launch pretrain on specific nodes (e.g. nodes 0-1)
./scripts/launch_cluster.sh start --nodes 0-1 --mode pretrain --name "<experiment_name>"

# Launch pretrain on a single node (e.g. node 2)
./scripts/launch_cluster.sh start --nodes 2 --mode pretrain --name "<experiment_name>"

# Launch pretrain with custom model and data
./scripts/launch_cluster.sh start --nodes all --mode pretrain --name "<experiment_name>" \
    --model Qwen3_6-35B-A3B --data pretrain/oldpretrainnochange

# Launch pretrain annealing (loads pretrain final checkpoint automatically)
./scripts/launch_cluster.sh start --nodes all --mode pretrain_annealing --name "<experiment_name>" --annealing_name "<annealing_name>"

# Launch SFT training (loads pretrain_annealing final checkpoint)
./scripts/launch_cluster.sh start --nodes all --mode sft --name "<experiment_name>" --annealing_name "<annealing_name>" --sft_name "<sft_name>"

# Launch SFT with custom configs
./scripts/launch_cluster.sh start --nodes all --mode sft --name "<experiment_name>" \
    --annealing_name "<annealing_name>" --sft_name "<sft_name>" \
    --model Qwen3_6-35B-A3B --training sft/origin --optimizer sft/origin --data sft/msmarco_mqa

# Launch SFT training skipping annealing (loads pretrain final checkpoint directly)
./scripts/launch_cluster.sh start --nodes all --mode sft --name "<experiment_name>" --annealing_name null --sft_name "<sft_name>"

# Available config override flags:
#   --nodes <i-j|i|all>         Node selection (required): range, single, or all
#   --model <name>              e.g. Qwen3_6-27B, Qwen3_6-35B-A3B, Qwen3-30B-A3B-Instruct-2507
#   --m2p_transformer <name>    e.g. full_prenorm_gatedlastnorm, moe_default
#   --training <mode/name>      e.g. sft/origin, pretrain/origin (must match --mode)
#   --optimizer <mode/name>     e.g. sft/origin, pretrain/origin (must match --mode)
#   --data <mode/name>          e.g. sft/msmarco_mqa, pretrain/oldpretrain (must match --mode)
#   --debug <name>              e.g. origin
#   --tokenizer <name>          e.g. origin
#   --detach_state <name>       e.g. origin, full
#   --force_overwrite           Force resume even if config selections differ from checkpoint
#   --evaluation_baseline       Run base LLM evaluation (no hypernetwork) on val set, then exit
#   --evaluation_export_lora    Run val set by repo, export per-repo PEFT LoRA adapters, then exit
#   --export_lora_max_traj <N>  Max trajectories per repo (required for --evaluation_export_lora)

# Check training status
./scripts/launch_cluster.sh status --nodes all

# Check training status with GPU info
./scripts/launch_cluster.sh status --nodes all --verbose

# Stop training on specific nodes
./scripts/launch_cluster.sh stop --nodes 0-1

# Stop training on all nodes
./scripts/launch_cluster.sh stop --nodes all

# ============================================================
# Evaluation modes (special one-shot runs, then exit)
# ============================================================

# Evaluation Baseline: run base LLM evaluation (no hypernetwork/LoRA) on val set, then exit
./scripts/launch_cluster.sh start --nodes all --mode pretrain --name "<experiment_name>" --parallel tp --tp_size 4 \
    --evaluation_baseline \
    --data pretrain/trajectory_all_transfer --detach_state full --model Qwen3_6-27B

# Evaluation Export LoRA: run val set grouped by repo, export per-repo PEFT LoRA adapters, then exit
# --export_lora_max_traj <N> is REQUIRED (max trajectories per repo)
# Output: ./save/<experiment_name>/<repo_name>/adapter_model.safetensors + adapter_config.json
./scripts/launch_cluster.sh start --nodes all --mode pretrain --name "<experiment_name>" --parallel tp --tp_size 4 \
    --evaluation_export_lora --export_lora_max_traj 50 \
    --data pretrain/trajectory_all_transfer --detach_state full --model Qwen3_6-27B

# PP mode also supported for both evaluation modes:
./scripts/launch_cluster.sh start --nodes all --mode pretrain --name "<experiment_name>" \
    --evaluation_baseline \
    --data pretrain/trajectory_all_transfer --detach_state full --model Qwen3_6-27B

./scripts/launch_cluster.sh start --nodes all --mode pretrain --name "<experiment_name>" \
    --evaluation_export_lora --export_lora_max_traj 50 \
    --data pretrain/trajectory_all_transfer --detach_state full --model Qwen3_6-27B

# ============================================================
# Multi-node TP (Tensor Parallel) training
# ============================================================
# Uses launch_cluster.sh with --parallel tp flag.
# Default: TP=2, DP=4 per node. Override with --tp_size.

# Launch TP pretrain training (default TP=2, DP=4 per node)
./scripts/launch_cluster.sh start --nodes all --mode pretrain --name "<experiment_name>" --parallel tp

# Launch TP pretrain with custom TP size (TP=4, DP=2 per node)
./scripts/launch_cluster.sh start --nodes all --mode pretrain --name "<experiment_name>" --parallel tp --tp_size 4

# Run on nodes 0-1 with TP=4 while running another job on nodes 2-3
./scripts/launch_cluster.sh start --nodes 0-1 --mode pretrain --name "<exp_A>" --parallel tp --tp_size 4
./scripts/launch_cluster.sh start --nodes 2-3 --mode pretrain --name "<exp_B>" --parallel tp --tp_size 4

# Launch TP pretrain with custom model and data
./scripts/launch_cluster.sh start --nodes all --mode pretrain --name "<experiment_name>" --parallel tp \
    --model Qwen3_6-35B-A3B --data pretrain/oldpretrainnochange

# Launch TP pretrain annealing
./scripts/launch_cluster.sh start --nodes all --mode pretrain_annealing --name "<experiment_name>" \
    --annealing_name "<annealing_name>" --parallel tp

# Launch TP SFT training
./scripts/launch_cluster.sh start --nodes all --mode sft --name "<experiment_name>" \
    --annealing_name "<annealing_name>" --sft_name "<sft_name>" --parallel tp

# Launch TP SFT with custom configs and TP=4
./scripts/launch_cluster.sh start --nodes all --mode sft --name "<experiment_name>" \
    --annealing_name "<annealing_name>" --sft_name "<sft_name>" --parallel tp --tp_size 4 \
    --model Qwen3_6-35B-A3B --training sft/origin --optimizer sft/origin --data sft/msmarco_mqa

# TP size recommendations:
#   TP=2 DP=4: Best throughput for Qwen3.6-27B (~6x PP speed)
#   TP=4 DP=2: Good for larger models or longer sequences (~4x PP speed)
#   TP=8 DP=1: Full TP, requires KV-head replication, slower on single node

# Check TP training status (same as PP)
./scripts/launch_cluster.sh status --nodes all
./scripts/launch_cluster.sh status --nodes all --verbose

# Stop TP training (same as PP)
./scripts/launch_cluster.sh stop --nodes all

# Logs are stored in logs/<nodes_spec>/ subdirectory:
#   --nodes all   -> logs/all/node_0.log, logs/all/node_1.log, ...
#   --nodes 0-1   -> logs/0-1/node_0.log, logs/0-1/node_1.log
#   --nodes 2-3   -> logs/2-3/node_0.log, logs/2-3/node_1.log
#   --nodes 2     -> logs/2/node_0.log


##########################################################################################################
# Install environment on all nodes (run this first to setup dependencies)
./scripts/setup_env.sh cluster_nodes/nodes.txt install

# Check installation process status on all nodes
./scripts/setup_env.sh cluster_nodes/nodes.txt status

# Stop installation processes on all nodes
./scripts/setup_env.sh cluster_nodes/nodes.txt stop
