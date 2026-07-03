#!/bin/bash
# Cluster launch script for multi-node training
# Usage: ./launch_cluster.sh <action> --nodes <i-j|i|all> --mode <pretrain|pretrain_annealing|sft> --name <name> [options]
# Actions: start, stop, status
#
# Node selection (reads from cluster_nodes/nodes.txt):
#   --nodes i-j               Use nodes i through j (0-indexed lines in nodes.txt)
#   --nodes i                 Use only node i
#   --nodes all               Use all nodes
#
# Config selection options (override defaults in main_*.yaml):
#   --model <name>              e.g. Qwen3_6-35B-A3B
#   --m2p_transformer <name>   e.g. full_prenorm_gatedlastnorm
#   --training <mode/name>     e.g. sft/origin (must match --mode)
#   --optimizer <mode/name>    e.g. sft/origin (must match --mode)
#   --data <mode/name>         e.g. sft/msmarco_mqa (must match --mode)
#   --debug <name>             e.g. origin
#   --tokenizer <name>         e.g. origin
#   --detach_state <name>      e.g. origin, full
#   --parallel <pp|tp>         Parallelism strategy (default: pp)
#   --tp_size <N>              Tensor parallel size (default: 2, only for --parallel tp)
#   --sp_size <N>              Sequence parallel size (default: 1, only for --parallel tp)

set -e

# Save the full launch command for wandb tracking
LAUNCH_CMD="$0 $*"

ACTION=${1:-start}

# Parse optional flags
TRAINING_MODE=""
EXP_NAME=""
ANNEALING_NAME=""
SFT_NAME=""
DATA_CONFIG=""
MODEL_CONFIG=""
TRAINING_CONFIG=""
OPTIMIZER_CONFIG=""
M2P_TRANSFORMER_CONFIG=""
DEBUG_CONFIG=""
TOKENIZER_CONFIG=""
DETACH_STATE_CONFIG=""
FORCE_OVERWRITE=""
PARALLEL_MODE="pp"
TP_SIZE="2"
SP_SIZE="1"
NODES_SPEC=""
VERBOSE=""
EVALUATION_BASELINE=""
EVALUATION_EXPORT_LORA=""
EXPORT_LORA_MAX_TRAJ=""
shift 1 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)
            NODES_SPEC="$2"
            shift 2
            ;;
        --mode)
            TRAINING_MODE="$2"
            shift 2
            ;;
        --name)
            EXP_NAME="$2"
            shift 2
            ;;
        --annealing_name)
            ANNEALING_NAME="$2"
            shift 2
            ;;
        --sft_name)
            SFT_NAME="$2"
            shift 2
            ;;
        --data)
            DATA_CONFIG="$2"
            shift 2
            ;;
        --model)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --training)
            TRAINING_CONFIG="$2"
            shift 2
            ;;
        --optimizer)
            OPTIMIZER_CONFIG="$2"
            shift 2
            ;;
        --m2p_transformer)
            M2P_TRANSFORMER_CONFIG="$2"
            shift 2
            ;;
        --debug)
            DEBUG_CONFIG="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER_CONFIG="$2"
            shift 2
            ;;
        --detach_state)
            DETACH_STATE_CONFIG="$2"
            shift 2
            ;;
        --force_overwrite)
            FORCE_OVERWRITE="1"
            shift 1
            ;;
        --parallel)
            PARALLEL_MODE="$2"
            shift 2
            ;;
        --tp_size)
            TP_SIZE="$2"
            shift 2
            ;;
        --sp_size)
            SP_SIZE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="1"
            shift 1
            ;;
        --evaluation_baseline)
            EVALUATION_BASELINE="1"
            shift 1
            ;;
        --evaluation_export_lora)
            EVALUATION_EXPORT_LORA="1"
            shift 1
            ;;
        --export_lora_max_traj)
            EXPORT_LORA_MAX_TRAJ="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 <action> --nodes <i-j|i|all> --mode <pretrain|pretrain_annealing|sft> --name <name> [options]"
            echo ""
            echo "Node selection:"
            echo "  --nodes i-j               Use nodes i through j"
            echo "  --nodes i                 Use only node i"
            echo "  --nodes all               Use all nodes"
            echo ""
            echo "Config selection options:"
            echo "  --model <name>              Model config (e.g. Qwen3_6-35B-A3B)"
            echo "  --m2p_transformer <name>    M2P transformer config"
            echo "  --training <mode/name>      Training config (must match --mode prefix)"
            echo "  --optimizer <mode/name>     Optimizer config (must match --mode prefix)"
            echo "  --data <mode/name>          Data config (must match --mode prefix)"
            echo "  --debug <name>              Debug config"
            echo "  --tokenizer <name>          Tokenizer config (e.g. origin)"
            echo "  --detach_state <name>       DetachState config (e.g. origin, full)"
            echo "  --force_overwrite           Force resume even if configs differ from checkpoint"
            echo "  --parallel <pp|tp>          Parallelism strategy (default: pp)"
            echo "  --tp_size <N>               Tensor parallel size (default: 2, only for --parallel tp)"
            echo "  --sp_size <N>               Sequence parallel size (default: 1, only for --parallel tp)"
            echo "  --evaluation_baseline       Run only one baseline evaluation (base LLM, no hypernetwork) then exit"
            echo "  --evaluation_export_lora    Run evaluation and export LoRA adapters per repo then exit"
            echo "  --export_lora_max_traj <N>  Max trajectories per repo for export_lora mode (required)"
            echo "  --verbose                   Show GPU memory/utilization in status"
            exit 1
            ;;
    esac
done

# --nodes is mandatory
if [ -z "$NODES_SPEC" ]; then
    echo "Error: --nodes <i-j|i|all> is required."
    echo "Usage: $0 <action> --nodes <i-j|i|all> --mode <pretrain|pretrain_annealing|sft> --name <name> [options]"
    exit 1
fi

# For 'start' action, --mode and --name are mandatory
if [ "$ACTION" = "start" ]; then
    if [ -z "$TRAINING_MODE" ]; then
        echo "Error: --mode <pretrain|pretrain_annealing|sft> is required for 'start' action."
        echo "Usage: $0 start --nodes <i-j|i|all> --mode <pretrain|pretrain_annealing|sft> --name <name> [options]"
        exit 1
    fi
    if [ "$TRAINING_MODE" != "pretrain" ] && [ "$TRAINING_MODE" != "pretrain_annealing" ] && [ "$TRAINING_MODE" != "sft" ]; then
        echo "Error: --mode must be 'pretrain', 'pretrain_annealing', or 'sft', got '$TRAINING_MODE'."
        exit 1
    fi
    if [ "$PARALLEL_MODE" != "pp" ] && [ "$PARALLEL_MODE" != "tp" ]; then
        echo "Error: --parallel must be 'pp' or 'tp', got '$PARALLEL_MODE'."
        exit 1
    fi
    if [ "$PARALLEL_MODE" = "pp" ] && [ "$SP_SIZE" -gt 1 ]; then
        echo "Error: Sequence parallelism (--sp_size $SP_SIZE) is not supported with pipeline parallelism (--parallel pp)."
        echo "  PP + SP is not yet implemented. Use --parallel tp with --sp_size."
        exit 1
    fi
    if [ -z "$EXP_NAME" ]; then
        echo "Error: --name <name> is required for 'start' action."
        echo "Usage: $0 start --nodes <i-j|i|all> --mode <pretrain|pretrain_annealing|sft> --name <name> [options]"
        exit 1
    fi
    # For pretrain_annealing mode, --annealing_name is mandatory
    if [ "$TRAINING_MODE" = "pretrain_annealing" ] && [ -z "$ANNEALING_NAME" ]; then
        echo "Error: --annealing_name <annealing_name> is required for pretrain_annealing mode."
        exit 1
    fi
    # For SFT mode, --annealing_name and --sft_name are mandatory
    if [ "$TRAINING_MODE" = "sft" ]; then
        if [ -z "$ANNEALING_NAME" ]; then
            echo "Error: --annealing_name <annealing_name|null> is required for SFT mode."
            exit 1
        fi
        if [ -z "$SFT_NAME" ]; then
            echo "Error: --sft_name <sft_name> is required for SFT mode."
            exit 1
        fi
    fi

    # --- Validate mode-prefixed configs ---
    # For training, optimizer, data: if specified, must start with "${TRAINING_MODE}/"
    validate_mode_prefix() {
        local config_name="$1"
        local config_value="$2"
        local mode="$3"
        if [ -n "$config_value" ]; then
            if [[ "$config_value" != "${mode}/"* ]]; then
                echo "Error: --${config_name} must be prefixed with '${mode}/' for mode '${mode}'."
                echo "  Got: --${config_name} ${config_value}"
                echo "  Expected: --${config_name} ${mode}/<config_name>"
                exit 1
            fi
        fi
    }
    validate_mode_prefix "training" "$TRAINING_CONFIG" "$TRAINING_MODE"
    validate_mode_prefix "optimizer" "$OPTIMIZER_CONFIG" "$TRAINING_MODE"
    validate_mode_prefix "data" "$DATA_CONFIG" "$TRAINING_MODE"

    # Compose WANDB_RUN_NAME based on mode
    if [ "$TRAINING_MODE" = "sft" ]; then
        if [ "$ANNEALING_NAME" = "null" ]; then
            WANDB_RUN_NAME="${EXP_NAME}_null_${SFT_NAME}"
        else
            WANDB_RUN_NAME="${EXP_NAME}_${ANNEALING_NAME}_${SFT_NAME}"
        fi
    elif [ "$TRAINING_MODE" = "pretrain_annealing" ]; then
        WANDB_RUN_NAME="${EXP_NAME}_${ANNEALING_NAME}"
    else
        WANDB_RUN_NAME="${EXP_NAME}"
    fi
fi

# --- Configurable parameters ---
# Working directory on remote nodes (where the project root is)
# Override with environment variable: WORK_DIR=/path/to/project ./launch_cluster.sh
WORK_DIR=${WORK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
PIPELINE_STAGES=${PIPELINE_STAGES:-8}
SSH_PORT=${SSH_PORT:-36000}
LOG_DIR="logs"

# Determine log subdirectory name from --nodes spec
LOG_SUBDIR="$NODES_SPEC"
LOG_PATH="$LOG_DIR/$LOG_SUBDIR"

# Resolve the directory where this script lives
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SSH_HELPER="$SCRIPT_DIR/ssh_helper.py"

# Verify ssh_helper.py and paramiko are available
if [ ! -f "$SSH_HELPER" ]; then
    echo "Error: $SSH_HELPER not found."
    exit 1
fi
if ! python3 -c "import paramiko" &>/dev/null; then
    echo "Error: python3 paramiko module not found."
    echo "Install it with: pip3 install paramiko"
    exit 1
fi

# Function to get password securely (for interactive 'prompt' mode only)
get_password() {
    read -sp "Enter SSH password for all nodes: " GLOBAL_PASSWORD
    echo
}

# Fixed config file: cluster_nodes/nodes.txt
CONFIG_FILE="$WORK_DIR/cluster_nodes/nodes.txt"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found."
    echo "Please create cluster_nodes/nodes.txt with node configuration."
    exit 1
fi

# Read all nodes from config file into arrays (indexed by line order)
ALL_NODE_IPS=()
ALL_NODE_USERS=()
ALL_NODE_KEYS=()
ALL_NODE_PASSWORDS=()

while IFS= read -r line; do
    # Skip comments and empty lines
    [[ $line =~ ^# ]] || [[ -z $line ]] && continue
    
    read -r ip _rank user auth <<< "$line"
    ALL_NODE_IPS+=("$ip")
    ALL_NODE_USERS+=("${user:-$USER}")
    
    # Check authentication method
    if [ "$auth" = "password" ]; then
        ALL_NODE_KEYS+=("")
        ALL_NODE_PASSWORDS+=("prompt")
    elif [ -n "$auth" ] && [ ! -f "$auth" ]; then
        ALL_NODE_KEYS+=("")
        ALL_NODE_PASSWORDS+=("$auth")
    elif [ -n "$auth" ]; then
        ALL_NODE_KEYS+=("$auth")
        ALL_NODE_PASSWORDS+=("")
    else
        ALL_NODE_KEYS+=("")
        ALL_NODE_PASSWORDS+=("")
    fi
done < "$CONFIG_FILE"

TOTAL_AVAILABLE=${#ALL_NODE_IPS[@]}

# Parse --nodes spec to determine which nodes to use
SELECTED_INDICES=()
if [ "$NODES_SPEC" = "all" ]; then
    for ((idx=0; idx<TOTAL_AVAILABLE; idx++)); do
        SELECTED_INDICES+=($idx)
    done
elif [[ "$NODES_SPEC" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    NODE_START=${BASH_REMATCH[1]}
    NODE_END=${BASH_REMATCH[2]}
    if [ $NODE_START -gt $NODE_END ]; then
        echo "Error: Invalid --nodes range: $NODES_SPEC (start > end)"
        exit 1
    fi
    if [ $NODE_END -ge $TOTAL_AVAILABLE ]; then
        echo "Error: Node index $NODE_END out of range (max: $((TOTAL_AVAILABLE-1)))"
        exit 1
    fi
    for ((idx=NODE_START; idx<=NODE_END; idx++)); do
        SELECTED_INDICES+=($idx)
    done
elif [[ "$NODES_SPEC" =~ ^[0-9]+$ ]]; then
    if [ $NODES_SPEC -ge $TOTAL_AVAILABLE ]; then
        echo "Error: Node index $NODES_SPEC out of range (max: $((TOTAL_AVAILABLE-1)))"
        exit 1
    fi
    SELECTED_INDICES+=($NODES_SPEC)
else
    echo "Error: Invalid --nodes format: '$NODES_SPEC'"
    echo "  Supported formats: i-j (range), i (single), all"
    exit 1
fi

# Build the actual node arrays from selected indices, assigning ranks 0,1,2,...
NODES=()
RANKS=()
USERS=()
KEYS=()
PASSWORDS=()
NEW_RANK=0
for idx in "${SELECTED_INDICES[@]}"; do
    NODES+=("${ALL_NODE_IPS[$idx]}")
    RANKS+=($NEW_RANK)
    USERS+=("${ALL_NODE_USERS[$idx]}")
    KEYS+=("${ALL_NODE_KEYS[$idx]}")
    PASSWORDS+=("${ALL_NODE_PASSWORDS[$idx]}")
    NEW_RANK=$((NEW_RANK + 1))
done

TOTAL_NODES=${#NODES[@]}
MASTER_IP="${NODES[0]}"

# Function to run a remote SSH command via ssh_helper.py (paramiko)
# Usage: remote_exec <user> <host> <key> <password_val> <command>
remote_exec() {
    local user="$1"
    local host="$2"
    local key="$3"
    local password_val="$4"
    local command="$5"
    
    # Resolve actual password
    local actual_pw="$password_val"
    if [ "$password_val" = "prompt" ]; then
        actual_pw="$GLOBAL_PASSWORD"
    fi
    
    if [ -n "$actual_pw" ]; then
        python3 "$SSH_HELPER" --port "$SSH_PORT" "$user" "$host" "$actual_pw" "$command"
    elif [ -n "$key" ]; then
        python3 "$SSH_HELPER" --port "$SSH_PORT" --key "$key" "$user" "$host" "$command"
    else
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$SSH_PORT" ${user}@${host} "$command"
    fi
}

# Function to check SSH connectivity
# Usage: remote_check <user> <host> <key> <password_val>
remote_check() {
    local user="$1"
    local host="$2"
    local key="$3"
    local password_val="$4"
    
    local actual_pw="$password_val"
    if [ "$password_val" = "prompt" ]; then
        actual_pw="$GLOBAL_PASSWORD"
    fi
    
    if [ -n "$actual_pw" ]; then
        python3 "$SSH_HELPER" --port "$SSH_PORT" --check "$user" "$host" "$actual_pw" &>/dev/null
    elif [ -n "$key" ]; then
        python3 "$SSH_HELPER" --port "$SSH_PORT" --check --key "$key" "$user" "$host" &>/dev/null
    else
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$SSH_PORT" ${user}@${host} "echo ok" &>/dev/null
    fi
}

case "$ACTION" in
    "start")
        # Get password if needed (only for interactive 'prompt' mode)
        NEED_PASSWORD=false
        for password in "${PASSWORDS[@]}"; do
            if [ "$password" = "prompt" ]; then
                NEED_PASSWORD=true
                break
            fi
        done
        
        if [ "$NEED_PASSWORD" = true ]; then
            get_password
        fi
        
        # Check if old training processes are still running on any selected node
        OLD_RUNNING=false
        for i in "${!NODES[@]}"; do
            node="${NODES[$i]}"
            user="${USERS[$i]}"
            key="${KEYS[$i]}"
            password_flag="${PASSWORDS[$i]}"
            if remote_exec "$user" "$node" "$key" "$password_flag" 'ps aux | grep -E "torchrun.*meta_train|python.*meta_train" | grep -v grep | grep -v ssh_helper | grep -q .' 2>/dev/null; then
                echo "Error: Training processes still running on node $node (index ${SELECTED_INDICES[$i]})"
                OLD_RUNNING=true
            fi
        done
        if [ "$OLD_RUNNING" = true ]; then
            echo ""
            echo "Please stop old processes first:"
            echo "  $0 stop --nodes $NODES_SPEC"
            exit 1
        fi
        
        # Clear the log subdirectory for this node selection before launching
        # (logs are temporary; persistent records are uploaded to wandb)
        # Note: only remove contents, not the directory itself, to avoid
        # cephfs race conditions that could corrupt the parent logs/ directory.
        if [ -d "$WORK_DIR/$LOG_PATH" ]; then
            find "$WORK_DIR/$LOG_PATH" -mindepth 1 -delete 2>/dev/null || true
        fi
        mkdir -p "$WORK_DIR/$LOG_PATH"

        # --- Resolve defaults from main_*.yaml for any unset config variables ---
        _CONFIGS_DIR="$WORK_DIR/configs"
        if [ "$TRAINING_MODE" = "sft" ]; then
            _MAIN_YAML="$_CONFIGS_DIR/main_sft.yaml"
        elif [ "$TRAINING_MODE" = "pretrain_annealing" ]; then
            _MAIN_YAML="$_CONFIGS_DIR/main_pretrain_annealing.yaml"
        else
            _MAIN_YAML="$_CONFIGS_DIR/main_pretrain.yaml"
        fi
        _get_yaml_default() {
            local key="$1"
            grep -E "^\s*-\s+${key}:" "$_MAIN_YAML" 2>/dev/null | sed -E 's/^\s*-\s+'"${key}"':\s*([^ #]+).*/\1/' | head -1
        }
        # Fill in any unset config variables from the main yaml defaults
        [ -z "$MODEL_CONFIG" ] && MODEL_CONFIG="$(_get_yaml_default model)"
        [ -z "$M2P_TRANSFORMER_CONFIG" ] && M2P_TRANSFORMER_CONFIG="$(_get_yaml_default m2p_transformer)"
        [ -z "$TRAINING_CONFIG" ] && TRAINING_CONFIG="$(_get_yaml_default training)"
        [ -z "$OPTIMIZER_CONFIG" ] && OPTIMIZER_CONFIG="$(_get_yaml_default optimizer)"
        [ -z "$DATA_CONFIG" ] && DATA_CONFIG="$(_get_yaml_default data)"
        [ -z "$DEBUG_CONFIG" ] && DEBUG_CONFIG="$(_get_yaml_default debug)"
        [ -z "$TOKENIZER_CONFIG" ] && TOKENIZER_CONFIG="$(_get_yaml_default tokenizer)"
        [ -z "$DETACH_STATE_CONFIG" ] && DETACH_STATE_CONFIG="$(_get_yaml_default detach_state)"

        # Run the entire launch process in a background subshell so the
        # terminal returns immediately. All output goes to a launcher log.
        LAUNCHER_LOG="$WORK_DIR/$LOG_PATH/launcher.log"
        mkdir -p "$WORK_DIR/$LOG_PATH" 2>/dev/null || true
        
        (
        echo "Launching training on $TOTAL_NODES nodes..."
        echo "Master node: $MASTER_IP"
        echo "Work directory: $WORK_DIR"
        echo "Training mode: $TRAINING_MODE"
        echo "Experiment name: $EXP_NAME"
        echo "Annealing name: ${ANNEALING_NAME:-N/A}"
        echo "SFT name: ${SFT_NAME:-N/A}"
        echo "Config selections (from CLI or main yaml defaults):"
        echo "  model: $MODEL_CONFIG"
        echo "  m2p_transformer: $M2P_TRANSFORMER_CONFIG"
        echo "  training: $TRAINING_CONFIG"
        echo "  optimizer: $OPTIMIZER_CONFIG"
        echo "  data: $DATA_CONFIG"
        echo "  debug: $DEBUG_CONFIG"
        echo "  tokenizer: $TOKENIZER_CONFIG"
        echo "  detach_state: $DETACH_STATE_CONFIG"
        echo "Parallel mode: $PARALLEL_MODE"
        echo "Nodes spec: $NODES_SPEC"
        echo "Log path: $LOG_PATH"
        if [ "$PARALLEL_MODE" = "tp" ]; then
            echo "TP size: $TP_SIZE"
            echo "SP size: $SP_SIZE"
            echo "DP per node: $((8 / (TP_SIZE * SP_SIZE)))"
            echo "Total DP: $(( (8 / (TP_SIZE * SP_SIZE)) * TOTAL_NODES ))"
        else
            echo "Pipeline stages: $PIPELINE_STAGES"
        fi
        echo "Wandb run name: $WANDB_RUN_NAME"
        echo "Launch command: $LAUNCH_CMD"
        
        # Pre-launch: verify connectivity to all nodes
        echo "Verifying connectivity to all nodes... (via paramiko)"
        for i in "${!NODES[@]}"; do
            node="${NODES[$i]}"
            user="${USERS[$i]}"
            key="${KEYS[$i]}"
            password_flag="${PASSWORDS[$i]}"
            
            if ! remote_check "$user" "$node" "$key" "$password_flag"; then
                echo "Error: Cannot connect to node ${RANKS[$i]} at ${user}@${node}"
                exit 1
            fi
            echo "  Node ${RANKS[$i]} ($node): reachable"
        done
        echo "All nodes reachable."
        
        # Start training on all nodes in parallel
        for i in "${!NODES[@]}"; do
            node="${NODES[$i]}"
            rank="${RANKS[$i]}"
            user="${USERS[$i]}"
            key="${KEYS[$i]}"
            password_flag="${PASSWORDS[$i]}"
            
            echo "Starting node $rank on $node"
            if [ "$PARALLEL_MODE" = "tp" ]; then
                remote_exec "$user" "$node" "$key" "$password_flag" \
"mkdir -p $WORK_DIR/$LOG_PATH && cd $WORK_DIR && export WANDB_NAME='$WANDB_RUN_NAME' && export TRAINING_MODE='$TRAINING_MODE' && export EXP_NAME='$EXP_NAME' && export ANNEALING_NAME='$ANNEALING_NAME' && export SFT_NAME='$SFT_NAME' && export DATA_CONFIG='$DATA_CONFIG' && export MODEL_CONFIG='$MODEL_CONFIG' && export TRAINING_CONFIG='$TRAINING_CONFIG' && export OPTIMIZER_CONFIG='$OPTIMIZER_CONFIG' && export M2P_TRANSFORMER_CONFIG='$M2P_TRANSFORMER_CONFIG' && export DEBUG_CONFIG='$DEBUG_CONFIG' && export TOKENIZER_CONFIG='$TOKENIZER_CONFIG' && export DETACH_STATE_CONFIG='$DETACH_STATE_CONFIG' && export LAUNCH_CMD='$LAUNCH_CMD' && export FORCE_OVERWRITE='$FORCE_OVERWRITE' && export EVALUATION_BASELINE='$EVALUATION_BASELINE' && export EVALUATION_EXPORT_LORA='$EVALUATION_EXPORT_LORA' && export EXPORT_LORA_MAX_TRAJ='$EXPORT_LORA_MAX_TRAJ' && export LOG_SUBDIR='$LOG_SUBDIR' && export DEBUG_RESUME='${DEBUG_RESUME:-}' && nohup ./scripts/run_training_tp.sh $TOTAL_NODES $rank $MASTER_IP $TP_SIZE $SP_SIZE > $LOG_PATH/node_${rank}.log 2>&1 &" &
            else
                remote_exec "$user" "$node" "$key" "$password_flag" \
"mkdir -p $WORK_DIR/$LOG_PATH && cd $WORK_DIR && export WANDB_NAME='$WANDB_RUN_NAME' && export TRAINING_MODE='$TRAINING_MODE' && export EXP_NAME='$EXP_NAME' && export ANNEALING_NAME='$ANNEALING_NAME' && export SFT_NAME='$SFT_NAME' && export DATA_CONFIG='$DATA_CONFIG' && export MODEL_CONFIG='$MODEL_CONFIG' && export TRAINING_CONFIG='$TRAINING_CONFIG' && export OPTIMIZER_CONFIG='$OPTIMIZER_CONFIG' && export M2P_TRANSFORMER_CONFIG='$M2P_TRANSFORMER_CONFIG' && export DEBUG_CONFIG='$DEBUG_CONFIG' && export TOKENIZER_CONFIG='$TOKENIZER_CONFIG' && export DETACH_STATE_CONFIG='$DETACH_STATE_CONFIG' && export LAUNCH_CMD='$LAUNCH_CMD' && export FORCE_OVERWRITE='$FORCE_OVERWRITE' && export EVALUATION_BASELINE='$EVALUATION_BASELINE' && export EVALUATION_EXPORT_LORA='$EVALUATION_EXPORT_LORA' && export EXPORT_LORA_MAX_TRAJ='$EXPORT_LORA_MAX_TRAJ' && export LOG_SUBDIR='$LOG_SUBDIR' && export DEBUG_RESUME='${DEBUG_RESUME:-}' && nohup ./scripts/run_training.sh $TOTAL_NODES $rank $MASTER_IP $PIPELINE_STAGES > $LOG_PATH/node_${rank}.log 2>&1 &" &
            fi
        done
        
        wait
        echo "All nodes started successfully"
        echo "Logs are at: $WORK_DIR/$LOG_PATH/node_<rank>.log on each node"
        ) >> "$LAUNCHER_LOG" 2>&1 &
        
        echo "Launching in background. Check progress: tail -f $LAUNCHER_LOG"
        disown
        ;;
    
    "stop")
        echo "Stopping training on $TOTAL_NODES nodes (--nodes $NODES_SPEC)..."
        for i in "${!NODES[@]}"; do
            node="${NODES[$i]}"
            user="${USERS[$i]}"
            key="${KEYS[$i]}"
            password_flag="${PASSWORDS[$i]}"
            
            # Kill torchrun and all its child processes (worker processes)
            # torchrun spawns multiple python workers that won't die when
            # only the parent is killed, so we need to match all of them.
            remote_exec "$user" "$node" "$key" "$password_flag" \
                'PIDS=$(ps aux | grep -E "torchrun.*meta_train|python.*meta_train|run_training\.sh" | grep -v grep | grep -v ssh_helper | awk "{print \$2}"); if [ -n "$PIDS" ]; then echo "$PIDS" | xargs kill 2>/dev/null; sleep 1; echo "$PIDS" | xargs kill -9 2>/dev/null; fi; true' || true
            echo "  Node ${RANKS[$i]} ($node): stopped"
        done
        ;;
    
    "status")
        echo "Checking status of $TOTAL_NODES nodes (--nodes $NODES_SPEC)..."
        echo ""
        for i in "${!NODES[@]}"; do
            node="${NODES[$i]}"
            user="${USERS[$i]}"
            key="${KEYS[$i]}"
            password_flag="${PASSWORDS[$i]}"
            
            if remote_exec "$user" "$node" "$key" "$password_flag" 'ps aux | grep -E "torchrun.*meta_train|python.*meta_train" | grep -v grep | grep -v ssh_helper | grep -q .' 2>/dev/null; then
                echo "Node ${RANKS[$i]} ($node): RUNNING"
            else
                echo "Node ${RANKS[$i]} ($node): STOPPED"
            fi
            
            # Show GPU status only with --verbose
            if [ -n "$VERBOSE" ]; then
                echo "  GPU status:"
                gpu_info=$(remote_exec "$user" "$node" "$key" "$password_flag" 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits' 2>/dev/null) || true
                if [ -n "$gpu_info" ]; then
                    while IFS= read -r gpu_line; do
                        IFS=',' read -r gpu_idx gpu_util mem_used mem_total <<< "$gpu_line"
                        gpu_idx=$(echo "$gpu_idx" | xargs)
                        gpu_util=$(echo "$gpu_util" | xargs)
                        mem_used=$(echo "$mem_used" | xargs)
                        mem_total=$(echo "$mem_total" | xargs)
                        echo "    GPU $gpu_idx: ${gpu_util}% util, ${mem_used}/${mem_total} MiB"
                    done <<< "$gpu_info"
                else
                    echo "    (unable to query GPU)"
                fi
            fi
            echo ""
        done
        ;;
    
    *)
        echo "Usage: $0 <action> --nodes <i-j|i|all> [options]"
        exit 1
        ;;
esac