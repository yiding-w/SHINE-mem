#!/bin/bash
# Stop debug tests on cluster nodes
# Usage: ./scripts/stop_debug_test.sh --nodes <i-j|i|all>
#
# Examples:
#   ./scripts/stop_debug_test.sh --nodes all
#   ./scripts/stop_debug_test.sh --nodes 0-1

set -e

# Parse arguments
NODES_SPEC=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)
            NODES_SPEC="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --nodes <i-j|i|all>"
            exit 1
            ;;
    esac
done

# --nodes is mandatory
if [ -z "$NODES_SPEC" ]; then
    echo "Error: --nodes <i-j|i|all> is required."
    echo "Usage: $0 --nodes <i-j|i|all>"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORK_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
SSH_HELPER="$SCRIPT_DIR/ssh_helper.py"
SSH_PORT=${SSH_PORT:-36000}

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
    [[ $line =~ ^# ]] || [[ -z $line ]] && continue
    read -r ip _rank user auth <<< "$line"
    ALL_NODE_IPS+=("$ip")
    ALL_NODE_USERS+=("${user:-$USER}")
    if [ -n "$auth" ] && [ -f "$auth" ]; then
        ALL_NODE_KEYS+=("$auth")
        ALL_NODE_PASSWORDS+=("")
    elif [ -n "$auth" ]; then
        ALL_NODE_KEYS+=("")
        ALL_NODE_PASSWORDS+=("$auth")
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

# Build the actual node arrays from selected indices
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

# Function to run remote command
remote_exec() {
    local user="$1" host="$2" key="$3" pw="$4" cmd="$5"
    if [ -n "$pw" ]; then
        python3 "$SSH_HELPER" --port "$SSH_PORT" "$user" "$host" "$pw" "$cmd"
    elif [ -n "$key" ]; then
        python3 "$SSH_HELPER" --port "$SSH_PORT" --key "$key" "$user" "$host" "$cmd"
    else
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$SSH_PORT" ${user}@${host} "$cmd"
    fi
}

echo "Stopping debug tests on ${#NODES[@]} nodes (--nodes $NODES_SPEC)..."
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    user="${USERS[$i]}"
    key="${KEYS[$i]}"
    pw="${PASSWORDS[$i]}"

    remote_exec "$user" "$node" "$key" "$pw" \
        'PIDS=$(ps aux | grep -E "torchrun.*meta_train_debug|python.*meta_train_debug|run_debug_test\.sh" | grep -v grep | grep -v ssh_helper | awk "{print \$2}"); if [ -n "$PIDS" ]; then echo "$PIDS" | xargs kill 2>/dev/null; sleep 1; echo "$PIDS" | xargs kill -9 2>/dev/null; fi; true' || true
    echo "  Node ${RANKS[$i]} ($node): stopped"
done

echo ""
echo "All debug test processes stopped."
