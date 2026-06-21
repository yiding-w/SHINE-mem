#!/bin/bash
# Launch debug tests on cluster nodes
# Usage: ./scripts/launch_debug_test.sh --nodes <i-j|i|all> [options]
#
# Node selection (reads from cluster_nodes/nodes.txt):
#   --nodes i-j               Use nodes i through j (0-indexed lines in nodes.txt)
#   --nodes i                 Use only node i
#   --nodes all               Use all nodes
#
# Options:
#   --test <name>             Test name: all, creation, write_read, e2e, detach, reset, checkpoint, dp (default: all)
#   --tp_size <N>             TP/PP size (default: 8)
#   --parallel <pp|tp>        Parallel mode (default: pp)
#   --skip <tests>            Space-separated list of tests to skip
#   --skip_sub <subs>         Space-separated list of sub-tests to skip
#
# Examples:
#   ./scripts/launch_debug_test.sh --nodes all
#   ./scripts/launch_debug_test.sh --nodes all --test checkpoint
#   ./scripts/launch_debug_test.sh --nodes 0-1 --test consistency --tp_size 2 --parallel tp
#   ./scripts/launch_debug_test.sh --nodes all --skip "creation reset"
#   ./scripts/launch_debug_test.sh --nodes all --skip_sub "2d 6c 7c"

set -e

# Parse arguments
NODES_SPEC=""
TEST_NAME="all"
TP_SIZE="8"
PARALLEL_MODE="pp"
SKIP_TESTS=""
SKIP_SUB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)
            NODES_SPEC="$2"
            shift 2
            ;;
        --test)
            TEST_NAME="$2"
            shift 2
            ;;
        --tp_size)
            TP_SIZE="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL_MODE="$2"
            shift 2
            ;;
        --skip)
            SKIP_TESTS="$2"
            shift 2
            ;;
        --skip_sub)
            SKIP_SUB="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --nodes <i-j|i|all> [--test <name>] [--tp_size <N>] [--parallel <pp|tp>] [--skip <tests>] [--skip_sub <subs>]"
            exit 1
            ;;
    esac
done

# --nodes is mandatory
if [ -z "$NODES_SPEC" ]; then
    echo "Error: --nodes <i-j|i|all> is required."
    echo "Usage: $0 --nodes <i-j|i|all> [--test <name>] [--tp_size <N>] [--parallel <pp|tp>] [--skip <tests>] [--skip_sub <subs>]"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORK_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
SSH_HELPER="$SCRIPT_DIR/ssh_helper.py"
SSH_PORT=${SSH_PORT:-36000}
LOG_DIR="logs"

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

echo "=== Launching Debug Tests on $TOTAL_NODES nodes ==="
echo "Master: $MASTER_IP"
echo "Test: $TEST_NAME"
echo "TP size: $TP_SIZE"
echo "Parallel: $PARALLEL_MODE"
if [ -n "$SKIP_TESTS" ]; then
    echo "Skip tests: $SKIP_TESTS"
fi
if [ -n "$SKIP_SUB" ]; then
    echo "Skip sub-tests: $SKIP_SUB"
fi

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

# Clear logs
rm -rf "$WORK_DIR/$LOG_DIR/debug_node_"*.log 2>/dev/null || true
mkdir -p "$WORK_DIR/$LOG_DIR"

# Launch on all nodes
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    rank="${RANKS[$i]}"
    user="${USERS[$i]}"
    key="${KEYS[$i]}"
    pw="${PASSWORDS[$i]}"

    echo "Starting debug test on node $rank ($node)..."
    remote_exec "$user" "$node" "$key" "$pw" \
        "cd $WORK_DIR && nohup ./scripts/run_debug_test.sh $TOTAL_NODES $rank $MASTER_IP $TP_SIZE $TEST_NAME $PARALLEL_MODE \"$SKIP_TESTS\" \"$SKIP_SUB\" > $LOG_DIR/debug_node_${rank}.log 2>&1 &" &
done

wait
echo ""
echo "All nodes started. Monitor with:"
echo "  tail -f $WORK_DIR/$LOG_DIR/debug_node_0.log"
echo ""
echo "Stop with:"
echo "  ./scripts/stop_debug_test.sh --nodes $NODES_SPEC"
