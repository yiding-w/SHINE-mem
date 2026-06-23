#!/bin/bash
# Multi-node environment setup script
# Usage: ./setup_env.sh <config_file> [action]
# Actions: install, check, cleanup

set -e

CONFIG_FILE=${1:-cluster_nodes.txt}
ACTION=${2:-install}

# --- Configurable parameters ---
WORK_DIR=${WORK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
SSH_PORT=${SSH_PORT:-36000}
LOG_DIR=${LOG_DIR:-logs}

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

# Default cluster configuration
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found."
    echo "Create one with the format:"
    echo "# node_ip node_rank ssh_user [ssh_key|password]"
    exit 1
fi

MASTER_IP=""
NODES=()
RANKS=()
USERS=()
KEYS=()
PASSWORDS=()

# Read cluster configuration
while IFS= read -r line; do
    # Skip comments and empty lines
    [[ $line =~ ^# ]] || [[ -z $line ]] && continue
    
    read -r ip rank user auth <<< "$line"
    NODES+=("$ip")
    RANKS+=("$rank")
    USERS+=("${user:-$USER}")
    
    # Check authentication method
    if [ "$auth" = "password" ]; then
        # Interactive password prompt mode
        KEYS+=("")
        PASSWORDS+=("prompt")
    elif [ -n "$auth" ] && [ ! -f "$auth" ]; then
        # Inline password specified directly in config
        KEYS+=("")
        PASSWORDS+=("$auth")
    elif [ -n "$auth" ]; then
        # SSH key file
        KEYS+=("$auth")
        PASSWORDS+=("")
    else
        KEYS+=("")
        PASSWORDS+=("")
    fi
    
    if [ "$rank" -eq 0 ]; then
        MASTER_IP="$ip"
    fi
done < "$CONFIG_FILE"

TOTAL_NODES=${#NODES[@]}

if [ -z "$MASTER_IP" ]; then
    echo "Error: No master node (rank 0) found in configuration"
    exit 1
fi

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

# Environment setup commands
INSTALL_COMMANDS=(
    "echo 'Running custom install.sh script...'"
    "cd $WORK_DIR && ./scripts/install.sh"
)

CHECK_COMMANDS=(
    "echo 'Checking Python installation...'"
    "python3 --version"
    "echo 'Checking CUDA availability...'"
    "python3 -c \"import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count())\""
    "echo 'Checking MPI...'"
    "which mpirun"
    "echo 'Checking SSH...'"
    "which ssh"
)

# Status check commands - check if install.sh is running
STATUS_COMMANDS=(
    "echo '=== Installation Status Check ==='"
    "echo 'Checking if install.sh is running...'"
    "ps aux | grep -E 'install\.sh|pip install' | grep -v grep | grep -v ssh_helper || echo 'No install processes found'"
    "echo ''"
    "echo 'Checking Python installation processes...'"
    "ps aux | grep -E 'python.*install|pip' | grep -v grep | grep -v ssh_helper || echo 'No Python install processes found'"
    "echo ''"
    "echo 'Checking if install.sh exists and is executable...'"
    "if [ -x \"$WORK_DIR/scripts/install.sh\" ]; then echo 'install.sh is executable'; else echo 'install.sh not found or not executable'; fi"
    "echo ''"
    "echo 'Checking if Python packages are already installed...'"
    "python3 -c \"packages=['torch','transformers','datasets']; [print(f'{p}: ✓' if __import__(p) else f'{p}: ✗') for p in packages]\" 2>/dev/null || echo 'Python check failed - Python may not be available'")

# Stop commands - kill install processes
STOP_COMMANDS=(
    "echo 'Stopping installation processes...'"
    "PIDS=\$(ps aux | grep -E 'install\.sh|pip install' | grep -v grep | grep -v ssh_helper | awk '{print \$2}'); if [ -n \"\$PIDS\" ]; then echo \"Killing PIDs: \$PIDS\"; echo \"\$PIDS\" | xargs kill 2>/dev/null; sleep 1; echo \"\$PIDS\" | xargs kill -9 2>/dev/null; fi"
    "echo 'Stopping Python installation processes...'"
    "PIDS=\$(ps aux | grep -E 'python.*install|pip' | grep -v grep | grep -v ssh_helper | awk '{print \$2}'); if [ -n \"\$PIDS\" ]; then echo \"Killing PIDs: \$PIDS\"; echo \"\$PIDS\" | xargs kill 2>/dev/null; sleep 1; echo \"\$PIDS\" | xargs kill -9 2>/dev/null; fi"
    "echo 'All installation processes stopped'"
)



case "$ACTION" in
    "install")
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
        
        echo "Setting up environment on $TOTAL_NODES nodes..."
        echo "Master node: $MASTER_IP"
        
        # Verify connectivity to all nodes
        echo "Verifying connectivity to all nodes..."
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
        
        # Execute installation commands on all nodes in parallel
        for i in "${!NODES[@]}"; do
            node="${NODES[$i]}"
            rank="${RANKS[$i]}"
            user="${USERS[$i]}"
            key="${KEYS[$i]}"
            password_flag="${PASSWORDS[$i]}"
            
            echo "Setting up environment on node $rank ($node)"
            
            # Combine all install commands into one command
            install_script=$(IFS=";"; echo "${INSTALL_COMMANDS[*]}")
            
            remote_exec "$user" "$node" "$key" "$password_flag" "$install_script" &
        done
        
        wait
        echo "Environment setup completed on all nodes"
        ;;
    
    "check")
        echo "Checking environment on $TOTAL_NODES nodes..."
        
        for i in "${!NODES[@]}"; do
            node="${NODES[$i]}"
            rank="${RANKS[$i]}"
            user="${USERS[$i]}"
            key="${KEYS[$i]}"
            password_flag="${PASSWORDS[$i]}"
            
            echo "=== Node $rank ($node) ==="
            
            # Combine all check commands into one command
            check_script=$(IFS=";"; echo "${CHECK_COMMANDS[*]}")
            
            remote_exec "$user" "$node" "$key" "$password_flag" "$check_script"
            echo
        done
        ;;
    
    "status")
        echo "Checking installation status on $TOTAL_NODES nodes..."
        
        for i in "${!NODES[@]}"; do
            node="${NODES[$i]}"
            rank="${RANKS[$i]}"
            user="${USERS[$i]}"
            key="${KEYS[$i]}"
            password_flag="${PASSWORDS[$i]}"
            
            echo "=== Node $rank ($node) ==="
            
            # Combine all status commands into one command
            status_script=$(IFS=";"; echo "${STATUS_COMMANDS[*]}")
            
            remote_exec "$user" "$node" "$key" "$password_flag" "$status_script"
            echo
        done
        ;;
    
    "stop")
        echo "Stopping installation processes on $TOTAL_NODES nodes..."
        
        for i in "${!NODES[@]}"; do
            node="${NODES[$i]}"
            rank="${RANKS[$i]}"
            user="${USERS[$i]}"
            key="${KEYS[$i]}"
            password_flag="${PASSWORDS[$i]}"
            
            echo "Stopping installation on node $rank ($node)"
            
            # Combine all stop commands into one command
            stop_script=$(IFS=";"; echo "${STOP_COMMANDS[*]}")
            
            remote_exec "$user" "$node" "$key" "$password_flag" "$stop_script" &
        done
        
        wait
        echo "Installation processes stopped on all nodes"
        ;;
    
    *)
        echo "Usage: $0 <config_file> [install|check|status|stop]"
        echo "  install: Install required packages and dependencies"
        echo "  check:   Check current environment status"
        echo "  status:  Check if installation processes are running"
        echo "  stop:    Stop all installation processes"
        exit 1
        ;;
esac