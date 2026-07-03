#!/bin/bash
# =============================================================================
# Batch Training Runner (Plan B: Directory-based Queue)
# =============================================================================
# Uses a directory-based queue system with three states:
#   pending/  — jobs waiting to run (you can add/remove/reorder these)
#   running/  — the currently executing job (auto-managed)
#   done/     — successfully completed jobs (auto-managed)
#   failed/   — failed jobs (auto-managed)
#
# Usage:
#   ./scripts/run_batch.sh start [options]       — start the batch runner
#   ./scripts/run_batch.sh stop                  — stop batch and all child processes
#   ./scripts/run_batch.sh status                — show current batch status
#   ./scripts/run_batch.sh list [pending|running|done|failed|all]  — list queue contents
#   ./scripts/run_batch.sh add <command> [--priority <N>]  — add a job to pending queue
#   ./scripts/run_batch.sh remove <job_file_or_number>     — remove a job from pending
#   ./scripts/run_batch.sh reorder <job_file_or_number> <new_position>  — reorder pending job
#   ./scripts/run_batch.sh import <command_list_file>      — import jobs from a file
#   ./scripts/run_batch.sh init                  — initialize queue directories
#   ./scripts/run_batch.sh clear [pending|done|failed|all] — clear queue(s)
#
# The batch runner uses flock-based locking to prevent race conditions:
#   When you run add/remove/reorder commands, the main loop is temporarily
#   blocked until the edit is complete, ensuring no job transitions from
#   pending to running during your edit.
#
# Job file format:
#   Each .job file in pending/ contains the command to execute (one line).
#   Files are named with a numeric prefix for ordering: 001_name.job
#
# Options (for 'start' command):
#   --nodes <spec>              Node spec for status checks (default: all)
#   --poll-interval <seconds>   Seconds between status polls (default: 60)
#   --cooldown <seconds>        Seconds to wait after job ends before next (default: 30)
#   --gpu-idle-threshold <%>    GPU util% below which is idle (default: 10)
#   --foreground                Run in foreground (do not nohup/detach)
# =============================================================================

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# =============================================================================
# Queue directories and lock file
# =============================================================================
QUEUE_DIR="$PROJECT_DIR/logs/.batch_queue"
PENDING_DIR="$QUEUE_DIR/pending"
RUNNING_DIR="$QUEUE_DIR/running"
DONE_DIR="$QUEUE_DIR/done"
FAILED_DIR="$QUEUE_DIR/failed"
LOCK_FILE="$QUEUE_DIR/.queue.lock"
BATCH_STATE_FILE="$QUEUE_DIR/.batch_state"
BATCH_PID_FILE="$QUEUE_DIR/.batch.pid"

# =============================================================================
# Helper: ensure queue directories exist
# =============================================================================
ensure_queue_dirs() {
    mkdir -p "$PENDING_DIR" "$RUNNING_DIR" "$DONE_DIR" "$FAILED_DIR"
    touch "$LOCK_FILE"
}

# =============================================================================
# Helper: acquire queue lock (blocks main loop during edits)
# Usage: exec {LOCK_FD}>$LOCK_FILE; flock $LOCK_FD
# =============================================================================
acquire_lock() {
    exec {LOCK_FD}>"$LOCK_FILE"
    flock "$LOCK_FD"
}

release_lock() {
    flock -u "$LOCK_FD"
    exec {LOCK_FD}>&-
}

# =============================================================================
# Helper: get next available sequence number in pending
# =============================================================================
get_next_seq() {
    local max_seq=0
    for f in "$PENDING_DIR"/*.job; do
        [[ -f "$f" ]] || continue
        local basename=$(basename "$f")
        local seq_str="${basename%%_*}"
        local seq_num
        seq_num=$((10#$seq_str)) || continue
        if (( seq_num > max_seq )); then
            max_seq=$seq_num
        fi
    done
    echo $((max_seq + 1))
}

# =============================================================================
# Helper: list jobs in a directory with formatted output
# =============================================================================
list_queue_dir() {
    local dir="$1"
    local label="$2"
    local count=0

    echo "  ┌─ $label"
    local files=("$dir"/*.job)
    for f in "${files[@]}"; do
        [[ -f "$f" ]] || continue
        count=$((count + 1))
        local bname=$(basename "$f")
        local cmd=$(head -1 "$f" | sed 's/#.*//')
        cmd=$(echo "$cmd" | xargs 2>/dev/null || echo "$cmd")
        # Truncate command for display
        if (( ${#cmd} > 80 )); then
            cmd="${cmd:0:77}..."
        fi
        printf "  │ %3d) %-20s  %s\n" "$count" "$bname" "$cmd"
    done
    if (( count == 0 )); then
        echo "  │  (empty)"
    fi
    echo "  └─ Total: $count"
    echo ""
}

# =============================================================================
# Helper: find a job file by number or filename in pending
# =============================================================================
find_pending_job() {
    local query="$1"
    local files=()
    
    for f in "$PENDING_DIR"/*.job; do
        [[ -f "$f" ]] || continue
        files+=("$f")
    done
    # Sort the array
    IFS=$'\n' files=($(sort <<<"${files[*]}")); unset IFS
    
    if [[ ${#files[@]} -eq 0 ]]; then
        echo ""
        return
    fi
    
    # Try as a number (1-based index)
    if [[ "$query" =~ ^[0-9]+$ ]]; then
        local idx=$((query - 1))
        if (( idx >= 0 && idx < ${#files[@]} )); then
            echo "${files[$idx]}"
            return
        fi
    fi
    
    # Try as filename (exact or partial match)
    for f in "${files[@]}"; do
        local basename=$(basename "$f")
        if [[ "$basename" == "$query" || "$basename" == "${query}.job" ]]; then
            echo "$f"
            return
        fi
    done
    
    # Try partial match
    for f in "${files[@]}"; do
        local basename=$(basename "$f")
        if [[ "$basename" == *"$query"* ]]; then
            echo "$f"
            return
        fi
    done
    
    echo ""
}

# =============================================================================
# Helper: renumber all pending jobs sequentially
# =============================================================================
renumber_pending() {
    local files=()
    for f in "$PENDING_DIR"/*.job; do
        [[ -f "$f" ]] || continue
        files+=("$f")
    done
    IFS=$'\n' files=($(sort <<<"${files[*]}")); unset IFS
    
    local seq=1
    for f in "${files[@]}"; do
        local basename=$(basename "$f")
        # Extract the name part after the sequence number
        local name_part="${basename#*_}"
        if [[ "$name_part" == "$basename" ]]; then
            # No underscore found, use the whole name
            name_part="$basename"
        fi
        local new_name=$(printf "%03d_%s" $seq "$name_part")
        if [[ "$basename" != "$new_name" ]]; then
            mv "$f" "$PENDING_DIR/$new_name"
        fi
        seq=$((seq + 1))
    done
}

# =============================================================================
# Helper: generate a short name from a command
# =============================================================================
cmd_to_name() {
    local cmd="$1"
    local name=""
    
    # Try to extract --name parameter
    if [[ "$cmd" =~ --name[[:space:]]+\"?([^\"[:space:]]+)\"? ]]; then
        name="${BASH_REMATCH[1]}"
    elif [[ "$cmd" =~ trajectory_all_transfer.*--preprocess ]]; then
        name="preprocess"
    elif [[ "$cmd" =~ ([^/[:space:]]+)\.py ]]; then
        name="${BASH_REMATCH[1]}"
    else
        # Use first meaningful word
        name=$(echo "$cmd" | awk '{print $NF}' | sed 's/[^a-zA-Z0-9_-]//g' | head -c 30)
    fi
    
    # Sanitize
    name=$(echo "$name" | sed 's/[^a-zA-Z0-9_-]/_/g' | head -c 50)
    echo "${name:-job}"
}

# =============================================================================
# Subcommand: init
# =============================================================================
cmd_init() {
    local force=false
    if [[ "${1:-}" == "--force" ]]; then
        force=true
    fi

    if [[ "$force" == false && -d "$PENDING_DIR" && -d "$RUNNING_DIR" && -d "$DONE_DIR" && -d "$FAILED_DIR" ]]; then
        echo "Queue already initialized at: $QUEUE_DIR"
        echo "  (use './scripts/run_batch.sh init --force' to destroy and reinitialize)"
        return 0
    fi

    # Clean slate: remove everything and recreate
    rm -rf "$QUEUE_DIR"
    ensure_queue_dirs
    echo "Queue directories initialized at: $QUEUE_DIR"
    echo "  pending/  — add jobs here"
    echo "  running/  — currently executing (auto-managed)"
    echo "  done/     — completed jobs (auto-managed)"
    echo "  failed/   — failed jobs (auto-managed)"
}

# =============================================================================
# Subcommand: list
# =============================================================================
cmd_list() {
    ensure_queue_dirs
    local filter="${1:-all}"
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════════════╗"
    echo "║                        BATCH QUEUE STATUS                                ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    case "$filter" in
        pending)
            list_queue_dir "$PENDING_DIR" "PENDING (waiting to run)"
            ;;
        running)
            list_queue_dir "$RUNNING_DIR" "RUNNING (currently executing)"
            ;;
        done)
            list_queue_dir "$DONE_DIR" "DONE (completed successfully)"
            ;;
        failed)
            list_queue_dir "$FAILED_DIR" "FAILED"
            ;;
        all)
            list_queue_dir "$RUNNING_DIR" "RUNNING (currently executing)"
            list_queue_dir "$PENDING_DIR" "PENDING (waiting to run)"
            list_queue_dir "$DONE_DIR" "DONE (completed successfully)"
            list_queue_dir "$FAILED_DIR" "FAILED"
            ;;
        *)
            echo "Unknown filter: $filter"
            echo "Usage: $0 list [pending|running|done|failed|all]"
            exit 1
            ;;
    esac
}

# =============================================================================
# Subcommand: add
# =============================================================================
cmd_add() {
    ensure_queue_dirs
    
    local command=""
    local priority=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --priority)
                priority="$2"
                shift 2
                ;;
            *)
                if [[ -z "$command" ]]; then
                    command="$1"
                else
                    command="$command $1"
                fi
                shift
                ;;
        esac
    done
    
    if [[ -z "$command" ]]; then
        echo "Error: no command specified."
        echo "Usage: $0 add <command> [--priority <N>]"
        echo "Example: $0 add './scripts/launch_cluster.sh start --nodes all --mode pretrain --name exp1'"
        exit 1
    fi
    
    # Acquire lock to prevent race condition
    acquire_lock
    
    local name=$(cmd_to_name "$command")
    
    if [[ -n "$priority" ]]; then
        # Insert at specific position
        local seq=$(printf "%03d" "$priority")
        local filename="${seq}_${name}.job"
        echo "$command" > "$PENDING_DIR/$filename"
        # Renumber to avoid conflicts
        renumber_pending
    else
        # Append at end
        local next_seq=$(get_next_seq)
        local seq=$(printf "%03d" "$next_seq")
        local filename="${seq}_${name}.job"
        echo "$command" > "$PENDING_DIR/$filename"
    fi
    
    release_lock
    
    echo "✅ Job added to pending queue:"
    echo "   Command: $command"
    echo ""
    echo "Current pending queue:"
    list_queue_dir "$PENDING_DIR" "PENDING"
}

# =============================================================================
# Subcommand: remove
# =============================================================================
cmd_remove() {
    ensure_queue_dirs
    
    local query="${1:-}"
    if [[ -z "$query" ]]; then
        echo "Error: specify job number or filename to remove."
        echo "Usage: $0 remove <number_or_filename>"
        echo ""
        echo "Current pending queue:"
        list_queue_dir "$PENDING_DIR" "PENDING"
        exit 1
    fi
    
    # Acquire lock
    acquire_lock
    
    local job_file=$(find_pending_job "$query")
    if [[ -z "$job_file" || ! -f "$job_file" ]]; then
        release_lock
        echo "Error: job '$query' not found in pending queue."
        echo ""
        echo "Current pending queue:"
        list_queue_dir "$PENDING_DIR" "PENDING"
        exit 1
    fi
    
    local basename=$(basename "$job_file")
    local cmd=$(head -1 "$job_file")
    rm -f "$job_file"
    
    # Renumber remaining
    renumber_pending
    
    release_lock
    
    echo "✅ Job removed from pending queue:"
    echo "   File: $basename"
    echo "   Command: $cmd"
    echo ""
    echo "Current pending queue:"
    list_queue_dir "$PENDING_DIR" "PENDING"
}

# =============================================================================
# Subcommand: reorder
# =============================================================================
cmd_reorder() {
    ensure_queue_dirs
    
    local query="${1:-}"
    local new_pos="${2:-}"
    
    if [[ -z "$query" || -z "$new_pos" ]]; then
        echo "Error: specify job and new position."
        echo "Usage: $0 reorder <number_or_filename> <new_position>"
        echo ""
        echo "Current pending queue:"
        list_queue_dir "$PENDING_DIR" "PENDING"
        exit 1
    fi
    
    # Acquire lock
    acquire_lock
    
    local job_file=$(find_pending_job "$query")
    if [[ -z "$job_file" || ! -f "$job_file" ]]; then
        release_lock
        echo "Error: job '$query' not found in pending queue."
        exit 1
    fi
    
    # Get all pending files sorted
    local files=()
    for f in "$PENDING_DIR"/*.job; do
        [[ -f "$f" ]] || continue
        files+=("$f")
    done
    IFS=$'\n' files=($(sort <<<"${files[*]}")); unset IFS
    
    local total=${#files[@]}
    if (( new_pos < 1 || new_pos > total )); then
        release_lock
        echo "Error: new position must be between 1 and $total."
        exit 1
    fi
    
    # Remove the job from the array, then insert at new position
    local temp_file=$(mktemp)
    cp "$job_file" "$temp_file"
    rm -f "$job_file"
    
    # Renumber without the removed job
    renumber_pending
    
    # Now insert at the desired position
    # Get current files again
    files=()
    for f in "$PENDING_DIR"/*.job; do
        [[ -f "$f" ]] || continue
        files+=("$f")
    done
    IFS=$'\n' files=($(sort <<<"${files[*]}")); unset IFS
    
    # Extract name part from original file
    local orig_basename=$(basename "$job_file")
    local name_part="${orig_basename#*_}"
    
    # Shift files at and after new_pos to make room
    local new_total=${#files[@]}
    for (( j=new_total-1; j>=new_pos-1; j-- )); do
        local f="${files[$j]}"
        local fb=$(basename "$f")
        local np="${fb#*_}"
        local new_seq=$(printf "%03d" $((j + 2)))
        mv "$f" "$PENDING_DIR/${new_seq}_${np}"
    done
    
    # Place the job at new_pos
    local insert_seq=$(printf "%03d" "$new_pos")
    mv "$temp_file" "$PENDING_DIR/${insert_seq}_${name_part}"
    
    # Final renumber to clean up
    renumber_pending
    
    release_lock
    
    echo "✅ Job reordered to position $new_pos."
    echo ""
    echo "Current pending queue:"
    list_queue_dir "$PENDING_DIR" "PENDING"
}

# =============================================================================
# Subcommand: import (from a command list file)
# =============================================================================
cmd_import() {
    ensure_queue_dirs
    
    local file="${1:-}"
    if [[ -z "$file" || ! -f "$file" ]]; then
        echo "Error: specify a valid command list file."
        echo "Usage: $0 import <command_list_file>"
        exit 1
    fi
    
    # Acquire lock
    acquire_lock
    
    local count=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip blank lines and comments
        # Only strip leading # comments (lines starting with #)
        if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "$(echo "$line" | xargs 2>/dev/null)" ]]; then
            continue
        fi
        line=$(echo "$line" | xargs 2>/dev/null || echo "$line")
        if [[ -n "$line" ]]; then
            local name=$(cmd_to_name "$line")
            local next_seq=$(get_next_seq)
            local seq=$(printf "%03d" "$next_seq")
            local filename="${seq}_${name}.job"
            echo "$line" > "$PENDING_DIR/$filename"
            count=$((count + 1))
        fi
    done < "$file"
    
    release_lock
    
    echo "✅ Imported $count jobs from: $file"
    echo ""
    echo "Current pending queue:"
    list_queue_dir "$PENDING_DIR" "PENDING"
}

# =============================================================================
# Subcommand: clear
# =============================================================================
cmd_clear() {
    ensure_queue_dirs
    local target="${1:-}"
    
    if [[ -z "$target" ]]; then
        echo "Usage: $0 clear [pending|done|failed|all]"
        exit 1
    fi
    
    # Acquire lock for pending operations
    acquire_lock
    
    case "$target" in
        pending)
            rm -f "$PENDING_DIR"/*.job 2>/dev/null
            echo "✅ Cleared pending queue."
            ;;
        done)
            rm -f "$DONE_DIR"/*.job 2>/dev/null
            echo "✅ Cleared done queue."
            ;;
        failed)
            rm -f "$FAILED_DIR"/*.job 2>/dev/null
            echo "✅ Cleared failed queue."
            ;;
        all)
            rm -f "$PENDING_DIR"/*.job 2>/dev/null
            rm -f "$DONE_DIR"/*.job 2>/dev/null
            rm -f "$FAILED_DIR"/*.job 2>/dev/null
            echo "✅ Cleared all queues (pending, done, failed)."
            ;;
        *)
            release_lock
            echo "Unknown target: $target"
            echo "Usage: $0 clear [pending|done|failed|all]"
            exit 1
            ;;
    esac
    
    release_lock
}

# =============================================================================
# Subcommand: status
# =============================================================================
cmd_status() {
    ensure_queue_dirs
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════════════╗"
    echo "║                      BATCH RUN STATUS                                    ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Check if batch is running
    local batch_alive=false
    if [[ -f "$BATCH_PID_FILE" ]]; then
        local pid=$(cat "$BATCH_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            batch_alive=true
        fi
    fi
    
    if [[ "$batch_alive" == "true" ]]; then
        echo "  Batch Status: ✅ RUNNING (PID: $pid)"
    else
        echo "  Batch Status: ❌ NOT RUNNING"
    fi
    
    # Read state file if exists
    if [[ -f "$BATCH_STATE_FILE" ]]; then
        source "$BATCH_STATE_FILE"
        echo "  Started:      ${BATCH_START_TIME_STR:-N/A}"
        echo "  Log file:     ${BATCH_LOG_FILE:-N/A}"
        if [[ -n "${CURRENT_JOB_START_EPOCH:-}" && "$batch_alive" == "true" ]]; then
            local elapsed=$(( $(date +%s) - CURRENT_JOB_START_EPOCH ))
            local hours=$((elapsed / 3600))
            local mins=$(( (elapsed % 3600) / 60 ))
            local secs=$((elapsed % 60))
            printf "  Job elapsed:  %02d:%02d:%02d\n" $hours $mins $secs
        fi
    fi
    echo ""
    
    # Count jobs in each queue
    local pending_count=0 running_count=0 done_count=0 failed_count=0
    for _f in "$PENDING_DIR"/*.job; do [[ -f "$_f" ]] && pending_count=$((pending_count+1)); done
    for _f in "$RUNNING_DIR"/*.job; do [[ -f "$_f" ]] && running_count=$((running_count+1)); done
    for _f in "$DONE_DIR"/*.job; do [[ -f "$_f" ]] && done_count=$((done_count+1)); done
    for _f in "$FAILED_DIR"/*.job; do [[ -f "$_f" ]] && failed_count=$((failed_count+1)); done
    
    echo "  Queue Summary:"
    echo "    Pending:  $pending_count"
    echo "    Running:  $running_count"
    echo "    Done:     $done_count"
    echo "    Failed:   $failed_count"
    echo ""
    
    # Show running job
    if (( running_count > 0 )); then
        echo "  Currently Running:"
        for f in "$RUNNING_DIR"/*.job; do
            [[ -f "$f" ]] || continue
            local cmd=$(head -1 "$f")
            echo "    $(basename "$f"): $cmd"
        done
        echo ""
    fi
    
    # Show next pending
    if (( pending_count > 0 )); then
        local next=""
        for _f in "$PENDING_DIR"/*.job; do
            if [[ -f "$_f" ]]; then next="$_f"; break; fi
        done
        if [[ -n "$next" ]]; then
            echo "  Next up: $(basename "$next")"
            echo "    Command: $(head -1 "$next")"
        fi
        echo ""
    fi
    
    # Show related processes (concise summary)
    echo "  Related processes:"
    local launch_pid=$(ps -ef | grep "[l]aunch_cluster\.sh start" | awk 'NR==1{print $2}')
    local train_count=$(ps -ef | grep "[m]eta_train\.py" | wc -l)
    local torchrun_pid=$(ps -ef | grep "[t]orchrun.*meta_train" | awk 'NR==1{print $2}')
    local train_cmd=$(ps -ef | grep "[t]orchrun.*meta_train" | head -1 | sed 's/^.*meta_train.py/meta_train.py/')

    if [[ -n "$launch_pid" ]]; then
        echo "    launch_cluster PID: $launch_pid"
    fi
    if [[ -n "$torchrun_pid" ]]; then
        echo "    torchrun PID: $torchrun_pid"
        if [[ -n "$train_cmd" ]]; then
            echo "    training args: $train_cmd"
        fi
    fi
    if (( train_count > 0 )); then
        echo "    meta_train.py workers: $train_count processes"
    fi
    if [[ -z "$launch_pid" && -z "$torchrun_pid" && "$train_count" -eq 0 ]]; then
        echo "    (none)"
    fi
    echo ""
}

# =============================================================================
# Subcommand: stop
# =============================================================================
cmd_stop() {
    ensure_queue_dirs
    echo "Stopping batch run and all child processes..."
    echo ""
    
    local stopped_something=false
    
    # 1. Kill batch main process
    if [[ -f "$BATCH_PID_FILE" ]]; then
        local pid=$(cat "$BATCH_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Killing batch main process (PID $pid)..."
            kill -9 "$pid" 2>/dev/null || true
            stopped_something=true
        fi
        rm -f "$BATCH_PID_FILE"
    fi
    
    # 2. Move running jobs back to pending (at the front)
    for f in "$RUNNING_DIR"/*.job; do
        [[ -f "$f" ]] || continue
        local basename=$(basename "$f")
        mv "$f" "$PENDING_DIR/000_${basename}"
        echo "  Moved running job back to pending: $basename"
    done
    renumber_pending
    
    # 3. Kill preprocessing processes
    local preprocess_pids=$(pgrep -f "trajectory_all_transfer.py.*--preprocess" 2>/dev/null || true)
    if [[ -n "$preprocess_pids" ]]; then
        echo "  Killing preprocessing processes..."
        while read pid; do
            [[ -z "$pid" ]] && continue
            local pgid=$(ps -o pgid= -p $pid 2>/dev/null | tr -d ' ')
            if [[ -n "$pgid" && "$pgid" != "0" && "$pgid" != "1" ]]; then
                kill -9 -"$pgid" 2>/dev/null || true
            fi
            kill -9 "$pid" 2>/dev/null || true
        done <<< "$preprocess_pids"
        stopped_something=true
    fi
    
    # 4. Stop training cluster
    if pgrep -f "meta_train\.py" >/dev/null 2>&1 || pgrep -f "torchrun.*meta_train" >/dev/null 2>&1; then
        echo "  Stopping training cluster..."
        ./scripts/launch_cluster.sh stop --nodes all 2>/dev/null || true
        sleep 2
        pkill -9 -f "meta_train\.py" 2>/dev/null || true
        pkill -9 -f "torchrun.*meta_train" 2>/dev/null || true
        stopped_something=true
    fi
    
    # 5. Kill any remaining launch_cluster start processes
    pkill -9 -f "launch_cluster.sh start" 2>/dev/null || true
    
    # 6. Clean up state
    rm -f "$BATCH_STATE_FILE"
    
    if [[ "$stopped_something" == "true" ]]; then
        echo ""
        echo "  ✅ All batch processes stopped."
    else
        echo "  No batch processes found running."
    fi
    
    # Verify
    sleep 2
    local remaining=$(ps -ef | grep -E "trajectory_all_transfer|meta_train\.py" | grep -v grep | wc -l)
    if (( remaining > 0 )); then
        echo ""
        echo "  Some processes still alive, retrying kill..."
        pkill -9 -f "trajectory_all_transfer" 2>/dev/null || true
        pkill -9 -f "meta_train\.py" 2>/dev/null || true
        sleep 2
        remaining=$(ps -ef | grep -E "trajectory_all_transfer|meta_train\.py" | grep -v grep | wc -l)
        if (( remaining > 0 )); then
            echo "  ⚠️  WARNING: $remaining related processes still alive."
            ps -ef | grep -E "trajectory_all_transfer|meta_train\.py" | grep -v grep | awk '{printf "    PID %-8s %s\n", $2, substr($0, index($0,$8))}' | head -10
        else
            echo "  ✅ All remaining processes killed on retry."
        fi
    fi
}

# =============================================================================
# Subcommand: start (main batch loop)
# =============================================================================
cmd_start() {
    ensure_queue_dirs
    
    # Check if already running (skip if we are the nohup-relaunched instance)
    if [[ "${_RUN_BATCH_NOHUP:-}" != "1" ]] && [[ -f "$BATCH_PID_FILE" ]]; then
        local existing_pid=$(cat "$BATCH_PID_FILE")
        if [[ "$existing_pid" != "$$" ]] && kill -0 "$existing_pid" 2>/dev/null; then
            echo "Error: Batch runner is already running (PID $existing_pid)."
            echo "Use './scripts/run_batch.sh stop' to stop it first."
            exit 1
        fi
    fi
    
    # Parse start options
    local POLL_INTERVAL=60
    local NODES_SPEC="all"
    local GPU_IDLE_THRESHOLD=10
    local COOLDOWN=30
    local FOREGROUND=false
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --nodes)
                NODES_SPEC="$2"
                shift 2
                ;;
            --poll-interval)
                POLL_INTERVAL="$2"
                shift 2
                ;;
            --cooldown)
                COOLDOWN="$2"
                shift 2
                ;;
            --gpu-idle-threshold)
                GPU_IDLE_THRESHOLD="$2"
                shift 2
                ;;
            --foreground)
                FOREGROUND=true
                shift
                ;;
            *)
                echo "Unknown option for start: $1"
                exit 1
                ;;
        esac
    done
    
    # Auto nohup if not in foreground mode
    if [[ "${_RUN_BATCH_NOHUP:-}" != "1" && "$FOREGROUND" == "false" ]]; then
        local _TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        mkdir -p "$PROJECT_DIR/logs"
        local _NOHUP_LOG="$PROJECT_DIR/logs/batch_run_${_TIMESTAMP}.log"
        
        echo "Starting batch runner in background (nohup)..."
        echo "Log file: $_NOHUP_LOG"
        echo "Monitor with: tail -f $_NOHUP_LOG"
        echo ""
        
        export _RUN_BATCH_NOHUP=1
        export _RUN_BATCH_TIMESTAMP="$_TIMESTAMP"
        export _RUN_BATCH_POLL_INTERVAL="$POLL_INTERVAL"
        export _RUN_BATCH_NODES_SPEC="$NODES_SPEC"
        export _RUN_BATCH_GPU_IDLE_THRESHOLD="$GPU_IDLE_THRESHOLD"
        export _RUN_BATCH_COOLDOWN="$COOLDOWN"
        nohup bash "${BASH_SOURCE[0]}" start --foreground > "$_NOHUP_LOG" 2>&1 &
        local BATCH_PID=$!
        echo "$BATCH_PID" > "$BATCH_PID_FILE"
        
        echo "Batch runner PID: $BATCH_PID"
        echo "To stop: ./scripts/run_batch.sh stop"
        
        sleep 0.5
        if kill -0 "$BATCH_PID" 2>/dev/null; then
            echo "Check status: ./scripts/run_batch.sh status"
            echo "List queue:   ./scripts/run_batch.sh list"
        fi
        exit 0
    fi
    
    # Restore options from environment (when re-exec'd via nohup)
    if [[ -n "${_RUN_BATCH_POLL_INTERVAL:-}" ]]; then
        POLL_INTERVAL="$_RUN_BATCH_POLL_INTERVAL"
        NODES_SPEC="$_RUN_BATCH_NODES_SPEC"
        GPU_IDLE_THRESHOLD="$_RUN_BATCH_GPU_IDLE_THRESHOLD"
        COOLDOWN="$_RUN_BATCH_COOLDOWN"
    fi
    
    # =========================================================================
    # Setup logging
    # =========================================================================
    local TIMESTAMP="${_RUN_BATCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
    local LOG_DIR="$PROJECT_DIR/logs"
    mkdir -p "$LOG_DIR"
    local BATCH_LOG="$LOG_DIR/batch_run_${TIMESTAMP}.log"
    
    # Save PID
    echo $$ > "$BATCH_PID_FILE"
    
    log() {
        local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
        echo "$msg"
    }
    
    log_separator() {
        echo "============================================================================="
    }
    
    # =========================================================================
    # Helper: write state file
    # =========================================================================
    write_state() {
        cat > "$BATCH_STATE_FILE" <<-STATEEOF
BATCH_PID=$$
BATCH_START_TIME_STR="${BATCH_START_TIME_STR:-$(date '+%Y-%m-%d %H:%M:%S')}"
BATCH_LOG_FILE="$BATCH_LOG"
CURRENT_JOB_FILE="${CURRENT_JOB_FILE:-}"
CURRENT_JOB_CMD="${CURRENT_JOB_CMD:-}"
CURRENT_JOB_TYPE="${CURRENT_JOB_TYPE:-}"
CURRENT_JOB_START_EPOCH="${CURRENT_JOB_START_EPOCH:-}"
STATEEOF
    }
    
    # =========================================================================
    # Helper: check if training is running
    # =========================================================================
    is_training_running() {
        local status_output
        status_output=$(./scripts/launch_cluster.sh status --nodes "$NODES_SPEC" 2>&1) || true
        if echo "$status_output" | grep -q "RUNNING"; then
            return 0
        else
            return 1
        fi
    }
    
    # =========================================================================
    # Helper: check if GPUs are idle
    # =========================================================================
    are_gpus_idle() {
        local status_output
        status_output=$(./scripts/launch_cluster.sh status --nodes "$NODES_SPEC" --verbose 2>&1) || true
        
        local max_util=0
        while IFS= read -r line; do
            if [[ "$line" =~ ([0-9]+)%\ util ]]; then
                local util="${BASH_REMATCH[1]}"
                if (( util > max_util )); then
                    max_util=$util
                fi
            fi
        done <<< "$status_output"
        
        if (( max_util < GPU_IDLE_THRESHOLD )); then
            return 0
        else
            return 1
        fi
    }
    
    # =========================================================================
    # Helper: wait for training to complete
    # =========================================================================
    wait_for_completion() {
        local job_name="$1"
        local start_time="$2"
        
        sleep 10
        
        local retries=0
        while ! is_training_running; do
            retries=$((retries + 1))
            if (( retries > 6 )); then
                log "  WARNING: Job '$job_name' never started (no RUNNING process after 60s)"
                return 1
            fi
            sleep 10
        done
        
        log "  Job '$job_name' confirmed running. Polling every ${POLL_INTERVAL}s..."
        
        while is_training_running; do
            local elapsed=$(( $(date +%s) - start_time ))
            local hours=$((elapsed / 3600))
            local mins=$(( (elapsed % 3600) / 60 ))
            printf "\r  [Running] Elapsed: %02d:%02d:%02d" $hours $mins $((elapsed % 60))
            sleep "$POLL_INTERVAL"
        done
        echo ""
        
        return 0
    }
    
    # =========================================================================
    # Helper: check job exit status from logs
    # =========================================================================
    check_job_exit_status() {
        local log_dir="$PROJECT_DIR/logs/$NODES_SPEC"
        
        if [[ ! -d "$log_dir" ]]; then
            echo "UNKNOWN"
            return
        fi
        
        local node_log="$log_dir/node_0.log"
        if [[ ! -f "$node_log" ]]; then
            echo "UNKNOWN"
            return
        fi
        
        local tail_content
        tail_content=$(tail -50 "$node_log" 2>/dev/null) || true
        
        if echo "$tail_content" | grep -qiE "error|exception|traceback|NCCL.*error|OOM|killed|segfault|SIGKILL|SIGTERM"; then
            if echo "$tail_content" | grep -qiE "training complete|training finished|all steps done|saving final"; then
                echo "SUCCESS (with warnings)"
            else
                echo "FAILED"
            fi
        elif echo "$tail_content" | grep -qiE "training complete|training finished|all steps done|saving final|checkpoint saved"; then
            echo "SUCCESS"
        else
            echo "UNKNOWN"
        fi
    }
    
    # =========================================================================
    # Pre-flight: ensure no training is currently running
    # =========================================================================
    if is_training_running; then
        log "WARNING: Training is currently running on nodes '$NODES_SPEC'."
        log "Waiting for existing training to finish before starting batch..."
        while is_training_running; do
            sleep "$POLL_INTERVAL"
        done
        log "Existing training finished. Proceeding with batch."
        sleep "$COOLDOWN"
    fi
    
    # =========================================================================
    # Main loop: process jobs from pending queue
    # =========================================================================
    local BATCH_START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
    local BATCH_START_EPOCH=$(date +%s)
    local COMPLETED=0
    local FAILED_COUNT=0
    
    log "Batch runner started."
    log "Nodes spec: $NODES_SPEC | Poll interval: ${POLL_INTERVAL}s | Cooldown: ${COOLDOWN}s"
    log "Queue dir: $QUEUE_DIR"
    log "Log file: $BATCH_LOG"
    log_separator
    
    write_state
    
    while true; do
        # Acquire lock to safely pick next job
        exec {LOCK_FD}>"$LOCK_FILE"
        flock "$LOCK_FD"
        
        # Find next pending job (sorted by filename)
        local next_job=""
        for f in "$PENDING_DIR"/*.job; do
            if [[ -f "$f" ]]; then
                next_job="$f"
                break
            fi
        done
        
        if [[ -z "$next_job" ]]; then
            # No more jobs
            flock -u "$LOCK_FD"
            exec {LOCK_FD}>&-
            break
        fi
        
        # Move job to running
        local job_basename=$(basename "$next_job")
        mv "$next_job" "$RUNNING_DIR/$job_basename"
        local job_file="$RUNNING_DIR/$job_basename"
        
        # Release lock — other commands can now safely edit pending/
        flock -u "$LOCK_FD"
        exec {LOCK_FD}>&-
        
        # Read command from job file
        local job_cmd=$(head -1 "$job_file" | sed 's/#.*//' | xargs)
        if [[ -z "$job_cmd" ]]; then
            # Empty job file, move to done and continue
            mv "$job_file" "$DONE_DIR/$job_basename"
            continue
        fi
        
        log_separator
        log "JOB [$job_basename] Starting..."
        log "  Command: $job_cmd"
        
        local job_start=$(date +%s)
        local job_start_str=$(date '+%Y-%m-%d %H:%M:%S')
        
        # Update state
        CURRENT_JOB_FILE="$job_basename"
        CURRENT_JOB_CMD="$job_cmd"
        CURRENT_JOB_START_EPOCH=$job_start
        
        # Detect command type
        if [[ "$job_cmd" == *"launch_cluster.sh start"* ]]; then
            # =============================================================
            # GPU Training command
            # =============================================================
            log "  Type: GPU training (async poll)"
            CURRENT_JOB_TYPE="GPU training"
            write_state
            
            eval "$job_cmd" 2>&1
            local launch_exit_code=$?
            
            if [[ $launch_exit_code -ne 0 ]]; then
                log "  ERROR: Launch command failed with exit code $launch_exit_code"
                # Append exit info to job file
                echo "# EXIT: LAUNCH_FAILED (code $launch_exit_code) at $(date '+%Y-%m-%d %H:%M:%S')" >> "$job_file"
                mv "$job_file" "$FAILED_DIR/$job_basename"
                FAILED_COUNT=$((FAILED_COUNT + 1))
                continue
            fi
            
            wait_for_completion "$job_basename" "$job_start"
            local wait_result=$?
            
            local job_end=$(date +%s)
            local duration=$((job_end - job_start))
            local hours=$((duration / 3600))
            local mins=$(( (duration % 3600) / 60 ))
            local secs=$((duration % 60))
            local duration_str=$(printf "%02d:%02d:%02d" $hours $mins $secs)
            
            local status=""
            if [[ $wait_result -ne 0 ]]; then
                status="FAILED (never started)"
            else
                status=$(check_job_exit_status)
            fi
            
            log "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
            log "  Duration: $duration_str ($duration seconds)"
            log "  Status:   $status"
            
            # Append result to job file
            echo "# RESULT: $status | Duration: $duration_str | Finished: $(date '+%Y-%m-%d %H:%M:%S')" >> "$job_file"
            
            if [[ "$status" == *"FAILED"* ]]; then
                mv "$job_file" "$FAILED_DIR/$job_basename"
                FAILED_COUNT=$((FAILED_COUNT + 1))
            else
                mv "$job_file" "$DONE_DIR/$job_basename"
                COMPLETED=$((COMPLETED + 1))
            fi
            
            # Cooldown
            local pending_remain=0
            for _f in "$PENDING_DIR"/*.job; do [[ -f "$_f" ]] && pending_remain=$((pending_remain+1)); done
            if (( pending_remain > 0 )); then
                log "  Cooling down for ${COOLDOWN}s before next job..."
                sleep "$COOLDOWN"
                
                if ! are_gpus_idle; then
                    log "  GPUs still busy, waiting for idle state..."
                    while ! are_gpus_idle; do
                        sleep 10
                    done
                    log "  GPUs are now idle."
                fi
            fi
        else
            # =============================================================
            # Preprocessing/sync command
            # =============================================================
            log "  Type: Preprocessing/sync command"
            CURRENT_JOB_TYPE="Preprocessing/sync"
            write_state
            
            eval "$job_cmd" 2>&1 &
            local cmd_pid=$!
            wait $cmd_pid 2>/dev/null
            local cmd_exit_code=$?
            
            local job_end=$(date +%s)
            local duration=$((job_end - job_start))
            local hours=$((duration / 3600))
            local mins=$(( (duration % 3600) / 60 ))
            local secs=$((duration % 60))
            local duration_str=$(printf "%02d:%02d:%02d" $hours $mins $secs)
            
            local status=""
            if [[ $cmd_exit_code -eq 0 ]]; then
                status="SUCCESS"
            else
                status="FAILED (exit code: $cmd_exit_code)"
            fi
            
            log "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
            log "  Duration: $duration_str ($duration seconds)"
            log "  Status:   $status"
            
            # Append result to job file
            echo "# RESULT: $status | Duration: $duration_str | Finished: $(date '+%Y-%m-%d %H:%M:%S')" >> "$job_file"
            
            if [[ $cmd_exit_code -eq 0 ]]; then
                mv "$job_file" "$DONE_DIR/$job_basename"
                COMPLETED=$((COMPLETED + 1))
            else
                mv "$job_file" "$FAILED_DIR/$job_basename"
                FAILED_COUNT=$((FAILED_COUNT + 1))
            fi
        fi
        
        # Update state
        CURRENT_JOB_FILE=""
        CURRENT_JOB_CMD=""
        CURRENT_JOB_START_EPOCH=""
        write_state
    done
    
    # =========================================================================
    # Summary
    # =========================================================================
    local BATCH_END_EPOCH=$(date +%s)
    local BATCH_DURATION=$((BATCH_END_EPOCH - BATCH_START_EPOCH))
    local batch_hours=$((BATCH_DURATION / 3600))
    local batch_mins=$(( (BATCH_DURATION % 3600) / 60 ))
    local batch_secs=$((BATCH_DURATION % 60))
    
    log_separator
    log ""
    log "╔═══════════════════════════════════════════════════════════════════════════╗"
    log "║                        BATCH RUN COMPLETE                                 ║"
    log "╚═══════════════════════════════════════════════════════════════════════════╝"
    log ""
    log "Total batch duration: $(printf '%02d:%02d:%02d' $batch_hours $batch_mins $batch_secs)"
    log "Completed: $COMPLETED | Failed: $FAILED_COUNT"
    log ""
    
    # Show done/failed summary
    log "Completed jobs:"
    for f in "$DONE_DIR"/*.job; do
        [[ -f "$f" ]] || continue
        local result_line=$(grep "^# RESULT:" "$f" | tail -1)
        log "  ✅ $(basename "$f") ${result_line#\# RESULT: }"
    done
    
    if (( FAILED_COUNT > 0 )); then
        log ""
        log "Failed jobs:"
    for f in "$FAILED_DIR"/*.job; do
            [[ -f "$f" ]] || continue
            local result_line=$(grep "^# RESULT:\|^# EXIT:" "$f" | tail -1)
            log "  ❌ $(basename "$f") ${result_line#\# }"
        done
    fi
    
    log ""
    log "Full log: $BATCH_LOG"
    log_separator
    
    # Clean up
    rm -f "$BATCH_STATE_FILE"
    rm -f "$BATCH_PID_FILE"
    
    if (( FAILED_COUNT > 0 )); then
        exit 1
    fi
    exit 0
}

# =============================================================================
# Main: dispatch subcommand
# =============================================================================
SUBCOMMAND="${1:-}"

if [[ -z "$SUBCOMMAND" ]]; then
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start [options]              Start the batch runner"
    echo "  stop                         Stop batch and all child processes"
    echo "  status                       Show current batch status"
    echo "  list [pending|running|done|failed|all]  List queue contents"
    echo "  add <command> [--priority N] Add a job to pending queue"
    echo "  remove <number_or_name>      Remove a job from pending queue"
    echo "  reorder <number_or_name> <new_pos>  Move a pending job to new position"
    echo "  import <command_list_file>   Import jobs from a file"
    echo "  init                         Initialize queue directories"
    echo "  clear [pending|done|failed|all]  Clear queue(s)"
    echo ""
    echo "Examples:"
    echo "  $0 init"
    echo "  $0 import ./scripts/run_command_list.sh"
    echo "  $0 start --nodes all --poll-interval 60"
    echo "  $0 list"
    echo "  $0 add './scripts/launch_cluster.sh start --nodes all --name exp3'"
    echo "  $0 remove 2"
    echo "  $0 reorder 3 1"
    echo "  $0 status"
    echo "  $0 stop"
    exit 0
fi

shift

case "$SUBCOMMAND" in
    init)
        cmd_init "$@"
        ;;
    start)
        cmd_start "$@"
        ;;
    stop)
        cmd_stop
        ;;
    status)
        cmd_status
        ;;
    list)
        cmd_list "$@"
        ;;
    add)
        cmd_add "$@"
        ;;
    remove)
        cmd_remove "$@"
        ;;
    reorder)
        cmd_reorder "$@"
        ;;
    import)
        cmd_import "$@"
        ;;
    clear)
        cmd_clear "$@"
        ;;
    *)
        echo "Unknown command: $SUBCOMMAND"
        echo "Run '$0' without arguments to see usage."
        exit 1
        ;;
esac
