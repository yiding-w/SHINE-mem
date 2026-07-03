#!/bin/bash
# =============================================================================
# Command List for run_batch.sh
# =============================================================================
# One command per line. Blank lines and lines starting with # are ignored.
#
# Two types of commands are supported (auto-detected):
#   1. GPU training: lines containing "launch_cluster.sh start"
#      -> launches async, polls until training finishes, then cooldown
#   2. Preprocessing/other: any other command (e.g. python scripts)
#      -> runs synchronously, waits for exit, then immediately proceeds
#
# Usage:
#   ./scripts/run_batch.sh ./scripts/run_command_list.sh --nodes all --poll-interval 60
# =============================================================================

# python mydatasets/pretrain/trajectory_all_transfer.py --preprocess --model_path ./models/Qwen3.6-27B/

# ./scripts/launch_cluster.sh start --nodes all --mode pretrain --parallel tp --tp_size 4 --name trajectory_all_transfer_18k_tp_600step_10repo_detachstatenohyper --data pretrain/trajectory_all_transfer --detach_state full_static_hypernetwork_reset_threshold_1.0 --training pretrain/savevram   --optimizer pretrain/lr1e-4 --m2p_transformer full_prenorm_gatedlastnorm --model Qwen3_6-27B

# ./scripts/launch_cluster.sh start --nodes all --mode pretrain --parallel tp --tp_size 4 --name trajectory_all_transfer_18k_tp_600step_10repo_detachstatenormal --data pretrain/trajectory_all_transfer --detach_state full_reset_threshold_1.0 --training pretrain/savevram   --optimizer pretrain/lr1e-4 --m2p_transformer full_prenorm_gatedlastnorm --model Qwen3_6-27B

# ./scripts/launch_cluster.sh start --nodes all --mode pretrain --parallel tp --tp_size 4 --name trajectory_all_transfer_18k_tp_600step_10repo_detachstatewtransform --data pretrain/trajectory_all_transfer --detach_state full_compressedmlp --training pretrain/savevram   --optimizer pretrain/lr1e-4 --m2p_transformer full_prenorm_gatedlastnorm --model Qwen3_6-27B

# ./scripts/launch_cluster.sh start --nodes all --mode pretrain --parallel tp --tp_size 4 --name trajectory_all_transfer_18k_tp_600step_10repo_detachstateresetevery --data pretrain/trajectory_all_transfer --detach_state full_resetevery --training pretrain/savevram   --optimizer pretrain/lr1e-4 --m2p_transformer full_prenorm_gatedlastnorm --model Qwen3_6-27B


# ./scripts/launch_cluster.sh start --nodes 2-3 --mode pretrain --parallel tp --tp_size 4 --name trajectory_all_transfer_v2_16k_tp --data pretrain/trajectory_all_transfer_v2 --detach_state full_reset_threshold_1.0 --training pretrain/savevram   --optimizer pretrain/lr1e-4 --m2p_transformer full_prenorm_gatedlastnorm --model Qwen3_6-27B

# ./scripts/launch_cluster.sh start --nodes all --mode pretrain --parallel tp --tp_size 4 --sp_size 2 --name trajectory_all_transfer_v2_tp_sp --data pretrain/trajectory_all_transfer_v2 --detach_state full_reset_threshold_1.0 --training pretrain/savevram --optimizer pretrain/lr1e-4 --m2p_transformer full_prenorm_gatedlastnorm --model Qwen3_6-27B



 ./scripts/launch_cluster.sh start --nodes all --mode pretrain --parallel tp --tp_size 4 --sp_size 2 --name trajectory_all_transfer_v2_long_tp_sp --data pretrain/trajectory_all_transfer_v2_long --detach_state full_reset_threshold_10.0 --training pretrain/savevram --optimizer pretrain/lr3e-4 --m2p_transformer full_prenorm_gatedlastnorm --model Qwen3_6-27B