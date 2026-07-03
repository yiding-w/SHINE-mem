#!/bin/bash
# Run the multilayer memory benchmark in background
cd /apdcephfs_zwfy/share_303937731/xiyuanwang/liuyewei/SHINE_V2_tmp
torchrun --nproc_per_node=8 --master_port=29524 tests/test_multilayer_memory.py > tests/benchmark_results.txt 2>&1
echo "BENCHMARK_COMPLETE" >> tests/benchmark_results.txt
