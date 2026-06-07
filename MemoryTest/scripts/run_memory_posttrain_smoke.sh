#!/bin/bash

python -m MemoryTest.prepare_data.prepare_memory_data \
  --input MemoryTest/json_data/semantic_facts.json \
  --seed 42 \
  --generate-synthetic-train 100

python -m MemoryTest.training.run_lora_upper_bound \
  --config MemoryTest/config/case_test.yaml \
  --facts-path MemoryTest/json_data/semantic_facts.json \
  --test-file MemoryTest/json_data/splits/semantic_test.json \
  --selection-mode head \
  --ranks 8 \
  --num-facts-list 4 \
  --num-trials 1 \
  --epochs 1 \
  --output MemoryTest/results/lora_upper_bound_smoke.json
