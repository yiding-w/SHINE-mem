# MemoryTest Post-Training Runs

Run these commands from the repository root on the server. The local workspace does not need to execute them.

## Entry Points

The runnable scripts are grouped by responsibility:

```text
MemoryTest/prepare_data/prepare_memory_data.py
MemoryTest/prepare_data/generate_capacity_data.py
MemoryTest/training/run_lora_upper_bound.py
MemoryTest/training/posttrain_shine_memory.py
MemoryTest/evaluation/eval_shine_memory.py
MemoryTest/comparisons/*.py
MemoryTest/scripts/*.sh
```

Prefer `python -m ...` from the repository root so imports resolve consistently on the server.

## 1. Prepare Splits

```bash
python -m MemoryTest.prepare_data.prepare_memory_data \
  --input MemoryTest/json_data/semantic_facts.json \
  --seed 42 \
  --generate-synthetic-train 5000 \
  --generate-synthetic-test 1000
```

Outputs:

```text
MemoryTest/json_data/splits/semantic_train.json
MemoryTest/json_data/splits/semantic_val.json
MemoryTest/json_data/splits/semantic_test.json
MemoryTest/json_data/splits/synthetic_train.json
MemoryTest/json_data/splits/semantic_train_augmented.json
MemoryTest/json_data/splits/synthetic_test.json
MemoryTest/json_data/splits/semantic_test_augmented.json
MemoryTest/json_data/splits/split_meta.json
```

## 2. LoRA Upper Bound

The default selection mode is `head`, so the upper-bound training facts are the first N rows of `semantic_facts.json`, matching the legacy compare scripts.

```bash
python -m MemoryTest.training.run_lora_upper_bound \
  --config MemoryTest/config/case_test.yaml \
  --facts-path MemoryTest/json_data/semantic_facts.json \
  --test-file MemoryTest/json_data/splits/semantic_test.json \
  --selection-mode head \
  --ranks 8 16 32 \
  --num-facts-list 4 8 20 \
  --num-trials 1 \
  --output MemoryTest/results/lora_upper_bound.json
```

Smoke test:

```bash
python -m MemoryTest.training.run_lora_upper_bound \
  --config MemoryTest/config/case_test.yaml \
  --facts-path MemoryTest/json_data/semantic_facts.json \
  --selection-mode head \
  --ranks 8 \
  --num-facts-list 20 \
  --num-trials 1 \
  --epochs 1 \
  --output MemoryTest/results/lora_upper_bound_smoke.json
```

Available:

```bash
python -m MemoryTest.training.run_lora_upper_bound   --config MemoryTest/config/case_test.yaml   --facts-path MemoryTest/json_data/semantic_facts.json   --selection-mode head   --ranks 8   --num-facts-list 20   --num-trials 1   --epochs 20   --batch-size 4   --learning-rate 5e-4   --variants-per-fact 5   --save-loras   --output MemoryTest/results/lora_upper_bound_rank8_20facts_best.json
```

## 3. Evaluate Original SHINE

```bash
python -m MemoryTest.evaluation.eval_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /path/to/original_shine_checkpoint \
  --test-file MemoryTest/json_data/splits/semantic_test_augmented.json \
  --num-facts-list 1 2 4 8 12 20 \
  --num-trials 10 \
  --include-baselines \
  --output MemoryTest/results/shine_original_memory_eval.json

  
python -m MemoryTest.evaluation.eval_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /home/wangyiding/SHINE-mem/checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150/train/checkpoint-epoch-1 \
  --test-file MemoryTest/json_data/splits/semantic_test_augmented.json.json \
  --num-facts-list 1 2 4 8 12 20 \
  --num-trials 5 \
  --output MemoryTest/results/shine_original_memory_eval.json
```

## 4. Post-Train SHINE

```bash
python -m MemoryTest.training.posttrain_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /path/to/original_shine_checkpoint \
  --train-file MemoryTest/json_data/splits/semantic_train_augmented.json \
  --val-file MemoryTest/json_data/splits/semantic_val.json \
  --output-dir MemoryTest/checkpoints/shine_memory_posttrain \
  --fact-counts 1 2 4 8 12 20 \
  --qa-per-context 4 \
  --torch-dtype bf16 \
  --use-gradient-checkpoint \
  --use-contrastive \
  --use-reconstruction

python -m MemoryTest.training.posttrain_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /home/wangyiding/SHINE-mem/checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150/train/checkpoint-epoch-1 \
  --train-file MemoryTest/json_data/splits/semantic_train_augmented.json \
  --val-file MemoryTest/json_data/splits/semantic_val.json \
  --output-dir MemoryTest/checkpoints/shine_memory_posttrain \
  --fact-counts 1 2 4 8 12 20 \
  --qa-per-context 4 \
  --max-steps 2000 \
  --learning-rate 1e-5 \
  --eval-every 500 \
  --use-contrastive \
  --use-reconstruction
```

Single-card 80GB memory-friendly start:

```bash
python -m MemoryTest.training.posttrain_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /home/wangyiding/SHINE-mem/checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150/train/checkpoint-epoch-1 \
  --train-file MemoryTest/json_data/splits/semantic_train_augmented.json \
  --val-file MemoryTest/json_data/splits/semantic_val.json \
  --output-dir MemoryTest/checkpoints/shine_memory_posttrain \
  --fact-counts 1 2 4 8 12 \
  --qa-per-context 4 \
  --max-steps 2000 \
  --learning-rate 1e-5 \
  --eval-every 500 \
  --answer-max-length 256 \
  --context-max-length 1024 \
  --conversation-max-length 512 \
  --use-contrastive \
  --use-reconstruction \
  --torch-dtype bf16 \
  --use-gradient-checkpoint
```

If this fits, add capacity and auxiliary losses back in order: first `--fact-counts 1 2 4 8 12 20`, then `--qa-per-context 2`, then `--use-contrastive`, and finally `--use-reconstruction`.

The script saves:

```text
MemoryTest/checkpoints/shine_memory_posttrain/latest/
MemoryTest/checkpoints/shine_memory_posttrain/best/
MemoryTest/checkpoints/shine_memory_posttrain/shine_posttrain_train_log.jsonl
MemoryTest/checkpoints/shine_memory_posttrain/summary.json
```

Resume:

```bash
python -m MemoryTest.training.posttrain_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /path/to/original_shine_checkpoint \
  --resume \
  --output-dir MemoryTest/checkpoints/shine_memory_posttrain
```

## 5. Evaluate Post-Trained SHINE

```bash
python -m MemoryTest.evaluation.eval_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --baseline-checkpoint-dir /path/to/original_shine_checkpoint \
  --checkpoint-dir MemoryTest/checkpoints/shine_memory_posttrain/best \
  --test-file MemoryTest/json_data/splits/semantic_test_augmented.json \
  --num-facts-list 1 2 4 8 12 20 \
  --num-trials 10 \
  --include-baselines \
  --output MemoryTest/results/shine_posttrained_memory_eval.json
```

## Legacy Compare Scripts

The old compare scripts now live under `MemoryTest/comparisons`:

```bash
python -m MemoryTest.comparisons.compare_update_capacity --merge-method sum
python -m MemoryTest.comparisons.compare_distractor_effect --merge-method sum
python -m MemoryTest.comparisons.compare_density_budget_effect --merge-method sum
python -m MemoryTest.comparisons.compare_baselines
```

Important: `metanetwork.transformer_cfg.num_layers=4` is the number of layers in the M2P/meta-network transformer, not the number of Qwen decoder layers receiving LoRA. The generated LoRA is applied to every Qwen decoder layer at q/k/v/o and gate/up/down projections.
