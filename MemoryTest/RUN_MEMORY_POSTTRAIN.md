# MemoryTest Post-Training Runs

Run these commands from the repository root on the server. The local workspace does not need to execute them.

## Entry Points

The runnable scripts are grouped by responsibility:

```text
MemoryTest/prepare_data/prepare_memory_data.py
MemoryTest/prepare_data/generate_capacity_data.py
MemoryTest/training/run_lora_upper_bound.py
MemoryTest/training/run_shine_initialized_lora_sft.py
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

QA-SFT upper bound directly trains on question-answer records:

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

NTP upper bound trains only on fact/context text and still evaluates with QA generation:

```bash
python -m MemoryTest.training.run_lora_upper_bound \
  --config MemoryTest/config/case_test.yaml \
  --facts-path MemoryTest/json_data/semantic_facts.json \
  --test-file MemoryTest/json_data/splits/semantic_test_augmented.json \
  --selection-mode head \
  --training-objective ntp \
  --ntp-record-mode both \
  --ntp-context-format mixed \
  --ntp-context-variants 5 \
  --ranks 8 \
  --num-facts-list 4 8 20 \
  --num-trials 1 \
  --epochs 20 \
  --batch-size 2 \
  --learning-rate 5e-4 \
  --output MemoryTest/results/lora_upper_bound_ntp.json
```

Smoke test:

```bash
python -m MemoryTest.training.run_lora_upper_bound \
  --config MemoryTest/config/case_test.yaml \
  --facts-path MemoryTest/json_data/semantic_facts.json \
  --selection-mode head \
  --ranks 8 \
  --num-facts-list 4 \
  --num-trials 1 \
  --epochs 1 \
  --output MemoryTest/results/lora_upper_bound_smoke.json
```

## 3. SHINE-Initialized LoRA SFT

This freezes SHINE and Qwen, generates `LoRA = SHINE(context)`, then updates only that LoRA dictionary for a small QA-SFT or NTP run. It measures whether SHINE is a better initialization than a random LoRA upper bound.

Short QA-SFT adaptation:

```bash
python -m MemoryTest.training.run_shine_initialized_lora_sft \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /path/to/original_shine_checkpoint \
  --facts-path MemoryTest/json_data/semantic_facts.json \
  --selection-mode head \
  --training-objective qa_sft \
  --num-facts-list 20 \
  --num-trials 1 \
  --epochs 3 \
  --batch-size 2 \
  --learning-rate 5e-4 \
  --output MemoryTest/results/shine_initialized_lora_qa_sft.json
```

Short NTP adaptation:

```bash
python -m MemoryTest.training.run_shine_initialized_lora_sft \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /path/to/original_shine_checkpoint \
  --facts-path MemoryTest/json_data/semantic_facts.json \
  --selection-mode head \
  --training-objective ntp \
  --ntp-record-mode both \
  --ntp-context-format mixed \
  --ntp-context-variants 3 \
  --num-facts-list 20 \
  --num-trials 1 \
  --epochs 3 \
  --batch-size 2 \
  --learning-rate 5e-4 \
  --output MemoryTest/results/shine_initialized_lora_ntp_sft.json
```

The output has both `shine_init_result` and `adapted_train_result`. `train_stats.best_epoch` tells which adaptation epoch was best.

## 4. Evaluate Original SHINE

```bash
python -m MemoryTest.evaluation.eval_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /path/to/original_shine_checkpoint \
  --test-file MemoryTest/json_data/splits/semantic_test_augmented.json \
  --num-facts-list 1 2 4 8 12 20 \
  --num-trials 10 \
  --include-baselines \
  --output MemoryTest/results/shine_original_memory_eval.json
```

## 5. Post-Train SHINE

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
```

Single-card 80GB memory-friendly start:

```bash
python -m MemoryTest.training.posttrain_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /path/to/original_shine_checkpoint \
  --train-file MemoryTest/json_data/splits/semantic_train_augmented.json \
  --val-file MemoryTest/json_data/splits/semantic_val.json \
  --output-dir MemoryTest/checkpoints/shine_memory_posttrain \
  --fact-counts 1 2 4 8 12 \
  --qa-per-context 1 \
  --answer-max-length 256 \
  --context-max-length 768 \
  --torch-dtype bf16 \
  --use-gradient-checkpoint
```

If this fits, add capacity and auxiliary losses back in order: first `--fact-counts 1 2 4 8 12 20`, then `--qa-per-context 2`, then `--use-contrastive`, and finally `--use-reconstruction`.

`--use-gradient-checkpoint` is applied to the context-to-LoRA generation path. The supervised answer/contrastive/reconstruction forward uses the generated LoRA and keeps checkpointing off by default, because per-layer checkpointing can backprop through the same generated LoRA graph multiple times. Only enable `--use-answer-gradient-checkpoint` for debugging experiments.

If loss becomes NaN, start more conservatively by freezing the loaded Meta-LoRA and clamping generated LoRA values:

```bash
python -m MemoryTest.training.posttrain_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /path/to/original_shine_checkpoint \
  --train-file MemoryTest/json_data/splits/semantic_train_augmented.json \
  --val-file MemoryTest/json_data/splits/semantic_val.json \
  --output-dir MemoryTest/checkpoints/shine_memory_posttrain \
  --fact-counts 1 2 4 8 \
  --eval-fact-counts 1 2 4 8 \
  --qa-per-context 2 \
  --max-steps 2000 \
  --learning-rate 2e-6 \
  --grad-clip-norm 0.5 \
  --answer-max-length 256 \
  --context-max-length 1024 \
  --conversation-max-length 512 \
  --torch-dtype bf16 \
  --use-gradient-checkpoint \
  --freeze-metalora \
  --generated-lora-clamp 5.0
```

After this is stable, try unfreezing Meta-LoRA with a smaller LR such as `--metalora-learning-rate 5e-7`.

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

## 6. Evaluate Post-Trained SHINE

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
