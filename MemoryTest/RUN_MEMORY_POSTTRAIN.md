# MemoryTest Post-Training Runs

Run these commands from the repository root on the server. The local workspace does not need to execute them.

## Entry Points

The runnable scripts are grouped by responsibility:

```text
MemoryTest/prepare_data/prepare_memory_data.py
MemoryTest/prepare_data/generate_capacity_data.py
MemoryTest/training/run_lora_upper_bound.py
MemoryTest/training/run_shine_initialized_lora_sft.py
MemoryTest/teacher_lora/build_teacher_lora_bank.py
MemoryTest/training/posttrain_shine_teacher_lora.py
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

```bash
python -m MemoryTest.training.run_lora_upper_bound \
  --config MemoryTest/config/case_test.yaml \
  --facts-path MemoryTest/json_data/semantic_facts.json \
  --selection-mode head \
  --ranks 8 \
  --num-facts-list 20 \
  --num-trials 1 \
  --epochs 20 \
  --batch-size 2 \
  --learning-rate 5e-4 \
  --output MemoryTest/results/lora_upper_bound_rank8_20facts_best.json
```

NTP upper bound trains only on fact/context text and still evaluates with QA generation:

```bash
python -m MemoryTest.training.run_lora_upper_bound \
  --config MemoryTest/config/case_test.yaml \
  --facts-path MemoryTest/json_data/semantic_facts.json \
  --selection-mode head \
  --training-objective ntp \
  --ntp-record-mode both \
  --ntp-context-format mixed \
  --ntp-context-variants 5 \
  --ranks 8 \
  --num-facts-list 20 \
  --num-trials 1 \
  --epochs 20 \
  --batch-size 2 \
  --learning-rate 5e-4 \
  --output MemoryTest/results/lora_upper_bound_ntp_rank8_20facts_best.json
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

## 3. SHINE-Initialized LoRA SFT

This freezes SHINE and Qwen, generates `LoRA = SHINE(context)`, then updates only that LoRA dictionary for a small QA-SFT or NTP run. It measures whether SHINE is a better initialization than a random LoRA upper bound.

Short QA-SFT adaptation:

```bash
python -m MemoryTest.training.run_shine_initialized_lora_sft \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /home/wangyiding/SHINE-mem/checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150/train/checkpoint-epoch-1 \
  --facts-path MemoryTest/json_data/semantic_facts.json \
  --selection-mode head \
  --training-objective qa_sft \
  --num-facts-list 20 \
  --num-trials 1 \
  --epochs 1 \
  --batch-size 2 \
  --learning-rate 5e-4 \
  --output MemoryTest/results/shine_initialized_lora_qa_sft_epoch1.json
```

Short NTP adaptation:

```bash
python -m MemoryTest.training.run_shine_initialized_lora_sft \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /home/wangyiding/SHINE-mem/checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150/train/checkpoint-epoch-1 \
  --facts-path MemoryTest/json_data/semantic_facts.json \
  --selection-mode head \
  --training-objective ntp \
  --ntp-record-mode both \
  --ntp-context-format mixed \
  --ntp-context-variants 3 \
  --num-facts-list 20 \
  --num-trials 1 \
  --epochs 10 \
  --batch-size 2 \
  --learning-rate 5e-4 \
  --output MemoryTest/results/shine_initialized_lora_ntp_sft_epoch10.json
```

The output has both `shine_init_result` and `adapted_train_result`. `train_stats.best_epoch` tells which adaptation epoch was best.

## 4. Teacher-LoRA Alignment Post-Train

This is a separate experiment from plain answer-CE post-training. First build an offline teacher LoRA bank. Each entry samples a context, trains an ordinary QA-SFT LoRA teacher for that context, and saves the best LoRA.

Small teacher bank smoke run:

```bash
python -m MemoryTest.teacher_lora.build_teacher_lora_bank \
  --config MemoryTest/config/case_test.yaml \
  --facts-path MemoryTest/json_data/splits/semantic_train_augmented.json \
  --output-dir MemoryTest/teacher_loras/qa_sft_rank8_smoke \
  --rank 8 \
  --fact-counts 4 8 20 \
  --contexts-per-count 2 \
  --context-sampling random \
  --training-objective qa_sft \
  --variants-per-fact 5 \
  --epochs 3 \
  --batch-size 2 \
  --learning-rate 5e-4
```

Larger first run:

```bash
python -m MemoryTest.teacher_lora.build_teacher_lora_bank \
  --config MemoryTest/config/case_test.yaml \
  --facts-path MemoryTest/json_data/splits/semantic_train_augmented.json \
  --output-dir MemoryTest/teacher_loras/qa_sft_rank8 \
  --rank 8 \
  --fact-counts 4 8 20 \
  --contexts-per-count 50 \
  --context-sampling random \
  --training-objective qa_sft \
  --variants-per-fact 5 \
  --epochs 3 \
  --batch-size 2 \
  --learning-rate 5e-4
```

Then train SHINE to generate LoRA close to those teacher LoRAs. SHINE/Qwen loading is the same as other post-training runs; Qwen stays frozen, while the trainable SHINE parts follow the same flags as `posttrain_shine_memory.py`.

```bash
python -m MemoryTest.training.posttrain_shine_teacher_lora \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /path/to/original_shine_checkpoint \
  --teacher-bank-dir MemoryTest/teacher_loras/qa_sft_rank8 \
  --val-file MemoryTest/json_data/splits/semantic_val.json \
  --output-dir MemoryTest/checkpoints/shine_teacher_lora_posttrain \
  --fact-counts 4 8 20 \
  --max-steps 2000 \
  --learning-rate 1e-6 \
  --teacher-align-weight 1.0 \
  --answer-weight 0.1 \
  --qa-per-context 4 \
  --context-max-length 1024 \
  --conversation-max-length 512 \
  --answer-max-length 256 \
  --torch-dtype bf16 \
  --use-gradient-checkpoint
```

The alignment loss compares the actual low-rank update `Delta W = A @ B` module by module, not the raw `A/B` factors. It uses a low-rank Frobenius formula, so it does not materialize full FFN delta matrices. Set `--answer-weight 0` to train only with teacher parameter alignment.

Outputs:

```text
MemoryTest/teacher_loras/qa_sft_rank8/index.json
MemoryTest/teacher_loras/qa_sft_rank8/context_000001/meta.json
MemoryTest/teacher_loras/qa_sft_rank8/context_000001/teacher_lora.pt
MemoryTest/checkpoints/shine_teacher_lora_posttrain/latest/
MemoryTest/checkpoints/shine_teacher_lora_posttrain/best/
MemoryTest/checkpoints/shine_teacher_lora_posttrain/shine_teacher_lora_train_log.jsonl
MemoryTest/checkpoints/shine_teacher_lora_posttrain/summary.json
```

## 5. Evaluate Original SHINE

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

## 6. Post-Train SHINE

This entry point now trains recurrent SHINE. Each optimizer step processes several context chunks. The backbone stores
only the current memory tokens' per-layer K/V and carries those tensors into the next chunk. Context queries are blocked
from attending to old memory; current memory queries can attend to both the current context and the old memory cache.
Memory slots use fixed RoPE positions starting at `context_max_length`.

By default the original Qwen weights and zero memory-token embeddings stay frozen, while both the loaded Metalora and
the SHINE hypernetwork are trained. `--recurrent-steps 4` uses full four-chunk BPTT; set
`--detach-recurrent-memory-every 1` only for a read-only-cache ablation that does not train earlier writers from later
losses. Per-layer hidden/K/V RMS statistics and current/previous norm ratios are written to the JSONL training log.

### Training data format

The canonical file is a `shine_recurrent_v1` JSON object. Data declares only text, order, and QA targets; objective,
loss weights, reconstruction scope, and source sampling probabilities remain command-line choices.

```json
{
  "schema": "shine_recurrent_v1",
  "streams": [
    {
      "stream_id": "alice",
      "turns": [
        {
          "turn_id": "alice_01",
          "text": "Alice moved to Paris.",
          "qa": [{"question": "Where does Alice live?", "answer": "Paris"}]
        },
        {
          "turn_id": "alice_02",
          "text": "Alice started working at Google.",
          "qa": [{"question": "Where does Alice work?", "answer": "Google"}]
        }
      ]
    }
  ],
  "facts": [
    {
      "id": "semantic_0001",
      "text": "Xander Hayes works as a gardener.",
      "qa": [{"question": "What is Xander Hayes's job?", "answer": "gardener"}]
    }
  ]
}
```

Turns retain array order and are never shuffled. For streams longer than `--recurrent-steps`, use
`--stream-window-policy contiguous` for an order-preserving random window, `prefix` for the beginning, or `full` for
the complete stream.

The legacy top-level `question` / `answer` pair is accepted and normalized into the new `qa` list. Only `id` and
`text` are needed for reconstruction-only pretraining with `--context-format natural`. `person`, `attribute` (or
`relation`), and top-level `answer` are additionally needed for structured fact contexts.

Existing top-level flat fact arrays remain valid and become a fact pool. The temporary flat `stream_id`/`turn` format
is also accepted. `--ordered-stream-probability 0.7` selects a real stream for 70% of optimizer steps and a synthetic
fact stream for the remaining 30%.

A directly editable example is available at `MemoryTest/config/recurrent_data.example.json`.

QA and reconstruction accumulation are training settings, not fields in the data. Use `--qa-scope current|cumulative`
and `--reconstruction-scope current|cumulative` to control them independently.

### Multi-Session Chat (MSC)

Download and extract the official MSC v0.1 release:

```bash
mkdir -p MemoryTest/raw_data/msc
curl -L https://parl.ai/downloads/msc/msc_v0.1.tar.gz \
  -o MemoryTest/raw_data/msc_v0.1.tar.gz
tar -xzf MemoryTest/raw_data/msc_v0.1.tar.gz \
  -C MemoryTest/raw_data/msc
```

Convert every split using the same Qwen tokenizer as the SHINE backbone. The limit is applied to each recurrent turn,
not to the complete multi-session trajectory. Session boundaries are always retained; a long session is split into
multiple chunks, while two sessions are never merged into one chunk.

```bash
python -m MemoryTest.prepare_data.prepare_msc_recurrent \
  --input-dir MemoryTest/raw_data/msc \
  --output-dir MemoryTest/json_data/msc_recurrent_2048 \
  --tokenizer /path/to/Qwen3-8B \
  --max-turn-tokens 2048 \
  --qa-tasks next_turn persona_extraction persona_summary
```

Use `--max-turn-tokens 4096` and a different output directory for a 4096-token collection. Omitting `--tokenizer`
is supported for data-pipeline debugging, but then the limit is only a whitespace-token approximation.

The converter performs the following leakage-safe reconstruction:

- Groups snapshots by `metadata.initial_data_id` and keeps only the longest cumulative snapshot.
- Restores chronological sessions from `previous_dialogs + dialog`.
- Writes only speaker-tagged original utterances to each turn's `text`.
- Never inserts top-level `personas`, `init_personas`, `newfact`, or `followup` into input text.
- Stores next-turn, per-chunk persona extraction, and session persona summary targets under `qa`, with explicit prompts
  and a `task` tag. They remain available for later experiments but do not select or weight a training loss.

The current recurrent trainer reads a complete turn before running QA readout. Therefore, do not train the stored
`next_turn` records through the current QA objective: their answers already occur in that turn's text. They are retained
for a future prefix-readout/SFT path. Persona extraction and summary targets are valid after-session readouts. This does
not affect the reconstruction-only command below, which ignores every QA record.

The generated directory contains `msc_train.json`, `msc_valid.json`, `msc_test.json`, and `manifest.json`. To train
strictly with memory reconstruction loss, use `--turn-objective reconstruction`; the stored QA records are not used:

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  -m MemoryTest.training.posttrain_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --device cuda --gpu-id 0 \
  --checkpoint-dir /path/to/SHINE-ift_mqa_1qa \
  --train-file MemoryTest/json_data/msc_recurrent_2048/msc_train.json \
  --val-file MemoryTest/json_data/msc_recurrent_2048/msc_valid.json \
  --output-dir MemoryTest/checkpoints/msc_reconstruction \
  --ordered-stream-probability 1.0 \
  --stream-window-policy full \
  --turn-supervision every \
  --turn-objective reconstruction \
  --reconstruction-scope cumulative \
  --qa-per-context 0 \
  --eval-every 0 \
  --context-max-length 2048 \
  --memory-position-offset 2048 \
  --answer-max-length 8192 \
  --learning-rate 1e-6 \
  --metalora-learning-rate 1e-6 \
  --torch-dtype bf16 \
  --use-gradient-checkpoint
```

`cumulative` is important for long-term retention: after recurrent update t, the memory must reconstruct every chunk
observed through t. With `current`, the model can discard previous memory and still minimize the loss. Set
`--answer-max-length` slightly above `statistics.max_stream_tokens_observed` in the generated split (allowing extra
room for the chat template and EOS). `--stream-window-policy full` processes the complete trajectory in chronological
order; in this mode `--recurrent-steps` does not truncate an ordered stream. If full-trajectory BPTT or cumulative
decoding is too large for the first run, start with `--stream-window-policy contiguous --recurrent-steps 2` and an
answer limit above the largest two-turn window, then scale up.

### Per-turn objectives

Supervise QA after every recurrent update:

```bash
python -m MemoryTest.training.posttrain_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /path/to/original_shine_checkpoint \
  --train-file /path/to/qa_facts.json \
  --val-file MemoryTest/json_data/splits/semantic_val.json \
  --output-dir MemoryTest/checkpoints/recurrent_qa \
  --recurrent-steps 4 \
  --turn-supervision every \
  --turn-objective qa \
  --qa-per-context 4 \
  --use-gradient-checkpoint
```

Run official-style `<RECON>` pretraining after every update. `current` repeats only the latest evidence chunk;
`cumulative` asks the current memory to repeat everything observed so far.

```bash
python -m MemoryTest.training.posttrain_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /path/to/original_shine_checkpoint \
  --train-file /path/to/pretrain_texts.json \
  --val-file MemoryTest/json_data/splits/semantic_val.json \
  --output-dir MemoryTest/checkpoints/recurrent_recon \
  --context-format natural \
  --recurrent-steps 4 \
  --turn-supervision every \
  --turn-objective reconstruction \
  --reconstruction-scope current \
  --use-gradient-checkpoint
```

Alternate QA and reconstruction independently across turns:

```bash
python -m MemoryTest.training.posttrain_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /path/to/original_shine_checkpoint \
  --train-file /path/to/qa_facts.json \
  --val-file MemoryTest/json_data/splits/semantic_val.json \
  --output-dir MemoryTest/checkpoints/recurrent_mixed \
  --recurrent-steps 4 \
  --turn-supervision every \
  --turn-objective mixed \
  --qa-turn-prob 0.5 \
  --reconstruction-scope cumulative \
  --use-gradient-checkpoint
```

Use `--turn-objective both --recon-weight 0.2` to apply QA and reconstruction in every turn. With
`--turn-supervision final` (the compatibility default), intermediate chunks update only recurrent K/V and only the
final turn performs hypernetwork readout and supervised loss.

```bash
python -m MemoryTest.training.posttrain_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir /path/to/original_shine_checkpoint \
  --train-file MemoryTest/json_data/splits/semantic_train_augmented.json \
  --val-file MemoryTest/json_data/splits/semantic_val.json \
  --output-dir MemoryTest/checkpoints/shine_memory_posttrain \
  --fact-counts 1 2 4 8 12 20 \
  --qa-per-context 4 \
  --recurrent-steps 4 \
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
  --recurrent-steps 4 \
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
  --output-dir MemoryTest/checkpoints/shine_memory_posttrain2 \
  --fact-counts 1 2 4 8 \
  --qa-per-context 4 \
  --recurrent-steps 2 \
  --max-steps 2000 \
  --learning-rate 2e-6 \
  --eval-every 500 \
  --answer-max-length 256 \
  --context-max-length 1024 \
  --conversation-max-length 512 \
  --torch-dtype bf16 \
  --use-gradient-checkpoint
```

If this fits, add capacity and auxiliary losses back in order: first `--fact-counts 1 2 4 8 12 20`, then `--qa-per-context 2`, then `--use-contrastive`, and finally `--use-reconstruction`.

`--use-gradient-checkpoint` is applied to the context-to-LoRA generation path. The supervised answer/contrastive/reconstruction forward uses the generated LoRA and keeps checkpointing off by default, because per-layer checkpointing can backprop through the same generated LoRA graph multiple times. Only enable `--use-answer-gradient-checkpoint` for debugging experiments.

If loss becomes NaN, first reduce both learning rates and clamp generated LoRA values while continuing to train both
the hypernetwork and Metalora:

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
  --metalora-learning-rate 1e-6 \
  --grad-clip-norm 0.5 \
  --answer-max-length 256 \
  --context-max-length 1024 \
  --conversation-max-length 512 \
  --torch-dtype bf16 \
  --use-gradient-checkpoint \
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

## 7. Evaluate Post-Trained SHINE

```bash
python -m MemoryTest.evaluation.eval_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --baseline-checkpoint-dir /path/to/original_shine_checkpoint \
  --checkpoint-dir MemoryTest/checkpoints/shine_memory_posttrain/best \
  --test-file MemoryTest/json_data/splits/semantic_test_augmented.json \
  --num-facts-list 1 2 4 8 12 20 \
  --num-trials 5 \
  --include-baselines \
  --output MemoryTest/results/shine_posttrained_memory_eval.json

python -m MemoryTest.evaluation.eval_shine_memory \
  --config MemoryTest/config/case_test.yaml \
  --checkpoint-dir MemoryTest/checkpoints/shine_memory_posttrain2/best \
  --test-file MemoryTest/json_data/splits/semantic_test_augmented.json \
  --num-facts-list 1 2 4 8 12 20 \
  --num-trials 5 \
  --output MemoryTest/results/shine_posttrained_memory_eval2.json
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
