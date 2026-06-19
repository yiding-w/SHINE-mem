# meta-past

RL fine-tuning of the SHINE context-conditioned hypernetwork on top of Qwen3-8B.
Continuation of the PAST work — instead of memorizing facts from a passage, the
hypernet should compile the context into a LoRA that gives the model genuine
multi-hop / reasoning skills.

## Layout

```
meta_past/
├── rl/                 # the RL training stack
│   ├── trainer.py        # DDP trainer with joint hypernet+rescore microbatching
│   ├── rollout.py        # batched rollout (B contexts × Q questions × K samples)
│   ├── vllm_engine.py    # per-rank co-located vLLM with sleep/wake
│   ├── lora_format.py    # SHINE loradict ↔ vLLM PEFT-name conversion
│   ├── advantages.py     # GRPO / RLOO / R++ advantage estimators
│   ├── losses.py         # token-mean REINFORCE policy loss
│   └── _verl_kernels.py  # vendored verl kernels (no verl pip dep)
├── reward/             # reward functions
│   ├── f1_reward.py      # SQuAD-style token F1
│   └── judge_reward.py   # LLM-as-judge (HTTP) — see scripts/judge_server.py
├── data/               # dataset loaders → SquadContext shape
│   ├── squad_contexts.py
│   └── musique_contexts.py
├── shine_adapter.py    # thin wrapper around third_party/SHINE
├── es/, anchor/, rollout/, utils/   # ES stack + shared helpers
└── config/             # yaml configs
    ├── rl_musique_grpo.yaml         # MuSiQue + F1 reward
    └── rl_musique_grpo_judge.yaml   # MuSiQue + LLM-judge reward

scripts/
├── train.py            # main RL launcher (torchrun-based)
├── judge_server.py     # FastAPI judge server (OpenAI or local vLLM backend)
├── phase1_es_squad.py  # ES launcher (separate stack)
└── ...                 # ES debug / eval helpers

examples/               # one-command runners
├── run_musique_f1.sh
└── run_musique_judge.sh

third_party/SHINE/      # SHINE submodule
```

## Setup

### Conda env

The RL stack runs in the `vllm_serve` conda env (torch 2.6 + vLLM 0.8.5 +
transformers 4.57; SHINE deps installed on top).

```bash
conda activate vllm_serve   # or use absolute path: /ceph/home/muhan01/.conda/envs/vllm_serve/bin
```

The ES stack lives in the separate `shine` env (torch 2.5.1 + vLLM 0.7.3) — see
the ES scripts under `scripts/phase*_es_squad.py` if you're running ES.

### Models / data

- Backbone: `~/huggingfacemodels/Qwen3-8B`
- SHINE checkpoint: `~/huggingfacemodels/SHINE-ift_mqa_1qa`
- SQuAD: `~/huggingfacedatasets/squad/plain_text/`
- MuSiQue: pulled from HuggingFace (`dgslibisey/MuSiQue`) on first run

## Quick start

### F1 reward (no external services)

```bash
bash examples/run_musique_f1.sh           # 8 GPUs, default
bash examples/run_musique_f1.sh 4         # 4 GPUs
```

### LLM-as-judge reward

`examples/run_musique_judge.sh` brings up the judge HTTP server in the
background (auto-cleanup on exit), waits for `/healthz`, then launches
torchrun.

```bash
# OpenAI gpt-4o-mini (default)
OPENAI_API_KEY=sk-... bash examples/run_musique_judge.sh

# Local Qwen3-32B served by vLLM (start it separately first)
vllm serve Qwen/Qwen3-32B-Instruct --port 8000 &  # spare GPU
bash examples/run_musique_judge.sh \
    --backend openai-compat \
    --judge-base-url http://127.0.0.1:8000/v1 \
    --judge-model Qwen3-32B-Instruct
```

### Manual (torchrun directly)

```bash
/ceph/home/muhan01/.conda/envs/vllm_serve/bin/torchrun \
    --nproc_per_node=8 --standalone \
    scripts/train.py \
    --config meta_past/config/rl_musique_grpo.yaml
```

`--nproc_per_node` must divide `rollout.contexts_per_step` (so each rank
gets an integer slice of the global batch).

## Architecture (RL)

verl-style **HybridEngine**: each torchrun rank owns one GPU and co-locates:

- A SHINE hypernet (frozen Qwen3-8B backbone copy + trainable M2P + metalora)
- A `vllm.LLM(enable_sleep_mode=True)` for sampling

Per training step:

```
wake_up vLLM → push LoRAs (in-memory via collective_rpc) → sample → sleep vLLM
            ↓
  rescore + per-chunk hypernet forward + per-chunk backward
            ↓
   all-reduce gradients (manual NCCL SUM, token-mean normalized) → optim.step
```

Sleep mode swaps vLLM's weights between GPU and CPU around the rollout
window so the same GPU is free for the heavy training forward+backward.
Joint per-chunk hypernet+rescore microbatching keeps peak HBM bounded by
chunk size, independent of `contexts_per_step`. See commit messages and
the trainer/rollout docstrings for memory / numerical details.

## Configs

Two MuSiQue configs ship by default; pick reward type:

| File                                 | Reward            | Use when                           |
|--------------------------------------|-------------------|------------------------------------|
| `rl_musique_grpo.yaml`               | F1 (token overlap)| Cheap, no external services        |
| `rl_musique_grpo_judge.yaml`         | LLM-as-judge      | Semantic correctness for paraphrased / multi-step answers |

Knobs you'll typically touch:

- `rollout.contexts_per_step` (B): global batch size in contexts
- `rollout.questions_per_context` (Q): MuSiQue is 1-question-per-context, so 1
- `rollout.rollouts_per_question` (K): GRPO group size — 16 default
- `rollout.{hypernet,rescore}_microbatch_contexts`: peak-HBM cap; 0 = no chunking
- `vllm.gpu_memory_utilization`: pool fraction for vLLM (it never releases this back to the driver, so smaller leaves more for training)
- `train.train_contexts`: 0 = use the entire training split with no repetition within an epoch
- `wandb.enabled`: defaults to true; set false to skip wandb init

## Logging

- **JSONL**: `runs/<out_dir>/train_log.jsonl` — every event the trainer emits
- **wandb**: keys mirror verl's namespacing (`actor/pg_loss`,
  `critic/{rewards,advantages}/{mean,min,max}`, `timing_s/*`,
  `response_length/*`, `val-core/<dataset>/reward/mean@1`, etc.) so verl
  dashboards can be reused

## Reward types

### F1 (`reward.type: f1`)
SQuAD token-level F1 against the gold answer + aliases. No external
service. Good for QA-style targets where surface-form match is fine.

### LLM-as-judge (`reward.type: judge`)
Runs `scripts/judge_server.py` as a FastAPI service. Each sample becomes
an HTTP POST `/evaluate` with `{question, reference, pred}`; the judge
returns `True`/`False` → `1.0` / `0.0`. Concurrent calls fan out via a
thread pool — at 1024 samples/step with 64-way concurrency the reward
phase is ~16 round-trips.

The wire format is identical to
`Long-Digestor-Experiments/reward_server.py`, so the same server can be
reused across both projects.

Backends:
- `openai`: forwards to OpenAI Chat Completions (default `gpt-4o-mini`).
  Needs `OPENAI_API_KEY`.
- `openai-compat`: forwards to a local OpenAI-compatible server (e.g. a
  `vllm serve` instance running Qwen3-32B-Instruct on a spare GPU). No
  API key, no per-call cost.

## Datasets

| `data.name` | Source                                  | Notes                                   |
|-------------|-----------------------------------------|-----------------------------------------|
| `squad`     | `~/huggingfacedatasets/squad/plain_text/` | Single-passage extractive QA (SHINE was pretrained on this — use as a smoke / baseline; train R will saturate near zero-shot heldout) |
| `musique`   | HuggingFace `dgslibisey/MuSiQue`        | Multi-hop QA, supporting paragraphs only (~250-400 tokens each). Tests reasoning, not span retrieval |

To plug in a new dataset, add `meta_past/data/<name>_contexts.py` with an
`iter_train_val(train_size, val_size)` that returns lists of
`SquadContext` (the dataclass is dataset-agnostic), then dispatch on
`data.name` in `scripts/train.py::_load_data`.

## Tests

```bash
/ceph/home/muhan01/.conda/envs/vllm_serve/bin/python -m pytest \
    tests/ --ignore=tests/test_shine_adapter.py -q
```

`test_shine_adapter.py` requires a live SHINE checkpoint and is skipped by
default. The other tests cover loradict slicing, batched rollout shapes,
verl kernels, ES perturbations, and anchor/advantage logic.
