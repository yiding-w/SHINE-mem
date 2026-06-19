# Implementation Plan: SHINE + ES Codebase

**Companion document to `proposal.md`.** This plan is self-contained for a code-writing agent: every external dependency has a URL, every SHINE call site has a file:line reference, and the first concrete action is listed at the end.

## 0. Design Principles

- **SHINE is an external dependency, not modified in place.** Include it as a git submodule under `third_party/SHINE` and call it through its public API only. Pinning to a commit SHA protects us from upstream drift.
- **ES loop is re-implemented, not copied from the ES at Scale repo.** The reference (`https://github.com/VsonicV/es-fine-tuning-paper`) is only ~72 lines of useful kernel code (`utils/worker_extn.py`); it assumes full-parameter iteration over `named_parameters`, no antithetic pairing, shared seed across layers (layer-correlated noise bug — see their issue #7), and HuggingFace-native rollout. We need antithetic pairing, per-tensor seed shift (iid noise), and a rollout that goes hypernetwork → LoRA → base LLM.
- **All rollouts use SHINE's HuggingFace `generate` path during ES training.** vLLM cannot host the hypernetwork forward pass; vLLM's fast path is only useful for downstream serving baselines.
- **Perturb all of $m_h$.** No mem_tokens-only fallback in the main code (see `proposal.md` §3.2).
- **Phase 1 is single-GPU.** Distributed comes later once the core loop has been debugged.

## 1. External Dependencies

### 1.1 Repositories and model checkpoints

| Resource | URL |
|---|---|
| SHINE source | https://github.com/Yewei-Liu/SHINE |
| SHINE pretrained checkpoint | https://huggingface.co/Yewei-Liu/SHINE-Pretrain |
| SHINE multi-QA IFT checkpoint | https://huggingface.co/Yewei-Liu/SHINE-ift_mqa |
| SHINE single-QA IFT checkpoint (**Phase 1 starting point**) | https://huggingface.co/Yewei-Liu/SHINE-ift_mqa_1qa |
| ES at Scale source (reference only) | https://github.com/VsonicV/es-fine-tuning-paper |
| ES at Scale paper | https://arxiv.org/abs/2509.24372 |
| SHINE paper | https://arxiv.org/abs/2602.06358 |
| PAST paper (our prior work) | https://arxiv.org/abs/2601.11258 |

### 1.2 SHINE public API (call sites we will use)

All paths relative to the SHINE repo root:

| Symbol | File:line | Purpose |
|---|---|---|
| `LoraQwen3ForCausalLM` | `LoraQwen.py:573-945` | Backbone with custom LoRA injection |
| `Metanetwork` | `metanetwork_family.py:121-169` | Hypernetwork wrapper (backbone + M2P + metalora) |
| `Metanetwork.generate_lora_dict` | `metanetwork_family.py:164-169` | **Context → LoRA dict (core call).** Inputs: `evidence_ids`, `evidence_attention_mask`, `metalora`. |
| `metanetwork.metamodel.generate` | via inherited HF generate, passes `loradict=...` | **LoRA-conditioned rollout.** Call with `loradict=<dict>`, `ignore_mem_token=True`, `do_sample=False`. |
| `utils.mysaveload.load_checkpoint` | `utils/mysaveload.py:28-49` | Loads `mem_tokens.pt`, `metanetwork.pth`, `metalora.pth` from a ckpt dir |
| `utils.mysaveload.save_checkpoint` | `utils/mysaveload.py:16-26` | Saves the above; reuse for our trained mem_tokens/M2P/metalora |
| `utils.myloradict.iter_learnable_tensors` | `utils/myloradict.py:4-22` | Iterates the leaf tensors in `metalora` — needed to flatten $m_h$ parameters for ES |
| `utils.myfreeze.freeze(metamodel)` | `utils/myfreeze.py` | Freezes the 8B backbone, unfreezes `mem_tokens`. Call after construction. |
| `LoraQwen3ForCausalLM.lora_params_numel(lora_r)` | `LoraQwen.py:730-738` | Used to compute `num_mem_token` at construction |
| `utils.mydataset.HumanDataset` / `HumanCollator` | `utils/mydataset.py:487-501, 1333-1384` | Minimal `{evidence, questions}` data pipeline |
| `calculate_f1.compute_f1` | `calculate_f1.py` | SQuAD-style F1 for fast rewards |
| `inference.ipynb` | notebook | **Canonical construction flow.** Mimic this exactly. |

### 1.3 Tensor access points for perturbation

```python
# All of m_h (= φ for ES). Union of three sources:
m2p_params = list(metanetwork.metanetwork.parameters())      # M2P transformer
mem_tokens = metanetwork.metamodel.model.mem_tokens          # Parameter leaf
metalora_leaves = list(iter_learnable_tensors(metalora))     # metalora leaves

# Frozen backbone — must NOT be perturbed:
# everything else under metanetwork.metamodel.*
```

### 1.4 Python environment

Match SHINE's `README.md` lines 64–68: `python==3.12`, `torch==2.5.1+cu124`, `transformers==4.57.1`, `datasets==4.4.1`, `hydra-core==1.3.2`, plus:
- `openai>=1.0` (reward judge, Batch API)
- `vllm>=0.6` (optional, only for serving baselines outside the ES loop)
- `pytest`, `tensorboard`, `wandb` (optional)

## 2. Repository Layout

```
meta-past/
├── README.md
├── pyproject.toml
├── proposal.md                       # research proposal (existing)
├── implementation_plan.md            # this file
├── third_party/
│   └── SHINE/                        # git submodule, pinned commit
├── meta_past/
│   ├── __init__.py
│   ├── shine_adapter.py              # thin wrapper over SHINE's Metanetwork
│   ├── es/
│   │   ├── __init__.py
│   │   ├── noise.py                  # seed → noise, deterministic, fp32 by default
│   │   ├── perturb.py                # in-place ± σ·ε application and exact restore
│   │   ├── update.py                 # antithetic grad, z-score / rank normalization
│   │   └── trainer.py                # main ES loop
│   ├── rollout/
│   │   ├── __init__.py
│   │   ├── base.py                   # Rollout abstract interface
│   │   └── squad_rollout.py          # SQuAD single-passage rollout
│   ├── reward/
│   │   ├── __init__.py
│   │   ├── f1_reward.py              # uses SHINE's calculate_f1
│   │   ├── judge_reward.py           # GPT-4.1 via OpenAI Batch API
│   │   └── learned_reward.py         # Phase 2: small distilled reward model
│   ├── anchor/
│   │   ├── __init__.py
│   │   └── frobenius.py              # soft L2 anchor to pretrained φ
│   ├── data/
│   │   ├── __init__.py
│   │   └── squad_contexts.py         # HumanDataset-backed SQuAD context loader
│   ├── config/
│   │   ├── defaults.yaml
│   │   └── es_squad_phase1.yaml
│   └── utils/
│       ├── checkpoint.py             # wraps SHINE save/load for our runs
│       ├── logging.py                # reward curves, held-out metrics
│       └── seeding.py                # deterministic outer-loop seeding
├── scripts/
│   ├── phase0_sanity.py              # zero-shot eval on 10 SQuAD passages
│   ├── phase1_es_squad.sh            # ES training entry point
│   ├── phase1_baseline_ift.sh        # SHINE + IFT-on-SQuAD baseline
│   └── eval.py                       # held-out evaluation
└── tests/
    ├── test_shine_adapter.py         # load ckpt, generate LoRA, shapes correct
    ├── test_es_perturb.py            # perturb + restore → bit-exact (fp32) / ULP-close (bf16)
    ├── test_es_update.py             # ES converges on a 2D toy problem
    └── test_rollout.py               # produces valid reward in [0, 1]
```

## 3. Core Modules

### 3.1 `meta_past/shine_adapter.py`

Goal: hide SHINE's Hydra / DDP machinery behind a single class usable from a single process.

```python
class ShineHypernet:
    """Thin wrapper around SHINE's Metanetwork for ES training.

    Construction mirrors inference.ipynb exactly:
      1. Load Qwen3 tokenizer and add the three SHINE special tokens
         (<RECON>, <COMP>, <NOTHING>).
      2. Compute num_mem_token from lora_params_numel (formula at
         SHINE meta_train_parallel.py:451-457).
      3. Construct LoraQwen3ForCausalLM with num_mem_token.
      4. resize_token_embeddings(len(tokenizer)).
      5. Wrap in Metanetwork.
      6. load_checkpoint(metanetwork, ckpt_dir, device) → returns
         (metanetwork, metalora, _).
      7. freeze(metamodel); assert only mem_tokens is trainable under metamodel.
    """

    def __init__(self, ckpt_dir: str, device: str,
                 backbone: str = "Qwen/Qwen3-8B",
                 lora_r: int = 8, metalora_r: int = 128): ...

    def all_perturbable_params(self) -> list[tuple[str, torch.Tensor]]:
        """Flat list of (name, tensor) covering:
          - M2P transformer: metanetwork.metanetwork.parameters()
          - mem_tokens:      metanetwork.metamodel.model.mem_tokens
          - metalora leaves: iter_learnable_tensors(metalora)

        Names MUST be stable across calls (used as seed-shift keys).
        Returns references to the live tensors; mutating them mutates the model.
        """

    def generate_lora(self, evidence_ids, evidence_mask) -> dict:
        """metanetwork.generate_lora_dict(...) — deterministic given φ."""

    @torch.no_grad()
    def answer(self, loradict, input_ids, attention_mask, **gen_kwargs):
        """metamodel.generate(..., loradict=..., ignore_mem_token=True,
                              do_sample=False). Wrap with no_grad; do NOT use
                              inference_mode because @torch.compile on the
                              metanetwork forward can conflict with it."""

    def save(self, out_dir: str):
        """Delegates to SHINE's save_checkpoint."""
```

**Critical construction gotchas (from code review):**
- `num_mem_token` is not the YAML default (4). It is computed as
  `num_mem_token = lora_params_numel(lora_r) * mean_pool_size / (hidden_size * num_hidden_layers)`.
  For Qwen3-8B with `lora_r=8` this is 148. If you skip this, shapes do not match the released checkpoint and loading fails silently.
- `resize_token_embeddings` must be called **before** loading the checkpoint; the `<RECON> / <COMP> / <NOTHING>` tokens are in the released embeddings.
- After construction, assert exactly three parameter groups are trainable. Anything else means the backbone leaked.

### 3.2 `meta_past/es/noise.py`

```python
def make_noise(shape, seed: int, device, dtype=torch.float32) -> torch.Tensor:
    """Deterministic Gaussian noise via torch.Generator on the param's device.

    dtype defaults to fp32 regardless of param dtype. Rationale:
      - Many SHINE params live in bf16/fp16.
      - bf16 with σ=1e-3 quantizes small tensors (e.g. mem_tokens values
        ~1e-3 magnitude) to zero, silently killing the ES signal.
      - We cast the fp32 noise to the param dtype only at the add step.
    """
    gen = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(shape, generator=gen, dtype=dtype, device=device)
```

### 3.3 `meta_past/es/perturb.py`

```python
class InPlacePerturber:
    """Applies ±σ·ε to a flat list of (name, tensor), in-place and reversibly.

    Uses per-tensor seed shift — ε_i for tensor k is generated from
    (base_seed + i * NUM_TENSORS + k) to avoid the layer-correlated noise
    bug from the ES-at-Scale default variant (see their issue #7 and the
    _iid.py counterpart).
    """

    def __init__(self, params: list[tuple[str, torch.Tensor]],
                 sigma: float, noise_dtype=torch.float32):
        self.params = params
        self.sigma = sigma
        self.noise_dtype = noise_dtype

    def _tensor_seed(self, base_seed: int, tensor_idx: int) -> int:
        # Stable per-tensor shift; overflow-safe 32-bit
        return (int(base_seed) * 1_000_003 + tensor_idx) & 0x7FFFFFFF

    def apply(self, base_seed: int, sign: int):
        for k, (_, p) in enumerate(self.params):
            eps = make_noise(p.shape, self._tensor_seed(base_seed, k),
                             p.device, self.noise_dtype)
            p.data.add_((sign * self.sigma) * eps.to(p.dtype))

    def restore(self, base_seed: int, sign: int):
        self.apply(base_seed, -sign)
```

### 3.4 `meta_past/es/update.py`

```python
def antithetic_es_grad(
    rewards_plus: list[float],
    rewards_minus: list[float],
    base_seeds: list[int],
    params: list[tuple[str, torch.Tensor]],
    sigma: float,
    normalize: str = "zscore",   # "zscore" | "rank" | "none"
) -> list[torch.Tensor]:
    """Returns a list of per-parameter gradients.

    ĝ = (1 / (N·σ)) · Σ_i r̂_i · ε_i
    where r̂_i = (R_i^+ − R_i^−) / 2, normalized across i.

    ε_i is regenerated from base_seeds[i] via the same seed-shift rule as
    InPlacePerturber.
    """
```

**vs ES at Scale defaults:**
- Antithetic pairing: **we do**, they do not. Roughly halves gradient variance.
- Rank transform: optional here; theirs uses z-score only.
- Per-tensor seed: **ours uses** seed shift; theirs has the layer-correlated default.

### 3.5 `meta_past/es/trainer.py`

```python
class ESTrainer:
    def __init__(self, hypernet: ShineHypernet, rollout: RolloutFn,
                 reward_fn: RewardFn, cfg: ESConfig,
                 anchor: AnchorFn | None = None):
        self.hypernet = hypernet
        self.perturber = InPlacePerturber(
            hypernet.all_perturbable_params(), sigma=cfg.sigma)
        self.rng = np.random.default_rng(cfg.seed)
        ...

    def step(self):
        contexts = self.sample_contexts(self.cfg.batch_size)
        seeds = self.rng.integers(0, 2**31, size=self.cfg.N // 2).tolist()

        R_plus, R_minus = [], []
        for s in seeds:
            self.perturber.apply(s, +1)
            R_plus.append(self._eval_batch(contexts))
            self.perturber.restore(s, +1)

            self.perturber.apply(s, -1)
            R_minus.append(self._eval_batch(contexts))
            self.perturber.restore(s, -1)

        grads = antithetic_es_grad(
            R_plus, R_minus, seeds,
            self.perturber.params, self.cfg.sigma,
            normalize=self.cfg.reward_norm,
        )
        for (name, p), g in zip(self.perturber.params, grads):
            p.data.add_(self.cfg.lr * g)

        if self.cfg.anchor_coef > 0:
            self.anchor.apply_step(
                self.perturber.params, self.cfg.lr,
                self.cfg.anchor_coef_at_step(self.step_idx),
            )

        return {"mean_R": (np.mean(R_plus) + np.mean(R_minus)) / 2}

    @torch.no_grad()
    def _eval_batch(self, contexts) -> float:
        rewards = []
        for ctx in contexts:
            loradict = self.hypernet.generate_lora(
                ctx.evidence_ids, ctx.evidence_mask)
            for q in ctx.questions:
                out = self.hypernet.answer(
                    loradict, q.input_ids, q.attention_mask,
                    max_new_tokens=self.cfg.max_new_tokens,
                )
                rewards.append(self.reward_fn(out, q.reference))
        return float(np.mean(rewards))
```

Key implementation notes:
- `torch.no_grad()` around `_eval_batch`. Avoid `torch.inference_mode()` because `@torch.compile` on `Metanetwork.forward` (`metanetwork_family.py:150`) can conflict with inference mode.
- `metanetwork.metamodel.generate` already calls the non-compiled path; `generate_lora_dict` is the compiled one and is fine under `no_grad`.

### 3.6 `meta_past/rollout/squad_rollout.py`

Reuse SHINE's `HumanDataset` + `HumanCollator` pattern. Our loader returns batches of `{evidence_ids, evidence_mask, [{input_ids, attention_mask, reference}...]}`. Tokenization of questions must use the same custom Qwen3 chat template that SHINE uses at `meta_train_parallel.py:470` / `inference.ipynb` (copy it verbatim; do not substitute HF's default Qwen3 template).

### 3.7 `meta_past/reward/judge_reward.py`

- Default: OpenAI Batch API. ES is inherently batched (hundreds of rollouts per step), so Batch API's latency is acceptable. Half the cost of the standard endpoint.
- Prompt: reuse PAST's binary correctness prompt verbatim (see PAST paper §3 and supplementary).
- Fall back to `f1_reward` when debugging / for sanity checks — much cheaper but noisier.

### 3.8 `meta_past/anchor/frobenius.py`

```python
def apply_step(params, pretrained_snapshot, lr: float, coef: float):
    """φ ← φ − lr · coef · (φ − φ_pretrained). Applied AFTER the ES update."""
```

Snapshot is captured once at trainer construction (deep copy of initial tensors into CPU pinned memory to save GPU HBM; stream to GPU per-step if it fits).

Schedule: `coef(t) = max(coef_end, coef_start − (coef_start − coef_end) · t / decay_steps)`, with `coef_start=1.0`, `coef_end=0.1`, `decay_steps=300`. Adaptive roll-back: if held-out reward drops >2 F1 over 3 eval intervals, multiply `coef` by 2.

## 4. Training Entry Point and Config

`scripts/phase1_es_squad.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
python -m meta_past.es.trainer --config meta_past/config/es_squad_phase1.yaml
```

`meta_past/config/es_squad_phase1.yaml`:
```yaml
hypernet:
  ckpt_dir: ${HOME}/ckpts/SHINE-ift_mqa_1qa
  backbone: Qwen/Qwen3-8B
  lora_r: 8
  metalora_r: 128
  device: cuda:0

perturb:
  sigma: 0.005                    # start slightly above ES-at-Scale's 1e-3
  noise_dtype: float32

es:
  N: 30                           # 15 antithetic pairs
  lr: 0.005
  reward_norm: zscore             # "zscore" | "rank" | "none"

rollout:
  batch_size: 16                  # contexts per step
  questions_per_context: 4
  max_new_tokens: 64

reward:
  type: f1                        # "f1" | "judge" | "learned_rm"
  judge_model: gpt-4.1
  batch_api: true

anchor:
  coef_start: 1.0
  coef_end: 0.1
  decay_steps: 300

train:
  total_steps: 500
  eval_every: 25
  heldout_contexts: 50
  save_every: 50
  seed: 42
  out_dir: runs/phase1_es_squad
```

## 5. Testing Strategy

**Unit tests (seconds each, run on every change):**

| Test | What it checks |
|---|---|
| `test_shine_adapter::test_load_ckpt` | `ShineHypernet("SHINE-ift_mqa_1qa")` constructs, `all_perturbable_params()` returns >0 tensors, no backbone params leak in |
| `test_shine_adapter::test_generate_lora_shape` | On a dummy tokenized context, `generate_lora()` returns a nested dict whose leaves have shapes matching `lora_r=8` across all Qwen3 layers |
| `test_es_perturb::test_roundtrip_fp32` | After `apply(s, +1); restore(s, +1)`, every param equals its original (bit-exact for fp32) |
| `test_es_perturb::test_roundtrip_bf16` | Same, with tolerance 2× ULP for bf16 params |
| `test_es_update::test_2d_quadratic` | ES on $f(x) = -\|x\|^2$ reaches $\|x\| < 0.1$ within 50 steps |
| `test_rollout::test_squad_one_passage` | Given one SQuAD passage + one question, `_eval_batch` returns a float in [0, 1] |

**Phase 0 sanity (run before the first real training):**

`scripts/phase0_sanity.py`:
1. Load `SHINE-ift_mqa_1qa`.
2. Run zero-shot evaluation on 10 SQuAD passages; record F1. Compare against SHINE paper numbers as a sanity check.
3. Apply one random ES perturbation (σ=0.005) to all of $m_h$. Re-run evaluation.
4. F1 should change but not crash; if F1 drops below half of baseline, σ is too large.
5. Restore the parameters. Re-run. F1 should match step 2 exactly (fp32) or very closely (bf16).

**Phase 1 success criteria:**
- Training reward curve increases by at least 3 F1 points over 200 steps.
- Held-out context reward increases (not just training contexts).
- The fine-tuned $m_h$ matches or beats the `SHINE-ift_mqa_1qa` baseline on a held-out SQuAD slice.

## 6. First Concrete Action

Run in order:

```bash
cd /Users/stanleytang/Documents/coding/meta-past
git init
git submodule add https://github.com/Yewei-Liu/SHINE third_party/SHINE
cd third_party/SHINE && git rev-parse HEAD > ../../shine_commit.txt && cd ../..

# Python env (uv or conda)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch==2.5.1+cu124 transformers==4.57.1 \
    datasets==4.4.1 hydra-core==1.3.2 openai pytest
uv pip install -e third_party/SHINE  # if SHINE exposes a setup; else adjust PYTHONPATH

# Download the Phase 1 starting checkpoint
huggingface-cli download Yewei-Liu/SHINE-ift_mqa_1qa \
    --local-dir ~/ckpts/SHINE-ift_mqa_1qa

# Write meta_past/shine_adapter.py and tests/test_shine_adapter.py
# Run:
pytest tests/test_shine_adapter.py -v
```

When `test_shine_adapter` passes, the rest of the plan is unblocked.

## References

| Item | URL |
|---|---|
| PAST paper | https://arxiv.org/abs/2601.11258 |
| SHINE paper | https://arxiv.org/abs/2602.06358 |
| SHINE code | https://github.com/Yewei-Liu/SHINE |
| SHINE pretrained ckpt | https://huggingface.co/Yewei-Liu/SHINE-Pretrain |
| SHINE IFT ckpt (single-QA) | https://huggingface.co/Yewei-Liu/SHINE-ift_mqa_1qa |
| ES at Scale paper | https://arxiv.org/abs/2509.24372 |
| ES at Scale code | https://github.com/VsonicV/es-fine-tuning-paper |
| vLLM multi-LoRA docs | https://docs.vllm.ai/en/latest/features/lora/ |
| LoRAX | https://github.com/predibase/lorax |
| OpenAI ES original | https://github.com/openai/evolution-strategies-starter |
