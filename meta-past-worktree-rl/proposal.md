# PAST Follow-up: RL/ES Fine-Tuning of a Context-Conditioned Hypernetwork

## 1. Motivation

**Prior work this builds on:**

- **PAST (our previous paper)** — arXiv: https://arxiv.org/abs/2601.11258
  Finds that SFT and RL parameter updates are nearly orthogonal, and proposes linear "skill vector" injection for cross-domain transfer. Core insight: **knowledge is not enough** — teaching a model new facts is not the same as teaching it to use those facts.

- **SHINE (lab work)** — arXiv: https://arxiv.org/abs/2602.06358 · code: https://github.com/Yewei-Liu/SHINE
  A hypernetwork $m_h$ that maps a context $c$ to a LoRA in a single forward pass. Architecture: frozen Qwen3 backbone + Meta LoRA + M2P transformer + learnable memory embeddings. Pretraining objectives are reconstruction + completion (self-supervised).

- **ES at Scale** — arXiv: https://arxiv.org/abs/2509.24372 · code: https://github.com/VsonicV/es-fine-tuning-paper
  Demonstrates that Evolution Strategies work for full-parameter fine-tuning of billion-parameter LLMs with N=30, σ=0.001, greedy decoding. Outperforms PPO / GRPO / Dr.GRPO on sparse and long-horizon rewards, with better stability and less reward hacking.

**The problem we address:** SHINE's pretraining objective (recover the context) is not the same as the downstream objective (use the context correctly). This is the exact SFT-vs-RL gap that PAST already established, now in the hypernetwork regime. Our hypothesis: applying RL/ES on top of the pretrained SHINE checkpoint, with downstream-task reward, closes this gap without losing SHINE's cross-domain generalization.

## 2. Proposed Method

**Setup:**
- Base model $m$ (Qwen3-8B, frozen)
- Hypernetwork $m_h$ with parameters $\theta$, initialized from the SHINE pretrained (or IFT) checkpoint
- Perturbation scope $\phi = \theta$ — all trainable hypernetwork parameters (see §3.2)
- Context pool $\{c\}$; downstream task (SQuAD single-passage first)
- Reward $R(m, \text{LoRA}, c)$ — rollout-based, averaged over multiple questions per context

**ES training loop** (one step evaluates $N$ parameter perturbations, not $N$ contexts):

```
initialize θ from SHINE pretrain (or IFT) ckpt
for step t = 1, 2, ...:
    sample a batch of contexts {c_1, ..., c_B}
    sample N/2 seeds → reconstruct noise {ε_1, ..., ε_{N/2}}   # shape matches φ

    # ── Evaluate with antithetic pairing ──
    for i in 1..N/2:
        for sign in {+1, -1}:
            φ ← φ + sign · σ · ε_i                  # in-place perturbation
            for c_j in batch:
                LoRA_ij = m_h(c_j ; θ with perturbed φ)
                R_ij^sign = rollout(m, LoRA_ij, c_j)
            R_i^sign = mean_j R_ij^sign
            φ ← φ - sign · σ · ε_i                  # in-place restoration (same seed)

    # ── Antithetic finite-difference gradient estimate ──
    normalize rewards (z-score or rank)  →  r̂_i = (R_i^+ - R_i^-) / 2
    ĝ = (1 / (N · σ)) · Σ_i r̂_i · ε_i
    φ ← φ + α · ĝ
```

**Test time:** one forward pass $m_h(c) \to$ LoRA, deployed directly. Inference cost is negligible. This is the core advantage of the hypernetwork path, and RL/ES fine-tuning does not affect it.

## 3. Key Decisions

### 3.1 ES as the primary choice; RL feasible without re-pretraining

**Why ES first:**
- SHINE is a deterministic map. ES at Scale's "greedy decoding + parameter-space perturbation" regime matches this exactly — no architectural change, no re-pretraining.
- ES is more robust under sparse / delayed / long-horizon rewards (binary GPT-4.1-judged rewards in PAST are sparse).
- Simpler to implement: seed-based noise + in-place perturbation (see ES at Scale's `utils/worker_extn.py`).

**RL as fallback — and the cost is lower than we initially thought:**
- SHINE's `mem_tokens` (`LoraQwen.py:573`) are context-independent learnable parameters appended as a suffix to every context. They act as "query anchors" that extract information from the context via causal attention.
- We can treat `mem_tokens` as the mean of a Gaussian latent and add fresh noise at every forward pass. This turns $m_h$ into a stochastic policy that can be trained with REINFORCE/PPO — without re-pretraining.
- This removes the original concern that RL requires a new pretraining run.

### 3.2 Perturbation scope: perturb all of $m_h$

**Decision: $\phi$ = every trainable hypernetwork parameter.** This follows ES at Scale faithfully — they perturb every `named_parameter` of the model being trained.

- ES at Scale showed ES works on ~7B-parameter models with N=30; our $m_h$ total is ~700M–1B (M2P ≈ 400M, Meta LoRA ≈ several hundred M, mem_tokens ≈ 0.6M), comparable to or smaller than their setup. The parameter-count argument against full-scope perturbation is not empirically supported.
- Full-$m_h$ perturbation is the simplest and most expressive choice.

**What "all of $m_h$" means concretely:**
- `metanetwork.metanetwork.parameters()` — the M2P transformer
- `metanetwork.metamodel.model.mem_tokens` — the memory tokens (Parameter leaf)
- `iter_learnable_tensors(metalora)` — the `metalora` leaf tensors (nested dict, utility in `utils/myloradict.py`)
- Explicitly **excluded**: everything else under `metanetwork.metamodel.*` (the frozen Qwen3-8B backbone)

### 3.3 Anchor regularizer to the pretrained checkpoint

- SHINE's generalization comes from large-scale cross-domain pretraining. Unconstrained ES on a narrow downstream distribution risks catastrophic drift.
- Add a soft L2 penalty: $\|\phi - \phi_{\text{pretrained}}\|^2$, applied after each ES update as $\phi \leftarrow \phi - \alpha \lambda (\phi - \phi_\text{pretrained})$.
- **Schedule:** $\lambda$ decays linearly from 1.0 → 0.1 over the first 300 steps. If held-out reward degrades, roll $\lambda$ back up (adaptive).

## 4. Efficiency Analysis

### 4.1 Training cost

Per-step cost ≈ $N \times B \times K \times T$, where $N$ = population, $B$ = contexts per batch, $K$ = questions per context, $T$ = rollout length.

**Reward cost (GPT-4.1 as judge, 2026 pricing):**
- Standard: $2 / $8 per M input/output tokens. Batch API: $1 / $4. Cached input: $0.5 / M.
- Per-judge for a single-turn SQuAD answer ≈ 500 input + 50 output tokens ≈ $0.0014 (standard) or ~$0.0007 (batch).
- Back-of-envelope: N=30, B=16, K=4 → 1920 judge calls/step × $0.0007 ≈ **$1.35/step**. 500 steps ≈ **$675**.
- Phase 2 mitigation: train a Qwen3-1.7B LoRA reward model on 10k GPT-4.1-labeled samples (~$50 one-off), then judging is free.

**Rollout cost:**
- SHINE hypernetwork forward pass ≈ 0.3s — not the bottleneck.
- SQuAD single-turn is cheap. ToolBench multi-step agent + judge would explode; we defer it.
- Same rollout cost is borne by PPO/GRPO — this is not an ES-specific tax.

### 4.2 Per-example LoRA serving

- vLLM supports multi-LoRA via S-LoRA / Punica kernels (`docs.vllm.ai/en/latest/features/lora/`). `max_loras` 4–16 concurrent, `max_cpu_loras` in the thousands. Dynamic registration via `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`.
- **Phase 1:** don't use vLLM inside the ES loop — use SHINE's native HuggingFace `generate` path, because vLLM does not host the hypernetwork forward pass.
- **Eval/baseline comparison:** vLLM is fine for serving a baseline that freezes one LoRA per context.
- **If high-throughput serving becomes the bottleneck at Phase 3+:** consider LoRAX (https://github.com/predibase/lorax), which is stronger than vLLM under high adapter churn.

### 4.3 Test-time advantage (unchanged by ES)

- One hypernetwork forward pass → LoRA → deploy.
- Versus per-context SFT LoRA (backprop + multiple optimizer steps per new context), wall-clock differs by orders of magnitude.
- ES/RL fine-tuning of $m_h$ does **not** erode this test-time advantage; it just changes what the pretrained $m_h$ has learned.

## 5. Experimental Design

### 5.1 Phase 1: SQuAD single-passage

- 100–200 contexts, multiple questions per context.
- Reward: GPT-4.1 binary correctness via Batch API (fallback: F1 against gold answer, cheaper).
- ES: N=30, σ ∈ {0.001, 0.005}, antithetic pairing, z-score reward normalization, constant learning rate α.
- Perturbation scope: all of $m_h$.
- Run 200–500 steps, plot training + held-out reward curves.
- Kill-switch criterion: if training reward is flat after 100 steps, debug before scaling up.

### 5.2 Required baselines

| Baseline | Purpose |
|---|---|
| SHINE pretrained, zero-shot | Floor — starting point of our method |
| SHINE + IFT on task data | The **hardest** baseline. SHINE's released `ift_mqa_1qa` checkpoint already does IFT on QA. Beating this proves RL signal contributes beyond supervised fine-tuning on the same data. |
| Per-context SFT LoRA (one LoRA per context) | Ceiling for test-time adaptation quality |
| PAST skill vector | Direct comparison with our previous method |

### 5.3 Held-out evaluation — guards against reward hacking

- **Held-out contexts** — the model must not have learned to encode answers into LoRA during training.
- **Held-out task family** — run the fine-tuned $m_h$ on a task it wasn't trained on (e.g. trained on SQuAD, eval on HotpotQA). Confirms cross-domain generalization survives.

### 5.4 Narrative focus

- PAST's story: cross-domain transfer of skills.
- This paper's story: **making a hypernetwork actually learn to use context**.
- Go deep on one benchmark first (SQuAD), then pick one of {LooGLE, ToolBench} rather than spreading thin across three.

## 6. Confirmed Facts (from code inspection)

| Question | Answer | Source |
|---|---|---|
| SHINE `mem_tokens` structure | Shape `(num_mem_token, 4096)`; auto-computed `num_mem_token ≈ 148` for Qwen3-8B with lora_r=8; zero-initialized; concatenated as suffix to context; context-independent | `LoraQwen.py:573, 624, 664-694`; `meta_train_parallel.py:451-457` |
| SHINE trainable parameter groups | mem_tokens + M2P transformer + metalora. Backbone is strictly frozen. | `meta_train_parallel.py:580-595`; `utils/myfreeze.py` |
| SHINE training stages | pretrain → iftpwc (multi-QA IFT) → train (1-QA IFT) | `meta_train_parallel.py:517-557` |
| SHINE released checkpoints | `Yewei-Liu/SHINE-Pretrain`, `Yewei-Liu/SHINE-ift_mqa`, `Yewei-Liu/SHINE-ift_mqa_1qa` on HuggingFace | SHINE README lines 79–89 |
| SHINE LoRA application | Custom `LoraLinear` injection (not HF PEFT); LoRA dict hot-swappable per forward; `loradict=None` → no-op | `LoraQwen.py:33-156, 593-717, 778-848` |
| ES at Scale perturbation | Seed-based, in-place, no antithetic in default code, z-score normalization, N=30, σ=1e-3, α=5e-4 | `es_fine-tuning_conciseness.py:102-159, 283-310`; `utils/worker_extn.py:23-47` |
| ES at Scale noise quirk | Default variant shares one seed across all layers (layer-correlated). IID variant has per-tensor seed shift. | `es_fine-tuning_conciseness_iid.py:110-115, 137-142` |
| vLLM per-example LoRA | Supported (S-LoRA/Punica); dynamic loading via env flag. Expect 0.1–0.3× throughput with unique adapter per cold request. | docs.vllm.ai/en/latest/features/lora/ |
| GPT-4.1 pricing (2026) | Standard $2/$8 per M tokens; Batch API $1/$4; Cached input $0.5/M | OpenAI pricing page |
| RL-as-fallback cost | Does **not** require re-pretraining. Add Gaussian noise to `mem_tokens` at each forward, train with REINFORCE. | Derived from SHINE code inspection |

## 7. Empirical Findings During Phase 1 (pre-Phase-2)

These are measured, non-obvious outcomes from the first Phase 1 smoke runs. The
proposal's original §3.2 scope recommendation does not survive contact with
SHINE-ift_mqa_1qa; the relevant design decisions have been updated in code.

- **mem_tokens is a singularity.** `SHINE-ift_mqa_1qa/mem_tokens.pt` ships all
  zeros because SHINE's DDP optimizer filter (`not n.startswith("module.metamodel")`)
  excluded them from training. The downstream M2P was trained to read
  memory-states from attention-on-zero queries, so any mem_tokens perturbation
  ≥ 1e-4 collapses F1 to ~0. `ShineHypernet.all_perturbable_params()` now
  excludes them by default.
- **σ cliff at 1e-4 under uniform-σ, no signal below.** With 554 perturbable
  tensors whose RMS spans 5e-3 to 1.0, a uniform absolute σ gives small-RMS
  tensors disproportionate relative kicks. The signal band with uniform σ is
  [0.005, 0.008]; outside that it's either invariant or collapsed.
- **RMS-relative perturbation is necessary.** `InPlacePerturber(sigma_mode=
  "rms_relative")` uses σ_k = σ_rel · RMS(p_k), snapshotted at construction.
  The paired change in `antithetic_es_grad` / `one_sided_es_grad` is that
  gradient normalization divides by the *scalar* σ_rel (not per-tensor σ_k),
  so ĝ_k ∝ RMS_k · g_k and updates have uniform *relative* magnitude.
- **ES at Scale kernel ported.** `es/update.py:one_sided_es_grad` matches
  their `es_fine-tuning_conciseness.py` line-for-line (one-sided pop, zscore
  with 1e-8 floor). ESConfig.mode ∈ {"one_sided", "antithetic"}; default is
  one_sided. Antithetic kept as option but carries no clear win here.
- **Anchor neutralized the gradient in run #5.** Smoke run with σ_rel=0.1,
  N=20, antithetic, lr=0.001, anchor coef 1.0→0.1 kept heldout pinned at
  0.640 ± 0.005 over 30 steps (baseline 0.658). Run #6 disables the anchor
  to check whether the raw ES signal can move heldout at all.

## 8. Open Questions (need experiments)

- Exact anchor coefficient schedule ($\lambda$ decay curve, step count for the schedule).
- Choice of σ (0.001 vs 0.005 vs 0.01). Start with ES at Scale's default and sweep if needed.
- Whether antithetic pairing helps consistently (ES at Scale does not use it; we default to using it).
- Whether rank-based reward normalization beats z-score for our reward distribution.
- Whether to train from SHINE-Pretrain or SHINE-ift_mqa_1qa as the starting checkpoint (the latter has already seen QA data; cleaner ablation if starting from pretrain).

## 9. Next Steps

### Phase 0 — Setup
1. Clone SHINE (`https://github.com/Yewei-Liu/SHINE`) as a git submodule; pin to a specific commit.
2. Download `Yewei-Liu/SHINE-ift_mqa_1qa` from HuggingFace.
3. Stand up the project skeleton (see `implementation_plan.md`).
4. Run SHINE's inference notebook end-to-end to verify the environment and obtain a zero-shot SQuAD number.

### Phase 1 — Minimal ES experiment (SQuAD single-passage)
5. Implement ES training loop with seed-based in-place perturb / restore (reference: ES at Scale's `worker_extn.py`).
6. Perturbation scope = all $m_h$ parameters; antithetic pairing; N=30; σ ∈ {0.001, 0.005}; z-score reward normalization.
7. Run 200–500 steps; plot training and held-out reward curves.
8. Compare against the four baselines in §5.2.

### Phase 2 — Ablations and reward model
9. Train a small learned reward model to replace GPT-4.1 in the inner loop.
10. Anchor regularizer ablation (with / without / varying $\lambda$).
11. Fallback arm: REINFORCE with Gaussian noise on `mem_tokens`, head-to-head vs ES.
12. Extend to one of {LooGLE, ToolBench}.

### Phase 3 — Writing
13. Deep dive on the chosen benchmark, cross-context and cross-task generalization analysis, paper draft.

## References

| | |
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
