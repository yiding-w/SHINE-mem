# ControlSHINE

Training-free evaluation of externally specified knowledge-source control among:

- **BASE**: pretrained parametric knowledge;
- **MEMORY**: a SHINE-generated LoRA;
- **CONTEXT**: explicit text in the current prompt.

The first milestone does not train a controller. It measures whether the three
signals are individually recoverable and whether logit-residual scaling can
select the requested source.

## First milestone

For every fixed question, construct three distinct target answers:

```text
base_answer    = A
memory_answer  = B
context_answer = C
```

Run three source-only forwards:

```text
z_base    = Base(question)
z_memory  = Base + SHINE(memory)(question)
z_context = Base(context, question)
```

Then decode policies using:

```text
BASE:    z_base
MEMORY:  z_base + alpha_memory  * (z_memory  - z_base)
CONTEXT: z_base + alpha_context * (z_context - z_base)
```

The important first filter is **recoverability**. Control accuracy is reported
primarily on examples for which all three source-only runs recover their own
answer. End-to-end accuracy is reported on the full set separately.

## Planned layout

```text
ControlSHINE/
  README.md                 experiment scope and protocol
  configs/                  model, decoding, and scale-sweep configs
  data/
    raw/                    downloaded or generated source datasets
    processed/              normalized three-source JSONL
  scripts/
    build_synthetic.py      deterministic synthetic triples
    convert_counterfact.py  CounterFact adapter (planned)
    run_source_only.py      A/B/C source-only generations
    run_scale_sweep.py      CAD-style decoding (planned)
    evaluate.py             switching/isolation metrics (planned)
  controlshine/
    schema.py               validated sample representation
    prompts.py              source-specific prompt construction (planned)
    decoding.py             logit-residual decoding (planned)
    metrics.py              recoverability and control metrics (planned)
  tests/                    CPU unit tests for data and metrics
```

Model integration should reuse the loading and generated-LoRA interfaces in
`SHINE_V2-main/eval_memory_gen.py`, while keeping this experiment independent
of the training path.

## Data strategy

### Stage A: controlled synthetic facts

Start with short, unambiguous attributes such as access code, city, color,
occupation, date, and numeric value. Use fictional entities to avoid an unknown
base prior. For the main three-way experiment, also include real entities whose
base answer is first verified by the exact checkpoint.

Generation constraints:

1. A, B, and C must be distinct and tokenization difficulty should be balanced.
2. Answer assignment is randomly permuted across sources.
3. Entity, relation, answer, and question-template splits are recorded.
4. Memory and context use matched wording where possible.
5. Every record carries source-only expected answers and perturbations for
   isolation tests.

### Stage B: adapt existing conflict/editing datasets

- **CounterFact** is the best initial source for a known base fact plus a
  counterfactual replacement, paraphrases, and locality prompts. It supplies
  two semantic answers; a third distinct answer/context must be generated.
- **ConflictQA** is useful for selecting examples where a target model's
  parametric answer conflicts with supplied context. It is model-specific and
  still needs a separately generated Memory answer.
- **MQuAKE-CF** is useful after the atomic-fact phase for multi-hop propagation;
  it is not the first smoke test.
- **CONFLICTS** is useful later for realistic noisy or multi-document RAG
  contexts, not for the cleanest three-way identifiability test.

Existing datasets are therefore inputs to a conversion pipeline, not ready-made
ControlSHINE benchmarks. Each converted record must be revalidated against the
actual base checkpoint and the generated Memory LoRA.

Convert a server-side CounterFact file with:

```bash
python ControlSHINE/scripts/convert_counterfact.py \
  --input /data/yidingw/counterfact/counterfact.json \
  --output ControlSHINE/data/processed/counterfact_three_source.jsonl \
  --limit 1000
```

Omit `--limit` for the full conversion. A record can be skipped when no third
distinct answer is available from another record with the exact same rewrite
template; the command reports these counts.

## Canonical JSONL record

See `controlshine/schema.py`. Raw records contain provenance and construction
metadata. Predictions, logits, recoverability labels, and sweep results should
be written to separate run artifacts so that raw data stays immutable.

## Initial go/no-go evidence

- enough fully recoverable examples to make control meaningful;
- one global or per-conflict-type scale clearly improves strict three-way
  switching over unscaled source-only decoding;
- non-selected source perturbations have little effect;
- gains are not limited to a per-example oracle scale.

## Source-only checkpoint run

The evaluator is a standalone ControlSHINE program. It reproduces the original
`test_pretrain.py` construction order and loads the checkpoint directly. Start
with 10 examples on one GPU:

```bash
CUDA_VISIBLE_DEVICES=3 python -m ControlSHINE.scripts.run_source_only \
  --runtime-config ControlSHINE/configs/runtime_pretrain.yaml \
  --checkpoint-dir /home/wangyiding/SHINE-mem/checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150/pretrain/checkpoint-epoch-1 \
  --input ControlSHINE/data/processed/counterfact_three_source.jsonl \
  --output-dir ControlSHINE/runs/counterfact_smoke_10 \
  --limit 10 \
  --max-new-tokens 16
```
