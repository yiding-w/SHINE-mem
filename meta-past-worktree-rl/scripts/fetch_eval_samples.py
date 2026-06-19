"""Fetch one real sample from each candidate eval dataset.

For every dataset we also assemble the ``(context, question, references)``
triple as it would be presented at eval time:

  - Bucket T (training distribution): context / question / references
    pulled via the **project's own loader** (so the format is exactly
    what the trainer sees), not the raw HF dataset.
  - Bucket A (context-QA): context = the dataset's passage(s),
    question = the question, references = gold answers.
  - Bucket B (in-parameter few-shot): context = K=2 demos from the same
    task/subject, question = a different held-out item, references =
    that item's gold answer. (Real eval uses K determined by token
    budget; we show K=2 just so the structure is visible.)
  - Bucket C (zero-shot probes): context = "" (or a placeholder),
    question = the question, references = gold answers.
  - Bucket D (long context): same as bucket A but contexts are long;
    we display only the first ~1200 chars to keep the markdown readable.

Output: writes a JSON file (one entry per dataset) which the wrapping
markdown report consumes.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[1] / "eval_samples.json"


def truncate(s: str, n: int = 1200) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n].rstrip() + " …[truncated]"


def safe(fn):
    try:
        return fn()
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}", "_trace": traceback.format_exc()[-400:]}


# -------------------- Bucket T: training distribution --------------------
# These use the project's own loaders (meta_past.data.*) so the
# rendered (context, question, references) triple matches exactly what
# the trainer feeds the hypernet + LM.

def fetch_train_squad():
    import sys
    sys.path.insert(0, str(ROOT := Path(__file__).resolve().parents[1]))
    from meta_past.data.squad_contexts import iter_train_val
    train, _ = iter_train_val(train_size=1, val_size=1)
    c = train[0]
    return {
        "context": c.context,
        "question": c.qa[0].question,
        "references": list(c.qa[0].references),
        "_notes": f"context_id={c.context_id}, n_qa_per_context={len(c.qa)}. Trainer samples Q questions per context (config: questions_per_context).",
    }


def fetch_train_musique():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from meta_past.data.musique_contexts import iter_train_val
    train, _ = iter_train_val(train_size=1, val_size=1)
    c = train[0]
    return {
        "context": c.context,
        "question": c.qa[0].question,
        "references": list(c.qa[0].references),
        "_notes": f"context_id={c.context_id}, n_qa_per_context={len(c.qa)}. Each MuSiQue example bundles its supporting paragraphs ('# Title\\nbody' blocks) into one context with one multi-hop question.",
    }


def fetch_train_bbh():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from meta_past.data.bbh_contexts import iter_train_val
    train, _ = iter_train_val(train_size=1, val_size=1)
    c = train[0]
    return {
        "context": c.context,
        "question": c.qa[0].question,
        "references": list(c.qa[0].references),
        "_notes": (
            f"context_id={c.context_id}. Context = K demonstrations from the SAME BBH task family "
            f"(greedy-packed under a token budget); question = a different held-out item from that "
            f"family. The LoRA must encode which-task-this-is + how-to-solve-it."
        ),
    }


# -------------------- Bucket A: context-grounded QA --------------------

def fetch_hotpotqa():
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation",
                      trust_remote_code=True)
    ex = ds[0]
    sup = {(t, sid) for t, sid in
           zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"])}
    # Use only supporting paragraphs (mirrors how we'd budget context).
    blocks = []
    for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
        if any(t == title for (t, _) in sup):
            blocks.append(f"# {title}\n{''.join(sents)}")
    return {
        "context": "\n\n".join(blocks),
        "question": ex["question"],
        "references": [ex["answer"]],
        "_notes": f"id={ex['id']}, type={ex['type']}, level={ex['level']}, n_supporting={len(sup)}",
    }


def fetch_2wikimulti():
    from datasets import load_dataset
    import ast
    ds = load_dataset("xanhho/2WikiMultiHopQA", split="validation")
    ex = ds[0]
    # context is a JSON-ish string: [[title, [sent, sent, ...]], ...]
    ctx_raw = ex["context"]
    try:
        paras = json.loads(ctx_raw)
    except Exception:
        paras = ast.literal_eval(ctx_raw)
    sup = ex.get("supporting_facts", [])
    try:
        sup = json.loads(sup) if isinstance(sup, str) else sup
    except Exception:
        sup = ast.literal_eval(sup) if isinstance(sup, str) else sup
    sup_titles = {t for t, _sid in sup} if sup else set()
    blocks = []
    for title, sents in paras:
        if (not sup_titles) or (title in sup_titles):
            blocks.append(f"# {title}\n{''.join(sents)}")
    return {
        "context": "\n\n".join(blocks),
        "question": ex["question"],
        "references": [ex["answer"]],
        "_notes": f"id={ex['_id']}, type={ex['type']}, n_supporting={len(sup_titles)}",
    }


def fetch_drop():
    from datasets import load_dataset
    ds = load_dataset("ucinlp/drop", split="validation")
    ex = ds[0]
    ans = ex["answers_spans"]
    # DROP answer formats: number / spans / date.
    refs = []
    if ans.get("spans"):
        refs.extend(ans["spans"])
    if not refs:
        refs.append(str(ans))
    return {
        "context": ex["passage"],
        "question": ex["question"],
        "references": refs,
        "_notes": f"section_id={ex['section_id']}, query_id={ex['query_id']}",
    }


def fetch_narrativeqa():
    from datasets import load_dataset
    ds = load_dataset("deepmind/narrativeqa", split="validation")
    ex = ds[0]
    summary = ex["document"]["summary"]["text"]
    refs = [a["text"] for a in ex["answers"]]
    return {
        "context": summary,                            # we use the summary
        "question": ex["question"]["text"],
        "references": refs,
        "_notes": (
            f"doc_kind={ex['document']['kind']} (book/movie). "
            "We use the summary as context; full text is the alternative, "
            "but ranges 50k–100k tokens — out of scope unless we chunk."
        ),
    }


def fetch_pubmedqa():
    from datasets import load_dataset
    # 'pqa_labeled' is the 1000-example test split with yes/no/maybe labels.
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    ex = ds[0]
    ctx = "\n\n".join(ex["context"]["contexts"])
    return {
        "context": ctx,
        "question": ex["question"],
        "references": [ex["final_decision"]],          # yes / no / maybe
        "_notes": f"pubid={ex['pubid']}, long_answer (gold rationale) also available",
    }


def fetch_qasper():
    # `allenai/qasper` is a loading script which `datasets>=3.0` refuses
    # to run. There is no canonical parquet mirror as of 2026-05; users
    # currently grab the JSON from
    #   https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz
    # and parse it themselves. We surface this as a noted gap rather than
    # bend the survey schema to fit a shape we wouldn't actually eval on.
    raise RuntimeError(
        "qasper is loading-script only; needs manual JSON download from "
        "qasper-dataset.s3.us-west-2.amazonaws.com (no Hub mirror)"
    )


def fetch_boolq():
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="validation")
    ex = ds[0]
    return {
        "context": ex["passage"],
        "question": ex["question"],
        "references": ["yes" if ex["answer"] else "no"],
    }


def fetch_triviaqa():
    from datasets import load_dataset
    # Use the with-context split. Each example carries evidence
    # documents (wiki_context entity_pages + search results); we
    # concatenate the wiki entity pages as the eval context.
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation")
    ex = ds[0]
    pages = ex.get("entity_pages", {})
    titles = pages.get("title", []) if isinstance(pages, dict) else []
    bodies = pages.get("wiki_context", []) if isinstance(pages, dict) else []
    blocks = []
    for t, b in zip(titles, bodies):
        if not b:
            continue
        blocks.append(f"# {t}\n{b}")
    context = "\n\n".join(blocks) or "(no wiki evidence — fallback to search_results)"
    refs = []
    ans = ex.get("answer", {}) or {}
    if ans.get("value"):
        refs.append(ans["value"])
    refs.extend(ans.get("aliases", []) or [])
    return {
        "context": context,
        "question": ex["question"],
        "references": refs[:6],
        "_notes": f"qid={ex.get('question_id')}. Wiki entity pages concatenated as context; `answer.value` + `answer.aliases` all count as correct under F1/EM.",
    }


def fetch_newsqa():
    from datasets import load_dataset
    ds = load_dataset("lucadiliello/newsqa", split="validation")
    ex = ds[0]
    return {
        "context": ex["context"],
        "question": ex["question"],
        "references": list(ex["answers"]),
    }


# -------------------- Bucket B: in-parameter few-shot --------------------

def _fmt_mcq(input_text: str, choices: list[str], target_letter: str | None = None) -> str:
    s = input_text + "\nOptions:\n"
    for i, c in enumerate(choices):
        s += f"({chr(65+i)}) {c}\n"
    return s.rstrip() + (f"\nAnswer: ({target_letter})" if target_letter else "")


def fetch_mmlu():
    from datasets import load_dataset
    # 'astronomy' has a small dev split we can demo from + test for the query.
    ds_dev = load_dataset("cais/mmlu", "astronomy", split="dev")        # 5 demo items per subject
    ds_test = load_dataset("cais/mmlu", "astronomy", split="test")
    def fmt(ex, with_answer=False):
        return _fmt_mcq(
            ex["question"], ex["choices"],
            target_letter=chr(65 + ex["answer"]) if with_answer else None,
        )
    demos = "\n\n".join(fmt(ds_dev[i], with_answer=True) for i in range(2))
    q = ds_test[0]
    return {
        "context": demos,
        "question": fmt(q),
        "references": [chr(65 + q["answer"])],
        "_notes": f"subject=astronomy. K=2 demos shown; real eval can pack ≥5 (5-shot is standard for MMLU).",
    }


def fetch_mmlu_pro():
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    # MMLU-Pro has a 'category' field and a 'cot_content' field; build demos
    # from the same category as the query.
    cat = ds[0]["category"]
    same = [e for e in ds.select(range(40)) if e["category"] == cat]
    demo_items = same[1:3]
    query = same[0]
    def fmt(ex, with_answer=False):
        s = ex["question"] + "\nOptions:\n"
        for i, c in enumerate(ex["options"]):
            s += f"({chr(65+i)}) {c}\n"
        return s.rstrip() + (f"\nAnswer: ({ex['answer']})" if with_answer else "")
    demos = "\n\n".join(fmt(d, with_answer=True) for d in demo_items)
    return {
        "context": demos,
        "question": fmt(query),
        "references": [query["answer"]],
        "_notes": f"category={cat}. 10-option MCQ. Real eval: pack ~3-5 demos per category.",
    }


def fetch_agieval():
    from datasets import load_dataset
    # Pick LSAT-LR as a representative english reasoning subtest.
    for repo in ("hails/agieval-lsat-lr", "hails/agieval-sat-math"):
        try:
            ds = load_dataset(repo, split="test")
            break
        except Exception:
            continue
    ex = ds[0]
    # AGIEval items: {query, choices, gold}
    def fmt(e, with_answer=False):
        s = e["query"]
        if with_answer and e.get("gold") is not None:
            ans = e["gold"][0] if isinstance(e["gold"], list) else e["gold"]
            return s + f"\nAnswer: ({chr(65+ans) if isinstance(ans, int) else ans})"
        return s
    demos = "\n\n".join(fmt(ds[i], with_answer=True) for i in (1, 2))
    return {
        "context": demos,
        "question": fmt(ex),
        "references": [str(ex["gold"])],
        "_notes": "AGIEval english subset; format varies per subtest. Real eval: pack 2-3 demos.",
    }


def fetch_arc_challenge():
    from datasets import load_dataset
    ds_train = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    ds_test = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    def fmt(ex, with_answer=False):
        s = _fmt_mcq(ex["question"], ex["choices"]["text"],
                     target_letter=ex["answerKey"] if with_answer else None)
        return s
    demos = "\n\n".join(fmt(ds_train[i], with_answer=True) for i in range(2))
    q = ds_test[0]
    return {
        "context": demos,
        "question": fmt(q),
        "references": [q["answerKey"]],
    }


def fetch_openbookqa():
    from datasets import load_dataset
    ds_train = load_dataset("allenai/openbookqa", "main", split="train")
    ds_test = load_dataset("allenai/openbookqa", "main", split="test")
    def fmt(ex, with_answer=False):
        s = _fmt_mcq(ex["question_stem"], ex["choices"]["text"],
                     target_letter=ex["answerKey"] if with_answer else None)
        return s
    demos = "\n\n".join(fmt(ds_train[i], with_answer=True) for i in range(2))
    q = ds_test[0]
    return {
        "context": demos,
        "question": fmt(q),
        "references": [q["answerKey"]],
    }


def fetch_commonsenseqa():
    from datasets import load_dataset
    ds_train = load_dataset("tau/commonsense_qa", split="train")
    ds_val = load_dataset("tau/commonsense_qa", split="validation")
    def fmt(ex, with_answer=False):
        return _fmt_mcq(ex["question"], ex["choices"]["text"],
                        target_letter=ex["answerKey"] if with_answer else None)
    demos = "\n\n".join(fmt(ds_train[i], with_answer=True) for i in range(2))
    q = ds_val[0]
    return {
        "context": demos,
        "question": fmt(q),
        "references": [q["answerKey"]],
    }


def fetch_hellaswag():
    from datasets import load_dataset
    ds_train = load_dataset("Rowan/hellaswag", split="train")
    ds_val = load_dataset("Rowan/hellaswag", split="validation")
    def fmt(ex, with_answer=False):
        stem = ex["ctx"] + " ___ "
        s = stem + "\nOptions:\n"
        for i, c in enumerate(ex["endings"]):
            s += f"({chr(65+i)}) {c}\n"
        if with_answer:
            s += f"Answer: ({chr(65+int(ex['label']))})"
        return s.rstrip()
    demos = "\n\n".join(fmt(ds_train[i], with_answer=True) for i in range(2))
    q = ds_val[0]
    return {
        "context": demos,
        "question": fmt(q),
        "references": [chr(65+int(q["label"]))],
    }


def fetch_piqa():
    from datasets import load_dataset
    # `ybisk/piqa` is a loading-script dataset; use the parquet mirror.
    ds_train = load_dataset("lighteval/piqa", split="train")
    ds_val = load_dataset("lighteval/piqa", split="validation")
    def fmt(ex, with_answer=False):
        s = (ex["goal"] + "\nOptions:\n"
             f"(A) {ex['sol1']}\n(B) {ex['sol2']}\n")
        if with_answer:
            s += f"Answer: ({'A' if ex['label']==0 else 'B'})"
        return s.rstrip()
    demos = "\n\n".join(fmt(ds_train[i], with_answer=True) for i in range(2))
    q = ds_val[0]
    return {
        "context": demos,
        "question": fmt(q),
        "references": ["A" if q["label"]==0 else "B"],
    }


def fetch_gsm8k():
    from datasets import load_dataset
    ds_train = load_dataset("openai/gsm8k", "main", split="train")
    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    def fmt(ex, with_answer=False):
        s = "Question: " + ex["question"]
        if with_answer:
            s += "\nAnswer: " + ex["answer"]
        return s
    demos = "\n\n".join(fmt(ds_train[i], with_answer=True) for i in range(2))
    q = ds_test[0]
    # Gold answer string ends with '#### <number>'
    gold = q["answer"].split("####")[-1].strip()
    return {
        "context": demos,
        "question": "Question: " + q["question"] + "\nAnswer:",
        "references": [gold, q["answer"]],
        "_notes": "demos include the full CoT; eval scorer extracts the final number after '####'.",
    }


def fetch_math():
    from datasets import load_dataset
    for repo in ("hendrycks/competition_math", "lighteval/MATH", "HuggingFaceH4/MATH-500"):
        try:
            ds = load_dataset(repo, split="test")
            break
        except Exception:
            continue
    ex = ds[0]
    return {
        "context": "(demos would be drawn from the same `level` and `type`)",
        "question": ex.get("problem", ex.get("question", "")),
        "references": [ex.get("solution", ex.get("answer", ""))],
        "_notes": f"level={ex.get('level')}, type={ex.get('type')}. Reference is the full LaTeX solution; scorer extracts the boxed expression and uses sympy equivalence.",
    }


def fetch_strategyqa():
    from datasets import load_dataset
    for repo in ("ChilleD/StrategyQA", "voidful/StrategyQA", "wics/strategy-qa"):
        try:
            ds = load_dataset(repo, split="test")
            break
        except Exception:
            try:
                ds = load_dataset(repo, split="train")
                break
            except Exception:
                continue
    ex = ds[0]
    def fmt(e, with_answer=False):
        q = e.get("question", e.get("input", ""))
        ans = e.get("answer", None)
        s = "Question: " + q
        if with_answer and ans is not None:
            s += f"\nAnswer: {'yes' if ans is True else 'no' if ans is False else ans}"
        return s
    demos_src = [ds[i] for i in (1, 2)]
    demos = "\n\n".join(fmt(d, with_answer=True) for d in demos_src)
    return {
        "context": demos,
        "question": fmt(ex),
        "references": ["yes" if ex.get("answer") is True else "no" if ex.get("answer") is False else str(ex.get("answer"))],
    }


def fetch_logiqa():
    # LogiQA's official Hub repos (`lucasmccabe/logiqa`,
    # `EleutherAI/logiqa`, `baber/logiqa2`) are all loading-script
    # datasets, which `datasets>=3.0` refuses to run. The
    # `tasksource/logiqa-2.0-nli` parquet mirror exists but reframes the
    # task as NLI (premise/hypothesis/label), so it isn't directly usable
    # for the MCQ few-shot setup we'd want. Flag as a gap.
    raise RuntimeError(
        "logiqa: all MCQ Hub mirrors are loading-script (broken on "
        "datasets>=3.0); only tasksource/logiqa-2.0-nli (NLI reframe) is "
        "parquet-accessible. Skip or fetch raw JSON manually."
    )


def fetch_babi():
    from datasets import load_dataset
    # Schema: passage / question / answer / task. 20 task families; we
    # demo and query from the SAME family (task=1, "location") so the
    # display mirrors the real eval setup.
    ds_test = load_dataset("Muennighoff/babi", split="test").filter(
        lambda e: e["task"] == 1
    )
    ds_train = load_dataset("Muennighoff/babi", split="train").filter(
        lambda e: e["task"] == 1
    )
    def fmt(e, with_answer=False):
        s = "Story:\n" + e["passage"] + f"Q: {e['question']}"
        return s + (f"\nA: {e['answer']}" if with_answer else "")
    demos = "\n\n".join(fmt(ds_train[i], with_answer=True) for i in range(2))
    q = ds_test[0]
    return {
        "context": demos,
        "question": fmt(q),
        "references": [q["answer"]],
        "_notes": f"task={q['task']} (location/where-is-X). 20 task families total — real eval packs K demos per family.",
    }


def fetch_bigbench_non_hard():
    from datasets import load_dataset
    # Pick a few non-Hard BIG-Bench tasks via tasksource/bigbench.
    for cand in ("known_unknowns", "anachronisms", "cause_and_effect"):
        try:
            ds = load_dataset("tasksource/bigbench", cand, split="validation")
            ex = ds[0]
            def fmt(e, with_answer=False):
                stem = e.get("inputs", e.get("input", ""))
                targs = e.get("targets", [])
                s = stem
                if with_answer and targs:
                    s += f"\nAnswer: {targs[0]}"
                return s
            demos = "\n\n".join(fmt(ds[i], with_answer=True) for i in (1, 2))
            return {
                "context": demos,
                "question": fmt(ex),
                "references": list(ex.get("targets", [])),
                "_notes": f"task={cand}. Pick any non-BBH-Hard task here for OOD coverage.",
            }
        except Exception:
            continue
    raise RuntimeError("no non-Hard BIG-Bench task reachable")


def fetch_natural_instructions():
    from datasets import load_dataset
    for repo in ("Muennighoff/natural-instructions", "andersonbcdefg/super_natural_instructions"):
        try:
            ds = load_dataset(repo, split="test", streaming=True)
            it = iter(ds)
            ex = next(it)
            break
        except Exception:
            continue
    # Schema: {task_name, definition, inputs, targets, ...}
    return {
        "context": (
            "Task definition:\n" + str(ex.get("definition", "")) +
            "\n\nPositive examples (demos):\n" +
            json.dumps(ex.get("positive_examples", [])[:2], indent=2, ensure_ascii=False)[:1000]
        ),
        "question": str(ex.get("inputs", ex.get("input", ""))),
        "references": list(ex.get("targets", [ex.get("target", "")])) if ex.get("targets") else [ex.get("target", "")],
        "_notes": f"task={ex.get('task_name', '?')}. NatInst supplies a definition + positive examples — natural fit for SHINE context.",
    }


def fetch_truthfulqa_mc():
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    ex = ds[0]
    choices = ex["mc1_targets"]["choices"]
    labels = ex["mc1_targets"]["labels"]
    correct_idx = labels.index(1)
    return {
        "context": "(use K demos from other categories; here we show MC1 single-correct format)",
        "question": _fmt_mcq(ex["question"], choices),
        "references": [chr(65 + correct_idx)],
        "_notes": "mc1 = exactly one correct; mc2 = multiple correct (use ROC-AUC). Generation split also exists for free-form judging.",
    }


# -------------------- Bucket C: zero-shot probes --------------------

def fetch_gsm8k_zeroshot():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    q = ds[0]
    return {
        "context": "",                                  # zero-shot: empty context
        "question": q["question"],
        "references": [q["answer"].split("####")[-1].strip()],
        "_notes": "Same GSM8K data as bucket B but eval with empty context. Compares SHINE-LoRA vs base Qwen3 with no in-context help.",
    }


def fetch_mmlu_zeroshot():
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "astronomy", split="test")
    q = ds[0]
    return {
        "context": "",
        "question": _fmt_mcq(q["question"], q["choices"]),
        "references": [chr(65 + q["answer"])],
        "_notes": "Subject = astronomy. Same MMLU items, no demos in context.",
    }


def fetch_humaneval():
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    ex = ds[0]
    return {
        "context": "",
        "question": ex["prompt"],
        "references": [ex["canonical_solution"]],
        "_notes": (
            f"task_id={ex['task_id']}. Eval = run candidate completion through "
            f"the dataset's `test` field (a unit-test script). Pass@1 is the metric. "
            "Useful as a 'does SHINE break code abilities' side-effect check."
        ),
    }


def fetch_truthfulqa_gen():
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    ex = ds[0]
    return {
        "context": "",
        "question": ex["question"],
        "references": list(ex["correct_answers"]),
        "_notes": "Generation split; reference set = correct answers; judging usually done by GPT-judge or BLEURT.",
    }


# -------------------- Bucket D: long context --------------------

def fetch_longbench():
    from datasets import load_dataset
    # THUDM/LongBench (v1) is loading-script only; v2 has a Hub parquet
    # mirror. v2 differs: MCQ (4-option) over very long contexts (~25k
    # tokens). We use v2 here — it's still a useful long-context probe.
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    ex = ds[0]
    question_block = (
        ex["question"] + "\nOptions:\n"
        f"(A) {ex['choice_A']}\n(B) {ex['choice_B']}\n"
        f"(C) {ex['choice_C']}\n(D) {ex['choice_D']}"
    )
    return {
        "context": ex["context"],
        "question": question_block,
        "references": [ex["answer"]],
        "_notes": (
            f"domain={ex.get('domain')}, length_class={ex.get('length')}, "
            f"difficulty={ex.get('difficulty')}. v1 (free-form QA over 14 "
            "tasks) is loading-script only — v2 (MCQ over very long context) "
            "shown here."
        ),
    }


def fetch_quality():
    from datasets import load_dataset
    for repo in ("emozilla/quality", "tau/quality"):
        try:
            ds = load_dataset(repo, split="validation")
            break
        except Exception:
            continue
    ex = ds[0]
    return {
        "context": ex.get("article", ex.get("story", "")),
        "question": _fmt_mcq(ex.get("question", ""), ex.get("options", [])),
        "references": [chr(65 + ex.get("gold_label", 0))],
        "_notes": "~5k-token articles, 4-option MCQ. Pure long-context comprehension.",
    }


# -------------------- driver --------------------

DATASETS = {
    # Bucket A — context-grounded QA (includes training datasets that
    # share this format; bucket assignment is by format, not by whether
    # the dataset was seen during training).
    "train_squad":       ("A", fetch_train_squad),
    "train_musique":     ("A", fetch_train_musique),
    "hotpotqa":          ("A", fetch_hotpotqa),
    "2wikimulti":        ("A", fetch_2wikimulti),
    "drop":              ("A", fetch_drop),
    "narrativeqa":       ("A", fetch_narrativeqa),
    "pubmedqa":          ("A", fetch_pubmedqa),
    "boolq":             ("A", fetch_boolq),
    "triviaqa":          ("A", fetch_triviaqa),
    "newsqa":            ("A", fetch_newsqa),
    # Bucket B — in-parameter few-shot
    "train_bbh":         ("B", fetch_train_bbh),
    "mmlu":              ("B", fetch_mmlu),
    "mmlu_pro":          ("B", fetch_mmlu_pro),
    "agieval":           ("B", fetch_agieval),
    "arc_challenge":     ("B", fetch_arc_challenge),
    "openbookqa":        ("B", fetch_openbookqa),
    "commonsenseqa":     ("B", fetch_commonsenseqa),
    "hellaswag":         ("B", fetch_hellaswag),
    "piqa":              ("B", fetch_piqa),
    "gsm8k":             ("B", fetch_gsm8k),
    "strategyqa":        ("B", fetch_strategyqa),
    "babi":              ("B", fetch_babi),
    "bigbench_non_hard": ("B", fetch_bigbench_non_hard),
    "natural_instr":     ("B", fetch_natural_instructions),
    "truthfulqa_mc":     ("B", fetch_truthfulqa_mc),
    # Bucket C — zero-shot probes
    "gsm8k_zeroshot":    ("C", fetch_gsm8k_zeroshot),
    "mmlu_zeroshot":     ("C", fetch_mmlu_zeroshot),
    "humaneval":         ("C", fetch_humaneval),
    "truthfulqa_gen":    ("C", fetch_truthfulqa_gen),
}


def main():
    out: dict = {}
    for key, (bucket, fn) in DATASETS.items():
        print(f"[{bucket}] {key} ...", flush=True)
        result = safe(fn)
        if "_error" in result:
            print(f"  ERROR: {result['_error']}", flush=True)
        else:
            ctx_n = len(result.get("context", "") or "")
            q_n = len(result.get("question", "") or "")
            print(f"  OK  ctx_chars={ctx_n}  q_chars={q_n}  n_refs={len(result.get('references', []))}", flush=True)
        result["_bucket"] = bucket
        # truncate context for storage (we display truncated in markdown anyway)
        if isinstance(result.get("context"), str):
            result["context_full_chars"] = len(result["context"])
            result["context"] = truncate(result["context"], 1500)
        if isinstance(result.get("question"), str):
            result["question_full_chars"] = len(result["question"])
            result["question"] = truncate(result["question"], 1200)
        if isinstance(result.get("references"), list):
            result["references"] = [truncate(r, 600) for r in result["references"]][:6]
        out[key] = result

    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
