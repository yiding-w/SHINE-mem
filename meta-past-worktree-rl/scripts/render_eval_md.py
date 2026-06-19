"""Render the eval-dataset survey markdown from eval_samples.json.

Combines the static metadata (size, HF repo, scoring, rationale) with
the live sample fetched by scripts/fetch_eval_samples.py, so the
context / question / references shown for each dataset come from a real
row of the corresponding HF dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "eval_samples.json"
OUT_PATH = ROOT / "eval_datasets.md"


META = {
    # ─── Bucket A: context-grounded QA ──────────────────────────────
    # 训练用过的也放这里（按形态分桶，不按是否训练过分桶）。
    "train_squad": dict(
        title="SQuAD v1  🟡训练已用",
        bucket="A",
        hf="`rajpurkar/squad` (parquet 缓存于 `~/huggingfacemodels/squad/`)",
        size="~2k 唯一 passage @ validation",
        answer="抽取式短 span",
        scoring="F1（官方 SQuAD F1）",
        why=(
            "**SHINE 的预训练任务**——hypernet 在这上面学会"
            "「把 passage 压成 LoRA」的基础能力。"
            "RL config `rl_squad_grpo.yaml` 把它当 baseline / smoke run。"
            "**可作 eval**：在别的数据集上训过、想反过来测它有没有退化。"
        ),
        eval_setup=(
            "loader = `meta_past.data.squad_contexts.iter_train_val`。"
            "每个 SquadContext 包含 `.context`（一段 wiki passage）和 `.qa`（多个 QA pair）。"
        ),
    ),
    "train_musique": dict(
        title="MuSiQue  🟡训练已用",
        bucket="A",
        hf="`dgslibisey/MuSiQue` (parquet)",
        size="~17.5k validation contexts",
        answer="抽取式短 span",
        scoring="F1（同 SQuAD 规范）",
        why=(
            "**RL 训练主力之一**：多跳 QA，context 已把 supporting paragraphs 拼成"
            "「# Title\\nbody」blocks。配置：`rl_musique_grpo.yaml`（F1）或 "
            "`rl_musique_grpo_judge.yaml`（HTTP LLM judge）。"
            "**可作 eval**：用 BBH/SQuAD 训过后看多跳能力变化。"
        ),
        eval_setup=(
            "loader = `meta_past.data.musique_contexts.iter_train_val`。"
            "每个 MusiqueContext 一条 multi-hop question。"
        ),
    ),

    "hotpotqa": dict(
        title="HotpotQA (distractor)",
        bucket="A",
        hf="`hotpot_qa` (config=`distractor`)",
        size="~7.4k validation",
        answer="short span",
        scoring="F1 / EM (official)",
        why=(
            "多跳 QA 老牌数据集；与 MuSiQue 同精神但**不同发布**——MuSiQue 训过的话直接看泛化。"
            "Distractor 配置有 10 个 paragraph（2 支持 + 8 噪声），可挑战 retrieval。"
        ),
        eval_setup="context = 把 supporting paragraphs 拼起来；question = 自然语言问句；reference = 短答案字符串。",
    ),
    "2wikimulti": dict(
        title="2WikiMultihopQA",
        bucket="A",
        hf="`xanhho/2WikiMultiHopQA`",
        size="~12.5k validation",
        answer="short span",
        scoring="F1 / EM",
        why=(
            "也是多跳，但题目模板更显式（compositional / inference / bridge_comparison / comparison 四类），"
            "可看模型是否依赖训练里学到的模板。"
        ),
        eval_setup="context = supporting paragraphs；question = 复合问句；reference = 短答案字符串。",
    ),
    "drop": dict(
        title="DROP",
        bucket="A",
        hf="`ucinlp/drop`",
        size="~9.5k validation",
        answer="number / spans / date",
        scoring="数字 EM + spans F1（官方）",
        why=(
            "**离散推理**：计数、加减、排序、最大值——MuSiQue / SQuAD 完全没有这种形态。"
            "答案是 number / spans / date 之一，需要模型在 passage 上做算术。"
        ),
        eval_setup=(
            "context = 一段 passage（通常是新闻段落或维基百科条目）；question = 包含数字推理的问句；"
            "reference = 答案 span 列表 + 数字答案。"
        ),
    ),
    "narrativeqa": dict(
        title="NarrativeQA",
        bucket="A",
        hf="`deepmind/narrativeqa`",
        size="~3.5k validation",
        answer="自由短答",
        scoring="F1 / BLEU / METEOR",
        why=(
            "**Domain shift**：小说 / 电影剧本节选（vs. 训练里的维基百科）。"
            "我们用 summary 作 context 控制长度；full text 50k–100k tokens，需要 chunk。"
        ),
        eval_setup="context = 故事 summary（~1.5–2k chars）；question = 关于故事的问句；reference = 多个 paraphrase 答案。",
    ),
    "pubmedqa": dict(
        title="PubMedQA",
        bucket="A",
        hf="`qiaojin/PubMedQA` (config=`pqa_labeled`)",
        size="1000 test (labeled)",
        answer="yes / no / maybe",
        scoring="EM on label",
        why=(
            "**强领域迁移**：生物医学论文摘要 + 是非问句。"
            "Labeled subset 有人工标注 long answer（gold rationale），可二次评测。"
        ),
        eval_setup="context = abstract 拼成的段落；question = 二分类/三分类问句；reference = ['yes' | 'no' | 'maybe']。",
    ),
    "boolq": dict(
        title="BoolQ",
        bucket="A",
        hf="`google/boolq`",
        size="~3.3k validation",
        answer="yes / no",
        scoring="EM",
        why="一段 wiki + 是非题，最便宜的 sanity baseline；可看模型对 passage 的 yes-no 判断是否稳定。",
        eval_setup="context = passage；question = yes/no 自然语言问句；reference = ['yes'] or ['no']。",
    ),
    "triviaqa": dict(
        title="TriviaQA (RC, Wikipedia)",
        bucket="A",
        hf="`mandarjoshi/trivia_qa` (config=`rc.wikipedia`)",
        size="~11k validation",
        answer="short answer + aliases",
        scoring="F1 / EM（aliases 都算对）",
        why="与训练 QA 形态最像（事实型 + wiki 段落），可作弱 OOD 基线。",
        eval_setup=(
            "context = wiki entity pages 拼成的「# Title\\nbody」blocks（rc.wikipedia 配置内置）；"
            "question = 事实型问句；reference = 答案 value + aliases 列表（任一命中即算对）。"
            "每条样本的 context 通常很长（~30k chars），eval 时需截断到 `context_max_length=1024`。"
        ),
    ),
    "newsqa": dict(
        title="NewsQA",
        bucket="A",
        hf="`lucadiliello/newsqa`",
        size="~4.2k validation",
        answer="short span",
        scoring="F1",
        why="CNN 新闻文章 + 抽取式 QA；测新闻领域 vs. 训练里的维基段落。",
        eval_setup="context = 新闻原文；question = 文章问句；reference = 答案 span 列表。",
    ),
    # ─── Bucket B: in-parameter few-shot ────────────────────────────
    "train_bbh": dict(
        title="BIG-Bench Hard  🟡训练已用",
        bucket="B",
        hf="`maveriq/bigbenchhard` (27 tasks，每 task 250 条)",
        size="~5.2k 构造的 contexts（80% train split × 27 tasks）",
        answer="任务依赖：MCQ 字母 / yes/no / 数字 / 排序后的词表 / 多类型",
        scoring="F1（normalizer 折叠 `(A)`→`a`，对 BBH 短答案足够）",
        why=(
            "**RL 训练 in-parameter few-shot 主战场**：context = K 个 demonstration（greedy "
            "packed under token budget），question = 同任务 held-out 一题。"
            "LoRA 必须同时编码「这是哪个任务」+「怎么解」。"
            "配置：`rl_bbh_grpo.yaml`（`enable_thinking: false` 强制直答）。"
            "**可作 eval**：在别的 few-shot 数据集上训过、想测 BBH 上的能力保留。"
        ),
        eval_setup=(
            "loader = `meta_past.data.bbh_contexts.iter_train_val`。"
            "每个 BBHContext: `.context` = K 个 demo 串成「Input: …\\nAnswer: …」格式；"
            "`.qa[0]` = held-out 一题；reference = 该题 gold target。"
        ),
    ),
    "mmlu": dict(
        title="MMLU",
        bucket="B",
        hf="`cais/mmlu` (per-subject configs，共 57 个 subject)",
        size="~14k test，per-subject dev 用作 demo 池",
        answer="MCQ (A/B/C/D)",
        scoring="EM on letter",
        why=(
            "黄金标准 few-shot benchmark；57 个学科可做 per-subject 切分。"
            "K-shot demos 同 subject、test 另留——直接对标公开榜。"
        ),
        eval_setup=(
            "context = K 个同 subject 的 demo（含答案）；question = 同 subject 的 held-out 题；"
            "reference = 正确选项字母。展示用 K=2，实际跑 5-shot 是标准。"
        ),
    ),
    "mmlu_pro": dict(
        title="MMLU-Pro",
        bucket="B",
        hf="`TIGER-Lab/MMLU-Pro`",
        size="~12k test",
        answer="MCQ (A–J, 10 选项)",
        scoring="EM",
        why="MMLU 升级版，10 选项、更难、**抗记忆性更强**；包含 CoT reference。",
        eval_setup="context = K 个同 category 的 demo；question = 同 category held-out 题；reference = 选项字母。",
    ),
    "agieval": dict(
        title="AGIEval (English)",
        bucket="B",
        hf="`hails/agieval-lsat-lr` 等若干 subtest",
        size="数百 per subtest（LSAT-LR=510）",
        answer="MCQ / 数值（视 subtest）",
        scoring="EM",
        why=(
            "SAT / LSAT / 数学竞赛 / 法律考试，**完全不在 BBH 风格里**；"
            "测应试题题型上的泛化。"
        ),
        eval_setup="context = K 个同 subtest 的 demo；question = held-out 题；reference = 答案 index/字母。",
    ),
    "arc_challenge": dict(
        title="ARC-Challenge",
        bucket="B",
        hf="`allenai/ai2_arc` (config=`ARC-Challenge`)",
        size="1172 test",
        answer="MCQ (4 选项)",
        scoring="EM",
        why="小学科学题，干净、便宜、有 train split 提供 demo；适合做快速 sanity check。",
        eval_setup="context = K 个 train 集 demo；question = test 题；reference = 选项字母。",
    ),
    "openbookqa": dict(
        title="OpenBookQA",
        bucket="B",
        hf="`allenai/openbookqa` (config=`main`)",
        size="500 test",
        answer="MCQ (4 选项)",
        scoring="EM",
        why="科学常识 + 可选 knowledge book（`additional` config 提供）；小而稳定。",
        eval_setup="context = K demos（可选 knowledge facts 拼到前面）；question = test 题；reference = 选项字母。",
    ),
    "commonsenseqa": dict(
        title="CommonsenseQA",
        bucket="B",
        hf="`tau/commonsense_qa`",
        size="1221 dev",
        answer="MCQ (5 选项)",
        scoring="EM",
        why="ConceptNet-based 概念常识 MCQ；测概念关联推理。",
        eval_setup="context = K demos；question = held-out 题；reference = 选项字母。",
    ),
    "hellaswag": dict(
        title="HellaSwag",
        bucket="B",
        hf="`Rowan/hellaswag`",
        size="~10k validation",
        answer="MCQ (4 个续写)",
        scoring="EM",
        why="续写式 MCQ（给上下文，4 选其后续）；与 BBH 形态差异大，覆盖叙事推理。",
        eval_setup="context = K 续写式 demos；question = 续写题（要选 A/B/C/D）；reference = 选项字母。",
    ),
    "piqa": dict(
        title="PIQA",
        bucket="B",
        hf="`lighteval/piqa` (parquet mirror, 原版 `ybisk/piqa` loading-script broken)",
        size="1838 validation",
        answer="MCQ (2 选项)",
        scoring="EM",
        why="物理常识：两种实现方法选更靠谱的那个。形态简单、信号清晰。",
        eval_setup="context = K demos；question = goal + 2 选项；reference = 'A' or 'B'。",
    ),
    "gsm8k": dict(
        title="GSM8K",
        bucket="B",
        hf="`openai/gsm8k` (config=`main`)",
        size="1319 test",
        answer="数字（带 CoT solution）",
        scoring="numeric EM（从 `#### N` 提取）",
        why=(
            "小学数学应用题；K-shot demos 包含完整 CoT，测**数值推理 + 格式遵循**。"
            "BBH 里没有连续数字运算，这块是真正的 OOD。"
        ),
        eval_setup="context = K demos（含 CoT）；question = 新数学题；reference = 数字答案（scorer 提取 `####` 后数字）。",
    ),
    "strategyqa": dict(
        title="StrategyQA",
        bucket="B",
        hf="`ChilleD/StrategyQA` 等社区镜像",
        size="490 test（含推理链 facts）",
        answer="yes / no + 推理链",
        scoring="EM on yes/no",
        why="**隐式多步推理**是非题（问题表面是非，实际需要 2-3 步推理）；测推理链能力。",
        eval_setup="context = K 同分布 demo；question = 是非题（不含 facts）；reference = ['yes'] or ['no']。",
    ),
    "babi": dict(
        title="bAbI",
        bucket="B",
        hf="`Muennighoff/babi`",
        size="20 任务 × 各 1000 test",
        answer="单词 / 短答",
        scoring="EM",
        why="20 个合成推理任务（location / chaining / counting / ...）；**纯净 sanity check**，跑起来超快。",
        eval_setup="context = K 同任务 demo（含 story + Q + A）；question = 同任务 held-out story + Q；reference = 单词答案。",
    ),
    "bigbench_non_hard": dict(
        title="BIG-Bench (non-Hard)",
        bucket="B",
        hf="`tasksource/bigbench` (每个任务一个 config)",
        size="~200 任务，规模差异大",
        answer="任务依赖",
        scoring="任务依赖",
        why=(
            "BBH 训过的话，**剩下的 BIG-Bench 任务**就是天然 held-out。"
            "**注意**：要严格剔除 BBH 那 27 个 task。"
        ),
        eval_setup="context = K 同任务 demo；question = same task held-out；reference = `targets` 列表。",
    ),
    "natural_instr": dict(
        title="Super-NaturalInstructions (test split)",
        bucket="B",
        hf="`Muennighoff/natural-instructions`",
        size="119 测试任务、每任务 100 样本",
        answer="任务依赖（短文本居多）",
        scoring="ROUGE-L / EM",
        why=(
            "**purpose-built held-out task generalization**——任务本身就是为测 generalization 而设计的；"
            "**自带 task definition + positive examples**，与 SHINE 的 context 形态天然契合。"
        ),
        eval_setup="context = task definition + K positive examples（demos）；question = 新 input；reference = `targets` 列表。",
    ),
    "truthfulqa_mc": dict(
        title="TruthfulQA (MC1)",
        bucket="B",
        hf="`truthfulqa/truthful_qa` (config=`multiple_choice`)",
        size="817 validation",
        answer="MC1 = 单正确 / MC2 = 多正确",
        scoring="MC1 EM / MC2 ROC-AUC",
        why="设计上诱发幻觉的题目；测 SHINE 是否减损模型的真实性判断。",
        eval_setup=(
            "context = K demos（可选他类别）；question = question + N 选项；"
            "reference = MC1 正确选项字母。"
        ),
    ),
    # ─── Bucket C: zero-shot probes ─────────────────────────────────
    "gsm8k_zeroshot": dict(
        title="GSM8K (zero-shot)",
        bucket="C",
        hf="`openai/gsm8k` (config=`main`)",
        size="1319 test",
        answer="数字",
        scoring="numeric EM",
        why="同 GSM8K 数据，**空 context** 测 SHINE-LoRA 是否伤害基模型的数值推理。",
        eval_setup="context = '' (或 'You are a helpful assistant.')；question = 数学题；reference = 数字。",
    ),
    "mmlu_zeroshot": dict(
        title="MMLU (zero-shot)",
        bucket="C",
        hf="`cais/mmlu` (per-subject configs)",
        size="~14k test",
        answer="MCQ (A–D)",
        scoring="EM",
        why="同 MMLU 数据，**空 context** 测 SHINE-LoRA 是否扰乱通用知识检索。",
        eval_setup="context = ''；question = 题 + 4 选项；reference = 选项字母。",
    ),
    "humaneval": dict(
        title="HumanEval",
        bucket="C",
        hf="`openai_humaneval`",
        size="164 test",
        answer="函数完成 (Python)",
        scoring="Pass@1（执行 unit test）",
        why=(
            "训练完全不沾代码，**做 SHINE 副作用对照**最合适："
            "如果 LoRA 严重伤了代码能力，说明 hypernet 编码非任务专属。"
        ),
        eval_setup="context = ''；question = 函数签名 + docstring；reference = canonical 实现（仅供参考，评分跑 test）。",
    ),
    "truthfulqa_gen": dict(
        title="TruthfulQA (Generation)",
        bucket="C",
        hf="`truthfulqa/truthful_qa` (config=`generation`)",
        size="817 validation",
        answer="自由短答",
        scoring="GPT-judge / BLEURT / 人工",
        why="生成式版本，更贴近真实使用；评分较贵（需 judge model）。",
        eval_setup="context = ''；question = 问句；reference = correct_answers 列表（incorrect_answers 用作对比）。",
    ),
}


BUCKET_TITLES = {
    "A": "桶 A — 给 context 答题",
    "B": "桶 B — In-parameter few-shot (K demos → LoRA → 答 query)",
    "C": "桶 C — Zero-shot 探针（空 context 副作用对照）",
}

BUCKET_INTROS = {
    "A": (
        "考察 hypernet 能不能把 context 压成有用的 LoRA。"
        "Context = 数据集自带的 passage；question = 问句；reference = gold 答案。"
        "训练已经用过的数据集（SQuAD / MuSiQue）也放在这一桶——"
        "**桶不是按「是否训练过」分，按「数据形态」分**："
        "一次训练只用一个数据集，剩下的都可以反过来当 eval。"
    ),
    "B": (
        "考察 hypernet 能否把 **K 个 demo 编进 LoRA** 之后，模型在没见过的任务上能照搬这种 pattern induction。"
        "Context = 同任务/同 subject 的 K 个 demonstration（含答案）；question = 同任务的 held-out 一题；"
        "reference = 那一题的 gold 答案。"
        "\n\n"
        "**重要的两个评测旋钮**：\n\n"
        "1. **shot 数量可控（K）**：runner 应支持 `--shots 1/2/4/8/16` 任意配置，"
        "看 LoRA 容量随 demo 数量的曲线（饱和点 / 拐点在哪）。\n"
        "2. **必须和 vanilla ICL 对照**：同样 K 个 demo，但**直接拼到 prompt 里**走基模型 Qwen3（不挂 LoRA）。\n"
        "   - 同 K 下，*in-parameter few-shot* (SHINE LoRA) ≥ *prompt-level ICL* (base Qwen3) → 证明 hypernet 把 demos 编进了 LoRA。\n"
        "   - 同 K 下两者相当 → hypernet 没起作用，等同于 ICL。\n"
        "   - 同 K 下 LoRA 更差 → 训练有副作用。\n\n"
        "训练已用过的 BBH 也放在这一桶（同样原因）。"
    ),
    "C": (
        "考察 SHINE-LoRA **在没有 context 也没有 demos** 时是否伤害基模型。"
        "Context = `''`（或一段中性提示如 'You are a helpful assistant.'）；question = 直接的问题；reference = gold。"
        "和**未挂 LoRA 的 base Qwen3** 同条件跑作对照。"
    ),
}


def fence(text: str, lang: str = "") -> str:
    """Wrap text in a triple-backtick fence safely."""
    if not text:
        return "```\n(empty)\n```"
    # If the text contains ``` we escalate the fence.
    fence_str = "```"
    while fence_str in text:
        fence_str += "`"
    return f"{fence_str}{lang}\n{text}\n{fence_str}"


def render_entry(key: str, meta: dict, sample: dict) -> str:
    lines: list[str] = []
    lines.append(f"### {meta['title']}")
    lines.append("")
    lines.append(f"- **HF**: {meta['hf']}")
    lines.append(f"- **规模**: {meta['size']}")
    lines.append(f"- **答案类型**: {meta['answer']}")
    lines.append(f"- **评分**: {meta['scoring']}")
    lines.append(f"- **为什么选**: {meta['why']}")
    lines.append(f"- **Eval 数据组织**: {meta['eval_setup']}")
    lines.append("")

    if "_error" in sample:
        lines.append(f"> **真实样本拉取失败**: `{sample['_error']}`")
        lines.append("")
        return "\n".join(lines)

    lines.append("**真实样本**：")
    lines.append("")
    ctx_chars = sample.get("context_full_chars", len(sample.get("context", "") or ""))
    q_chars = sample.get("question_full_chars", len(sample.get("question", "") or ""))
    lines.append(f"_context_ ({ctx_chars} chars, 截到 1500 显示)：")
    lines.append("")
    lines.append(fence(sample.get("context", "") or "(empty)"))
    lines.append("")
    lines.append(f"_question_ ({q_chars} chars)：")
    lines.append("")
    lines.append(fence(sample.get("question", "") or "(empty)"))
    lines.append("")
    refs = sample.get("references", [])
    if refs:
        lines.append("_references_：")
        lines.append("")
        for r in refs:
            lines.append(f"- `{r}`")
        lines.append("")
    if sample.get("_notes"):
        lines.append(f"_notes_: {sample['_notes']}")
        lines.append("")
    return "\n".join(lines)


def render_overview_table() -> str:
    rows = []
    rows.append("| 数据集 | 桶 | HF | 规模 | 答案 | 评分 |")
    rows.append("|---|---|---|---|---|---|")
    for key, meta in META.items():
        # strip backticks from HF for cleaner cell, then re-add a single code span.
        hf = meta["hf"]
        rows.append(f"| {meta['title']} | {meta['bucket']} | {hf} | {meta['size']} | {meta['answer']} | {meta['scoring']} |")
    return "\n".join(rows)


def main():
    samples = json.loads(JSON_PATH.read_text())

    md: list[str] = []
    md.append("# 评测数据集调研")
    md.append("")
    md.append("> 目标：验证 SHINE-hypernet 训练后的能力，看是否真的学到东西、是否在训练分布之外也有提升。")
    md.append("> ")
    md.append("> 评测套件按「**数据形态**」分桶（不是按「是否训练过」分）。")
    md.append("> 一次训练只用其中一个数据集，剩下的全部都可以反过来当 eval——所以训练用过的（SQuAD / MuSiQue / BBH）"
              "也列在对应形态的桶里，用 🟡 标记。")
    md.append(">")
    md.append("> 每条都附一个**真实样本**展示数据组织形式（context / question / references）。")
    md.append("")

    md.append("## 概览表")
    md.append("")
    md.append(render_overview_table())
    md.append("")

    md.append("## 推荐 v1 套件")
    md.append("")
    md.append("覆盖最广 + 实现成本最低 + 信号最干净：")
    md.append("")
    md.append("- **桶 A**：HotpotQA + DROP + NarrativeQA + PubMedQA"
              "（+ SQuAD / MuSiQue 当 train-distribution 自检）")
    md.append("- **桶 B**：MMLU + ARC-Challenge + GSM8K + bAbI"
              "（+ BBH 当 train-distribution 自检）")
    md.append("- **桶 B 加分**：Super-NaturalInstructions (test split)")
    md.append("- **桶 C**：MMLU zero-shot + HumanEval")
    md.append("")
    md.append("**桶 B 必须做**：对每个数据集，跑 `--shots ∈ {1, 2, 4, 8, 16}` × `{SHINE-LoRA, base Qwen3 prompt-ICL}` 矩阵，"
              "把 K-vs-acc 两条曲线画出来——这是判断 hypernet 是否真的把 demos 编进 LoRA 的核心证据。")
    md.append("")

    for bucket in ("A", "B", "C"):
        md.append(f"## {BUCKET_TITLES[bucket]}")
        md.append("")
        md.append(BUCKET_INTROS[bucket])
        md.append("")
        for key, meta in META.items():
            if meta["bucket"] != bucket:
                continue
            sample = samples.get(key, {"_error": "no sample fetched"})
            md.append(render_entry(key, meta, sample))

    OUT_PATH.write_text("\n".join(md))
    print(f"Wrote {OUT_PATH}  ({len(md)} lines, {OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
