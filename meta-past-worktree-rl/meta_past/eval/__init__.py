"""Held-out evaluation harness for SHINE-hypernet.

Three eval modes per item:

  - ``shine``: hypernet(context) → LoRA, then vLLM.generate(question).
    The thing we're actually evaluating.
  - ``icl``:   no LoRA; (context + question) packed into the prompt
    and run through base Qwen3 via vLLM. Apples-to-apples control for
    bucket-B in-parameter few-shot claims (LoRA-compiled demos vs.
    prompt-stuffed demos at matched K).
  - ``zero``:  no LoRA, no context; just the question. Side-effect
    probe — does SHINE-LoRA break the base model's general abilities?

Datasets are organized by **format** (bucket A / B / C), not by whether
they were used during training. See ``eval_datasets.md`` for the full
survey.
"""
