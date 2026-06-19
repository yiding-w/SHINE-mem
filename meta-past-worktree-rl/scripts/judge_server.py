"""LLM-as-judge HTTP server.

Drop-in for the protocol consumed by ``meta_past.reward.judge_reward.HttpJudgeReward``
(same wire format as ``Long-Digestor-Experiments/reward_server.py``):

    POST /evaluate
    Body: {"question": str, "reference": str, "pred": str}
    Returns: {"result": "True" | "False"}

Two backends:

* ``--backend openai``: forward to OpenAI Chat Completions (default model
  ``gpt-4o-mini``). Needs ``OPENAI_API_KEY`` in env. Cheapest when call
  volume is moderate.
* ``--backend openai-compat``: forward to a local OpenAI-compatible server
  (a vLLM ``/v1/chat/completions`` endpoint, e.g. Qwen3-32B served on a
  spare GPU). No API key required. Highest throughput on-cluster.

Usage:

    # Cheap path — OpenAI gpt-4o-mini
    OPENAI_API_KEY=sk-... python scripts/judge_server.py --port 8124

    # Local Qwen3-32B judge (assumes you've ``vllm serve`` Qwen3-32B at :8000)
    python scripts/judge_server.py \\
        --backend openai-compat \\
        --base-url http://127.0.0.1:8000/v1 \\
        --model Qwen3-32B-Instruct \\
        --port 8124
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

logger = logging.getLogger("judge_server")


SYSTEM_PROMPT = (
    "You are a precise evaluator. Your task is to determine if the "
    "'Predicted Answer' is semantically the same as the 'Ground Truth' "
    "for the given 'Question'. Your entire response MUST be only the "
    "single word 'True' or the single word 'False'. Do not provide any "
    "explanation or punctuation."
)
USER_TEMPLATE = (
    "Question: {question}\nGround Truth: {reference}\nPredicted Answer: {pred}"
)


# Pydantic models live at module scope. Defining them inside ``main()``
# breaks FastAPI's request validation under Pydantic v2 — the
# ``TypeAdapter`` it builds for the request body keeps a ForwardRef to
# the class name and can't resolve it back to a function-local class,
# raising ``PydanticUserError: ... is not fully defined`` at request time.
try:
    from pydantic import BaseModel as _BaseModel  # type: ignore
except ImportError:  # checked at runtime in main()
    _BaseModel = object  # type: ignore[assignment, misc]


class JudgeRequest(_BaseModel):  # type: ignore[misc, valid-type]
    question: str
    reference: str
    pred: str


class JudgeResponse(_BaseModel):  # type: ignore[misc, valid-type]
    result: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8124)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--backend", choices=["openai", "openai-compat"], default="openai")
    p.add_argument("--base-url", default=None,
                   help="For backend=openai-compat: base URL of the OpenAI-compatible server.")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="Judge model id. For openai-compat, the path/name vllm exposes.")
    p.add_argument("--max-concurrency", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--timeout-s", type=float, default=30.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        from fastapi import Body, FastAPI, HTTPException
        from openai import AsyncOpenAI
        import uvicorn
    except ImportError as e:
        sys.exit(
            f"Missing dep: {e}. Install with `pip install fastapi uvicorn openai pydantic`."
        )

    if args.backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set; calls will fail.")
        client = AsyncOpenAI(api_key=api_key or None)
    else:
        if not args.base_url:
            sys.exit("--backend openai-compat requires --base-url")
        client = AsyncOpenAI(api_key="EMPTY", base_url=args.base_url)

    sem = asyncio.Semaphore(args.max_concurrency)
    app = FastAPI()

    async def _evaluate(req: JudgeRequest) -> str:
        async with sem:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(
                    question=req.question,
                    reference=req.reference,
                    pred=req.pred,
                )},
            ]
            resp: Any = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout_s,
            )
            text = (resp.choices[0].message.content or "").strip()
            verdict = "True" if text.lower().startswith("true") else "False"
            return verdict

    @app.post("/evaluate", response_model=JudgeResponse)
    async def evaluate(req: JudgeRequest = Body(...)) -> JudgeResponse:
        try:
            v = await _evaluate(req)
            return JudgeResponse(result=v)
        except Exception as e:
            logger.exception("evaluate failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "backend": args.backend, "model": args.model}

    logger.info(
        "judge server: backend=%s model=%s host=%s port=%d concurrency=%d",
        args.backend, args.model, args.host, args.port, args.max_concurrency,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
