import os
import time
from typing import List, Tuple

MODEL_NAME = "gpt-3.5-turbo-instruct"
MAX_RETRIES = 4


def get_client():
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")
    return OpenAI(api_key=key)


def _top1_token(client, prompt: str, model: str = MODEL_NAME) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            r = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=4,
                temperature=0,
                logprobs=1,
            )
            break
        except Exception as exc:
            # Invalid keys/models and other permanent errors must not be retried.
            from openai import APIConnectionError, APITimeoutError

            status = getattr(exc, "status_code", None)
            transient = (
                isinstance(exc, (APIConnectionError, APITimeoutError))
                or status in (408, 409, 429)
                or (isinstance(status, int) and status >= 500)
            )
            if not transient or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)
    choice = r.choices[0]
    toks = (choice.logprobs.tokens if getattr(choice, "logprobs", None) else None) or [
        choice.text
    ]
    for t in toks:
        if t.strip():
            return t.strip().lower()
    return ""


def compute_cr(
    client,
    factual_prompts: List[str],
    counterfactual_prompts: List[str],
    model: str = MODEL_NAME,
) -> Tuple[float, list]:
    assert len(factual_prompts) == len(counterfactual_prompts)
    if not factual_prompts:
        print("  [warn] No pairs loaded — returning CR=nan")
        return float("nan"), []

    changed, valid, rows = 0, 0, []
    for i, (p_fact, p_cf) in enumerate(zip(factual_prompts, counterfactual_prompts)):
        tok_fact = _top1_token(client, p_fact, model)
        tok_cf = _top1_token(client, p_cf, model)
        if tok_fact == "" or tok_cf == "":
            continue
        valid += 1
        did_change = int(tok_fact != tok_cf)
        changed += did_change
        rows.append(
            {
                "index": i,
                "factual": p_fact[:100],
                "counterfactual": p_cf[:100],
                "pred_factual": tok_fact,
                "pred_cf": tok_cf,
                "changed": did_change,
            }
        )
        if (i + 1) % 25 == 0:
            print(
                f"    ... {i+1}/{len(factual_prompts)}  "
                f"(CR so far: {changed/max(1,valid):.3f})"
            )

    cr = round(changed / valid, 4) if valid else float("nan")
    return cr, rows
