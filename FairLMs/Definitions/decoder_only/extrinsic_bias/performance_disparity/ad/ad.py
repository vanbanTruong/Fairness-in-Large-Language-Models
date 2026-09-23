import os
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import re

MODEL_NAME  = "davinci-002"
MAX_RETRIES = 4


def get_client():
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")
    return OpenAI(api_key=key)


def continuation_logprob(client, prompt: str, continuation: str,
                         model: str = MODEL_NAME) -> float:
    full = prompt + continuation
    start = len(prompt)
    for attempt in range(MAX_RETRIES):
        try:
            r = client.completions.create(
                model=model, prompt=full, max_tokens=0,
                echo=True, logprobs=1, temperature=0,
            )
            lp = r.choices[0].logprobs
            if lp is None or not lp.token_logprobs:
                return float("-inf")
            vals = [t for off, t in zip(lp.text_offset, lp.token_logprobs)
                    if off >= start and t is not None]
            if not vals:
                return float("-inf")
            return float(np.sum(vals) / len(vals))
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"    [warn] API failed: {e}")
                return float("-inf")
            time.sleep(2 ** attempt)
    return float("-inf")


def forced_choice(client, prompt: str, candidates: List[str],
                  model: str = MODEL_NAME) -> Tuple[Optional[str], List[float]]:
    lps = [continuation_logprob(client, prompt, c, model) for c in candidates]
    if all(not np.isfinite(l) for l in lps):
        return None, lps
    return candidates[int(np.argmax(lps))], lps


def generate_fewshot(client, prompt: str, max_tokens: int = 12,
                     model: str = MODEL_NAME, stop=("\n",)) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            r = client.completions.create(
                model=model, prompt=prompt, max_tokens=max_tokens,
                temperature=0, stop=list(stop),
            )
            return r.choices[0].text.strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"    [warn] API failed: {e}")
                return ""
            time.sleep(2 ** attempt)
    return ""


def exact_match(prediction: str, reference: str) -> float:
    p = prediction.lower().strip().rstrip(".")
    r = reference.lower().strip().rstrip(".")
    return 1.0 if p == r else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pt = set(re.sub(r"[^\w\s]", "", prediction.lower()).split())
    rt = set(re.sub(r"[^\w\s]", "", reference.lower()).split())
    if not pt or not rt:
        return 0.0
    common = pt & rt
    if not common:
        return 0.0
    prec = len(common) / len(pt)
    rec  = len(common) / len(rt)
    return 2 * prec * rec / (prec + rec)


def any_exact_match(prediction: str, references: List[str]) -> float:
    return 1.0 if any(exact_match(prediction, r) == 1.0 for r in references) else 0.0


def best_token_f1(prediction: str, references: List[str]) -> float:
    return max((token_f1(prediction, r) for r in references), default=0.0)


def compute_ad(scores_s: List[float], scores_sp: List[float]) -> Tuple[float, float, float]:
    acc_s  = float(np.mean(scores_s))  if scores_s  else float("nan")
    acc_sp = float(np.mean(scores_sp)) if scores_sp else float("nan")
    ad = abs(acc_s - acc_sp) if (np.isfinite(acc_s) and np.isfinite(acc_sp)) else float("nan")
    return acc_s, acc_sp, ad