import os
import time
from typing import Dict, List, Tuple

import numpy as np

MODEL_NAME  = "gpt-3.5-turbo-instruct"
MAX_RETRIES = 4
TOP_LOGPROBS = 5


def get_client():
    from openai import OpenAI
    key = os.environ.get("HF_TOKEN")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment.")
    return OpenAI(api_key=key)


def next_token_dist(client, prompt: str, model: str = MODEL_NAME
                    ) -> Dict[str, float]:
    for attempt in range(MAX_RETRIES):
        try:
            r = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=1,
                temperature=0,
                logprobs=TOP_LOGPROBS,
            )
            lp = r.choices[0].logprobs
            if lp is None or not lp.top_logprobs:
                return {}
            top = lp.top_logprobs[0]
            return {tok: float(np.exp(l)) for tok, l in top.items()}
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"    [warn] API failed: {e}")
                return {}
            time.sleep(2 ** attempt)
    return {}


def tvd(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    diff = sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
    
    res_p = max(0.0, 1.0 - sum(p.values()))
    res_q = max(0.0, 1.0 - sum(q.values()))
    diff += abs(res_p - res_q)
    return min(1.0, 0.5 * diff)


def compute_ctf(
    client,
    factual_prompts: List[str],
    counterfactual_prompts: List[str],
    model: str = MODEL_NAME,
) -> Tuple[float, list]:
    assert len(factual_prompts) == len(counterfactual_prompts)
    if not factual_prompts:
        return float("nan"), []

    vals, rows = [], []
    for i, (p_f, p_cf) in enumerate(zip(factual_prompts, counterfactual_prompts)):
        d_f  = next_token_dist(client, p_f,  model)
        d_cf = next_token_dist(client, p_cf, model)
        if not d_f or not d_cf:
            continue
        v = tvd(d_f, d_cf)
        vals.append(v)
        top_f  = max(d_f,  key=d_f.get)
        top_cf = max(d_cf, key=d_cf.get)
        rows.append({
            "index": i,
            "factual": p_f[-90:],
            "counterfactual": p_cf[-90:],
            "top_token_factual": repr(top_f),
            "top_token_cf": repr(top_cf),
            "p_top_factual": round(d_f[top_f], 4),
            "p_top_cf": round(d_cf[top_cf], 4),
            "argmax_flipped": int(top_f != top_cf),
            "tvd": round(v, 6),
        })
        if (i + 1) % 25 == 0:
            print(f"    ... {i+1}/{len(factual_prompts)}  "
                  f"(CTF so far: {np.mean(vals):.4f})")

    ctf = round(float(np.mean(vals)), 4) if vals else float("nan")
    return ctf, rows