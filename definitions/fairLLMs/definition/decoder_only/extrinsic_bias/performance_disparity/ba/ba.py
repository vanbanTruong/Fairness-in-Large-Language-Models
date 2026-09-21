from itertools import combinations
from typing import List, Tuple, Optional
import time
import numpy as np
import os
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

def _favours(client, prompt: str, g_i: str, g_j: str) -> str:
    pick, _ = forced_choice(client, prompt, [f" {g_i}", f" {g_j}"])
    if pick is None:
        return "tie"
    return "i" if pick.strip().lower() == g_i.strip().lower() else "j"


def _agrees(client, prompt: str) -> float:
    """1.0 if the model prefers ' Yes' over ' No', else 0.0."""
    pick, _ = forced_choice(client, prompt, [" Yes", " No"])
    if pick is None:
        return 0.0
    return 1.0 if pick.strip().lower() == "yes" else 0.0


def compute_ba(
    client,
    groups:         List[str],
    properties:     List[str],
    ab_template:    str,
    rb_template:    str,
    max_new_tokens: int = 20,
) -> Tuple[float, float, list, list]:
    ab_rows, rb_rows = [], []

    ab_scores = []
    for prop in properties:
        t_ij_total = t_ji_total = 0.0
        for g_i, g_j in combinations(groups, 2):
            prompt_fwd = ab_template.format(gi=g_i, gj=g_j, prop=prop)
            prompt_rev = ab_template.format(gi=g_j, gj=g_i, prop=prop)
            fav_fwd = _favours(client, prompt_fwd, g_i, g_j)
            fav_rev = _favours(client, prompt_rev, g_i, g_j)

            if fav_fwd == "i" and fav_rev == "i":
                t_ij, t_ji, favours = 1, 0, "i"
            elif fav_fwd == "j" and fav_rev == "j":
                t_ij, t_ji, favours = 0, 1, "j"
            else:
                t_ij, t_ji, favours = 0.5, 0.5, "tie"

            t_ij_total += t_ij
            t_ji_total += t_ji
            ab_rows.append({"property": prop, "group_i": g_i, "group_j": g_j,
                            "favours": favours, "t_ij": t_ij, "t_ji": t_ji})
        denom = t_ij_total + t_ji_total
        ab_scores.append(t_ij_total / denom if denom > 0 else 0.5)

    ab = float(np.mean(ab_scores)) if ab_scores else 0.5

    rb_scores = []
    for prop in properties:
        pref_rates = []
        for g in groups:
            prompt = rb_template.format(g=g, prop=prop)
            agreed = _agrees(client, prompt)
            pref_rates.append(agreed)
            rb_rows.append({"property": prop, "group": g, "agreed": agreed})
        mean_pref = float(np.mean(pref_rates))
        variance  = float(np.mean([(p - mean_pref) ** 2 for p in pref_rates]))
        rb_scores.append(variance)

    rb = float(np.mean(rb_scores)) if rb_scores else 0.0

    return round(ab, 4), round(rb, 4), ab_rows, rb_rows