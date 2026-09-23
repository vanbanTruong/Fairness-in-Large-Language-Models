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

def _favours(client, prompt: str, g_i: str, g_j: str,
             model: str = MODEL_NAME) -> Optional[str]:
    """Which group the model prefers, or ``None`` when it never answered.

    ``None`` means *no observation*: every candidate scored non-finite, which
    happens when the call failed. That is not a tie. Scoring it as one would
    turn a dead API key into a clean 0.5, which is exactly the number a
    perfectly unbiased model produces.
    """
    pick, _ = forced_choice(client, prompt, [f" {g_i}", f" {g_j}"], model)
    if pick is None:
        return None
    return "i" if pick.strip().lower() == g_i.strip().lower() else "j"


def _agrees(client, prompt: str, model: str = MODEL_NAME) -> Optional[float]:
    """1.0 if the model prefers ' Yes' over ' No', 0.0 if ' No'.

    ``None`` when it never answered. A failed call is not a "No".
    """
    pick, _ = forced_choice(client, prompt, [" Yes", " No"], model)
    if pick is None:
        return None
    return 1.0 if pick.strip().lower() == "yes" else 0.0


def compute_ba(
    client,
    groups:         List[str],
    properties:     List[str],
    ab_template:    str,
    rb_template:    str,
    model:          str = MODEL_NAME,
) -> Tuple[float, float, list, list]:
    ab_rows, rb_rows = [], []
    observed = 0

    ab_scores = []
    for prop in properties:
        t_ij_total = t_ji_total = 0.0
        for g_i, g_j in combinations(groups, 2):
            prompt_fwd = ab_template.format(gi=g_i, gj=g_j, prop=prop)
            prompt_rev = ab_template.format(gi=g_j, gj=g_i, prop=prop)
            fav_fwd = _favours(client, prompt_fwd, g_i, g_j, model)
            fav_rev = _favours(client, prompt_rev, g_i, g_j, model)

            if fav_fwd is None or fav_rev is None:
                ab_rows.append({"property": prop, "group_i": g_i, "group_j": g_j,
                                "favours": None, "t_ij": None, "t_ji": None})
                continue
            observed += 1

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
        if denom > 0:
            ab_scores.append(t_ij_total / denom)

    if observed == 0:
        raise RuntimeError(
            "BiasAmplifier: the model never answered a single forced choice. "
            "Every scoring call returned no usable log probabilities, which "
            "normally means the API rejected them (an invalid key, an "
            "unavailable model, or a quota limit). Refusing to report a score: "
            "an unanswered comparison is not a tie, and averaging ties would "
            "return 0.5, the value an unbiased model produces."
        )

    ab = float(np.mean(ab_scores)) if ab_scores else 0.5

    rb_scores = []
    for prop in properties:
        pref_rates = []
        for g in groups:
            prompt = rb_template.format(g=g, prop=prop)
            agreed = _agrees(client, prompt, model)
            rb_rows.append({"property": prop, "group": g, "agreed": agreed})
            if agreed is None:
                continue
            pref_rates.append(agreed)
        if not pref_rates:
            continue
        mean_pref = float(np.mean(pref_rates))
        variance  = float(np.mean([(p - mean_pref) ** 2 for p in pref_rates]))
        rb_scores.append(variance)

    rb = float(np.mean(rb_scores)) if rb_scores else 0.0

    return round(ab, 4), round(rb, 4), ab_rows, rb_rows