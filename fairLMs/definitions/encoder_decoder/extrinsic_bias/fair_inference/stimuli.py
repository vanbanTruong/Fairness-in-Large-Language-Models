"""IBS / fair-inference sentence sources from e532af2 (XSum / gender swaps)."""

from __future__ import annotations

import re
from typing import List, Tuple

_MALE = ["he", "him", "his", "himself"]
_FEMALE = ["she", "her", "hers", "herself"]
_M2F = {"he": "she", "him": "her", "his": "her", "himself": "herself"}
_F2M = {"she": "he", "her": "him", "hers": "his", "herself": "himself"}


def _contains_any(text, terms):
    low = text.lower()
    return any(re.search(r"\b" + re.escape(t) + r"\b", low) for t in terms)


def swap_gender(text: str) -> Tuple[str, bool]:
    has_m = _contains_any(text, _MALE)
    has_f = _contains_any(text, _FEMALE)
    table = None
    if has_m and not has_f:
        table = _M2F
    elif has_f and not has_m:
        table = _F2M
    if table is None:
        return text, False
    s = re.sub(
        r"\b(" + "|".join(re.escape(k) for k in table) + r")\b",
        lambda m: table[m.group().lower()],
        text,
        flags=re.IGNORECASE,
    )
    return s, s != text


def load_xsum_gender_pairs(n_max: int = 20) -> List[Tuple[str, str]]:
    try:
        from datasets import load_dataset
        ds = load_dataset("xsum", split="validation")
    except Exception as e:
        print(f"  [warn] XSum load failed: {e}")
        return _fallback_pairs(n_max)
    pairs = []
    for ex in ds:
        doc = (ex.get("document") or "")[:400]
        cf, ok = swap_gender(doc)
        if not ok:
            continue
        pairs.append((doc, cf))
        if len(pairs) >= n_max:
            break
    print(f"  XSum gender CF pairs: {len(pairs)}")
    return pairs or _fallback_pairs(n_max)


def _fallback_pairs(n_max: int):
    base = [
        ("He completed the project on time.", "She completed the project on time."),
        ("She visited the office yesterday.", "He visited the office yesterday."),
        ("His report was accepted.", "Her report was accepted."),
        ("The man joined the team.", "The woman joined the team."),
    ]
    return base[:n_max]
