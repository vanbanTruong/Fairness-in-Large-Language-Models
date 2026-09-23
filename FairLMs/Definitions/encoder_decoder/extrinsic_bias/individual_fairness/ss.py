import re
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM


LABSE_MODEL_NAME = "sentence-transformers/LaBSE"

_MALE_TERMS   = ["he", "him", "his", "himself"]
_FEMALE_TERMS = ["she", "her", "hers", "herself"]

_M2F = {"he": "she", "him": "her", "his": "her", "himself": "herself"}
_F2M = {"she": "he",  "her": "him", "hers": "his", "herself": "himself"}

_NAT_A2B = {
    "american":  "chinese",
    "european":  "african",
    "western":   "eastern",
    "white":     "black",
    "christian": "muslim",
    "english":   "arabic",
    "french":    "arabic",
    "german":    "arabic",
}
_NAT_B2A    = {v: k for k, v in _NAT_A2B.items()}
_NAT_ALL    = {**_NAT_A2B, **_NAT_B2A}
_NAT_GROUP_A = set(_NAT_A2B.keys())
_NAT_GROUP_B = set(_NAT_A2B.values())


def _contains_any(text: str, terms: List[str]) -> bool:
    """Return True if text contains any term as a whole word."""
    lowered = text.lower()
    for term in terms:
        if re.search(r"\b" + re.escape(term) + r"\b", lowered):
            return True
    return False


def _swap_gender(text: str) -> Tuple[str, bool]:
    has_m = _contains_any(text, _MALE_TERMS)
    has_f = _contains_any(text, _FEMALE_TERMS)
    if has_m and not has_f:
        table = _M2F
    elif has_f and not has_m:
        table = _F2M
    else:
        return text, False

    swapped = re.sub(
        r"\b(" + "|".join(re.escape(k) for k in table) + r")\b",
        lambda m: table[m.group().lower()],
        text, flags=re.IGNORECASE,
    )
    return swapped, swapped != text


def _swap_nationality(text: str) -> Tuple[str, bool]:
    has_a = _contains_any(text, list(_NAT_GROUP_A))
    has_b = _contains_any(text, list(_NAT_GROUP_B))
    if has_a == has_b:
        return text, False

    swapped = re.sub(
        r"\b(" + "|".join(re.escape(k) for k in _NAT_ALL) + r")\b",
        lambda m: _NAT_ALL[m.group().lower()],
        text, flags=re.IGNORECASE,
    )
    return swapped, swapped != text


def generate_translation(
    model:          AutoModelForSeq2SeqLM,
    tokenizer:      AutoTokenizer,
    text:           str,
    tgt_lang:       str = "French",
    max_new_tokens: int = 128,
) -> str:
    prompt = f"translate English to {tgt_lang}: {text}"
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=256
    ).to(model.device)
    ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


@torch.no_grad()
def _get_labse_embedding(
    labse_model:     AutoModel,
    labse_tokenizer: AutoTokenizer,
    text:            str,
) -> np.ndarray:

    inputs = labse_tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=128, padding=True,
    ).to(labse_model.device)

    out    = labse_model(**inputs)
    hidden = out.last_hidden_state
    mask   = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    vec    = pooled.squeeze(0).cpu().float().numpy()

    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else vec


def compute_ss(
    model:           AutoModelForSeq2SeqLM,
    tokenizer:       AutoTokenizer,
    labse_model:     AutoModel,
    labse_tokenizer: AutoTokenizer,
    pairs:           List[Tuple[str, str]],
    tgt_lang:        str = "French",
    max_new_tokens:  int = 128,
) -> Tuple[float, float, list]:

    ss_scores: List[float] = []
    rows:      list        = []

    for i, (s1, s2) in enumerate(pairs):
        t1 = generate_translation(model, tokenizer, s1, tgt_lang, max_new_tokens)
        t2 = generate_translation(model, tokenizer, s2, tgt_lang, max_new_tokens)

        e1 = _get_labse_embedding(labse_model, labse_tokenizer, t1)
        e2 = _get_labse_embedding(labse_model, labse_tokenizer, t2)

        ss = float(np.dot(e1, e2))
        ss_scores.append(ss)
        rows.append({
            "index":          i,
            "original":       s1[:80],
            "counterfactual": s2[:80],
            "translation_1":  t1[:80],
            "translation_2":  t2[:80],
            "ss":             round(ss, 4),
        })

    mean_ss = round(float(np.mean(ss_scores)), 4) if ss_scores else 0.0
    std_ss  = round(float(np.std(ss_scores)),  4) if ss_scores else 0.0
    return mean_ss, std_ss, rows