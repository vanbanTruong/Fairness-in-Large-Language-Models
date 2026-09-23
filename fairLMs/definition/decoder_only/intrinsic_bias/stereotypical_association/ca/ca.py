import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def generate(
    model,
    tokenizer,
    prompt: str,
    n_samples:      int   = 20,
    max_new_tokens: int   = 30,
    temperature:    float = 0.9,
    top_p:          float = 0.95,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample            = True,
        temperature          = temperature,
        top_p                = top_p,
        max_new_tokens       = max_new_tokens,
        num_return_sequences = n_samples,
        pad_token_id         = tokenizer.eos_token_id,
    )
    prompt_len = inputs.input_ids.shape[1]
    return [
        tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
        for o in outputs
    ]


_WORD_RE_CACHE: Dict[str, "re.Pattern"] = {}


def _count_word(text_lower: str, word: str) -> int:
    w = word.lower()
    pat = _WORD_RE_CACHE.get(w)
    if pat is None:
        pat = _WORD_RE_CACHE[w] = re.compile(r"\b" + re.escape(w) + r"\b")
    return len(pat.findall(text_lower))


def observed_distribution(generations, group_terms) -> Optional[np.ndarray]:
    counts = np.zeros(len(group_terms), dtype=float)
    for i, (_grp, terms) in enumerate(group_terms.items()):
        for text in generations:
            lowered = text.lower()
            for t in terms:
                counts[i] += _count_word(lowered, t)

    total = counts.sum()
    if total == 0:
        return None
    return counts / total


def compute_ca(
    model,
    tokenizer,
    concepts,
    prompt_template,
    group_terms,
    n_samples: int = 20,
) -> Tuple[float, int, int]:
    k         = len(group_terms)
    reference = np.ones(k) / k
    tvds      = []
    n_skipped = 0

    for concept in concepts:
        prompt = prompt_template.format(concept=concept)
        gens   = generate(model, tokenizer, prompt, n_samples=n_samples)
        p_obs  = observed_distribution(gens, group_terms)
        if p_obs is None:
            n_skipped += 1
            continue
        tvds.append(0.5 * float(np.abs(p_obs - reference).sum()))

    if not tvds:
        return float("nan"), 0, n_skipped
    return round(float(np.mean(tvds)), 4), len(tvds), n_skipped