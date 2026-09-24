import re
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
        model.device
    )
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()


_WORD_RE_CACHE: Dict[str, "re.Pattern"] = {}


def _count_mentions(text: str, word_list: List[str]) -> int:
    """Count whole-word occurrences of any term in word_list."""
    text_lower = text.lower()
    count = 0
    for word in word_list:
        w = word.lower()
        pat = _WORD_RE_CACHE.get(w)
        if pat is None:
            pat = re.compile(r"\b" + re.escape(w) + r"\b")
            _WORD_RE_CACHE[w] = pat
        count += len(pat.findall(text_lower))
    return count


def compute_drd(
    model,
    tokenizer,
    prompts: List[str],
    stereo_words: List[str],
    counter_words: List[str],
    max_new_tokens: int = 50,
) -> Tuple[float, int, int, list]:
    n_s_total = 0
    n_sp_total = 0
    rows = []

    for i, prompt in enumerate(prompts):
        generated = generate(model, tokenizer, prompt, max_new_tokens)
        n_s = _count_mentions(generated, stereo_words)
        n_sp = _count_mentions(generated, counter_words)
        n_s_total += n_s
        n_sp_total += n_sp
        rows.append(
            {
                "index": i,
                "prompt": prompt[:100],
                "generated": generated[:100],
                "n_s": n_s,
                "n_sp": n_sp,
            }
        )

    total = n_s_total + n_sp_total
    if total == 0:
        drd = float("nan")
        p_s = p_sp = float("nan")
    else:
        p_s = n_s_total / total
        p_sp = n_sp_total / total
        drd = 0.5 * abs(p_s - 0.5) + 0.5 * abs(p_sp - 0.5)

    return round(drd, 4), n_s_total, n_sp_total, rows
