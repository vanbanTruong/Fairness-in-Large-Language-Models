from typing import List, Dict, Tuple

import numpy as np
import torch


def _token_logprobs(model, tokenizer, prompt: str, words: List[str]) -> Dict[str, float]:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits    = model(ids, use_cache=False).logits[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
    result = {}
    for word in words:
        toks = tokenizer.encode(" " + word, add_special_tokens=False)
        if not toks:
            toks = tokenizer.encode(word, add_special_tokens=False)
        if toks:
            result[word] = log_probs[toks[0]].item()
        else:
            result[word] = float("-inf")
    return result

def compute_dnp(
    model,
    tokenizer,
    prompts:       List[str],
    stereo_words:  List[str],
    counter_words: List[str],
    neutral_words: List[str],
) -> Tuple[float, float, float, list]:
    all_words = stereo_words + counter_words + neutral_words

    ps_list  = []
    psp_list = []
    pd_list  = []
    rows     = []

    for i, prompt in enumerate(prompts):
        lp_map = _token_logprobs(model, tokenizer, prompt, all_words)

        p_s  = sum(np.exp(lp_map.get(w, float("-inf")))
                   for w in stereo_words  if lp_map.get(w, float("-inf")) > float("-inf"))
        p_sp = sum(np.exp(lp_map.get(w, float("-inf")))
                   for w in counter_words if lp_map.get(w, float("-inf")) > float("-inf"))
        p_d  = sum(np.exp(lp_map.get(w, float("-inf")))
                   for w in neutral_words  if lp_map.get(w, float("-inf")) > float("-inf"))

        total = p_s + p_sp + p_d
        if total <= 0:
            ps_hat = psp_hat = pd_hat = 1.0 / 3.0
        else:
            ps_hat  = p_s  / total
            psp_hat = p_sp / total
            pd_hat  = p_d  / total

        ps_list.append(ps_hat)
        psp_list.append(psp_hat)
        pd_list.append(pd_hat)

        rows.append({
            "index":   i,
            "prompt":  prompt[:100],
            "p_s":     p_s,
            "p_sp":    p_sp,
            "p_d":     p_d,
            "ps_hat":  ps_hat,
            "psp_hat": psp_hat,
            "pd_hat":  pd_hat,
        })

    return (
        round(float(np.mean(ps_list)),  4),
        round(float(np.mean(psp_list)), 4),
        round(float(np.mean(pd_list)),  4),
        rows,
    )