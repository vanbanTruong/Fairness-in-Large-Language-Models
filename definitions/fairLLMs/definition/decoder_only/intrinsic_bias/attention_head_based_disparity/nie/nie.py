import math
import numpy as np
import torch
from typing import Dict, List


def _capture_acts(model, ids, pos):
    acts = {}
    handles = [
        block.attn.c_proj.register_forward_hook(
            lambda _m, inp, _o, li=li: acts.__setitem__(
                li, inp[0][:, pos, :].detach().clone())
        )
        for li, block in enumerate(model.transformer.h)
    ]
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    for h in handles:
        h.remove()
    return acts, logits


def _y(logits, probe):
    lp = torch.log_softmax(logits, dim=-1)
    return float(torch.exp(lp[probe["anti_token_id"]] - lp[probe["stereo_token_id"]]))


def compute_nie_matrix(model, tokenizer, DEVICE, probes, N_LAYERS, N_HEADS, HEAD_DIM):
    if not probes:
        return np.zeros((N_LAYERS, N_HEADS))
    nie_accum, n_valid = np.zeros((N_LAYERS, N_HEADS)), 0
    for probe in probes:
        prompt, cf_text = probe["prompt"], probe["cf_text"]
        if not (prompt and prompt.strip() and cf_text and cf_text.strip()):
            continue
        ids    = tokenizer.encode(prompt,  return_tensors="pt").to(DEVICE)
        cf_ids = tokenizer.encode(cf_text, return_tensors="pt").to(DEVICE)
        if ids.shape[1] == 0 or cf_ids.shape[1] == 0:
            continue
        base_acts, base_logits = _capture_acts(model, ids, ids.shape[1] - 1)
        cf_acts, _             = _capture_acts(model, cf_ids,
                                               min(ids.shape[1] - 1, cf_ids.shape[1] - 1))
        y_base = _y(base_logits, probe)
        if y_base == 0 or not math.isfinite(y_base):
            continue
        for l in range(N_LAYERS):
            for h in range(N_HEADS):
                s, e = h * HEAD_DIM, (h + 1) * HEAD_DIM
                repl = cf_acts[l][:, s:e]
                if torch.allclose(repl, base_acts[l][:, s:e]):
                    continue
                def pre_hook(_m, inp, _s=s, _e=e, _r=repl):
                    inp[0][:, -1, _s:_e] = _r

                handle = model.transformer.h[l].attn.c_proj.register_forward_pre_hook(pre_hook)
                with torch.no_grad():
                    y_int = _y(model(ids).logits[0, -1], probe)
                handle.remove()
                if y_int != 0 and math.isfinite(y_int):
                    nie_accum[l, h] += (y_int / y_base) - 1.0
        n_valid += 1
    return nie_accum / max(1, n_valid)


def compute_nie(nie, threshold=0.003):
    return round(float((np.abs(nie) > threshold).mean()),4)