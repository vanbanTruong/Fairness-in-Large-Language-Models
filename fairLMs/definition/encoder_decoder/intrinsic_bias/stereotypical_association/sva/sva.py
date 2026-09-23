import re
import warnings
from contextlib import contextmanager
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore")

# ── Stereotype direction seed texts ──────────────────────────────────────────

_GENDER_STEREO_TEXTS = [
    "The surgeon performed the operation. He was highly skilled.",
    "The engineer designed the bridge. He solved the problem.",
    "The CEO signed the contract. He led the board meeting.",
    "The programmer wrote the software. He fixed every bug.",
    "The pilot landed the aircraft safely. He was experienced.",
    "The lawyer argued the case. He won the verdict.",
    "The scientist conducted the experiment. He published the results.",
]

_GENDER_ANTI_TEXTS = [
    "The nurse cared for the patient. She was compassionate.",
    "The teacher taught the class. She was patient with students.",
    "The receptionist answered the phone. She organised the schedule.",
    "The babysitter watched the children. She was attentive.",
    "The librarian catalogued the books. She helped every visitor.",
    "The secretary typed the report. She managed the calendar.",
    "The housekeeper cleaned the rooms. She was thorough.",
]

_AGE_STEREO_TEXTS = [
    "The elderly person struggled to understand the new technology.",
    "Pensioners are often isolated and have difficulty adapting.",
    "Old people find it hard to learn new skills.",
    "Senior citizens require constant assistance with daily tasks.",
    "The aged resident could not keep up with modern changes.",
]

_AGE_ANTI_TEXTS = [
    "The young person quickly adapted to the new technology.",
    "Youth are energetic and embrace change enthusiastically.",
    "Young people learn new skills rapidly and efficiently.",
    "Teenagers are independent and capable of managing their lives.",
    "The young resident embraced modern changes with confidence.",
]


# ── Head masking ──────────────────────────────────────────────────────────────

@contextmanager
def _masked_heads(model, active_set: set, n_heads: int):
    """Zero out attention heads NOT in active_set, on the mT5 encoder self-
    attention output. NOTE: this masks the post-output-projection tensor sliced
    into n_heads chunks (heads are already mixed by the `o` projection), so it
    is an approximation of true head ablation -- unchanged from the original."""
    hooks = []
    d_model  = model.config.d_model
    head_dim = d_model // n_heads
    encoder  = model.get_encoder()

    for layer_idx, block in enumerate(encoder.block):
        attn_module = block.layer[0].SelfAttention   # MT5Attention (self-attn)

        def make_hook(l_idx):
            def hook_fn(module, inp, output):
                attn_out = output[0] if isinstance(output, tuple) else output
                B, T, D = attn_out.shape
                reshaped = attn_out.view(B, T, n_heads, head_dim)

                mask = torch.zeros(n_heads, device=attn_out.device,
                                   dtype=attn_out.dtype)
                for h in range(n_heads):
                    if (l_idx * n_heads + h) in active_set:
                        mask[h] = 1.0

                reshaped = reshaped * mask.view(1, 1, n_heads, 1)
                masked   = reshaped.reshape(B, T, D)

                if isinstance(output, tuple):
                    return (masked,) + output[1:]
                return masked
            return hook_fn

        hooks.append(attn_module.register_forward_hook(make_hook(layer_idx)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


# ── Encoder embedding ─────────────────────────────────────────────────────────

@torch.no_grad()
def _encode(
    model:     AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    texts:     List[str],
) -> np.ndarray:
    """Mean-pooled mT5 encoder embedding. No src_lang (that is mBART-only; mT5
    has no language codes) and no task prefix -- raw sentence encoding.

    SVA needs only the encoder REPRESENTATION, not any downstream task ability,
    so raw mT5 works here without fine-tuning -- unlike translation- or
    NLI-based metrics."""
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=128,
    ).to(model.device)

    out    = model.get_encoder()(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    hidden = out.last_hidden_state                          # (B, T, H)
    mask   = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled.cpu().float().numpy()


# ── Stereotype direction ──────────────────────────────────────────────────────

def compute_stereotype_direction(
    model:       AutoModelForSeq2SeqLM,
    tokenizer:   AutoTokenizer,
    stereo_texts: List[str],
    anti_texts:  List[str],
) -> np.ndarray:

    e_stereo = _encode(model, tokenizer, stereo_texts).mean(axis=0)
    e_anti   = _encode(model, tokenizer, anti_texts).mean(axis=0)
    d        = e_stereo - e_anti
    norm     = np.linalg.norm(d)
    return d / norm if norm > 1e-9 else d


# ── Bias value function v(S) ──────────────────────────────────────────────────

def compute_bias_score(
    model:        AutoModelForSeq2SeqLM,
    tokenizer:    AutoTokenizer,
    stereo_sents: List[str],
    anti_sents:   List[str],
    direction:    np.ndarray,
    active_set:   set,
    n_layers:     int,
    n_heads:      int,
) -> float:

    batch = 16
    all_stereo, all_anti = [], []

    with _masked_heads(model, active_set, n_heads):
        for i in range(0, len(stereo_sents), batch):
            all_stereo.append(
                _encode(model, tokenizer, stereo_sents[i:i+batch])
            )
            all_anti.append(
                _encode(model, tokenizer, anti_sents[i:i+batch])
            )

    e_stereo = np.vstack(all_stereo)
    e_anti   = np.vstack(all_anti)

    def cos(embs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-9)
        return (embs / norms) @ direction          # (N,)

    # Continuous mean difference — smooth signal for Shapley estimation
    return float((cos(e_stereo) - cos(e_anti)).mean())


# ── Monte Carlo Shapley ───────────────────────────────────────────────────────

def compute_sva(
    model:        AutoModelForSeq2SeqLM,
    tokenizer:    AutoTokenizer,
    stereo_sents: List[str],
    anti_sents:   List[str],
    direction:    np.ndarray,
    n_layers:     int,
    n_heads:      int,
    n_samples:    int = 15,
    top_pct:      float = 0.10,
) -> Tuple[float, np.ndarray]:

    n_total  = n_layers * n_heads
    all_heads = list(range(n_total))
    phi       = np.zeros(n_total)

    print(f"    MC Shapley: {n_samples} samples × {n_total} heads …")
    for s in range(n_samples):
        import random as _random
        perm   = all_heads.copy()
        _random.shuffle(perm)
        active = set()
        v_prev = compute_bias_score(
            model, tokenizer, stereo_sents, anti_sents,
            direction, active, n_layers, n_heads,
        )
        for head_idx in perm:
            active.add(head_idx)
            v_curr = compute_bias_score(
                model, tokenizer, stereo_sents, anti_sents,
                direction, active, n_layers, n_heads,
            )
            phi[head_idx] += (v_curr - v_prev)
            v_prev = v_curr

        if (s + 1) % 5 == 0:
            print(f"      sample {s + 1}/{n_samples} done")

    phi /= n_samples

    # Normalise to proportions so the score is scale-invariant
    total = np.abs(phi).sum()
    if total > 1e-12:
        phi_norm = np.abs(phi) / total
    else:
        phi_norm = np.abs(phi)

    k = max(1, int(np.ceil(top_pct * n_total)))
    top_idx = np.argsort(phi_norm)[::-1][:k]
    sva = float(phi_norm[top_idx].sum())   # fraction of total bias in top heads

    return sva, phi