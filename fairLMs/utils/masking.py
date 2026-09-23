"""Masked-token probability helpers (DisCo, LPBS, CBS, and related)."""

from __future__ import annotations

import logging
import math

import numpy as np
import torch

log = logging.getLogger(__name__)


def get_top_k_predictions(pipe, sentence, k=3):
    preds = pipe(sentence, top_k=k)
    if preds and isinstance(preds[0], list):
        preds = preds[0]
    return {p["token_str"]: p["score"] for p in preds}


def get_mask_fill_probs(sentence, tokenizer, model, mask_position=0):
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    mask_positions = torch.where(input_ids == tokenizer.mask_token_id)[1]

    if len(mask_positions) == 0:
        raise ValueError(f"No [MASK] token found in sentence: {sentence!r}")

    pos = mask_positions[mask_position].item()

    with torch.no_grad():
        logits = model(input_ids, return_dict=True).logits

    return logits[0, pos, :].softmax(dim=0)


def get_token_prob(probs, token, tokenizer):
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid == tokenizer.unk_token_id:
        log.warning("Token '%s' is unknown to the tokenizer; using epsilon.", token)
        return 1e-10
    return probs[tid].item()


def get_multitoken_log_prob(sentence, term, tokenizer, model):
    sub_tokens = tokenizer.tokenize(term)
    if not sub_tokens:
        log.warning("Term '%s' tokenizes to nothing; returning log(epsilon).", term)
        return math.log(1e-10)

    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    mask_positions = torch.where(input_ids == tokenizer.mask_token_id)[1]

    if len(mask_positions) != len(sub_tokens):
        raise ValueError(
            f"Term '{term}' has {len(sub_tokens)} sub-tokens but sentence "
            f"has {len(mask_positions)} [MASK] tokens. Use build_masked_sentence "
            f"to construct the sentence correctly."
        )

    with torch.no_grad():
        logits = model(input_ids, return_dict=True).logits

    log_probs = []
    for pos, sub_token in zip(mask_positions, sub_tokens):
        tid = tokenizer.convert_tokens_to_ids(sub_token)
        if tid == tokenizer.unk_token_id:
            log.warning(
                "Sub-token '%s' of term '%s' is unknown; using epsilon.",
                sub_token,
                term,
            )
            log_probs.append(math.log(1e-10))
        else:
            prob = logits[0, pos.item(), :].softmax(dim=0)[tid].item()
            log_probs.append(math.log(prob + 1e-10))

    return float(np.mean(log_probs))


def build_masked_sentence(template, term_placeholder, term, tokenizer):
    n_tokens = len(tokenizer.tokenize(term))
    mask_str = " ".join([tokenizer.mask_token] * n_tokens)
    return template.replace(term_placeholder, mask_str)
