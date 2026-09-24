"""Embedding helpers for encoder and encoder-decoder models."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def encode_sentence(model, tokenizer, text, pooling="mean", device="cpu"):
    """Encode a sentence (or batch) to a single vector using a transformer encoder."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state
    if pooling == "cls":
        return hidden[:, 0, :].squeeze().cpu().numpy()
    if pooling == "mean":
        mask = inputs["attention_mask"].unsqueeze(-1).type_as(hidden)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return (summed / counts).squeeze().cpu().numpy()
    raise ValueError(f"unknown pooling '{pooling}'")


def encode_in_context(
    model, tokenizer, word, contexts, pooling="cls", device="cpu"
):
    vecs = []
    for sentence in contexts:
        vec = encode_sentence(model, tokenizer, sentence, pooling, device=device)
        vecs.append(vec)
    return np.array(vecs)


def embed_encoder_hidden(
    model,
    tokenizer,
    text,
    pooling="mean",
    max_length=128,
    device: Optional[torch.device] = None,
):
    """Mean/CLS pool over an encoder-decoder model's encoder hidden states."""
    if device is None:
        device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(**inputs)
        hidden = encoder_outputs.last_hidden_state

    if pooling == "mean":
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        vector = (summed / counts).squeeze()
    else:
        vector = hidden[:, 0, :].squeeze()

    return vector.detach().cpu().numpy().astype(np.float32)
