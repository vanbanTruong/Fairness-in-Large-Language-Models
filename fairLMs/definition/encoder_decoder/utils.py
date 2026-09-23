"""Backward-compatible encoder-decoder embedding helpers.

Prefer ``fairLMs.utils.embed_encoder_hidden`` and
``fairLMs.models.HuggingFaceModel`` in new code.
"""

from __future__ import annotations

import numpy as np

from fairLMs.models import HuggingFaceModel
from fairLMs.utils import embed_encoder_hidden

_tokenizer = None
_model = None
_MODEL_NAME = "google/mt5-base"
_MAX_LENGTH = 128
_POOLING = "mean"


def _load_model(model_name):
    """Load and cache the tokenizer and encoder-decoder model."""
    global _tokenizer, _model, _MODEL_NAME
    if _tokenizer is None or _model is None or model_name != _MODEL_NAME:
        loaded = HuggingFaceModel(model_name, task="seq2seq").load()
        _tokenizer = loaded.tokenizer
        _model = loaded.model
        _MODEL_NAME = model_name


def embed_sentence(text):
    _load_model(_MODEL_NAME)
    return embed_encoder_hidden(
        _model,
        _tokenizer,
        text,
        pooling=_POOLING,
        max_length=_MAX_LENGTH,
    ).astype(np.float32)
