import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Defaults match the seq2seq model most encoder-decoder metrics in this package use.
_MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
_MAX_LENGTH = 512
_POOLING = "mean"

_tokenizer = None
_model = None


def _load_model(model_name):
    """Load and cache the tokenizer and encoder-decoder model."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _model.eval()
    return _tokenizer, _model


def embed_sentence(text, model_name=_MODEL_NAME, pooling=_POOLING,
                   max_length=_MAX_LENGTH):
    """Encode text to a single vector using the model's encoder stack."""
    tokenizer, model = _load_model(model_name)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )

    with torch.no_grad():
        encoder_outputs = model.encoder(**inputs)
        hidden = encoder_outputs.last_hidden_state

    if pooling == "mean":
        mask = inputs["attention_mask"].unsqueeze(-1).type_as(hidden)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        vector = (summed / counts).squeeze()
    elif pooling == "cls":
        vector = hidden[:, 0, :].squeeze()
    else:
        raise ValueError(f"unknown pooling '{pooling}'")

    return vector.cpu().numpy().astype(np.float32)
