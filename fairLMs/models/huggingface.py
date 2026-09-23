"""Hugging Face model loading helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import torch

from fairLMs.models.base import LoadedModel, ModelAdapter

TaskName = str  # "mlm" | "encoder" | "sequence_classification" | "seq2seq" | "causal"


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _from_pretrained(loader, model_name: str, **kwargs):
    token = hf_token()
    if token and "token" not in kwargs:
        kwargs["token"] = token
    return loader(model_name, **kwargs)


class HuggingFaceModel(ModelAdapter):
    """Load common Hugging Face architectures used across fairLMs metrics.

    Parameters
    ----------
    model_name:
        Hub id or local path (e.g. ``bert-base-uncased``).
    task:
        One of ``mlm``, ``encoder``, ``sequence_classification``,
        ``seq2seq``, ``causal``.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        task: TaskName = "mlm",
        device: Optional[Union[str, torch.device]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        torch_dtype: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.name = model_name
        self.task = task
        self.device = resolve_device(device)
        self.model_kwargs = dict(model_kwargs or {})
        self.tokenizer_kwargs = dict(tokenizer_kwargs or {})
        if torch_dtype is not None:
            self.model_kwargs.setdefault("torch_dtype", torch_dtype)
        self._loaded: Optional[LoadedModel] = None

    def load(self) -> LoadedModel:
        if self._loaded is not None:
            return self._loaded

        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        tokenizer_kwargs = dict(self.tokenizer_kwargs)
        if "revision" in self.model_kwargs:
            tokenizer_kwargs.setdefault("revision", self.model_kwargs["revision"])
        tokenizer = _from_pretrained(
            AutoTokenizer.from_pretrained,
            self.model_name,
            **tokenizer_kwargs,
        )

        if self.task == "mlm":
            model_cls = AutoModelForMaskedLM
        elif self.task == "encoder":
            model_cls = AutoModel
        elif self.task == "sequence_classification":
            model_cls = AutoModelForSequenceClassification
        elif self.task == "seq2seq":
            model_cls = AutoModelForSeq2SeqLM
        elif self.task == "causal":
            model_cls = AutoModelForCausalLM
        else:
            raise ValueError(
                f"Unknown task '{self.task}'. Expected mlm|encoder|"
                "sequence_classification|seq2seq|causal."
            )

        model_kwargs = dict(self.model_kwargs)
        if self.task == "causal":
            model_kwargs.setdefault("device_map", None)

        model = _from_pretrained(
            model_cls.from_pretrained,
            self.model_name,
            **model_kwargs,
        )
        dispatched = bool(getattr(model, "hf_device_map", None))
        quantized = bool(
            getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "is_loaded_in_4bit", False)
        )
        if not dispatched and not quantized:
            model.to(self.device)
        # Input device follows the input embedding, not the first visible GPU.
        embedding = model.get_input_embeddings()
        if embedding is not None and hasattr(embedding, "weight"):
            self.device = embedding.weight.device
        if self.device.type == "meta":
            raise ValueError(
                "Input embeddings are offloaded to meta; supply a supported device_map."
            )
        model.eval()

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        self._loaded = LoadedModel(
            name=self.model_name,
            tokenizer=tokenizer,
            model=model,
            device=self.device,
            task=self.task,
        )
        return self._loaded


def load_masked_lm(
    model_name: str = "bert-base-uncased",
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> LoadedModel:
    """Convenience: BERT-style masked LM (replaces per-leaf ``load_bert``)."""
    return HuggingFaceModel(
        model_name=model_name, task="mlm", device=device, **kwargs
    ).load()


def load_encoder(
    model_name: str = "bert-base-uncased",
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> LoadedModel:
    return HuggingFaceModel(
        model_name=model_name, task="encoder", device=device, **kwargs
    ).load()


def load_sequence_classifier(
    model_name: str,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> LoadedModel:
    return HuggingFaceModel(
        model_name=model_name,
        task="sequence_classification",
        device=device,
        **kwargs,
    ).load()


def load_seq2seq(
    model_name: str,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> LoadedModel:
    return HuggingFaceModel(
        model_name=model_name, task="seq2seq", device=device, **kwargs
    ).load()


def load_causal_lm(
    model_name: str,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> LoadedModel:
    return HuggingFaceModel(
        model_name=model_name, task="causal", device=device, **kwargs
    ).load()
