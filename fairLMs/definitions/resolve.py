"""Helpers to normalize model / dataset arguments for metric wrappers."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch

from fairLMs.definitions.core.applicability import check_applicability
from fairLMs.definitions.models.base import LoadedModel, ModelAdapter
from fairLMs.definitions.models.openai import OpenAILoadedModel, OpenAIModel

#: The heads :class:`~fairLMs.definitions.models.HuggingFaceModel` can attach to a local
#: checkpoint. A ``task`` outside this set (``openai``) is a different kind of
#: deployment, not a different head.
LOCAL_TASKS = frozenset(
    {"mlm", "encoder", "sequence_classification", "seq2seq", "causal"}
)


def get_examples(dataset: Any) -> Optional[List[Any]]:
    """Load examples from a FairnessDataset or pass through a sequence."""
    if dataset is None:
        return None
    if hasattr(dataset, "load") and callable(dataset.load):
        return list(dataset.load())
    if isinstance(dataset, (list, tuple)):
        return list(dataset)
    # Hugging Face Dataset / other iterables
    try:
        return list(dataset)
    except TypeError as exc:
        raise TypeError(
            f"dataset must be a FairnessDataset or sequence, got {type(dataset)}"
        ) from exc


def check_task(loaded: Any, metric: Any) -> None:
    """Refuse a model loaded for the wrong head.

    A metric declares ``required_task``; a :class:`~fairLMs.definitions.models.LoadedModel`
    records the ``task`` it was loaded with. When both are known and disagree,
    the metric would otherwise fail deep in its own numerics: with an
    ``AttributeError`` on a missing ``.logits``, or, worse, with plausible
    numbers read off a randomly initialized head. Fail here instead, naming
    both sides.

    Silent when either side is unknown: raw ``(tokenizer, model)`` tuples carry
    no task, and metrics that work on any head leave ``required_task`` as
    ``None``.

    Also silent when the model is not a locally loaded checkpoint at all. This
    check is about *which head is attached*, and "reload with a different head"
    is the wrong advice for an API-served model that has no head to choose. That
    case is refused by :func:`check_model_applicability`, which can name the
    quantity the deployment cannot produce.
    """
    required = getattr(metric, "required_task", None)
    if required is None:
        return
    actual = getattr(loaded, "task", None)
    if actual is None or actual == required:
        return
    if actual not in LOCAL_TASKS:
        return
    name = type(metric).__name__ if not isinstance(metric, str) else metric
    raise TypeError(
        f"{name} requires a model loaded with task={required!r}, got "
        f"task={actual!r}. Reload the checkpoint as "
        f"HuggingFaceModel(name, task={required!r}). The task selects which "
        f"head is attached, and this metric reads a quantity that "
        f"task={actual!r} does not expose."
    )


def check_model_applicability(model: Any, metric: Any) -> None:
    """Refuse a model that cannot produce what *metric* declared it reads.

    ``check_task`` catches the wrong head on a local checkpoint. This catches
    the broader case the paper names: a deployment that cannot expose the
    quantity at all, such as an API-served decoder asked for ``hidden_states``.
    Evidence is checked by each metric against its own ``data``, so only the
    model side is matched here.

    Silent whenever either side is unknown, exactly as :func:`check_task` is.
    """
    if metric is None:
        return
    check_applicability(metric, model)


def get_tokenizer_model(
    model: Any,
    tokenizer: Any = None,
    *,
    metric: Any = None,
) -> Tuple[Any, Any, torch.device]:
    """Return ``(tokenizer, model, device)`` from adapters or raw objects.

    Pass ``metric=self`` from a metric to have the model's ``task`` checked
    against that metric's ``required_task``. See :func:`check_task`.
    """
    if model is None:
        raise TypeError("model is required for this metric")

    if isinstance(model, ModelAdapter):
        # Both matched against the adapter before load(): a refusal is only
        # worth anything if it lands before the weights are pulled. Adapters
        # already carry the `task` tag both checks read.
        check_task(model, metric)
        check_model_applicability(model, metric)
        loaded = model.load()
        check_task(loaded, metric)
        return loaded.tokenizer, loaded.model, loaded.device

    if isinstance(model, LoadedModel):
        check_task(model, metric)
        check_model_applicability(model, metric)
        return model.tokenizer, model.model, model.device

    if isinstance(model, (tuple, list)) and len(model) >= 2:
        tok, mod = model[0], model[1]
        if len(model) >= 3 and model[2] is not None:
            device = (
                torch.device(model[2])
                if not isinstance(model[2], torch.device)
                else model[2]
            )
        else:
            device = next(mod.parameters()).device
        return tok, mod, device

    if tokenizer is not None:
        device = next(model.parameters()).device
        return tokenizer, model, device

    raise TypeError(
        "Pass a HuggingFaceModel / LoadedModel, a (tokenizer, model) tuple, "
        "or model=... with tokenizer=..."
    )


def get_openai_bundle(model: Any, *, metric: Any = None) -> OpenAILoadedModel:
    """Return an OpenAI client bundle from an adapter or loaded object."""
    if model is None:
        model = OpenAIModel()
    check_model_applicability(model, metric)
    if isinstance(model, OpenAIModel):
        bundle = model.load()
    elif isinstance(model, OpenAILoadedModel):
        bundle = model
    elif hasattr(model, "completions"):
        bundle = OpenAILoadedModel(name="openai", client=model, model="davinci-002")
    else:
        bundle = None
    if bundle is not None:
        if not callable(
            getattr(getattr(bundle.client, "completions", None), "create", None)
        ):
            raise TypeError(
                "This metric requires a Completions API client with completions.create()."
            )
        return bundle
    raise TypeError("Pass an OpenAIModel / OpenAILoadedModel, or a raw OpenAI client")


def require_kwargs(kwargs: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in kwargs or kwargs[k] is None]
    if missing:
        raise TypeError(f"Missing required argument(s): {', '.join(missing)}")
