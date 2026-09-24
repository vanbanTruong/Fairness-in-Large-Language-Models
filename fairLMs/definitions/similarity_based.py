"""Similarity-based association metrics: WEAT, SEAT, CEAT.

These three are the reference implementation of the library's target API shape:

* ``__init__`` takes **keyword-only configuration**, stored verbatim, so
  :meth:`~fairLMs.definitions.base.FairnessMetric.get_params` works and metrics can
  be cloned or swept.
* ``compute(model, data)`` takes exactly two positional arguments. Structured
  inputs arrive as validated containers from :mod:`fairLMs.definitions.data`
  instead of a loose bag of ``**kwargs``.
* Unknown keyword arguments raise ``TypeError`` rather than being silently
  ignored.

The pre-existing ``T1_terms=...`` / ``T1_contexts=...`` keyword style still
works but emits a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from fairLMs.definitions.encoder_only.intrinsic_bias.similarity_based.ceat.ceat import (
    compute_ceat,
)
from fairLMs.definitions.encoder_only.intrinsic_bias.similarity_based.seat.seat import (
    compute_seat,
)
from fairLMs.definitions.encoder_only.intrinsic_bias.similarity_based.weat.weat import (
    compute_weat,
)
from fairLMs.definitions.base import FairnessMetric, MetricResult
from fairLMs.definitions.data import ContextSets, VectorSets, WordSets
from fairLMs.definitions.resolve import get_tokenizer_model
from fairLMs.definitions.utils import encode_sentence

# Legacy keyword aliases, kept working for one deprecation cycle.
_LEGACY_TERM_KEYS = {
    "target_1": ("T1_terms", "T1"),
    "target_2": ("T2_terms", "T2"),
    "attribute_1": ("A_terms", "A1_terms", "A1"),
    "attribute_2": ("B_terms", "A2_terms", "A2"),
}
_LEGACY_VEC_KEYS = {
    "target_1": ("T1_vecs",),
    "target_2": ("T2_vecs",),
    "attribute_1": ("A_vecs", "A1_vecs"),
    "attribute_2": ("B_vecs", "A2_vecs"),
}
_LEGACY_CONTEXT_KEYS = {
    "target_1": ("T1_contexts",),
    "target_2": ("T2_contexts",),
    "attribute_1": ("A1_contexts",),
    "attribute_2": ("A2_contexts",),
}
_DICT_KEYS = {
    "target_1": ("target_1", "t1", "T1"),
    "target_2": ("target_2", "t2", "T2"),
    "attribute_1": ("attribute_1", "a1", "A1", "a", "A"),
    "attribute_2": ("attribute_2", "a2", "A2", "b", "B"),
}


def _warn_legacy(metric: str, keys: Sequence[str], replacement: str) -> None:
    warnings.warn(
        f"Passing {', '.join(sorted(keys))} to {metric}.compute() is deprecated "
        f"and will be removed in a future release. Pass {replacement} as the "
        f"second argument instead, e.g. "
        f"{metric}().compute(model, {replacement}(...)).",
        DeprecationWarning,
        stacklevel=3,
    )


def _gather(kwargs: Mapping[str, Any], key_map: Mapping[str, Sequence[str]]):
    """Pull one value per role out of legacy kwargs. Returns (values, used_keys)."""
    values, used = {}, []
    for role, aliases in key_map.items():
        for alias in aliases:
            if kwargs.get(alias) is not None:
                values[role] = kwargs[alias]
                used.append(alias)
                break
    return values, used


def _from_mapping(obj: Mapping[str, Any], container):
    """Build a container from a dict keyed by any accepted role alias."""
    values = {}
    for role, aliases in _DICT_KEYS.items():
        for alias in aliases:
            if alias in obj and obj[alias] is not None:
                values[role] = obj[alias]
                break
    if len(values) != 4:
        missing = [r for r in _DICT_KEYS if r not in values]
        raise ValueError(
            f"Mapping is missing {', '.join(missing)}. Provide all four roles "
            f"(target_1, target_2, attribute_1, attribute_2) or pass a "
            f"{container.__name__} directly."
        )
    return container(**values)


def _unwrap(data: Any) -> Any:
    """Resolve a dataset-like object down to a plain container or mapping."""
    if data is not None and not isinstance(
        data, (WordSets, VectorSets, ContextSets, Mapping)
    ) and hasattr(data, "load"):
        return data.load()
    return data


class WEAT(FairnessMetric):
    """Word Embedding Association Test (Caliskan et al., 2017).

    Parameters
    ----------
    pooling:
        How to pool encoder hidden states when embedding terms
        (``"mean"`` or ``"cls"``). Ignored when ``data`` is a
        :class:`~fairLMs.definitions.data.VectorSets`.
    n_samples:
        Permutation samples for the p-value.
    seed:
        Seed for the permutation test, making the p-value reproducible without
        mutating numpy's global RNG. ``None`` draws from OS entropy. Has no
        effect when the term sets are small enough for exact enumeration.

    Examples
    --------
    >>> from fairLMs.definitions.data import WordSets
    >>> WEAT().compute(model, WordSets(t1, t2, a1, a2))          # doctest: +SKIP

    With precomputed embeddings and no model:

    >>> from fairLMs.definitions.data import VectorSets
    >>> WEAT().compute(None, VectorSets(v1, v2, va, vb))         # doctest: +SKIP
    """

    name = "weat"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)
    required_task = "encoder"
    requires = frozenset({"hidden_states", "local_tokenizer"})
    accepts = (WordSets, VectorSets)

    def __init__(
        self,
        *,
        pooling: str = "mean",
        n_samples: int = 10_000,
        seed: Optional[int] = None,
    ):
        self.pooling = pooling
        self.n_samples = n_samples
        self.seed = seed

    def _embed(self, hf_model, tokenizer, terms, device, pooling) -> np.ndarray:
        return np.asarray(
            [
                np.asarray(
                    encode_sentence(
                        hf_model, tokenizer, term, pooling=pooling, device=device
                    )
                )
                for term in terms
            ]
        )

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = _unwrap(data if data is not None else legacy.pop("dataset", None))

        if data is None:
            vecs, used_v = _gather(legacy, _LEGACY_VEC_KEYS)
            terms, used_t = _gather(legacy, _LEGACY_TERM_KEYS)
            if len(vecs) == 4:
                _warn_legacy("WEAT", used_v, "VectorSets")
                data = VectorSets(**vecs)
                legacy = {k: v for k, v in legacy.items() if k not in used_v}
            elif len(terms) == 4:
                _warn_legacy("WEAT", used_t, "WordSets")
                data = WordSets(**terms)
                legacy = {k: v for k, v in legacy.items() if k not in used_t}
            else:
                raise ValueError(
                    "WEAT needs four term sets. Pass a WordSets (with a model to "
                    "embed them) or a VectorSets of precomputed embeddings as the "
                    "second argument."
                )

        self._reject_unknown_kwargs(legacy, "n_samples", "pooling", "seed")
        n_samples = legacy.get("n_samples", self.n_samples)
        pooling = legacy.get("pooling", self.pooling)
        seed = legacy.get("seed", self.seed)

        if isinstance(data, Mapping):
            data = _from_mapping(data, WordSets)

        if isinstance(data, VectorSets):
            T1, T2, A, B = (
                data.target_1,
                data.target_2,
                data.attribute_1,
                data.attribute_2,
            )
            embedded_with = None
        elif isinstance(data, WordSets):
            if model is None:
                raise ValueError(
                    "WEAT needs a model to embed WordSets. Pass an encoder model, "
                    "or supply precomputed embeddings as a VectorSets."
                )
            tok, hf_model, device = get_tokenizer_model(model, tokenizer, metric=self)
            T1 = self._embed(hf_model, tok, data.target_1, device, pooling)
            T2 = self._embed(hf_model, tok, data.target_2, device, pooling)
            A = self._embed(hf_model, tok, data.attribute_1, device, pooling)
            B = self._embed(hf_model, tok, data.attribute_2, device, pooling)
            embedded_with = getattr(model, "name", None) or type(model).__name__
        else:
            raise TypeError(
                f"WEAT.compute() expects a WordSets, VectorSets, or mapping as "
                f"data, got {type(data).__name__}."
            )

        d, p = compute_weat(T1, T2, A, B, n_samples=n_samples, seed=seed)
        return MetricResult(
            score=float(d),
            details={
                "effect_size": float(d),
                "p_value": float(p),
                "n_targets": (len(T1), len(T2)),
                "n_attributes": (len(A), len(B)),
                "n_samples": n_samples,
                "seed": seed,
                "pooling": pooling if embedded_with else None,
                "embedded_with": embedded_with,
            },
        )


class SEAT(FairnessMetric):
    """Sentence Encoder Association Test (May et al., 2019).

    Terms are substituted into neutral templates, encoded, and averaged per term
    before the WEAT statistic is applied.

    Parameters
    ----------
    pooling:
        Encoder pooling strategy (``"mean"`` or ``"cls"``).
    n_samples:
        Permutation samples for the p-value.
    templates:
        Sentence templates containing one ``{}`` slot. ``None`` uses the
        built-in neutral set.
    seed:
        Seed for the permutation test, making the p-value reproducible without
        mutating numpy's global RNG. ``None`` draws from OS entropy. Has no
        effect when the term sets are small enough for exact enumeration.

    Examples
    --------
    >>> SEAT(n_samples=1_000, seed=0).get_params()               # doctest: +SKIP
    {'n_samples': 1000, 'pooling': 'mean', 'seed': 0, 'templates': None}
    """

    name = "seat"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)
    required_task = "encoder"
    requires = frozenset({"hidden_states", "local_tokenizer"})
    accepts = (WordSets,)

    def __init__(
        self,
        *,
        pooling: str = "mean",
        n_samples: int = 10_000,
        templates: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ):
        self.pooling = pooling
        self.n_samples = n_samples
        self.templates = templates
        self.seed = seed

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = _unwrap(data if data is not None else legacy.pop("dataset", None))

        if data is None:
            terms, used = _gather(legacy, _LEGACY_TERM_KEYS)
            if len(terms) != 4:
                raise ValueError(
                    "SEAT needs four term sets. Pass a WordSets as the second "
                    "argument, e.g. SEAT().compute(model, WordSets(t1, t2, a1, a2))."
                )
            _warn_legacy("SEAT", used, "WordSets")
            data = WordSets(**terms)
            legacy = {k: v for k, v in legacy.items() if k not in used}

        self._reject_unknown_kwargs(
            legacy, "n_samples", "pooling", "templates", "seed"
        )
        n_samples = legacy.get("n_samples", self.n_samples)
        pooling = legacy.get("pooling", self.pooling)
        templates = legacy.get("templates", self.templates)
        seed = legacy.get("seed", self.seed)

        if isinstance(data, Mapping):
            data = _from_mapping(data, WordSets)
        if not isinstance(data, WordSets):
            raise TypeError(
                f"SEAT.compute() expects a WordSets or mapping as data, got "
                f"{type(data).__name__}."
            )
        data.require_balanced_targets("SEAT")

        if model is None:
            raise ValueError("SEAT requires an encoder model to embed templates.")
        tok, hf_model, device = get_tokenizer_model(model, tokenizer, metric=self)

        effect, p = compute_seat(
            hf_model,
            tok,
            data.target_1,
            data.target_2,
            data.attribute_1,
            data.attribute_2,
            templates=templates,
            pooling=pooling,
            n_samples=n_samples,
            device=device,
            seed=seed,
        )
        return MetricResult(
            score=float(effect),
            details={
                "effect_size": float(effect),
                "p_value": float(p),
                "n_templates": len(templates) if templates is not None else None,
                "pooling": pooling,
                "n_samples": n_samples,
                "seed": seed,
            },
        )


class CEAT(FairnessMetric):
    """Contextualized Embedding Association Test (Guo & Caliskan, 2021).

    Samples ``sample_size`` contexts per term for ``n_trials`` trials and pools
    the per-trial effect sizes with a DerSimonian-Laird random-effects model.

    Parameters
    ----------
    pooling:
        Encoder pooling strategy (``"cls"`` or ``"mean"``).
    sample_size:
        Contexts sampled per term per trial. Every term must have at least this
        many contexts.
    n_trials:
        Number of resampling trials pooled into the combined effect size.
    seed:
        Seed for the resampling RNG.
    """

    name = "ceat"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)
    required_task = "encoder"
    requires = frozenset({"hidden_states", "local_tokenizer"})
    accepts = (ContextSets,)

    def __init__(
        self,
        *,
        pooling: str = "cls",
        sample_size: int = 10,
        n_trials: int = 100,
        seed: Optional[int] = None,
    ):
        self.pooling = pooling
        self.sample_size = sample_size
        self.n_trials = n_trials
        self.seed = seed

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = _unwrap(data if data is not None else legacy.pop("dataset", None))

        if data is None:
            ctxs, used = _gather(legacy, _LEGACY_CONTEXT_KEYS)
            if len(ctxs) != 4:
                raise ValueError(
                    "CEAT needs four context sets. Pass a ContextSets as the "
                    "second argument, e.g. "
                    "CEAT().compute(model, ContextSets(t1, t2, a1, a2))."
                )
            _warn_legacy("CEAT", used, "ContextSets")
            data = ContextSets(**ctxs)
            legacy = {k: v for k, v in legacy.items() if k not in used}

        self._reject_unknown_kwargs(
            legacy, "pooling", "sample_size", "n_trials", "seed"
        )
        pooling = legacy.get("pooling", self.pooling)
        sample_size = legacy.get("sample_size", self.sample_size)
        n_trials = legacy.get("n_trials", self.n_trials)
        seed = legacy.get("seed", self.seed)

        if isinstance(data, Mapping) and not isinstance(data, ContextSets):
            data = _from_mapping(data, ContextSets)
        if not isinstance(data, ContextSets):
            raise TypeError(
                f"CEAT.compute() expects a ContextSets or mapping as data, got "
                f"{type(data).__name__}."
            )
        data.require_contexts(sample_size, "CEAT")

        if model is None:
            raise ValueError("CEAT requires an encoder model to embed contexts.")
        tok, hf_model, device = get_tokenizer_model(model, tokenizer, metric=self)

        result = compute_ceat(
            hf_model,
            tok,
            dict(data.target_1),
            dict(data.target_2),
            dict(data.attribute_1),
            dict(data.attribute_2),
            pooling=pooling,
            sample_size=sample_size,
            n_trials=n_trials,
            seed=seed,
            device=device,
        )
        details = dict(result)
        details.update(
            {
                "pooling": pooling,
                "sample_size": sample_size,
                "n_trials": n_trials,
                "min_contexts": data.min_contexts(),
            }
        )
        return MetricResult(score=float(result["CES"]), details=details)
