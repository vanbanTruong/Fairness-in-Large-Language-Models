"""Attention-head disparity metrics: GBE, NIE."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from fairLMs.definitions.decoder_only.intrinsic_bias.attention_head_based_disparity.gbe.gbe import (
    compute_gbe,
    compute_gbe_matrix,
)
from fairLMs.definitions.decoder_only.intrinsic_bias.attention_head_based_disparity.nie.nie import (
    compute_nie,
    compute_nie_matrix,
)
from fairLMs.definitions._compat import take, unwrap, warn_legacy
from fairLMs.definitions.base import FairnessMetric, MetricResult
from fairLMs.definitions.data import ProbeSet, WordSets
from fairLMs.definitions.resolve import get_tokenizer_model


def _derive_head_shape(config, n_layers, n_heads, head_dim):
    """Fill in transformer shape from ``model.config`` where not supplied."""
    if n_layers is None:
        n_layers = getattr(config, "n_layer", None) or getattr(
            config, "num_hidden_layers", None
        )
    if n_heads is None:
        n_heads = getattr(config, "n_head", None) or getattr(
            config, "num_attention_heads", None
        )
    if head_dim is None:
        hidden = getattr(config, "n_embd", None) or getattr(config, "hidden_size", None)
        if hidden is not None and n_heads:
            head_dim = int(hidden) // int(n_heads)
    if None in (n_layers, n_heads, head_dim):
        raise ValueError(
            "Could not derive n_layers/n_heads/head_dim from model.config; pass "
            "them to the constructor explicitly."
        )
    return int(n_layers), int(n_heads), int(head_dim)


class GradientBasedBiasEstimation(FairnessMetric):
    """Gradient-based bias on decoder Value heads (GBE).

    ``data`` is a :class:`~fairLMs.definitions.data.WordSets`: ``target_1`` /
    ``target_2`` are the contrasted groups (X, Y) and ``attribute_1`` /
    ``attribute_2`` the attribute poles (A, B).

    Parameters
    ----------
    loss_scale:
        Scaling applied to the contrastive loss before differentiating.
    verbose:
        Print per-layer progress.
    gbe_matrix:
        Skip the model pass and score this precomputed ``(n_layers, n_heads)``
        matrix instead.
    """

    name = "gradient_based_bias_estimation"
    bias_type = "intrinsic"
    architectures = ("decoder_only",)
    required_task = "causal"
    requires = frozenset({"gradients", "hidden_states", "local_tokenizer"})
    accepts = (WordSets,)

    def __init__(
        self, *, loss_scale: float = 1.0, verbose: bool = False, gbe_matrix: Any = None
    ):
        self.loss_scale = loss_scale
        self.verbose = verbose
        self.gbe_matrix = gbe_matrix

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = unwrap(data if data is not None else legacy.pop("dataset", None), WordSets)

        if data is None:
            x, k1 = take(legacy, "X")
            y, k2 = take(legacy, "Y")
            a, k3 = take(legacy, "A")
            b, k4 = take(legacy, "B")
            if not (None in (x, y, a, b)):
                warn_legacy("GradientBasedBiasEstimation", [k1, k2, k3, k4], "WordSets")
                for key in (k1, k2, k3, k4):
                    legacy.pop(key, None)
                data = WordSets(x, y, a, b)

        self._reject_unknown_kwargs(legacy, "loss_scale", "verbose", "gbe_matrix")
        matrix = legacy.get("gbe_matrix", self.gbe_matrix)

        if matrix is not None:
            matrix = np.asarray(matrix)
            derived = False
        else:
            if data is None:
                raise ValueError(
                    "GradientBasedBiasEstimation requires four word sets. Pass a "
                    "WordSets as the second argument, e.g. compute(model, "
                    'WordSets(["man"], ["woman"], ["career"], ["family"])), '
                    "or supply gbe_matrix= to score a precomputed matrix."
                )
            if not isinstance(data, WordSets):
                raise TypeError(
                    f"GradientBasedBiasEstimation expects a WordSets as data, got "
                    f"{type(data).__name__}."
                )
            tok, hf_model, device = get_tokenizer_model(model, tokenizer, metric=self)
            matrix = compute_gbe_matrix(
                hf_model,
                tok,
                device,
                list(data.target_1),
                list(data.target_2),
                list(data.attribute_1),
                list(data.attribute_2),
                loss_scale=legacy.get("loss_scale", self.loss_scale),
                verbose=legacy.get("verbose", self.verbose),
            )
            derived = True

        return MetricResult(
            score=float(compute_gbe(matrix)),
            details={
                "gbe_matrix": matrix,
                "shape": tuple(np.shape(matrix)),
                "matrix_computed": derived,
            },
        )


class NaturalIndirectEffect(FairnessMetric):
    """Natural Indirect Effect / attention mediation (NIE).

    ``data`` is a :class:`~fairLMs.definitions.data.ProbeSet` whose probes each carry
    ``prompt``, ``cf_text``, ``stereo_token_id`` and ``anti_token_id``.

    ``n_layers`` / ``n_heads`` / ``head_dim`` are derived from ``model.config``
    when omitted; they used to be required arguments even though the model
    already knows them.

    Parameters
    ----------
    threshold:
        Absolute NIE above which a head counts as mediating.
    n_layers, n_heads, head_dim:
        Optional shape overrides.
    nie:
        Skip the model pass and score this precomputed NIE matrix instead.
    """

    name = "natural_indirect_effect"
    bias_type = "intrinsic"
    architectures = ("decoder_only",)
    required_task = "causal"
    requires = frozenset({"hidden_states", "token_logprobs", "local_tokenizer"})
    accepts = (ProbeSet,)

    def __init__(
        self,
        *,
        threshold: float = 0.003,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        nie: Any = None,
    ):
        self.threshold = threshold
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.nie = nie

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = unwrap(data if data is not None else legacy.pop("dataset", None), ProbeSet)

        if data is None:
            probes, key = take(legacy, "probes")
            if probes is not None:
                warn_legacy("NaturalIndirectEffect", [key], "ProbeSet")
                legacy.pop(key, None)
                data = ProbeSet(probes)

        self._reject_unknown_kwargs(
            legacy, "threshold", "n_layers", "n_heads", "head_dim", "nie",
            "N_LAYERS", "N_HEADS", "HEAD_DIM",
        )
        nie = legacy.get("nie", self.nie)

        if nie is not None:
            nie = np.asarray(nie)
            shape = tuple(np.shape(nie))
            derived = False
        else:
            if data is None:
                raise ValueError(
                    "NaturalIndirectEffect requires probes. Pass a ProbeSet as the "
                    "second argument, e.g. compute(model, ProbeSet([{...}])), or "
                    "supply nie= to score a precomputed matrix."
                )
            if not isinstance(data, ProbeSet):
                data = ProbeSet(data)
            tok, hf_model, device = get_tokenizer_model(model, tokenizer, metric=self)
            n_layers, n_heads, head_dim = _derive_head_shape(
                hf_model.config,
                legacy.get("n_layers", legacy.get("N_LAYERS", self.n_layers)),
                legacy.get("n_heads", legacy.get("N_HEADS", self.n_heads)),
                legacy.get("head_dim", legacy.get("HEAD_DIM", self.head_dim)),
            )
            nie = compute_nie_matrix(
                hf_model, tok, device, list(data.probes), n_layers, n_heads, head_dim
            )
            shape = (n_layers, n_heads)
            derived = True

        threshold = legacy.get("threshold", self.threshold)
        return MetricResult(
            score=float(compute_nie(nie, threshold=threshold)),
            details={
                "nie": nie,
                "shape": shape,
                "threshold": threshold,
                "matrix_computed": derived,
            },
        )
