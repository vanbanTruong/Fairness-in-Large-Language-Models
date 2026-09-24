"""Plain-function metrics, in the style of ``sklearn.metrics``.

Five of the library's metrics never touch a model; they score predictions you
already have. For those, a class plus ``.compute()`` is ceremony, so they are
also exposed here as functions with sklearn's ``(y_true, y_pred, ...)`` argument
order:

>>> from fairLMs.definitions.functional import equal_opportunity_gap
>>> equal_opportunity_gap([1, 1, 0], [1, 0, 0], ["A", "B", "A"], g1="A", g2="B")
1.0

Each returns a bare ``float`` (like ``accuracy_score``). Use the class form when
you want the full :class:`~fairLMs.definitions.base.MetricResult` with diagnostics,
or when you need parameter introspection for a sweep.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from fairLMs.definitions.data import GroupPredictions, ScorePair
from fairLMs.definitions.encoder_extrinsic import (
    ContextBasedDisparityScore,
    EqualOpportunityGap,
    FairInferenceScore,
)
from fairLMs.definitions.encoder_decoder_extrinsic import InferenceBiasScore
from fairLMs.definitions.performance_disparity import AccuracyDisparity

__all__ = [
    "equal_opportunity_gap",
    "accuracy_disparity",
    "inference_bias_score",
    "fair_inference_score",
    "context_based_disparity",
]


def equal_opportunity_gap(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    groups: Sequence[Any],
    *,
    g1: Any = None,
    g2: Any = None,
    positive_label: Any = 1,
) -> float:
    """True-positive-rate gap between two groups.

    Parameters
    ----------
    y_true, y_pred, groups:
        Equal-length sequences: ground truth, prediction, and protected group.
    g1, g2:
        Groups to compare. Defaults to the first two groups seen in ``groups``.
    positive_label:
        Label treated as the positive outcome.

    Returns
    -------
    float
        ``TPR(g1) - TPR(g2)`` (signed; reverse groups to reverse the sign), or ``nan`` when a group has no positives.
    """
    return float(
        EqualOpportunityGap(g1=g1, g2=g2, positive_label=positive_label)
        .compute(None, GroupPredictions(y_true, y_pred, groups))
        .score
    )


def accuracy_disparity(
    scores_stereotype: Sequence[float],
    scores_counter: Sequence[float],
) -> float:
    """Absolute accuracy gap between stereotyped and counter-stereotyped items."""
    return float(
        AccuracyDisparity()
        .compute(None, ScorePair(scores_stereotype, scores_counter))
        .score
    )


def inference_bias_score(predictions: Sequence[Sequence[Any]]) -> float:
    """Idealized Bias Score over ``(label, prediction)`` pairs."""
    return float(InferenceBiasScore().compute(None, predictions).score)


def fair_inference_score(predictions: Sequence[Mapping[str, float]]) -> float:
    """Fraction-neutral rate over NLI probability records."""
    return float(FairInferenceScore().compute(None, predictions).score)


def context_based_disparity(
    outputs: Sequence[Mapping[str, Any]],
    *,
    score: str = "s_dis",
) -> float:
    """BBQ-style context disparity, ``"s_dis"`` (default) or ``"s_amb"``."""
    return float(ContextBasedDisparityScore(score=score).compute(None, outputs).score)
