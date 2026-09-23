"""Encoder-only extrinsic metrics: fair inference, equal opportunity, BBQ disparity.

None of these three needs a model; they score predictions you already have, so
``model`` is accepted and ignored for signature uniformity. Plain-function
equivalents live in :mod:`fairLMs.metrics.functional`, in the style of
``sklearn.metrics``.
"""

from __future__ import annotations

from typing import Any

from fairLMs.definition.encoder_only.extrinsic_bias.context_based_disparity.context_based import (  # noqa: E501
    compute_s_amb,
    compute_s_dis,
)
from fairLMs.definition.encoder_only.extrinsic_bias.equal_opportunity.equal_opportunity import (  # noqa: E501
    gap_g_y,
)
from fairLMs.definition.encoder_only.extrinsic_bias.fair_inference.fair_inference import (  # noqa: E501
    evaluate_fair_inference,
)
from fairLMs.metrics._compat import as_examples, require_mapping_keys, take, unwrap, warn_legacy
from fairLMs.metrics.base import FairnessMetric, MetricResult
from fairLMs.metrics.data import GroupPredictions, RECORD_CORPUS


class FairInferenceScore(FairnessMetric):
    """Dev et al. fair-inference rates over NLI prediction records.

    ``data`` is a sequence of dicts with class-probability keys including
    ``neutral``. The reported ``score`` is the fraction-neutral rate ``fn``;
    ``details`` carries ``nn``, ``t05`` and ``t07``.
    """

    name = "fair_inference_score"
    bias_type = "extrinsic"
    architectures = ("encoder_only",)
    requires = frozenset(set())
    accepts = RECORD_CORPUS

    def compute(
        self, model: Any = None, data: Any = None, **legacy: Any
    ) -> MetricResult:
        data = unwrap(data if data is not None else legacy.pop("dataset", None))
        if data is None:
            data, key = take(legacy, "predictions")
            if data is None:
                raise ValueError(
                    "FairInferenceScore requires NLI prediction records. Pass them "
                    "as the second argument, e.g. FairInferenceScore().compute("
                    'None, [{"entailment": .1, "neutral": .8, "contradiction": .1}]).'
                )
            warn_legacy("FairInferenceScore", [key], "the prediction records")
            legacy.pop(key, None)
        self._reject_unknown_kwargs(legacy)

        records = as_examples(
            data, "FairInferenceScore", "a sequence of NLI probability dicts"
        )
        require_mapping_keys(records, "FairInferenceScore", "neutral")
        nn, fn, t05, t07 = evaluate_fair_inference(records)
        return MetricResult(
            score=float(fn),
            details={"nn": nn, "fn": fn, "t05": t05, "t07": t07, "n": len(records)},
        )


class EqualOpportunityGap(FairnessMetric):
    """True-positive-rate gap between two groups (equal opportunity).

    ``data`` is a :class:`~fairLMs.metrics.data.GroupPredictions`.

    Parameters
    ----------
    g1, g2:
        The two group labels to compare. ``None`` uses the first two groups
        appearing in the data.
    positive_label:
        The label treated as the positive outcome.
    """

    name = "equal_opportunity_gap"
    bias_type = "extrinsic"
    architectures = ("encoder_only",)
    requires = frozenset(set())
    accepts = (GroupPredictions,)

    def __init__(
        self, *, g1: Any = None, g2: Any = None, positive_label: Any = 1
    ):
        self.g1 = g1
        self.g2 = g2
        self.positive_label = positive_label

    def compute(
        self, model: Any = None, data: Any = None, **legacy: Any
    ) -> MetricResult:
        data = unwrap(
            data if data is not None else legacy.pop("dataset", None), GroupPredictions
        )

        if data is None:
            y_true, k1 = take(legacy, "y_true")
            y_pred, k2 = take(legacy, "y_pred")
            groups, k3 = take(legacy, "groups")
            if None in (y_true, y_pred, groups):
                raise ValueError(
                    "EqualOpportunityGap requires y_true, y_pred and groups. Pass a "
                    "GroupPredictions as the second argument, e.g. "
                    "EqualOpportunityGap(g1='A', g2='B').compute(None, "
                    "GroupPredictions(y_true, y_pred, groups))."
                )
            warn_legacy("EqualOpportunityGap", [k1, k2, k3], "GroupPredictions")
            for key in (k1, k2, k3):
                legacy.pop(key, None)
            data = GroupPredictions(y_true, y_pred, groups)

        self._reject_unknown_kwargs(legacy, "g1", "g2", "y", "positive_label")
        g1 = legacy.get("g1", self.g1)
        g2 = legacy.get("g2", self.g2)
        # ``y=`` was the old name for the positive label.
        positive = legacy.get("y", legacy.get("positive_label", self.positive_label))

        if not isinstance(data, GroupPredictions):
            data = GroupPredictions.from_examples(
                as_examples(
                    data, "EqualOpportunityGap", "a GroupPredictions or sequence of dicts"
                )
            )

        if g1 is None or g2 is None:
            present = data.unique_groups()
            if len(present) < 2:
                raise ValueError(
                    f"EqualOpportunityGap needs two groups to compare but the data "
                    f"contains only {present!r}. Pass g1=/g2= explicitly."
                )
            g1 = g1 if g1 is not None else present[0]
            g2 = g2 if g2 is not None else present[1]

        for label, value in (("g1", g1), ("g2", g2)):
            if value not in data.groups:
                raise ValueError(
                    f"EqualOpportunityGap: {label}={value!r} does not appear in the "
                    f"data. Present groups: {list(data.unique_groups())}."
                )

        result = gap_g_y(
            list(data.y_true),
            list(data.y_pred),
            list(data.groups),
            g1,
            g2,
            y=positive,
        )
        details = dict(result._asdict())
        details.update({"g1": g1, "g2": g2, "positive_label": positive})
        return MetricResult(score=float(result.gap), details=details)


class ContextBasedDisparityScore(FairnessMetric):
    """BBQ-style context-based disparity (``S_DIS`` / ``S_AMB``).

    ``data`` is a sequence of BBQ result dicts with ``cond``
    (``"disambig"``/``"ambig"``), ``output`` and ``expected``.

    Parameters
    ----------
    score:
        Which statistic to report as ``score``: ``"s_dis"`` or ``"s_amb"``.
        Both are always present in ``details``.
    """

    name = "context_based_disparity"
    bias_type = "extrinsic"
    architectures = ("encoder_only",)
    requires = frozenset(set())
    accepts = RECORD_CORPUS

    def __init__(self, *, score: str = "s_dis"):
        # Validation lives in compute(), not __init__, per the sklearn contract:
        # __init__ must not reject values so that clone/get_params stay safe.
        self.score = score

    def compute(
        self, model: Any = None, data: Any = None, **legacy: Any
    ) -> MetricResult:
        data = unwrap(data if data is not None else legacy.pop("dataset", None))
        if data is None:
            data, key = take(legacy, "outputs")
            if data is None:
                raise ValueError(
                    "ContextBasedDisparityScore requires BBQ-style result dicts. "
                    "Pass them as the second argument."
                )
            warn_legacy("ContextBasedDisparityScore", [key], "the result dicts")
            legacy.pop(key, None)

        self._reject_unknown_kwargs(legacy, "also_s_amb", "score")
        which = legacy.get("score", self.score)
        if which not in ("s_dis", "s_amb"):
            raise ValueError(
                f"score must be 's_dis' or 's_amb', got {which!r}."
            )

        outputs = as_examples(
            data, "ContextBasedDisparityScore", "a sequence of BBQ result dicts"
        )
        require_mapping_keys(outputs, "ContextBasedDisparityScore", "cond", "output")

        s_dis, n_dis, n_non_unk, n_biased = compute_s_dis(outputs)
        s_amb, acc_amb, n_amb = compute_s_amb(outputs, s_dis)
        details = {
            "s_dis": s_dis,
            "s_amb": s_amb,
            "accuracy_ambig": acc_amb,
            "n_disambig": n_dis,
            "n_ambig": n_amb,
            "n_non_unknown": n_non_unk,
            "n_biased": n_biased,
            "score_basis": which,
        }
        return MetricResult(
            score=float(s_amb if which == "s_amb" else s_dis), details=details
        )
