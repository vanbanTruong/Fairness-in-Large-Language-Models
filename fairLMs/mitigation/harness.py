"""Before/after comparison across a declared metric set.

Comparing a base and a mitigated system "needs no new code" because an
intra-processing result *is* a :class:`~fairLMs.definitions.models.base.ModelAdapter`. This
module is only the thin helper that runs a declared metric set against both and
reports the deltas.

Three rules, all of them refusals:

* **Fairness and utility deltas are reported separately.** They are not
  commensurable and combining them hides the trade-off that is the entire point
  of measuring both.
* **Intrinsic and extrinsic deltas are reported separately.** An intrinsic
  improvement is not evidence of an extrinsic one.
* **There is no composite "mitigation effectiveness score."** No defensible
  aggregation exists and the paper does not claim one. :class:`ComparisonReport`
  therefore has no ``__float__`` and exposes no overall number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from fairLMs.definitions import METRIC_REGISTRY, FairnessMetric
from fairLMs.definitions.core.provenance import json_safe, model_provenance
from fairLMs.datasets.diagnostics._utils import freeze_json_mapping, thaw_json
import copy
import math

__all__ = [
    "ComparisonReport",
    "MetricDelta",
    "MetricEvaluation",
    "compare_before_after",
]


@dataclass(frozen=True, kw_only=True)
class MetricDelta:
    """One metric's score before and after mitigation."""

    metric: str
    bias_type: str
    before: Optional[float]
    after: Optional[float]
    error: Optional[str] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(json_safe(self.provenance), path="provenance"),
        )

    @property
    def delta(self) -> Optional[float]:
        """``after - before``, or ``None`` when either side failed."""
        if self.before is None or self.after is None:
            return None
        value = self.after - self.before
        return value if math.isfinite(value) else None

    def to_dict(self) -> dict:
        return json_safe(
            {
                "metric": self.metric,
                "bias_type": self.bias_type,
                "before": self.before,
                "after": self.after,
                "delta": self.delta,
                "error": self.error,
                "provenance": thaw_json(self.provenance),
            }
        )


@dataclass(frozen=True, kw_only=True)
class ComparisonReport:
    """Before/after deltas, partitioned and never aggregated.

    Deliberately not float-convertible: there is no single number here.
    """

    fairness: Sequence[MetricDelta] = ()
    utility: Sequence[MetricDelta] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(json_safe(self.provenance), path="provenance"),
        )

    @property
    def intrinsic(self) -> tuple:
        """Fairness deltas from intrinsic metrics."""
        return tuple(d for d in self.fairness if d.bias_type == "intrinsic")

    @property
    def extrinsic(self) -> tuple:
        """Fairness deltas from extrinsic metrics."""
        return tuple(d for d in self.fairness if d.bias_type == "extrinsic")

    def to_dict(self) -> dict:
        return {
            "fairness": {
                "intrinsic": [d.to_dict() for d in self.intrinsic],
                "extrinsic": [d.to_dict() for d in self.extrinsic],
            },
            "utility": [d.to_dict() for d in self.utility],
            "provenance": thaw_json(self.provenance),
            "note": (
                "Fairness and utility are reported separately, as are intrinsic "
                "and extrinsic fairness. There is no composite effectiveness "
                "score: no defensible aggregation exists."
            ),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(
            self.to_dict(), allow_nan=False, indent=indent, sort_keys=True
        )


@dataclass(frozen=True, kw_only=True)
class MetricEvaluation:
    """A configured metric and explicit evidence handoff for a comparison.

    Model-backed metrics reuse ``data``. For precomputed metrics, supply
    ``after_data`` or ``evidence_factory(model)`` to generate predictions for
    each system. The factory runs in the baseline/candidate phase respectively.
    """

    metric: FairnessMetric
    data: Any = None
    after_data: Any = None
    evidence_factory: Any = None


def compare_before_after(
    base_model, mitigated_model, *, metrics, utility=None, provenance=None
):
    """Compare configured metrics; retain missing/failed/undefined results.

    ``metrics`` maps names to evidence (default configuration) or to a
    ``MetricEvaluation``. Model-free metrics require explicit before/after
    evidence. Fairness, utility, and intrinsic/extrinsic changes remain separate.
    A removable adapter is deactivated before measuring the baseline and is
    removed in a finally block after the candidate phase.
    """
    unknown = [
        name
        for name, value in metrics.items()
        if name not in METRIC_REGISTRY and not isinstance(value, MetricEvaluation)
    ]
    if unknown:
        raise KeyError(f"unknown metric(s): {', '.join(sorted(unknown))}")
    entries = {}
    for name, value in metrics.items():
        entry = (
            value
            if isinstance(value, MetricEvaluation)
            else MetricEvaluation(metric=METRIC_REGISTRY[name](), data=value)
        )
        if not isinstance(entry.metric, FairnessMetric):
            raise TypeError(
                "MetricEvaluation.metric must be a FairnessMetric instance."
            )
        entries[name] = entry

    remove = getattr(mitigated_model, "remove", None)
    if callable(remove):
        remove()  # Never measure a baseline through an already-installed hook.

    def evaluate(entry, model, after):
        try:
            if entry.evidence_factory is not None:
                data = entry.evidence_factory(model)
            elif not entry.metric.requires and entry.after_data is None:
                raise ValueError(
                    "Precomputed metric comparison needs explicit after_data or evidence_factory; the model argument does not create new predictions."
                )
            else:
                data = (
                    entry.after_data
                    if after and entry.after_data is not None
                    else entry.data
                )
            metric = copy.deepcopy(entry.metric)
            result = metric.compute(model, data)
            score = float(result)
            if not math.isfinite(score):
                reason = (
                    result.details.get("reason")
                    or "metric returned an undefined or non-finite score"
                )
                return None, reason, json_safe(result.provenance)
            return score, None, json_safe(result.provenance)
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}", {}

    def utility_value(function, model):
        try:
            score = float(function(model))
            if not math.isfinite(score):
                raise ValueError("utility returned a non-finite score")
            return score, None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def errors(before, after):
        messages = []
        if before:
            messages.append(f"before: {before}")
        if after:
            messages.append(f"after: {after}")
        return "; ".join(messages) or None

    try:
        baselines = {
            name: evaluate(entries[name], base_model, False) for name in sorted(entries)
        }
        utility_baselines = {
            name: utility_value(function, base_model)
            for name, function in sorted((utility or {}).items())
        }
        fairness = []
        for name in sorted(entries):
            before, before_error, before_meta = baselines[name]
            after, after_error, after_meta = evaluate(
                entries[name], mitigated_model, True
            )
            fairness.append(
                MetricDelta(
                    metric=name,
                    bias_type=entries[name].metric.bias_type,
                    before=before,
                    after=after,
                    error=errors(before_error, after_error),
                    provenance=freeze_json_mapping(
                        {"before": before_meta, "after": after_meta}, path="provenance"
                    ),
                )
            )
        utility_deltas = []
        for name, function in sorted((utility or {}).items()):
            before, before_error = utility_baselines[name]
            after, after_error = utility_value(function, mitigated_model)
            utility_deltas.append(
                MetricDelta(
                    metric=name,
                    bias_type="utility",
                    before=before,
                    after=after,
                    error=errors(before_error, after_error),
                )
            )
    finally:
        if callable(remove):
            remove()  # Cleanup failure must be visible to the caller.
    return ComparisonReport(
        fairness=tuple(fairness),
        utility=tuple(utility_deltas),
        provenance={
            "base_model": model_provenance(base_model),
            "candidate_model": model_provenance(mitigated_model),
            "user": json_safe(provenance or {}),
        },
    )
