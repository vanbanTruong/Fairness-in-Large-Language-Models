"""Row-level scoring-instrument diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from itertools import combinations
from numbers import Real
from sys import float_info
from typing import Any, ClassVar

from fairLMs.datasets.diagnostics._utils import (
    freeze_json_mapping,
    normalize_enum,
    require_nonempty_string,
    thaw_json,
)

from .base import (
    ComponentPlan,
    ComponentResult,
    DatasetDiagnostic,
    DiagnosticReport,
    DiagnosticStatus,
)
from .evidence import PairedScores, ScoredGroups
from .spec import DatasetAuditSpec, TargetKind

_DESCRIPTIVE_INTERPRETATION = (
    "The group mean score gap is descriptive evidence about one declared "
    "scoring instrument and audit context, not an automatic fairness pass/fail "
    "judgment or a causal attribution to group identity."
)
_SCALE_INTERPRETATION = (
    "The value is reported in the scorer's native score units and is not "
    "automatically comparable across unrelated scorer scales or datasets."
)
_RATE_INTERPRETATION = (
    "The group event-rate gap is descriptive and threshold-dependent; it is "
    "not automatically a fairness pass/fail rule, an error-rate metric, or a "
    "causal effect of group identity."
)
_AGGREGATE_CONFOUNDING = (
    "Group-level rates can reflect content, topic, and coverage differences; "
    "the same serialized score-to-event rule is applied to every group."
)
_RATE_CONTEXT_INTERPRETATION = (
    "The value is only interpretable within the declared dataset, scorer, "
    "score-to-event rule, and audit context; it is not automatically comparable "
    "across datasets, scorers, or thresholds."
)
_RATE_UNIT_INTERPRETATION = (
    "The value uses proportion units: for example, 0.12 is a 12-percentage-point "
    "absolute gap, not a 12 percent relative change."
)
_DISTRIBUTION_INTERPRETATION = (
    "The empirical Wasserstein distance is a descriptive, symmetric comparison "
    "of unpaired group score distributions, not a causal or counterfactual "
    "effect, an error-rate metric, or a fairness pass/fail rule."
)
_DISTRIBUTION_CONFOUNDING = (
    "Group score distributions can reflect content, topic, coverage, repeated "
    "observations, and sample-size differences as well as scorer behavior."
)
_DISTRIBUTION_GEOMETRY = (
    "The distance uses the scorer's native numeric geometry and is not normalized "
    "by score_range; arbitrary category codes or ranks without meaningful spacing "
    "are not valid score inputs."
)
_DISTRIBUTION_CONTEXT_INTERPRETATION = (
    "The value is only interpretable within the declared dataset and scorer "
    "snapshot and is not automatically comparable across datasets, scorers, "
    "rescalings, filtering rules, or score transformations."
)
_POINT_ESTIMATE_INTERPRETATION = (
    "This component reports an empirical point estimate without a confidence "
    "interval or statistical-significance claim."
)
_COUNTERFACTUAL_INTERPRETATION = (
    "The mean absolute paired score difference is a descriptive instrument-"
    "sensitivity estimate; causal isolation is justified only when the declared "
    "pairs truly differ solely in the intended identity intervention."
)
_PAIRING_VALIDITY_INTERPRETATION = (
    "The package validates pair completeness and condition roles but cannot "
    "verify from scores alone that paired source items are minimal contrasts; "
    "the declared pairing basis remains an external measurement assumption."
)
_COUNTERFACTUAL_DIRECTION_INTERPRETATION = (
    "The primary value averages absolute within-pair changes and is symmetric; "
    "the role-ordered signed mean is secondary detail and must not replace the "
    "paper-aligned absolute estimand."
)
_COUNTERFACTUAL_SCALE_INTERPRETATION = (
    "Every complete pair receives equal mass, and differences remain in the "
    "scorer's native units; values are not automatically comparable across "
    "scorers, score transformations, datasets, or pairing protocols."
)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _evidence_metadata(evidence: ScoredGroups | PairedScores) -> dict[str, Any]:
    metadata = {
        "axis": evidence.axis,
        "score_name": evidence.score_name,
        "score_range": evidence.score_range,
        "source": evidence.source,
        "provenance": dict(evidence.provenance),
    }
    if isinstance(evidence, PairedScores):
        metadata.update(
            {
                "condition_roles": evidence.condition_roles,
                "pair_count": evidence.pair_count,
                "pairing_basis": evidence.pairing_basis,
            }
        )
    return metadata


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{field_name} must be a real number, got {type(value).__name__}."
        )
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be representable as a finite float."
        ) from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


class ScoreRateDirection(str, Enum):
    """Which side of a threshold is treated as the declared event."""

    HIGHER = "higher"
    LOWER = "lower"


@dataclass(frozen=True, kw_only=True)
class ScoreRateTransform:
    """Explicit, portable threshold rule that maps one score to an event."""

    event_name: str
    threshold: float
    direction: ScoreRateDirection
    inclusive: bool
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_name",
            require_nonempty_string(self.event_name, "event_name"),
        )
        object.__setattr__(
            self,
            "threshold",
            _finite_float(self.threshold, "threshold"),
        )
        object.__setattr__(
            self,
            "direction",
            normalize_enum(self.direction, ScoreRateDirection, "direction"),
        )
        if not isinstance(self.inclusive, bool):
            raise TypeError(
                f"inclusive must be a boolean, got {type(self.inclusive).__name__}."
            )
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    @property
    def operator(self) -> str:
        """Return the exact comparison operator represented by this rule."""
        if self.direction is ScoreRateDirection.HIGHER:
            return ">=" if self.inclusive else ">"
        return "<=" if self.inclusive else "<"

    @property
    def paper_alignment(self) -> str:
        """Classify the rule against the paper's formal threshold definition."""
        if self.direction is ScoreRateDirection.HIGHER and self.inclusive:
            return "paper_exact"
        if self.direction is ScoreRateDirection.LOWER and self.inclusive:
            return "generalized_lower_tail_analogue"
        return "generalized_boundary_extension"

    @property
    def alignment_assumption(self) -> str:
        """Return a report-ready statement of the rule's methodological scope."""
        if self.paper_alignment == "paper_exact":
            return (
                "The score-to-event transform exactly matches the paper's "
                "formal score >= threshold definition."
            )
        if self.paper_alignment == "generalized_lower_tail_analogue":
            return (
                "The lower-inclusive transform is a generalized package "
                "extension documented by the paper only as a lower-tail "
                "analogue, not as its formal positive-rate equation."
            )
        return (
            "The exclusive threshold boundary is a generalized package "
            "extension, not the paper's formal score >= threshold definition."
        )

    def matches(self, score: float) -> bool:
        """Return whether one finite score satisfies the declared event rule."""
        value = _finite_float(score, "score")
        if self.direction is ScoreRateDirection.HIGHER:
            if self.inclusive:
                return value >= self.threshold
            return value > self.threshold
        if self.inclusive:
            return value <= self.threshold
        return value < self.threshold

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation of the complete rule."""
        return {
            "event_name": self.event_name,
            "threshold": self.threshold,
            "direction": self.direction.value,
            "inclusive": self.inclusive,
            "operator": self.operator,
            "paper_alignment": self.paper_alignment,
            "provenance": thaw_json(self.provenance),
        }


def _stable_mean(values: list[float]) -> float:
    """Return a finite mean without overflowing on repeated large values."""
    try:
        mean = math.fsum(values) / len(values)
    except OverflowError:
        scale = max(abs(value) for value in values)
        if scale == 0.0:  # pragma: no cover - zero values cannot overflow fsum
            return 0.0
        mean = scale * (math.fsum(value / scale for value in values) / len(values))
    if not math.isfinite(mean):
        raise ArithmeticError("group mean was not representable as a finite float")
    return mean


def _weighted_score_interval(
    lower: float,
    upper: float,
    mass_difference_numerator: int,
    mass_denominator: int,
) -> float:
    """Return one finite ECDF-area contribution without avoidable overflow."""
    if mass_difference_numerator == 0:
        return 0.0
    weight = mass_difference_numerator / mass_denominator
    width = upper - lower
    if math.isfinite(width):
        contribution = width * weight
    else:
        scale = max(abs(lower), abs(upper))
        normalized_width = (upper / scale) - (lower / scale)
        contribution = (scale * weight) * normalized_width
    if not math.isfinite(contribution):
        raise ArithmeticError(
            "Wasserstein interval contribution was not representable as a "
            "finite float"
        )
    return contribution


def _exact_empirical_wasserstein_1d(
    left_scores: list[float],
    right_scores: list[float],
) -> float:
    """Compute W1 exactly over binary-float inputs before one final rounding."""
    left = sorted(left_scores)
    right = sorted(right_scores)
    left_size = len(left)
    right_size = len(right)
    mass_denominator = left_size * right_size
    left_index = 0
    right_index = 0
    current = min(left[0], right[0])

    while left_index < left_size and left[left_index] == current:
        left_index += 1
    while right_index < right_size and right[right_index] == current:
        right_index += 1

    weighted_width_sum = Fraction(0)
    while left_index < left_size or right_index < right_size:
        next_left = left[left_index] if left_index < left_size else math.inf
        next_right = right[right_index] if right_index < right_size else math.inf
        next_score = min(next_left, next_right)
        mass_difference_numerator = abs(
            left_index * right_size - right_index * left_size
        )
        if mass_difference_numerator:
            exact_width = Fraction.from_float(next_score) - Fraction.from_float(current)
            weighted_width_sum += exact_width * mass_difference_numerator
        current = next_score
        while left_index < left_size and left[left_index] == current:
            left_index += 1
        while right_index < right_size and right[right_index] == current:
            right_index += 1

    exact_distance = weighted_width_sum / mass_denominator
    try:
        distance = float(exact_distance)
    except OverflowError as exc:
        raise ArithmeticError(
            "Wasserstein distance was not representable as a finite float"
        ) from exc
    if not math.isfinite(distance) or (exact_distance > 0 and distance == 0.0):
        raise ArithmeticError(
            "Wasserstein distance was not representable as a positive finite float"
        )
    return distance


def _empirical_wasserstein_1d(
    left_scores: list[float],
    right_scores: list[float],
) -> float:
    """Compute empirical 1-Wasserstein distance by exact ECDF mass counts."""
    if not left_scores or not right_scores:
        raise ValueError("empirical Wasserstein distance requires two non-empty groups")

    left = sorted(left_scores)
    right = sorted(right_scores)
    left_size = len(left)
    right_size = len(right)
    mass_denominator = left_size * right_size
    left_index = 0
    right_index = 0
    current = min(left[0], right[0])

    while left_index < left_size and left[left_index] == current:
        left_index += 1
    while right_index < right_size and right[right_index] == current:
        right_index += 1

    contributions = []
    requires_exact_fallback = False
    while left_index < left_size or right_index < right_size:
        next_left = left[left_index] if left_index < left_size else math.inf
        next_right = right[right_index] if right_index < right_size else math.inf
        next_score = min(next_left, next_right)
        mass_difference_numerator = abs(
            left_index * right_size - right_index * left_size
        )
        try:
            contribution = _weighted_score_interval(
                current,
                next_score,
                mass_difference_numerator,
                mass_denominator,
            )
        except ArithmeticError:
            return _exact_empirical_wasserstein_1d(left, right)
        contributions.append(contribution)
        if (
            mass_difference_numerator
            and next_score > current
            and contribution < float_info.min
        ):
            # Multiplying each tiny interval by its probability mass can round
            # too early. Re-evaluate all intervals exactly and round only once.
            requires_exact_fallback = True
        current = next_score
        while left_index < left_size and left[left_index] == current:
            left_index += 1
        while right_index < right_size and right[right_index] == current:
            right_index += 1

    if requires_exact_fallback:
        return _exact_empirical_wasserstein_1d(left, right)

    try:
        distance = math.fsum(contributions)
    except OverflowError as exc:
        raise ArithmeticError(
            "Wasserstein distance was not representable as a finite float"
        ) from exc
    if not math.isfinite(distance):
        raise ArithmeticError(
            "Wasserstein distance was not representable as a finite float"
        )
    return distance


@dataclass(frozen=True, kw_only=True)
class ScorerMeanGap(DatasetDiagnostic):
    """Maximum absolute pairwise difference between group mean scores."""

    name: ClassVar[str] = "score_mean_gap"

    def plan(
        self,
        evidence: ScoredGroups,
        spec: DatasetAuditSpec,
    ) -> ComponentPlan:
        """Plan the scorer component without performing arithmetic."""
        if not isinstance(evidence, ScoredGroups):
            raise TypeError(
                f"evidence must be ScoredGroups, got {type(evidence).__name__}."
            )
        if not isinstance(spec, DatasetAuditSpec):
            raise TypeError(
                f"spec must be DatasetAuditSpec, got {type(spec).__name__}."
            )

        requested = spec.requested_components
        if requested is not None and self.name not in requested:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.NOT_APPLICABLE,
                reason_code="component_not_requested",
                reason=(
                    "The score_mean_gap component was not requested by the "
                    "audit specification."
                ),
                details={
                    "axis": evidence.axis,
                    "score_name": evidence.score_name,
                },
            )

        if spec.target_kind is not TargetKind.SCORE_TABLE:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.NOT_APPLICABLE,
                reason_code="target_kind_not_supported",
                reason=(
                    "score_mean_gap applies to row-level scorer/model score "
                    "tables, not target kind "
                    f"{_enum_value(spec.target_kind)!r}."
                ),
                details={
                    "axis": evidence.axis,
                    "score_name": evidence.score_name,
                },
            )

        return ComponentPlan(
            component=self.name,
            status=DiagnosticStatus.READY,
            details={
                "axis": evidence.axis,
                "score_name": evidence.score_name,
                "support": evidence.support,
            },
        )

    def compute(
        self,
        evidence: ScoredGroups,
        spec: DatasetAuditSpec,
    ) -> ComponentResult:
        """Compute the paper-aligned maximum absolute group mean score gap."""
        plan = self.plan(evidence, spec)
        if plan.status is not DiagnosticStatus.READY:
            return ComponentResult(
                component=plan.component,
                status=plan.status,
                details=dict(plan.details),
                provenance={"evidence": _evidence_metadata(evidence)},
                reason_code=plan.reason_code,
                reason=plan.reason,
            )

        grouped_scores = {group: [] for group in evidence.support}
        for group, score in zip(evidence.groups, evidence.scores):
            grouped_scores[group].append(score)

        group_counts = {group: len(grouped_scores[group]) for group in evidence.support}
        try:
            group_means = {
                group: _stable_mean(grouped_scores[group]) for group in evidence.support
            }
        except (ArithmeticError, OverflowError, ValueError) as exc:
            return ComponentResult(
                component=self.name,
                status=DiagnosticStatus.FAILED,
                details={
                    "axis": evidence.axis,
                    "score_name": evidence.score_name,
                    "score_range": evidence.score_range,
                    "sample_count": evidence.total,
                    "support": evidence.support,
                    "group_counts": group_counts,
                    "exception_type": type(exc).__name__,
                },
                provenance={"evidence": _evidence_metadata(evidence)},
                reason_code="numeric_computation_failed",
                reason=(
                    "At least one group mean could not be represented as a "
                    "finite floating-point value."
                ),
            )

        pairwise = []
        best_value = -math.inf
        argmax_groups: tuple[str, str] | None = None
        higher_group = None
        lower_group = None
        for left_group, right_group in combinations(evidence.support, 2):
            left_mean = group_means[left_group]
            right_mean = group_means[right_group]
            signed_gap = right_mean - left_mean
            if not math.isfinite(signed_gap):
                return ComponentResult(
                    component=self.name,
                    status=DiagnosticStatus.FAILED,
                    details={
                        "axis": evidence.axis,
                        "score_name": evidence.score_name,
                        "score_range": evidence.score_range,
                        "sample_count": evidence.total,
                        "support": evidence.support,
                        "group_counts": group_counts,
                        "group_means": group_means,
                        "unrepresentable_pair": {
                            "left_group": left_group,
                            "right_group": right_group,
                            "left_mean": left_mean,
                            "right_mean": right_mean,
                        },
                    },
                    provenance={"evidence": _evidence_metadata(evidence)},
                    reason_code="numeric_computation_failed",
                    reason=(
                        "A pairwise group-mean difference could not be "
                        "represented as a finite floating-point value."
                    ),
                )
            absolute_gap = abs(signed_gap)
            pairwise.append(
                {
                    "left_group": left_group,
                    "right_group": right_group,
                    "left_mean": left_mean,
                    "right_mean": right_mean,
                    "signed_gap_right_minus_left": signed_gap,
                    "absolute_gap": absolute_gap,
                }
            )
            if absolute_gap > best_value:
                best_value = absolute_gap
                argmax_groups = (left_group, right_group)
                if signed_gap > 0.0:
                    higher_group, lower_group = right_group, left_group
                elif signed_gap < 0.0:
                    higher_group, lower_group = left_group, right_group
                else:
                    higher_group = lower_group = None

        assert argmax_groups is not None
        return ComponentResult(
            component=self.name,
            status=DiagnosticStatus.READY,
            value=best_value,
            details={
                "axis": evidence.axis,
                "score_name": evidence.score_name,
                "score_range": evidence.score_range,
                "sample_count": evidence.total,
                "support": evidence.support,
                "group_counts": group_counts,
                "group_means": group_means,
                "pairwise_gaps": pairwise,
                "argmax_groups": argmax_groups,
                "higher_group": higher_group,
                "lower_group": lower_group,
                "aggregation": "maximum_absolute_pairwise_group_mean_difference",
                "unit": "score_units",
            },
            assumptions=(
                _DESCRIPTIVE_INTERPRETATION,
                _SCALE_INTERPRETATION,
                f"Target design stance: {_enum_value(spec.design_stance)}.",
            ),
            provenance={"evidence": _evidence_metadata(evidence)},
        )


@dataclass(frozen=True, kw_only=True)
class ScorerRateGap(DatasetDiagnostic):
    """Maximum absolute pairwise difference between group event rates."""

    transform: ScoreRateTransform | None = None
    name: ClassVar[str] = "score_rate_gap"

    def __post_init__(self) -> None:
        if self.transform is not None and not isinstance(
            self.transform, ScoreRateTransform
        ):
            raise TypeError(
                "transform must be ScoreRateTransform or None, "
                f"got {type(self.transform).__name__}."
            )

    def plan(
        self,
        evidence: ScoredGroups,
        spec: DatasetAuditSpec,
    ) -> ComponentPlan:
        """Plan the rate component without applying the threshold rule."""
        if not isinstance(evidence, ScoredGroups):
            raise TypeError(
                f"evidence must be ScoredGroups, got {type(evidence).__name__}."
            )
        if not isinstance(spec, DatasetAuditSpec):
            raise TypeError(
                f"spec must be DatasetAuditSpec, got {type(spec).__name__}."
            )

        if self.name not in spec.requested_components:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.NOT_APPLICABLE,
                reason_code="component_not_requested",
                reason=(
                    "The score_rate_gap component was not requested by the "
                    "audit specification."
                ),
                details={
                    "axis": evidence.axis,
                    "score_name": evidence.score_name,
                },
            )

        if spec.target_kind is not TargetKind.SCORE_TABLE:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.NOT_APPLICABLE,
                reason_code="target_kind_not_supported",
                reason=(
                    "score_rate_gap applies to row-level scorer/model score "
                    "tables, not target kind "
                    f"{_enum_value(spec.target_kind)!r}."
                ),
                details={
                    "axis": evidence.axis,
                    "score_name": evidence.score_name,
                },
            )

        if self.transform is None:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.BLOCKED,
                reason_code="missing_rate_transform",
                reason=(
                    "score_rate_gap requires an explicit score-to-event "
                    "threshold transform."
                ),
                details={
                    "axis": evidence.axis,
                    "score_name": evidence.score_name,
                },
            )

        if evidence.score_range is not None:
            lower, upper = evidence.score_range
            if self.transform.threshold < lower or self.transform.threshold > upper:
                return ComponentPlan(
                    component=self.name,
                    status=DiagnosticStatus.BLOCKED,
                    reason_code="threshold_outside_score_range",
                    reason=(
                        "The declared rate threshold must lie inside the "
                        "evidence score range."
                    ),
                    details={
                        "axis": evidence.axis,
                        "score_name": evidence.score_name,
                        "score_range": evidence.score_range,
                        "transform": self.transform.to_dict(),
                    },
                )

        return ComponentPlan(
            component=self.name,
            status=DiagnosticStatus.READY,
            details={
                "axis": evidence.axis,
                "score_name": evidence.score_name,
                "support": evidence.support,
                "transform": self.transform.to_dict(),
            },
        )

    def compute(
        self,
        evidence: ScoredGroups,
        spec: DatasetAuditSpec,
    ) -> ComponentResult:
        """Compute the paper-aligned maximum absolute group event-rate gap."""
        plan = self.plan(evidence, spec)
        provenance = {"evidence": _evidence_metadata(evidence)}
        if self.transform is not None:
            provenance["rate_transform"] = self.transform.to_dict()
        if plan.status is not DiagnosticStatus.READY:
            return ComponentResult(
                component=plan.component,
                status=plan.status,
                details=dict(plan.details),
                provenance=provenance,
                reason_code=plan.reason_code,
                reason=plan.reason,
            )

        assert self.transform is not None
        group_counts = {group: 0 for group in evidence.support}
        event_counts = {group: 0 for group in evidence.support}
        for group, score in zip(evidence.groups, evidence.scores):
            group_counts[group] += 1
            if self.transform.matches(score):
                event_counts[group] += 1
        group_rates = {
            group: event_counts[group] / group_counts[group]
            for group in evidence.support
        }

        pairwise = []
        best_value = -math.inf
        argmax_groups: tuple[str, str] | None = None
        higher_rate_group = None
        lower_rate_group = None
        for left_group, right_group in combinations(evidence.support, 2):
            left_rate = group_rates[left_group]
            right_rate = group_rates[right_group]
            signed_gap = right_rate - left_rate
            absolute_gap = abs(signed_gap)
            pairwise.append(
                {
                    "left_group": left_group,
                    "right_group": right_group,
                    "left_rate": left_rate,
                    "right_rate": right_rate,
                    "signed_gap_right_minus_left": signed_gap,
                    "absolute_gap": absolute_gap,
                }
            )
            if absolute_gap > best_value:
                best_value = absolute_gap
                argmax_groups = (left_group, right_group)
                if signed_gap > 0.0:
                    higher_rate_group, lower_rate_group = right_group, left_group
                elif signed_gap < 0.0:
                    higher_rate_group, lower_rate_group = left_group, right_group
                else:
                    higher_rate_group = lower_rate_group = None

        assert argmax_groups is not None
        return ComponentResult(
            component=self.name,
            status=DiagnosticStatus.READY,
            value=best_value,
            details={
                "axis": evidence.axis,
                "score_name": evidence.score_name,
                "score_range": evidence.score_range,
                "transform": self.transform.to_dict(),
                "sample_count": evidence.total,
                "support": evidence.support,
                "group_counts": group_counts,
                "event_counts": event_counts,
                "group_rates": group_rates,
                "pairwise_rate_gaps": pairwise,
                "argmax_groups": argmax_groups,
                "higher_rate_group": higher_rate_group,
                "lower_rate_group": lower_rate_group,
                "aggregation": "maximum_absolute_pairwise_group_rate_difference",
                "unit": "proportion",
            },
            assumptions=(
                _RATE_INTERPRETATION,
                _AGGREGATE_CONFOUNDING,
                _RATE_CONTEXT_INTERPRETATION,
                _RATE_UNIT_INTERPRETATION,
                self.transform.alignment_assumption,
                f"Target design stance: {_enum_value(spec.design_stance)}.",
            ),
            provenance=provenance,
        )


@dataclass(frozen=True, kw_only=True)
class ScorerWasserstein1Gap(DatasetDiagnostic):
    """Maximum pairwise empirical one-dimensional Wasserstein distance."""

    name: ClassVar[str] = "score_wasserstein_1_gap"

    def plan(
        self,
        evidence: ScoredGroups,
        spec: DatasetAuditSpec,
    ) -> ComponentPlan:
        """Plan the distribution component without performing arithmetic."""
        if not isinstance(evidence, ScoredGroups):
            raise TypeError(
                f"evidence must be ScoredGroups, got {type(evidence).__name__}."
            )
        if not isinstance(spec, DatasetAuditSpec):
            raise TypeError(
                f"spec must be DatasetAuditSpec, got {type(spec).__name__}."
            )

        if self.name not in spec.requested_components:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.NOT_APPLICABLE,
                reason_code="component_not_requested",
                reason=(
                    "The score_wasserstein_1_gap component was not requested "
                    "by the audit specification."
                ),
                details={
                    "axis": evidence.axis,
                    "score_name": evidence.score_name,
                },
            )

        if spec.target_kind is not TargetKind.SCORE_TABLE:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.NOT_APPLICABLE,
                reason_code="target_kind_not_supported",
                reason=(
                    "score_wasserstein_1_gap applies to row-level scorer/model "
                    "score tables, not target kind "
                    f"{_enum_value(spec.target_kind)!r}."
                ),
                details={
                    "axis": evidence.axis,
                    "score_name": evidence.score_name,
                },
            )

        return ComponentPlan(
            component=self.name,
            status=DiagnosticStatus.READY,
            details={
                "axis": evidence.axis,
                "score_name": evidence.score_name,
                "support": evidence.support,
            },
        )

    def compute(
        self,
        evidence: ScoredGroups,
        spec: DatasetAuditSpec,
    ) -> ComponentResult:
        """Compute the paper-aligned maximum pairwise empirical W1 distance."""
        plan = self.plan(evidence, spec)
        provenance = {"evidence": _evidence_metadata(evidence)}
        if plan.status is not DiagnosticStatus.READY:
            return ComponentResult(
                component=plan.component,
                status=plan.status,
                details=dict(plan.details),
                provenance=provenance,
                reason_code=plan.reason_code,
                reason=plan.reason,
            )

        grouped_scores = {group: [] for group in evidence.support}
        for group, score in zip(evidence.groups, evidence.scores):
            grouped_scores[group].append(score)
        group_counts = {group: len(grouped_scores[group]) for group in evidence.support}

        pairwise = []
        best_value = -math.inf
        argmax_groups: tuple[str, str] | None = None
        for left_group, right_group in combinations(evidence.support, 2):
            try:
                distance = _empirical_wasserstein_1d(
                    grouped_scores[left_group],
                    grouped_scores[right_group],
                )
            except (ArithmeticError, OverflowError, ValueError) as exc:
                return ComponentResult(
                    component=self.name,
                    status=DiagnosticStatus.FAILED,
                    details={
                        "axis": evidence.axis,
                        "score_name": evidence.score_name,
                        "score_range": evidence.score_range,
                        "sample_count": evidence.total,
                        "support": evidence.support,
                        "group_counts": group_counts,
                        "pairwise_wasserstein_1": pairwise,
                        "unrepresentable_pair": {
                            "left_group": left_group,
                            "right_group": right_group,
                            "left_sample_count": group_counts[left_group],
                            "right_sample_count": group_counts[right_group],
                        },
                        "exception_type": type(exc).__name__,
                    },
                    provenance=provenance,
                    reason_code="numeric_computation_failed",
                    reason=(
                        "A pairwise empirical Wasserstein distance could not be "
                        "represented faithfully as a finite floating-point value."
                    ),
                )

            pairwise.append(
                {
                    "left_group": left_group,
                    "right_group": right_group,
                    "left_sample_count": group_counts[left_group],
                    "right_sample_count": group_counts[right_group],
                    "wasserstein_1": distance,
                }
            )
            if distance > best_value:
                best_value = distance
                argmax_groups = (left_group, right_group)

        assert argmax_groups is not None
        return ComponentResult(
            component=self.name,
            status=DiagnosticStatus.READY,
            value=best_value,
            details={
                "axis": evidence.axis,
                "score_name": evidence.score_name,
                "score_range": evidence.score_range,
                "sample_count": evidence.total,
                "support": evidence.support,
                "group_counts": group_counts,
                "pairwise_wasserstein_1": pairwise,
                "argmax_groups": argmax_groups,
                "aggregation": (
                    "maximum_pairwise_one_dimensional_empirical_wasserstein_distance"
                ),
                "estimator": "empirical_uniform_mass_per_group",
                "ground_metric": "absolute_score_difference",
                "directionality": "symmetric",
                "normalization": "none",
                "unit": "score_units",
            },
            assumptions=(
                _DISTRIBUTION_INTERPRETATION,
                _DISTRIBUTION_CONFOUNDING,
                _DISTRIBUTION_GEOMETRY,
                _DISTRIBUTION_CONTEXT_INTERPRETATION,
                _POINT_ESTIMATE_INTERPRETATION,
                f"Target design stance: {_enum_value(spec.design_stance)}.",
            ),
            provenance=provenance,
        )


@dataclass(frozen=True, kw_only=True)
class ScorerCounterfactualSensitivity(DatasetDiagnostic):
    """Mean absolute score change within complete two-condition pairs."""

    name: ClassVar[str] = "score_counterfactual_sensitivity"

    def plan(
        self,
        evidence: PairedScores,
        spec: DatasetAuditSpec,
    ) -> ComponentPlan:
        """Plan the paired component without performing score arithmetic."""
        if not isinstance(evidence, PairedScores):
            raise TypeError(
                f"evidence must be PairedScores, got {type(evidence).__name__}."
            )
        if not isinstance(spec, DatasetAuditSpec):
            raise TypeError(
                f"spec must be DatasetAuditSpec, got {type(spec).__name__}."
            )

        if self.name not in spec.requested_components:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.NOT_APPLICABLE,
                reason_code="component_not_requested",
                reason=(
                    "The score_counterfactual_sensitivity component was not "
                    "requested by the audit specification."
                ),
                details={
                    "axis": evidence.axis,
                    "score_name": evidence.score_name,
                },
            )

        if spec.target_kind is not TargetKind.SCORE_TABLE:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.NOT_APPLICABLE,
                reason_code="target_kind_not_supported",
                reason=(
                    "score_counterfactual_sensitivity applies to row-level "
                    "paired scorer/model score tables, not target kind "
                    f"{_enum_value(spec.target_kind)!r}."
                ),
                details={
                    "axis": evidence.axis,
                    "score_name": evidence.score_name,
                },
            )

        return ComponentPlan(
            component=self.name,
            status=DiagnosticStatus.READY,
            details={
                "axis": evidence.axis,
                "score_name": evidence.score_name,
                "condition_roles": evidence.condition_roles,
                "pair_count": evidence.pair_count,
                "pairing_basis": evidence.pairing_basis,
            },
        )

    def compute(
        self,
        evidence: PairedScores,
        spec: DatasetAuditSpec,
    ) -> ComponentResult:
        """Compute the paper-aligned mean absolute within-pair score change."""
        plan = self.plan(evidence, spec)
        provenance = {"evidence": _evidence_metadata(evidence)}
        if plan.status is not DiagnosticStatus.READY:
            return ComponentResult(
                component=plan.component,
                status=plan.status,
                details=dict(plan.details),
                provenance=provenance,
                reason_code=plan.reason_code,
                reason=plan.reason,
            )

        first_role, second_role = evidence.condition_roles
        signed_differences = []
        absolute_differences = []
        for index in range(0, evidence.total, 2):
            first_score = evidence.scores[index]
            second_score = evidence.scores[index + 1]
            signed_difference = second_score - first_score
            if not math.isfinite(signed_difference):
                return ComponentResult(
                    component=self.name,
                    status=DiagnosticStatus.FAILED,
                    details={
                        "axis": evidence.axis,
                        "score_name": evidence.score_name,
                        "score_range": evidence.score_range,
                        "sample_count": evidence.total,
                        "pair_count": evidence.pair_count,
                        "condition_roles": evidence.condition_roles,
                        "pairing_basis": evidence.pairing_basis,
                        "unrepresentable_pair": {
                            "pair_id": evidence.pair_ids[index],
                            "first_role": first_role,
                            "second_role": second_role,
                            "first_score": first_score,
                            "second_score": second_score,
                        },
                    },
                    provenance=provenance,
                    reason_code="numeric_computation_failed",
                    reason=(
                        "A within-pair score difference could not be represented "
                        "as a finite floating-point value."
                    ),
                )
            signed_differences.append(signed_difference)
            absolute_differences.append(abs(signed_difference))

        try:
            mean_absolute_difference = _stable_mean(absolute_differences)
            mean_signed_difference = _stable_mean(signed_differences)
        except (ArithmeticError, OverflowError, ValueError) as exc:
            return ComponentResult(
                component=self.name,
                status=DiagnosticStatus.FAILED,
                details={
                    "axis": evidence.axis,
                    "score_name": evidence.score_name,
                    "score_range": evidence.score_range,
                    "sample_count": evidence.total,
                    "pair_count": evidence.pair_count,
                    "condition_roles": evidence.condition_roles,
                    "pairing_basis": evidence.pairing_basis,
                    "exception_type": type(exc).__name__,
                },
                provenance=provenance,
                reason_code="numeric_computation_failed",
                reason=(
                    "The paired score-difference mean could not be represented "
                    "as a finite floating-point value."
                ),
            )

        if mean_absolute_difference == 0.0 and any(absolute_differences):
            return ComponentResult(
                component=self.name,
                status=DiagnosticStatus.FAILED,
                details={
                    "axis": evidence.axis,
                    "score_name": evidence.score_name,
                    "score_range": evidence.score_range,
                    "sample_count": evidence.total,
                    "pair_count": evidence.pair_count,
                    "condition_roles": evidence.condition_roles,
                    "pairing_basis": evidence.pairing_basis,
                },
                provenance=provenance,
                reason_code="numeric_computation_failed",
                reason=(
                    "A positive counterfactual sensitivity was too small to be "
                    "represented as a nonzero floating-point value."
                ),
            )

        first_role_higher = sum(value < 0.0 for value in signed_differences)
        second_role_higher = sum(value > 0.0 for value in signed_differences)
        equal_score_pairs = sum(value == 0.0 for value in signed_differences)
        return ComponentResult(
            component=self.name,
            status=DiagnosticStatus.READY,
            value=mean_absolute_difference,
            details={
                "axis": evidence.axis,
                "score_name": evidence.score_name,
                "score_range": evidence.score_range,
                "sample_count": evidence.total,
                "pair_count": evidence.pair_count,
                "condition_roles": evidence.condition_roles,
                "condition_counts": {
                    first_role: evidence.pair_count,
                    second_role: evidence.pair_count,
                },
                "pairing_basis": evidence.pairing_basis,
                "mean_absolute_difference": mean_absolute_difference,
                "mean_signed_difference_second_minus_first": (mean_signed_difference),
                "minimum_absolute_difference": min(absolute_differences),
                "maximum_absolute_difference": max(absolute_differences),
                "first_role_higher_pair_count": first_role_higher,
                "second_role_higher_pair_count": second_role_higher,
                "equal_score_pair_count": equal_score_pairs,
                "aggregation": "mean_absolute_within_pair_score_difference",
                "estimator": "empirical_uniform_mass_per_pair",
                "signed_difference_orientation": (
                    "condition_roles[1]_minus_condition_roles[0]"
                ),
                "directionality": "symmetric_absolute_primary_value",
                "unit": "score_units",
            },
            assumptions=(
                _COUNTERFACTUAL_INTERPRETATION,
                _PAIRING_VALIDITY_INTERPRETATION,
                _COUNTERFACTUAL_DIRECTION_INTERPRETATION,
                _COUNTERFACTUAL_SCALE_INTERPRETATION,
                _POINT_ESTIMATE_INTERPRETATION,
                f"Target design stance: {_enum_value(spec.design_stance)}.",
            ),
            provenance=provenance,
        )


def _normalize_diagnostic_selection(
    diagnostic: DatasetDiagnostic | None,
    diagnostics: Sequence[DatasetDiagnostic] | None,
) -> tuple[DatasetDiagnostic, ...]:
    if diagnostic is not None and diagnostics is not None:
        raise TypeError("diagnostic and diagnostics cannot be provided together.")
    if diagnostics is None:
        selected = ScorerMeanGap() if diagnostic is None else diagnostic
        if not isinstance(selected, DatasetDiagnostic):
            raise TypeError(
                "diagnostic must be a DatasetDiagnostic, "
                f"got {type(selected).__name__}."
            )
        return (selected,)

    if isinstance(diagnostics, (str, bytes)) or not isinstance(diagnostics, Sequence):
        raise TypeError("diagnostics must be a non-empty ordered sequence.")
    selected_diagnostics = tuple(diagnostics)
    if not selected_diagnostics:
        raise ValueError("diagnostics must contain at least one component.")

    names = []
    for index, selected in enumerate(selected_diagnostics):
        if not isinstance(selected, DatasetDiagnostic):
            raise TypeError(
                f"diagnostics[{index}] must be a DatasetDiagnostic, "
                f"got {type(selected).__name__}."
            )
        names.append(
            require_nonempty_string(
                selected.name,
                f"diagnostics[{index}].name",
            )
        )
    if len(set(names)) != len(names):
        raise ValueError("diagnostics must not contain duplicate component names.")
    return selected_diagnostics


def _failure_provenance(
    selected: DatasetDiagnostic,
    evidence: ScoredGroups | PairedScores,
) -> dict[str, Any]:
    provenance = {"evidence": _evidence_metadata(evidence)}
    if isinstance(selected, ScorerRateGap) and selected.transform is not None:
        provenance["rate_transform"] = selected.transform.to_dict()
    return provenance


def audit_scores(
    evidence: ScoredGroups | PairedScores,
    spec: DatasetAuditSpec,
    diagnostic: DatasetDiagnostic | None = None,
    *,
    diagnostics: Sequence[DatasetDiagnostic] | None = None,
) -> DiagnosticReport:
    """Audit one score channel with one or more explicit diagnostics."""
    if (
        isinstance(evidence, PairedScores)
        and diagnostic is None
        and diagnostics is None
    ):
        raise TypeError(
            "PairedScores requires an explicit paired diagnostic selection; "
            "audit_scores(...) defaults to ScorerMeanGap only for ScoredGroups."
        )
    selected_diagnostics = _normalize_diagnostic_selection(diagnostic, diagnostics)
    results = {}
    for selected in selected_diagnostics:
        plan = selected.plan(evidence, spec)
        try:
            result = selected.compute(evidence, spec)
        except Exception as exc:
            if plan.status is not DiagnosticStatus.READY:
                raise
            result = ComponentResult(
                component=selected.name,
                status=DiagnosticStatus.FAILED,
                details={"exception_type": type(exc).__name__},
                provenance=_failure_provenance(selected, evidence),
                reason_code="computation_failed",
                reason=(
                    "Applicability was established, but the score diagnostic "
                    f"failed with {type(exc).__name__}."
                ),
            )
        results[selected.name] = result

    report_provenance = (
        {"diagnostic": selected_diagnostics[0].name}
        if len(selected_diagnostics) == 1
        else {"diagnostics": [item.name for item in selected_diagnostics]}
    )
    return DiagnosticReport(
        spec=spec,
        components=results,
        provenance=report_provenance,
    )


__all__ = [
    "ScoreRateDirection",
    "ScoreRateTransform",
    "ScorerCounterfactualSensitivity",
    "ScorerMeanGap",
    "ScorerRateGap",
    "ScorerWasserstein1Gap",
    "audit_scores",
]
