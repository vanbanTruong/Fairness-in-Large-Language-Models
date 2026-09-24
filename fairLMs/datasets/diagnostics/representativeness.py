"""Axis-level representativeness diagnostic.

The diagnostic compares observed category counts with an explicitly supplied
reference distribution.  It intentionally makes no judgment about whether a
particular amount of divergence is acceptable.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar, Mapping

from .base import (
    ComponentPlan,
    ComponentResult,
    DatasetDiagnostic,
    DiagnosticReport,
    DiagnosticStatus,
)
from .evidence import RepresentationEvidence
from .spec import DatasetAuditSpec, DesignStance, TargetKind

_DESCRIPTIVE_INTERPRETATION = (
    "Representativeness divergence is descriptive evidence, not an automatic "
    "fairness pass/fail judgment; its meaning depends on the benchmark design "
    "stance and the purpose of the reference distribution."
)
_STRESS_TEST_WARNING = (
    "The audit has a stress-test design stance: divergence from the reference "
    "may be intentional and must not be treated as an automatic fairness failure."
)


def _enum_value(value: Any) -> Any:
    """Return a stable scalar for enum-like public values."""

    return getattr(value, "value", value)


def _reference_metadata(reference: Any) -> dict[str, Any]:
    """Copy the descriptive reference fields into result provenance."""

    metadata: dict[str, Any] = {}
    for name in (
        "axis",
        "source",
        "purpose",
        "population",
        "geography",
        "period",
        "provenance",
        "input_probability_sum",
        "normalization_applied",
    ):
        value = getattr(reference, name)
        if isinstance(value, Mapping):
            value = dict(value)
        metadata[name] = _enum_value(value)
    return metadata


def _evidence_metadata(evidence: RepresentationEvidence) -> dict[str, Any]:
    """Copy evidence source metadata into result provenance."""

    return {
        "axis": evidence.axis,
        "source": evidence.source,
        "provenance": dict(evidence.provenance),
    }


def _stable_log_ratio(numerator: float, denominator: float) -> float:
    """Preserve canonical arithmetic, with a safe extreme-value fallback."""

    ratio = numerator / denominator
    if ratio > 0.0 and math.isfinite(ratio):
        return math.log(ratio)
    return math.log(numerator) - math.log(denominator)


@dataclass(frozen=True, kw_only=True)
class RepresentativenessBias(DatasetDiagnostic):
    """Smoothed KL divergence between observed and reference composition."""

    name: ClassVar[str] = "b_rep"
    smoothing_mass: float = 0.01

    def __post_init__(self) -> None:
        value = self.smoothing_mass
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                "smoothing_mass must be a real number (booleans are not accepted)"
            )
        try:
            converted = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                "smoothing_mass must be representable as a finite float"
            ) from exc
        if not math.isfinite(converted) or converted < sys.float_info.min:
            raise ValueError(
                "smoothing_mass must be a finite float at least as large as "
                f"sys.float_info.min ({sys.float_info.min!r})"
            )
        object.__setattr__(self, "smoothing_mass", converted)

    def plan(
        self,
        evidence: RepresentationEvidence,
        spec: DatasetAuditSpec,
    ) -> ComponentPlan:
        """Plan this component without performing the numeric calculation."""

        if not isinstance(evidence, RepresentationEvidence):
            raise TypeError(
                "evidence must be RepresentationEvidence, "
                f"got {type(evidence).__name__}."
            )
        if not isinstance(spec, DatasetAuditSpec):
            raise TypeError(
                "spec must be DatasetAuditSpec, "
                f"got {type(spec).__name__}."
            )

        requested = spec.requested_components
        if requested is not None and self.name not in requested:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.NOT_APPLICABLE,
                reason_code="component_not_requested",
                reason="The b_rep component was not requested by the audit specification.",
                details={"axis": evidence.axis},
            )

        if spec.target_kind is not TargetKind.BENCHMARK_DATASET:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.NOT_APPLICABLE,
                reason_code="target_kind_not_supported",
                reason=(
                    "b_rep applies to benchmark-dataset composition, not target "
                    f"kind {_enum_value(spec.target_kind)!r}."
                ),
                details={"axis": evidence.axis},
            )

        reference = spec.references.get(evidence.axis)
        if reference is None:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.BLOCKED,
                reason_code="missing_reference",
                reason=(
                    "An explicit reference distribution is required for axis "
                    f"{evidence.axis!r}."
                ),
                details={"axis": evidence.axis},
            )

        observed_support = set(evidence.counts)
        reference_support = set(reference.probabilities)
        if observed_support != reference_support:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.BLOCKED,
                reason_code="reference_support_mismatch",
                reason=(
                    "Observed counts and the reference distribution must use "
                    "exactly the same category support."
                ),
                details={
                    "axis": evidence.axis,
                    "observed_support": sorted(observed_support),
                    "reference_support": sorted(reference_support),
                    "missing_from_reference": sorted(
                        observed_support - reference_support
                    ),
                    "missing_from_observed": sorted(
                        reference_support - observed_support
                    ),
                },
            )

        return ComponentPlan(
            component=self.name,
            status=DiagnosticStatus.READY,
            details={
                "axis": evidence.axis,
                "support": sorted(observed_support),
            },
        )

    def compute(
        self,
        evidence: RepresentationEvidence,
        spec: DatasetAuditSpec,
    ) -> ComponentResult:
        """Compute smoothed ``KL(observed || reference)`` in nats."""

        plan = self.plan(evidence, spec)
        if plan.status is not DiagnosticStatus.READY:
            reference = spec.references.get(evidence.axis)
            provenance = {"evidence": _evidence_metadata(evidence)}
            if reference is not None:
                provenance["reference"] = _reference_metadata(reference)
            return ComponentResult(
                component=plan.component,
                status=plan.status,
                details=dict(plan.details),
                provenance=provenance,
                reason_code=plan.reason_code,
                reason=plan.reason,
            )

        reference = spec.references[evidence.axis]
        support = sorted(evidence.counts)
        sample_count = sum(evidence.counts.values())
        observed_counts = {
            category: evidence.counts[category] for category in support
        }
        observed_distribution = {
            category: observed_counts[category] / sample_count
            for category in support
        }
        reference_distribution = {
            category: reference.probabilities[category] for category in support
        }

        smoothing_per_category = self.smoothing_mass / len(support)
        if smoothing_per_category == 0.0:
            return ComponentResult(
                component=self.name,
                status=DiagnosticStatus.FAILED,
                details={
                    "axis": evidence.axis,
                    "support_size": len(support),
                    "smoothing_mass": self.smoothing_mass,
                },
                provenance={
                    "evidence": _evidence_metadata(evidence),
                    "reference": _reference_metadata(reference),
                },
                reason_code="smoothing_underflow",
                reason=(
                    "The per-category smoothing mass underflowed to zero for "
                    "this support."
                ),
            )
        smoothing_denominator = 1.0 + self.smoothing_mass
        smoothed_observed_distribution = {
            category: (
                observed_distribution[category] + smoothing_per_category
            )
            / smoothing_denominator
            for category in support
        }
        smoothed_reference_distribution = {
            category: (
                reference_distribution[category] + smoothing_per_category
            )
            / smoothing_denominator
            for category in support
        }
        contributions = {
            category: smoothed_observed_distribution[category]
            * _stable_log_ratio(
                smoothed_observed_distribution[category],
                smoothed_reference_distribution[category],
            )
            for category in support
        }
        value = math.fsum(contributions.values())
        if not math.isfinite(value) or value < -1e-12:
            return ComponentResult(
                component=self.name,
                status=DiagnosticStatus.FAILED,
                details={
                    "axis": evidence.axis,
                    "support": support,
                    "smoothing_mass": self.smoothing_mass,
                    "numeric_value": value if math.isfinite(value) else None,
                },
                provenance={
                    "evidence": _evidence_metadata(evidence),
                    "reference": _reference_metadata(reference),
                },
                reason_code="numeric_computation_failed",
                reason="The KL calculation did not produce a finite non-negative value.",
            )
        numeric_zero_clamped = value < 0.0
        if numeric_zero_clamped:
            value = 0.0

        return ComponentResult(
            component=self.name,
            status=DiagnosticStatus.READY,
            value=value,
            details={
                "axis": evidence.axis,
                "support": support,
                "sample_count": sample_count,
                "observed_counts": observed_counts,
                "observed_distribution": observed_distribution,
                "reference_distribution": reference_distribution,
                "smoothed_observed_distribution": smoothed_observed_distribution,
                "smoothed_reference_distribution": smoothed_reference_distribution,
                "contributions": contributions,
                "smoothing_mass": self.smoothing_mass,
                "smoothing_per_category": smoothing_per_category,
                "direction": "observed||reference",
                "unit": "nats",
                "numeric_zero_clamped": numeric_zero_clamped,
            },
            assumptions=(
                _DESCRIPTIVE_INTERPRETATION,
                f"Design stance: {_enum_value(spec.design_stance)}.",
                f"Reference purpose: {_enum_value(reference.purpose)}.",
            ),
            provenance={
                "evidence": _evidence_metadata(evidence),
                "reference": _reference_metadata(reference),
            },
        )


def audit_representativeness(
    evidence: RepresentationEvidence,
    spec: DatasetAuditSpec,
    diagnostic: RepresentativenessBias | None = None,
) -> DiagnosticReport:
    """Run the representativeness component and return a one-part report."""

    selected = diagnostic if diagnostic is not None else RepresentativenessBias()
    plan = selected.plan(evidence, spec)
    try:
        result = selected.compute(evidence, spec)
    except Exception as exc:
        if plan.status is not DiagnosticStatus.READY:
            raise
        reference = spec.references.get(evidence.axis)
        provenance = {"evidence": _evidence_metadata(evidence)}
        if reference is not None:
            provenance["reference"] = _reference_metadata(reference)
        result = ComponentResult(
            component=selected.name,
            status=DiagnosticStatus.FAILED,
            details={"exception_type": type(exc).__name__},
            provenance=provenance,
            reason_code="computation_failed",
            reason=(
                "Applicability was established, but the diagnostic computation "
                f"failed with {type(exc).__name__}."
            ),
        )
    warnings = (
        (_STRESS_TEST_WARNING,)
        if (
            spec.design_stance is DesignStance.STRESS_TEST
            and result.status is DiagnosticStatus.READY
        )
        else ()
    )
    return DiagnosticReport(
        spec=spec,
        components={selected.name: result},
        warnings=warnings,
        provenance={"diagnostic": selected.name},
    )


__all__ = ["RepresentativenessBias", "audit_representativeness"]
