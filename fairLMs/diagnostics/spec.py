"""Audit intent and reference contracts for fairness diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from fairLMs.diagnostics._utils import (
    freeze_json_mapping,
    normalize_enum,
    normalize_string_sequence,
    require_nonempty_string,
    thaw_json,
)

if TYPE_CHECKING:  # pragma: no cover - imported only by type checkers
    from fairLMs.diagnostics.base import DiagnosticStatus
    from fairLMs.diagnostics.leakage import LeakageExtractionConfig

_PROBABILITY_SUM_TOLERANCE = 1e-9


class TargetKind(str, Enum):
    """Kind of artifact supplied as diagnostic evidence."""

    BENCHMARK_DATASET = "benchmark_dataset"
    GENERATED_OUTPUT = "generated_output"
    SCORE_TABLE = "score_table"
    AGGREGATE_STATISTICS = "aggregate_statistics"


class DesignStance(str, Enum):
    """How the benchmark's composition is intended to be interpreted."""

    POPULATION_PROXY = "population_proxy"
    STRESS_TEST = "stress_test"


class ReferencePurpose(str, Enum):
    """Purpose claimed for a representativeness reference distribution."""

    POPULATION = "population"
    DESIGN_TARGET = "design_target"
    PROXY = "proxy"


def _validate_probability_mapping(
    value: Any,
) -> tuple[Mapping[str, float], float, bool]:
    if not isinstance(value, Mapping):
        raise TypeError(
            "probabilities must be a mapping of category label -> probability, "
            f"got {type(value).__name__}."
        )
    if len(value) < 2:
        raise ValueError("probabilities must contain at least two categories.")

    probabilities = {}
    for label, raw in value.items():
        require_nonempty_string(label, "probability category")
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise TypeError(
                f"probability for {label!r} must be a real number, "
                f"got {type(raw).__name__}."
            )
        try:
            probability = float(raw)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"probability for {label!r} must be representable as a float."
            ) from exc
        if not math.isfinite(probability):
            raise ValueError(f"probability for {label!r} must be finite.")
        if probability < 0.0:
            raise ValueError(f"probability for {label!r} must be non-negative.")
        probabilities[label] = probability

    total = math.fsum(probabilities.values())
    if not math.isclose(
        total,
        1.0,
        rel_tol=0.0,
        abs_tol=_PROBABILITY_SUM_TOLERANCE,
    ):
        raise ValueError(
            "probabilities must sum to 1.0 within absolute tolerance "
            f"{_PROBABILITY_SUM_TOLERANCE}; got {total!r}. Values outside "
            "that tolerance are not normalized automatically."
        )

    normalization_applied = total != 1.0
    if normalization_applied:
        # Values inside the declared tolerance are treated as floating-point
        # residue, but the stored object must still lie exactly on the simplex.
        # Put the final residual on the largest cell to minimize relative error.
        anchor = max(probabilities, key=lambda label: (probabilities[label], label))
        normalized = {
            label: probability / total
            for label, probability in probabilities.items()
            if label != anchor
        }
        normalized[anchor] = 1.0 - math.fsum(normalized.values())
        if normalized[anchor] < 0.0 or normalized[anchor] > 1.0:
            raise ValueError("probabilities could not be stably canonicalized.")
        probabilities = normalized

    return (
        MappingProxyType(dict(sorted(probabilities.items()))),
        total,
        normalization_applied,
    )


@dataclass(frozen=True, kw_only=True)
class ReferenceDistribution:
    """A declared comparison distribution and the provenance needed to use it."""

    axis: str
    probabilities: Mapping[str, float]
    source: str
    purpose: ReferencePurpose
    population: str
    geography: Optional[str] = None
    period: Optional[str] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    input_probability_sum: float = field(init=False)
    normalization_applied: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )
        object.__setattr__(
            self,
            "population",
            require_nonempty_string(self.population, "population"),
        )
        if self.geography is not None:
            object.__setattr__(
                self,
                "geography",
                require_nonempty_string(self.geography, "geography"),
            )
        if self.period is not None:
            object.__setattr__(
                self, "period", require_nonempty_string(self.period, "period")
            )
        object.__setattr__(
            self,
            "purpose",
            normalize_enum(self.purpose, ReferencePurpose, "purpose"),
        )
        probabilities, input_total, normalized = _validate_probability_mapping(
            self.probabilities
        )
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "input_probability_sum", input_total)
        object.__setattr__(self, "normalization_applied", normalized)
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    @property
    def support(self) -> tuple[str, ...]:
        """Sorted category support."""
        return tuple(self.probabilities)

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "axis": self.axis,
            "probabilities": dict(self.probabilities),
            "source": self.source,
            "purpose": self.purpose.value,
            "population": self.population,
            "geography": self.geography,
            "period": self.period,
            "provenance": thaw_json(self.provenance),
            "input_probability_sum": self.input_probability_sum,
            "normalization_applied": self.normalization_applied,
        }


@dataclass(frozen=True, kw_only=True)
class ComponentOverride:
    """Caller-declared applicability decision that can only suppress a component.

    An override is auditable but never generative: it may assert
    ``not_applicable`` or ``blocked`` and nothing else, so it can never
    manufacture a value for a component that did not run.
    """

    component: str
    status: "DiagnosticStatus"
    reason: str
    declared_reason_code: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Imported inside the method so the declared one-way import direction
        # (evidence -> spec -> base) is preserved at module scope, exactly as
        # ``DiagnosticReport.__post_init__`` imports ``DatasetAuditSpec``.
        from fairLMs.diagnostics.base import DiagnosticStatus

        object.__setattr__(
            self,
            "component",
            require_nonempty_string(self.component, "component"),
        )
        object.__setattr__(
            self, "reason", require_nonempty_string(self.reason, "reason")
        )
        object.__setattr__(
            self,
            "declared_reason_code",
            require_nonempty_string(
                self.declared_reason_code, "declared_reason_code"
            ),
        )
        status = normalize_enum(self.status, DiagnosticStatus, "status")
        if status not in (
            DiagnosticStatus.NOT_APPLICABLE,
            DiagnosticStatus.BLOCKED,
        ):
            raise ValueError(
                "a component override may only declare 'not_applicable' or "
                "'blocked'; a ready or failed status cannot be asserted by an "
                "override."
            )
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "component": self.component,
            "status": self.status.value,
            "reason": self.reason,
            "declared_reason_code": self.declared_reason_code,
            "provenance": thaw_json(self.provenance),
        }


@dataclass(frozen=True, kw_only=True)
class DatasetAuditSpec:
    """Explicit scientific intent for a dataset or result-table audit."""

    target_name: str
    target_kind: TargetKind
    task_family: str
    design_stance: DesignStance
    references: Mapping[str, ReferenceDistribution] = field(default_factory=dict)
    requested_components: Sequence[str] = ("b_rep",)
    protected_axes: Sequence[str] = ()
    leakage_extraction: Optional["LeakageExtractionConfig"] = None
    component_overrides: Mapping[str, ComponentOverride] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_name",
            require_nonempty_string(self.target_name, "target_name"),
        )
        object.__setattr__(
            self,
            "task_family",
            require_nonempty_string(self.task_family, "task_family"),
        )
        object.__setattr__(
            self,
            "target_kind",
            normalize_enum(self.target_kind, TargetKind, "target_kind"),
        )
        object.__setattr__(
            self,
            "design_stance",
            normalize_enum(self.design_stance, DesignStance, "design_stance"),
        )

        if not isinstance(self.references, Mapping):
            raise TypeError(
                "references must be a mapping of axis -> ReferenceDistribution."
            )
        references = {}
        for axis, reference in self.references.items():
            require_nonempty_string(axis, "reference axis")
            if not isinstance(reference, ReferenceDistribution):
                raise TypeError(
                    f"references[{axis!r}] must be a ReferenceDistribution, "
                    f"got {type(reference).__name__}."
                )
            if reference.axis != axis:
                raise ValueError(
                    f"reference mapping key {axis!r} does not match "
                    f"reference.axis {reference.axis!r}."
                )
            references[axis] = reference
        object.__setattr__(
            self, "references", MappingProxyType(dict(sorted(references.items())))
        )

        components = normalize_string_sequence(
            self.requested_components,
            field_name="requested_components",
            allow_empty=False,
        )
        if len(set(components)) != len(components):
            raise ValueError("requested_components must not contain duplicates.")
        object.__setattr__(self, "requested_components", components)

        protected_axes = normalize_string_sequence(
            self.protected_axes,
            field_name="protected_axes",
        )
        if len(set(protected_axes)) != len(protected_axes):
            raise ValueError("protected_axes must not contain duplicates.")
        object.__setattr__(self, "protected_axes", protected_axes)
        if protected_axes:
            for axis in self.references:
                if axis not in protected_axes:
                    raise ValueError(
                        f"references axis {axis!r} is not declared in "
                        f"protected_axes {list(protected_axes)!r}."
                    )

        if self.leakage_extraction is not None:
            # Imported inside the method to avoid a spec <-> leakage cycle.
            from fairLMs.diagnostics.leakage import LeakageExtractionConfig

            if not isinstance(self.leakage_extraction, LeakageExtractionConfig):
                raise TypeError(
                    "leakage_extraction must be a LeakageExtractionConfig or "
                    f"None, got {type(self.leakage_extraction).__name__}."
                )

        if not isinstance(self.component_overrides, Mapping):
            raise TypeError(
                "component_overrides must be a mapping of component name -> "
                "ComponentOverride."
            )
        overrides = {}
        for name, override in self.component_overrides.items():
            require_nonempty_string(name, "component_overrides key")
            if not isinstance(override, ComponentOverride):
                raise TypeError(
                    f"component_overrides[{name!r}] must be a "
                    f"ComponentOverride, got {type(override).__name__}."
                )
            if override.component != name:
                raise ValueError(
                    f"component_overrides key {name!r} does not match "
                    f"override.component {override.component!r}."
                )
            if name not in components:
                raise ValueError(
                    f"component_overrides names {name!r}, which is not in "
                    "requested_components; an override for a component that "
                    "never runs is a silent no-op."
                )
            overrides[name] = override
        object.__setattr__(
            self,
            "component_overrides",
            MappingProxyType(dict(sorted(overrides.items()))),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "target_name": self.target_name,
            "target_kind": self.target_kind.value,
            "task_family": self.task_family,
            "design_stance": self.design_stance.value,
            "references": {
                axis: reference.to_dict() for axis, reference in self.references.items()
            },
            "requested_components": list(self.requested_components),
            "protected_axes": list(self.protected_axes),
            "leakage_extraction": (
                None
                if self.leakage_extraction is None
                else self.leakage_extraction.to_dict()
            ),
            "component_overrides": {
                name: override.to_dict()
                for name, override in self.component_overrides.items()
            },
        }


__all__ = [
    "ComponentOverride",
    "DatasetAuditSpec",
    "DesignStance",
    "ReferenceDistribution",
    "ReferencePurpose",
    "TargetKind",
]
