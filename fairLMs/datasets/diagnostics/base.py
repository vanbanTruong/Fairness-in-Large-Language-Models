"""Structured result contracts for fairness diagnostics."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from fairLMs.datasets.diagnostics._utils import (
    freeze_json_mapping,
    normalize_enum,
    normalize_string_sequence,
    require_nonempty_string,
    thaw_json,
)

if TYPE_CHECKING:  # pragma: no cover
    from fairLMs.datasets.diagnostics.spec import DatasetAuditSpec


DIAGNOSTIC_SCHEMA_VERSION = "1.7"


class DiagnosticStatus(str, Enum):
    """Applicability/execution state of one diagnostic component."""

    READY = "ready"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"


class ReportStatus(str, Enum):
    """Derived aggregate state of a diagnostic report."""

    SUCCESS = "success"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"


def _normalize_reason(
    status: DiagnosticStatus,
    reason_code: Optional[str],
    reason: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    if status is DiagnosticStatus.READY:
        if reason_code is not None or reason is not None:
            raise ValueError("a ready component must not carry a failure reason.")
        return None, None
    if reason_code is None or reason is None:
        raise ValueError(
            f"a {status.value} component requires both reason_code and reason."
        )
    return (
        require_nonempty_string(reason_code, "reason_code"),
        require_nonempty_string(reason, "reason"),
    )


@dataclass(frozen=True, kw_only=True)
class ComponentPlan:
    """Pre-computation applicability decision for one component."""

    component: str
    status: DiagnosticStatus
    reason_code: Optional[str] = None
    reason: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component",
            require_nonempty_string(self.component, "component"),
        )
        status = normalize_enum(self.status, DiagnosticStatus, "status")
        if status is DiagnosticStatus.FAILED:
            raise ValueError("failed is an execution status and is invalid in a plan.")
        object.__setattr__(self, "status", status)
        reason_code, reason = _normalize_reason(status, self.reason_code, self.reason)
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "details",
            freeze_json_mapping(self.details, path="details"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "details": thaw_json(self.details),
        }


@dataclass(frozen=True, kw_only=True)
class ComponentResult:
    """Value and evidence for one diagnostic component."""

    component: str
    status: DiagnosticStatus
    value: Optional[float] = None
    details: Mapping[str, Any] = field(default_factory=dict)
    assumptions: Sequence[str] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    reason_code: Optional[str] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component",
            require_nonempty_string(self.component, "component"),
        )
        status = normalize_enum(self.status, DiagnosticStatus, "status")
        object.__setattr__(self, "status", status)

        if status is DiagnosticStatus.READY:
            if isinstance(self.value, bool) or not isinstance(self.value, Real):
                raise ValueError("a ready component requires a finite numeric value.")
            value = float(self.value)
            if not math.isfinite(value):
                raise ValueError("a ready component value must be finite.")
            object.__setattr__(self, "value", value)
        elif self.value is not None:
            raise ValueError(
                f"a {status.value} component must use value=None, not a numeric sentinel."
            )

        reason_code, reason = _normalize_reason(status, self.reason_code, self.reason)
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "details",
            freeze_json_mapping(self.details, path="details"),
        )
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )
        assumptions = normalize_string_sequence(
            self.assumptions,
            field_name="assumptions",
        )
        object.__setattr__(self, "assumptions", assumptions)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, complete JSON-safe component object."""
        return {
            "component": self.component,
            "status": self.status.value,
            "value": self.value,
            "details": thaw_json(self.details),
            "assumptions": list(self.assumptions),
            "provenance": thaw_json(self.provenance),
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


class DatasetDiagnostic(ABC):
    """Base protocol for diagnostics that inspect declared audit evidence."""

    name: str = "diagnostic"

    @abstractmethod
    def plan(self, evidence: Any, spec: "DatasetAuditSpec") -> ComponentPlan:
        """Decide whether this diagnostic is applicable before computation."""

    @abstractmethod
    def compute(self, evidence: Any, spec: "DatasetAuditSpec") -> ComponentResult:
        """Compute the diagnostic or return its non-ready applicability state."""


@dataclass(frozen=True, kw_only=True)
class DiagnosticReport:
    """Structured multi-component report; deliberately not float-convertible."""

    spec: "DatasetAuditSpec"
    components: Mapping[str, ComponentResult]
    warnings: Sequence[str] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    _spec_snapshot: Mapping[str, Any] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        from fairLMs.datasets.diagnostics.spec import DatasetAuditSpec

        if not isinstance(self.spec, DatasetAuditSpec):
            raise TypeError(
                "spec must be a DatasetAuditSpec, " f"got {type(self.spec).__name__}."
            )
        object.__setattr__(
            self,
            "_spec_snapshot",
            freeze_json_mapping(self.spec.to_dict(), path="spec"),
        )
        if not isinstance(self.components, Mapping):
            raise TypeError("components must be a mapping of name -> ComponentResult.")
        if not self.components:
            raise ValueError("components must contain at least one result.")
        components = {}
        for name, result in self.components.items():
            require_nonempty_string(name, "component mapping key")
            if not isinstance(result, ComponentResult):
                raise TypeError(
                    f"components[{name!r}] must be a ComponentResult, "
                    f"got {type(result).__name__}."
                )
            if result.component != name:
                raise ValueError(
                    f"component mapping key {name!r} does not match "
                    f"result.component {result.component!r}."
                )
            components[name] = result
        object.__setattr__(
            self, "components", MappingProxyType(dict(sorted(components.items())))
        )

        warnings = normalize_string_sequence(
            self.warnings,
            field_name="warnings",
        )
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    @property
    def status(self) -> ReportStatus:
        """Derive an aggregate state without inventing an aggregate score."""
        statuses = [result.status for result in self.components.values()]
        if all(status is DiagnosticStatus.READY for status in statuses):
            return ReportStatus.SUCCESS
        if any(status is DiagnosticStatus.READY for status in statuses):
            return ReportStatus.PARTIAL
        if any(status is DiagnosticStatus.FAILED for status in statuses):
            return ReportStatus.FAILED
        if any(status is DiagnosticStatus.BLOCKED for status in statuses):
            return ReportStatus.BLOCKED
        return ReportStatus.NOT_APPLICABLE

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh dictionary conforming to the current report schema."""
        return {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "status": self.status.value,
            "spec": thaw_json(self._spec_snapshot),
            "components": {
                name: result.to_dict() for name, result in self.components.items()
            },
            "warnings": list(self.warnings),
            "provenance": thaw_json(self.provenance),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Serialize deterministically using strict standard JSON numbers."""
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            indent=indent,
            sort_keys=True,
        )


__all__ = [
    "DIAGNOSTIC_SCHEMA_VERSION",
    "ComponentPlan",
    "ComponentResult",
    "DatasetDiagnostic",
    "DiagnosticReport",
    "DiagnosticStatus",
    "ReportStatus",
]
