"""Multi-evidence dataset audits over one declared protected axis.

``audit_dataset`` is the aggregate entry point for dataset-composition
diagnostics.  It plans and runs exactly the components the specification
*requested*, over exactly the evidence views the caller *supplied*, and returns
one :class:`~fairLMs.datasets.diagnostics.base.DiagnosticReport` containing every
requested component.

Three properties are structural rather than conventional:

* **Nothing is selected from a dataset name.**  ``spec.target_name`` and
  ``spec.task_family`` reach provenance and nothing else.
* **Registry membership never causes anything to run.**
  :data:`~fairLMs.datasets.diagnostics.registry.DIAGNOSTIC_REGISTRY` is never consulted
  here; ``spec.requested_components`` is the only candidate set.
* **A requested component with no suitable evidence is reported, never
  skipped.**  It becomes ``blocked`` or ``not_applicable`` with a precise
  reason code and ``value is None``; absent evidence is never a measured zero.

``audit_scores`` remains the separate entry point for row-level scorer results
(:class:`ScoredGroups` / :class:`PairedScores`), and this module never folds it
in.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Mapping, Optional, Sequence

from ._utils import require_nonempty_string
from .base import (
    ComponentPlan,
    ComponentResult,
    DatasetDiagnostic,
    DiagnosticReport,
    DiagnosticStatus,
)
from .construction import (
    DependencyDepthDisparity,
    GrammarConsistency,
    SemanticEquivalence,
    BACKEND_BLOCKED_WARNING,
    BACKEND_CONSTRUCTION_SLOTS,
    CONSTRUCTION_SLOTS,
    CONSTRUCTION_VECTOR_WARNING,
    INJECTED_PREDICATE_WARNING,
    TOKENIZATION_DIVERGENCE_WARNING,
    FramingDisparity,
    InjectedFramePredicate,
    LengthDisparity,
    MinimalPairResidual,
    OptionLengthBias,
    TemplateImbalance,
    _SLOT_VIEWS,
    _component_override,
    _non_ready_result,
    _protected_axes,
    _shared_plan_prefix,
    _view_not_supplied_result,
)
from .evidence import DatasetEvidence
from .leakage import LEAKAGE_INTENT_WARNING, ZERO_HIT_WARNING, StereotypeLeakage
from .representativeness import RepresentativenessBias
from .scoring import (
    ScorerCounterfactualSensitivity,
    ScorerMeanGap,
    ScorerRateGap,
    ScorerWasserstein1Gap,
)
from .spec import DatasetAuditSpec, DesignStance, TargetKind

from ._messages import STRESS_TEST_WARNING

# --------------------------------------------------------------------------
# Component vocabulary
# --------------------------------------------------------------------------

#: Dataset components that are not part of the construction vector.
_STANDALONE_DATASET_COMPONENTS: Final[tuple[str, ...]] = ("b_rep", "b_leak")

#: Row-level scorer components; they belong to ``audit_scores``, not here.
SCORE_COMPONENTS: Final[tuple[str, ...]] = tuple(
    sorted(
        diagnostic.name
        for diagnostic in (
            ScorerCounterfactualSensitivity,
            ScorerMeanGap,
            ScorerRateGap,
            ScorerWasserstein1Gap,
        )
    )
)

#: Every component name ``audit_dataset`` can plan, whatever its outcome.
KNOWN_DATASET_COMPONENTS: Final[tuple[str, ...]] = tuple(
    sorted(set(_STANDALONE_DATASET_COMPONENTS) | set(CONSTRUCTION_SLOTS))
)

_KNOWN_COMPONENTS: Final[frozenset[str]] = frozenset(
    set(KNOWN_DATASET_COMPONENTS) | set(SCORE_COMPONENTS)
)

#: The required evidence view for every non-leakage dataset component.
_COMPONENT_VIEWS: Final[Mapping[str, str]] = MappingProxyType(
    {"b_rep": "representation", **dict(_SLOT_VIEWS)}
)

#: ``b_leak`` accepts a count matrix or raw text, in this fixed precedence.
_LEAKAGE_VIEWS: Final[tuple[str, ...]] = ("association_counts", "texts")

_IMPLEMENTED_CLASSES: Final[Mapping[str, type]] = MappingProxyType(
    {
        "b_rep": RepresentativenessBias,
        "b_leak": StereotypeLeakage,
        "b_min": MinimalPairResidual,
        "b_diff_len": LengthDisparity,
        "b_frame": FramingDisparity,
        "b_opt": OptionLengthBias,
        "b_temp": TemplateImbalance,
        "b_equiv": SemanticEquivalence,
        "b_gram": GrammarConsistency,
        "b_diff_dep": DependencyDepthDisparity,
    }
)

_GENERIC_TARGET_KIND_REASON: Final[str] = (
    "{component} applies to benchmark-dataset construction, not target kind "
    "{kind!r}."
)
_TARGET_KIND_REASONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "b_rep": (
            "b_rep applies to benchmark-dataset composition, not target kind "
            "{kind!r}."
        ),
        "b_leak": (
            "b_leak applies to benchmark-dataset composition, not target kind "
            "{kind!r}. Association measured over generated output is output "
            "association, not dataset leakage."
        ),
    }
)


def _enum_value(value: Any) -> Any:
    """Return a stable scalar for enum-like public values."""
    return getattr(value, "value", value)


# --------------------------------------------------------------------------
# Planner-owned decisions
# --------------------------------------------------------------------------


def _plan_from_result(result: ComponentResult) -> ComponentPlan:
    """Mirror a planner-owned result as the plan that produced it.

    A planner-owned decision is never ``ready`` and never ``failed``, so the
    plan and the result always agree by construction rather than by a second,
    drift-prone code path.
    """
    return ComponentPlan(
        component=result.component,
        status=result.status,
        reason_code=result.reason_code,
        reason=result.reason,
        details=dict(result.details),
    )


def _override_result(component: str, *, override: Any) -> ComponentResult:
    """Record a caller-declared suppression; it can never produce a value."""
    return ComponentResult(
        component=component,
        status=override.status,
        details={
            "applicability_override": True,
            "declared_reason_code": override.declared_reason_code,
            "override_source": "spec",
        },
        provenance={"override": override.to_dict()},
        reason_code="applicability_override",
        reason=override.reason,
    )


def _target_kind_result(
    component: str, *, spec: DatasetAuditSpec, axis: str
) -> ComponentResult:
    template = _TARGET_KIND_REASONS.get(component, _GENERIC_TARGET_KIND_REASON)
    return ComponentResult(
        component=component,
        status=DiagnosticStatus.NOT_APPLICABLE,
        details={"axis": axis},
        reason_code="target_kind_not_supported",
        reason=template.format(
            component=component, kind=_enum_value(spec.target_kind)
        ),
    )


def _axis_not_declared_result(
    component: str, *, declared_axes: Sequence[str], axis: str
) -> ComponentResult:
    return ComponentResult(
        component=component,
        status=DiagnosticStatus.BLOCKED,
        details={"axis": axis, "protected_axes": list(declared_axes)},
        reason_code="axis_not_declared",
        reason=(
            f"Axis {axis!r} is not declared in protected_axes "
            f"{list(declared_axes)!r}."
        ),
    )


def _score_component_result(component: str, *, axis: str) -> ComponentResult:
    return ComponentResult(
        component=component,
        status=DiagnosticStatus.NOT_APPLICABLE,
        details={"axis": axis, "entry_point": "audit_scores"},
        reason_code="component_requires_score_evidence",
        reason=(
            f"{component} audits row-level scorer results; run it through "
            "audit_scores(...) with ScoredGroups or PairedScores evidence."
        ),
    )


def _missing_association_evidence_result(*, axis: str) -> ComponentResult:
    return ComponentResult(
        component="b_leak",
        status=DiagnosticStatus.BLOCKED,
        details={"axis": axis, "required_views": list(_LEAKAGE_VIEWS)},
        reason_code="missing_association_evidence",
        reason=(
            "b_leak requires either a complete association count matrix or "
            f"text evidence with an extraction configuration for axis {axis!r}."
        ),
    )


@dataclass(frozen=True, kw_only=True)
class _Resolution:
    """One component's resolved plan, and how its result will be produced."""

    component: str
    plan: ComponentPlan
    result: Optional[ComponentResult] = None
    diagnostic: Optional[DatasetDiagnostic] = None
    view: Optional[Any] = None
    view_name: Optional[str] = None


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def _component_order(spec: DatasetAuditSpec) -> tuple[str, ...]:
    """Resolve the execution order from the request, never from the registry."""
    requested = tuple(spec.requested_components)
    for name in requested:
        if name in _KNOWN_COMPONENTS:
            continue
        message = (
            f"Unknown requested component {name!r}. Available: "
            + ", ".join(sorted(_KNOWN_COMPONENTS))
        )
        if name == "b_constr":
            message += (
                ". b_constr is a component vector, not a component; request "
                "individual slots from CONSTRUCTION_SLOTS."
            )
        raise ValueError(message)

    ordered = [name for name in requested if name not in CONSTRUCTION_SLOTS]
    if any(name in CONSTRUCTION_SLOTS for name in requested):
        # Rule 2: whenever any slot is in play the vector is complete, so the
        # unrequested slots stay visible as not-applicable rather than absent.
        ordered.extend(CONSTRUCTION_SLOTS)
    return tuple(ordered)


def _normalize_diagnostics(
    diagnostics: Sequence[DatasetDiagnostic],
    *,
    spec: DatasetAuditSpec,
) -> dict[str, DatasetDiagnostic]:
    """Validate an explicit diagnostic selection against the request."""
    if isinstance(diagnostics, (str, bytes)) or not isinstance(
        diagnostics, Sequence
    ):
        raise TypeError("diagnostics must be an ordered sequence of diagnostics.")
    selected: dict[str, DatasetDiagnostic] = {}
    names: list[str] = []
    for index, diagnostic in enumerate(diagnostics):
        if not isinstance(diagnostic, DatasetDiagnostic):
            raise TypeError(
                f"diagnostics[{index}] must be a DatasetDiagnostic, "
                f"got {type(diagnostic).__name__}."
            )
        name = require_nonempty_string(diagnostic.name, f"diagnostics[{index}].name")
        if name not in spec.requested_components:
            raise ValueError(
                f"diagnostics[{index}].name {name!r} is not in "
                "spec.requested_components."
            )
        if name in SCORE_COMPONENTS:
            raise ValueError(
                f"diagnostics[{index}].name {name!r} audits row-level scorer "
                "results; run it through audit_scores(...), not "
                "audit_dataset(...)."
            )
        if name not in _IMPLEMENTED_CLASSES:
            raise ValueError(
                f"diagnostics[{index}].name {name!r} is not a dataset "
                f"component; expected one of {list(KNOWN_DATASET_COMPONENTS)}."
            )
        names.append(name)
        selected[name] = diagnostic
    if len(set(names)) != len(names):
        raise ValueError("diagnostics must not contain duplicate component names.")
    return selected


def _resolve_instances(
    component_order: Sequence[str],
    supplied: Mapping[str, DatasetDiagnostic],
    *,
    spec: DatasetAuditSpec,
) -> dict[str, DatasetDiagnostic]:
    """Default-construct every implemented component the caller did not supply.

    A default-constructed component that needs configuration then blocks
    honestly for its own missing configuration, which is the correct outcome
    rather than a failure.
    """
    instances: dict[str, DatasetDiagnostic] = {}
    for component in component_order:
        if component not in _IMPLEMENTED_CLASSES:
            continue
        if component in supplied:
            instances[component] = supplied[component]
            continue
        instances[component] = _IMPLEMENTED_CLASSES[component]()

    leakage = instances.get("b_leak")
    spec_config = getattr(spec, "leakage_extraction", None)
    if (
        leakage is not None
        and spec_config is not None
        and getattr(leakage, "extractor", None) is not None
        and leakage.extractor.config_digest != spec_config.config_digest
    ):
        raise ValueError(
            "the supplied diagnostic's extraction configuration does not "
            "match spec.leakage_extraction; supply one or make them identical."
        )
    return instances


# --------------------------------------------------------------------------
# The resolved audit
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class DatasetAudit:
    """A frozen, inspectable, resolved audit over one axis of one dataset."""

    evidence: DatasetEvidence
    spec: DatasetAuditSpec
    axis: str
    component_order: tuple[str, ...]
    diagnostics: Mapping[str, DatasetDiagnostic]

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, DatasetEvidence):
            raise TypeError(
                "evidence must be a DatasetEvidence, "
                f"got {type(self.evidence).__name__}."
            )
        if not isinstance(self.spec, DatasetAuditSpec):
            raise TypeError(
                f"spec must be a DatasetAuditSpec, got {type(self.spec).__name__}."
            )
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))

        order = tuple(self.component_order)
        if not order:
            raise ValueError("component_order must contain at least one component.")
        for component in order:
            require_nonempty_string(component, "component_order entry")
        if len(set(order)) != len(order):
            raise ValueError("component_order must not contain duplicates.")
        object.__setattr__(self, "component_order", order)

        if not isinstance(self.diagnostics, Mapping):
            raise TypeError(
                "diagnostics must be a mapping of component name -> "
                "DatasetDiagnostic."
            )
        instances: dict[str, DatasetDiagnostic] = {}
        for name, diagnostic in self.diagnostics.items():
            require_nonempty_string(name, "diagnostics key")
            if not isinstance(diagnostic, DatasetDiagnostic):
                raise TypeError(
                    f"diagnostics[{name!r}] must be a DatasetDiagnostic, "
                    f"got {type(diagnostic).__name__}."
                )
            if diagnostic.name != name:
                raise ValueError(
                    f"diagnostics key {name!r} does not match diagnostic.name "
                    f"{diagnostic.name!r}."
                )
            if name not in order:
                raise ValueError(
                    f"diagnostics names {name!r}, which is not in "
                    "component_order."
                )
            instances[name] = diagnostic
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(sorted(instances.items())))
        )

    # -- resolution ------------------------------------------------------

    def _view_for(self, component: str) -> tuple[Optional[Any], Optional[str]]:
        """Return the evidence view this component will read, if any."""
        if component == "b_leak":
            for view_name in _LEAKAGE_VIEWS:
                view = getattr(self.evidence, view_name).get(self.axis)
                if view is not None:
                    return view, view_name
            return None, None
        view_name = _COMPONENT_VIEWS.get(component)
        if view_name is None:
            return None, None
        return getattr(self.evidence, view_name).get(self.axis), view_name

    def _resolve_component(self, component: str) -> _Resolution:
        spec = self.spec
        axis = self.axis
        requested = spec.requested_components

        if component in SCORE_COMPONENTS:
            result = _score_component_result(component, axis=axis)
            return _Resolution(
                component=component, plan=_plan_from_result(result), result=result
            )

        if component not in requested:
            # Only an unrequested construction slot reaches here; it stays in
            # the eight-slot vector without running.
            plan = _shared_plan_prefix(
                component=component, spec=spec, axis=axis, check_axis=False
            )
            if plan is None:  # pragma: no cover - step 1 always matches here
                raise RuntimeError(
                    f"unrequested component {component!r} did not resolve to a "
                    "component_not_requested plan."
                )
            result = _non_ready_result(plan, spec=spec, provenance={})
            return _Resolution(
                component=component, plan=_plan_from_result(result), result=result
            )

        diagnostic = self.diagnostics.get(component)
        if diagnostic is None:
            raise KeyError(
                f"no diagnostic instance was resolved for component {component!r}."
            )

        view, view_name = self._view_for(component)

        # ``b_rep`` predates ``component_overrides`` and ``protected_axes``, so
        # the planner applies those two steps for it. Every other component
        # applies the full precedence inside its own ``plan()``, and delegating
        # keeps ``audit_dataset`` byte-identical to the dedicated entry point.
        planner_owned_prefix = component == "b_rep" or view is None

        if planner_owned_prefix:
            override = _component_override(spec, component)
            if override is not None:
                result = _override_result(component, override=override)
                return _Resolution(
                    component=component,
                    plan=_plan_from_result(result),
                    result=result,
                )
            if spec.target_kind is not TargetKind.BENCHMARK_DATASET:
                result = _target_kind_result(component, spec=spec, axis=axis)
                return _Resolution(
                    component=component,
                    plan=_plan_from_result(result),
                    result=result,
                )

        if view is None:
            if component == "b_leak":
                result = _missing_association_evidence_result(axis=axis)
            else:
                result = _view_not_supplied_result(
                    component, axis=axis, view=_COMPONENT_VIEWS[component]
                )
            return _Resolution(
                component=component, plan=_plan_from_result(result), result=result
            )

        if component == "b_rep":
            declared_axes = _protected_axes(spec)
            if declared_axes and axis not in declared_axes:
                result = _axis_not_declared_result(
                    component, declared_axes=declared_axes, axis=axis
                )
                return _Resolution(
                    component=component,
                    plan=_plan_from_result(result),
                    result=result,
                )

        return _Resolution(
            component=component,
            plan=diagnostic.plan(view, spec),
            diagnostic=diagnostic,
            view=view,
            view_name=view_name,
        )

    def _resolutions(self) -> tuple[_Resolution, ...]:
        return tuple(
            self._resolve_component(component) for component in self.component_order
        )

    def _views_used(self) -> dict[str, str]:
        used: dict[str, str] = {}
        for resolution in self._resolutions():
            if resolution.view_name is not None:
                used[resolution.component] = resolution.view_name
        return used

    # -- public surface --------------------------------------------------

    def plan(self) -> Mapping[str, ComponentPlan]:
        """Return every component's plan in execution order; no arithmetic runs."""
        plans = {
            resolution.component: resolution.plan
            for resolution in self._resolutions()
        }
        return MappingProxyType(plans)

    def run(self) -> DiagnosticReport:
        """Execute the resolved audit and return one multi-component report."""
        resolutions = self._resolutions()
        results: dict[str, ComponentResult] = {}
        views_used: dict[str, str] = {}

        for resolution in resolutions:
            if resolution.result is not None:
                results[resolution.component] = resolution.result
                continue
            diagnostic = resolution.diagnostic
            view = resolution.view
            if diagnostic is None or view is None:  # pragma: no cover
                raise RuntimeError(
                    f"component {resolution.component!r} resolved to neither a "
                    "planner-owned result nor a runnable diagnostic."
                )
            if resolution.view_name is not None:
                views_used[resolution.component] = resolution.view_name
            try:
                results[resolution.component] = diagnostic.compute(view, self.spec)
            except Exception as exc:
                if resolution.plan.status is not DiagnosticStatus.READY:
                    raise
                results[resolution.component] = ComponentResult(
                    component=resolution.component,
                    status=DiagnosticStatus.FAILED,
                    details={"exception_type": type(exc).__name__},
                    provenance={
                        "evidence": {"axis": view.axis, "source": view.source}
                    },
                    reason_code="computation_failed",
                    reason=(
                        "Applicability was established, but the diagnostic "
                        f"computation failed with {type(exc).__name__}."
                    ),
                )

        return DiagnosticReport(
            spec=self.spec,
            components=results,
            warnings=self._warnings(results),
            provenance=self._provenance(views_used),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the resolved selection, for an audit trail."""
        return {
            "target_name": self.evidence.target_name,
            "axis": self.axis,
            "component_order": list(self.component_order),
            "diagnostics": {
                name: type(instance).__qualname__
                for name, instance in self.diagnostics.items()
            },
            "evidence_views": list(self.evidence.available_views),
            "views_used": self._views_used(),
            "spec": self.spec.to_dict(),
        }

    # -- report assembly -------------------------------------------------

    def _warnings(self, results: Mapping[str, ComponentResult]) -> tuple[str, ...]:
        warnings: list[str] = []

        leakage = results.get("b_leak")
        if leakage is not None and leakage.status is DiagnosticStatus.READY:
            warnings.append(LEAKAGE_INTENT_WARNING)
            if leakage.details.get("zero_lexical_hits"):
                warnings.append(ZERO_HIT_WARNING)

        if any(slot in results for slot in CONSTRUCTION_SLOTS):
            warnings.append(CONSTRUCTION_VECTOR_WARNING)
            blocked = [
                slot
                for slot in BACKEND_CONSTRUCTION_SLOTS
                if slot in results
                and results[slot].status is DiagnosticStatus.BLOCKED
                and (results[slot].reason_code or "").endswith("_backend_unavailable")
            ]
            if blocked:
                ordered = [slot for slot in CONSTRUCTION_SLOTS if slot in blocked]
                warnings.append(
                    BACKEND_BLOCKED_WARNING.format(slots=", ".join(ordered))
                )
            length = results.get("b_diff_len")
            if length is not None and length.status is DiagnosticStatus.READY:
                warnings.append(TOKENIZATION_DIVERGENCE_WARNING)
            frame = results.get("b_frame")
            if (
                frame is not None
                and frame.status is DiagnosticStatus.READY
                and isinstance(
                    getattr(self.diagnostics.get("b_frame"), "predicate", None),
                    InjectedFramePredicate,
                )
            ):
                warnings.append(INJECTED_PREDICATE_WARNING)

        if self.spec.design_stance is DesignStance.STRESS_TEST and any(
            result.status is DiagnosticStatus.READY for result in results.values()
        ):
            warnings.append(STRESS_TEST_WARNING)

        deduplicated: list[str] = []
        for warning in warnings:
            if warning not in deduplicated:
                deduplicated.append(warning)
        return tuple(deduplicated)

    def _provenance(self, views_used: Mapping[str, str]) -> dict[str, Any]:
        available = self.evidence.views_for(self.axis)
        unused = sorted(set(available) - set(views_used.values()))
        provenance: dict[str, Any] = {
            "entry_point": "audit_dataset",
            "axis": self.axis,
            "target_name": self.evidence.target_name,
            "component_order": list(self.component_order),
            "diagnostics": {
                name: type(instance).__qualname__
                for name, instance in self.diagnostics.items()
            },
            "evidence_views": list(self.evidence.available_views),
            "views_used": dict(sorted(views_used.items())),
        }
        if unused:
            provenance["unused_evidence_views"] = unused
        if any(slot in self.component_order for slot in CONSTRUCTION_SLOTS):
            provenance["slot_order"] = list(CONSTRUCTION_SLOTS)
            provenance["backend_slots"] = list(BACKEND_CONSTRUCTION_SLOTS)
        return provenance


def audit_dataset(
    evidence: DatasetEvidence,
    spec: DatasetAuditSpec,
    *,
    axis: str,
    diagnostics: Sequence[DatasetDiagnostic] = (),
) -> DatasetAudit:
    """Resolve a dataset audit for one axis without computing anything yet.

    The returned :class:`DatasetAudit` exposes ``plan()`` and ``run()``, so an
    applicability decision can be inspected before any arithmetic happens.

    The audit is single-axis by design: every component keys off one
    ``evidence.axis``, and a rollup across axes is an explicit caller
    operation rather than hidden behaviour.
    """
    if not isinstance(evidence, DatasetEvidence):
        raise TypeError(
            f"evidence must be a DatasetEvidence, got {type(evidence).__name__}."
        )
    if not isinstance(spec, DatasetAuditSpec):
        raise TypeError(f"spec must be a DatasetAuditSpec, got {type(spec).__name__}.")
    axis = require_nonempty_string(axis, "axis")

    component_order = _component_order(spec)
    supplied = _normalize_diagnostics(diagnostics, spec=spec)
    instances = _resolve_instances(component_order, supplied, spec=spec)

    return DatasetAudit(
        evidence=evidence,
        spec=spec,
        axis=axis,
        component_order=component_order,
        diagnostics=instances,
    )


__all__ = ["DatasetAudit", "audit_dataset"]
