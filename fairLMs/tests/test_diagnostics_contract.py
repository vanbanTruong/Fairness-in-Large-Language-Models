"""Public result, serialization, registry, and import-boundary contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import fairLMs.diagnostics as diagnostics
from fairLMs.diagnostics import (
    DIAGNOSTIC_REGISTRY,
    DIAGNOSTIC_SCHEMA_VERSION,
    ComponentPlan,
    ComponentResult,
    DatasetAuditSpec,
    DesignStance,
    DiagnosticReport,
    DiagnosticStatus,
    PairedScores,
    ReferenceDistribution,
    ReferencePurpose,
    ReportStatus,
    RepresentativenessBias,
    ScorerCounterfactualSensitivity,
    ScorerMeanGap,
    ScorerRateGap,
    ScorerWasserstein1Gap,
    TargetKind,
    get_diagnostic,
    list_diagnostics,
)
from fairLMs.metrics import METRIC_REGISTRY

PUBLIC_NAMES = {
    "DIAGNOSTIC_SCHEMA_VERSION",
    "DiagnosticStatus",
    # Composes ScoredGroups with an outcome label. Lives here rather than in
    # fairLMs.mitigation because it composes a diagnostics type and both layers
    # use it: it is what the deferred label-membership rate transform needs.
    "LabeledScoredGroups",
    "PairedScores",
    "ReportStatus",
    "ComponentPlan",
    "ComponentResult",
    "DiagnosticReport",
    "DatasetDiagnostic",
    "TargetKind",
    "DesignStance",
    "ReferencePurpose",
    "ReferenceDistribution",
    "DatasetAuditSpec",
    "RepresentationEvidence",
    "RepresentativenessBias",
    "ScoreRateDirection",
    "ScoreRateTransform",
    "ScoredGroups",
    "ScorerCounterfactualSensitivity",
    "ScorerMeanGap",
    "ScorerRateGap",
    "ScorerWasserstein1Gap",
    "audit_representativeness",
    "audit_scores",
    "DIAGNOSTIC_REGISTRY",
    "list_diagnostics",
    "get_diagnostic",
    # Milestone 3/4 evidence containers and audit intent.
    "AssociationCounts",
    "TextEvidence",
    "GroupedTexts",
    "PairedTexts",
    "OptionItems",
    "TemplateGroups",
    "DatasetEvidence",
    "ComponentOverride",
    # b_leak.
    "TokenMatchAttribute",
    "LogBase",
    "LeakageExtractionConfig",
    "LeakageExtractionRecord",
    "LeakageExtractor",
    "SurfaceCooccurrenceExtractor",
    "EXTRACTOR_VERSION",
    "StereotypeLeakage",
    "audit_leakage",
    # The eight-slot construction vector.
    "TokenizationMode",
    "TokenizationRule",
    "IdentityMaskConfig",
    "OptionRoleContrast",
    "FrameMatchMode",
    "FramePredicate",
    "InjectedFramePredicate",
    "SELF_IDENTIFICATION_FRAME",
    "CONSTRUCTION_SLOTS",
    "LIGHTWEIGHT_CONSTRUCTION_SLOTS",
    "BACKEND_CONSTRUCTION_SLOTS",
    "CONSTRUCTION_BACKEND_REQUIREMENTS",
    "construction_vector",
    "MinimalPairResidual",
    "LengthDisparity",
    "OptionLengthBias",
    "FramingDisparity",
    "TemplateImbalance",
    # The backend-dependent construction slots and the protocols they accept.
    "SemanticEquivalence",
    "GrammarConsistency",
    "DependencyDepthDisparity",
    "EmbeddingBackend",
    "GrammarCheckerBackend",
    "DependencyParserBackend",
    "audit_construction",
    # The multi-evidence aggregate entry point.
    "DatasetAudit",
    "audit_dataset",
}


def _spec() -> DatasetAuditSpec:
    return DatasetAuditSpec(
        target_name="unregistered-contract-fixture",
        target_kind="benchmark_dataset",
        task_family="free_text",
        design_stance="population_proxy",
        references={},
        requested_components=("b_rep",),
    )


def _component(component: str, status: DiagnosticStatus) -> ComponentResult:
    if status is DiagnosticStatus.READY:
        return ComponentResult(component=component, status=status, value=0.0)
    return ComponentResult(
        component=component,
        status=status,
        reason_code=f"{status.value}_for_test",
        reason=f"The component is {status.value} in this contract test.",
    )


def _report(*statuses: DiagnosticStatus) -> DiagnosticReport:
    components = {
        f"component_{index}": _component(f"component_{index}", status)
        for index, status in enumerate(statuses)
    }
    return DiagnosticReport(spec=_spec(), components=components)


def test_expected_milestone_one_api_is_public():
    assert PUBLIC_NAMES == set(diagnostics.__all__)
    for name in PUBLIC_NAMES:
        assert getattr(diagnostics, name) is not None


def test_ready_result_accepts_numeric_zero_without_a_reason():
    result = ComponentResult(
        component="synthetic_zero",
        status=DiagnosticStatus.READY,
        value=0,
    )

    assert result.value == 0.0
    assert result.reason_code is None
    assert result.reason is None
    assert result.to_dict()["value"] == 0.0


@pytest.mark.parametrize(
    "status",
    [
        DiagnosticStatus.BLOCKED,
        DiagnosticStatus.NOT_APPLICABLE,
        DiagnosticStatus.FAILED,
    ],
)
def test_nonready_result_requires_no_value_and_an_explicit_reason(status):
    result = ComponentResult(
        component="synthetic_nonready",
        status=status,
        value=None,
        reason_code="deliberate_test_state",
        reason="This state is intentional in the contract test.",
    )

    assert result.value is None
    assert result.reason_code == "deliberate_test_state"
    assert result.reason


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": "ready", "value": None},
        {
            "status": "ready",
            "value": 0.0,
            "reason_code": "should_not_exist",
            "reason": "Ready values do not carry failure reasons.",
        },
        {"status": "blocked", "value": 0.0},
        {"status": "blocked", "value": None},
        {
            "status": "blocked",
            "value": None,
            "reason_code": "missing_human_reason",
        },
        {
            "status": "not_applicable",
            "value": 0.0,
            "reason_code": "wrong_target",
            "reason": "This diagnostic does not apply.",
        },
        {
            "status": "failed",
            "value": 1.0,
            "reason_code": "calculation_failed",
            "reason": "The calculation failed.",
        },
        {"status": "ready", "value": True},
        {"status": "ready", "value": float("nan")},
        {"status": "ready", "value": float("inf")},
        {"status": "ready", "value": float("-inf")},
        {"status": "ready", "value": object()},
    ],
)
def test_component_result_rejects_invalid_status_value_reason_combinations(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ComponentResult(component="invalid", **kwargs)


def test_component_plan_uses_the_same_applicability_reason_invariant():
    ready = ComponentPlan(component="b_rep", status="ready")
    blocked = ComponentPlan(
        component="b_rep",
        status="blocked",
        reason_code="missing_reference",
        reason="No reference distribution was supplied.",
    )

    assert ready.reason is None
    assert blocked.reason_code == "missing_reference"
    with pytest.raises(ValueError):
        ComponentPlan(
            component="b_rep",
            status="failed",
            reason_code="runtime_only",
            reason="Failed is not a planning state.",
        )


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ((DiagnosticStatus.READY,), ReportStatus.SUCCESS),
        (
            (DiagnosticStatus.READY, DiagnosticStatus.BLOCKED),
            ReportStatus.PARTIAL,
        ),
        (
            (DiagnosticStatus.READY, DiagnosticStatus.FAILED),
            ReportStatus.PARTIAL,
        ),
        (
            (DiagnosticStatus.BLOCKED, DiagnosticStatus.NOT_APPLICABLE),
            ReportStatus.BLOCKED,
        ),
        ((DiagnosticStatus.NOT_APPLICABLE,), ReportStatus.NOT_APPLICABLE),
        (
            (DiagnosticStatus.FAILED, DiagnosticStatus.BLOCKED),
            ReportStatus.FAILED,
        ),
    ],
)
def test_report_status_is_derived_from_component_statuses(statuses, expected):
    report = _report(*statuses)
    assert report.status is expected
    assert report.to_dict()["status"] == expected.value


def test_diagnostic_report_is_deliberately_not_float_convertible():
    with pytest.raises(TypeError):
        float(_report(DiagnosticStatus.READY))


def test_report_requires_the_frozen_dataset_audit_spec_contract():
    class MutableDuckSpec:
        def to_dict(self):
            return {"target_name": "mutable"}

    with pytest.raises(TypeError, match="DatasetAuditSpec"):
        DiagnosticReport(
            spec=MutableDuckSpec(),
            components={"b_rep": _component("b_rep", DiagnosticStatus.READY)},
        )


def test_spec_and_reference_serialization_shape_is_frozen_for_schema_one():
    reference = ReferenceDistribution(
        axis="community",
        probabilities={"amber": 0.5, "teal": 0.5},
        source="Synthetic reference",
        purpose=ReferencePurpose.DESIGN_TARGET,
        population="Synthetic population",
        geography="Example region",
        period="2040",
        provenance={"revision": 1},
    )
    spec = DatasetAuditSpec(
        target_name="unregistered-schema-fixture",
        target_kind=TargetKind.BENCHMARK_DATASET,
        task_family="free_text",
        design_stance=DesignStance.STRESS_TEST,
        references={"community": reference},
        requested_components=("b_rep",),
    )

    assert set(spec.to_dict()) == {
        "target_name",
        "target_kind",
        "task_family",
        "design_stance",
        "references",
        "requested_components",
        # Additive at schema 1.5: audit intent that audit_dataset needs.
        "protected_axes",
        "leakage_extraction",
        "component_overrides",
    }
    assert spec.to_dict()["protected_axes"] == []
    assert spec.to_dict()["leakage_extraction"] is None
    assert spec.to_dict()["component_overrides"] == {}
    assert set(spec.to_dict()["references"]["community"]) == {
        "axis",
        "probabilities",
        "source",
        "purpose",
        "population",
        "geography",
        "period",
        "provenance",
        "input_probability_sum",
        "normalization_applied",
    }


@pytest.mark.parametrize("unordered", [{"one", "two"}, iter(("one", "two"))])
def test_serialized_string_arrays_require_ordered_sequences(unordered):
    with pytest.raises(TypeError, match="ordered sequence"):
        ComponentResult(
            component="b_rep",
            status="ready",
            value=0.0,
            assumptions=unordered,
        )

    with pytest.raises(TypeError, match="ordered sequence"):
        DiagnosticReport(
            spec=_spec(),
            components={"b_rep": _component("b_rep", DiagnosticStatus.READY)},
            warnings=unordered,
        )

    with pytest.raises(TypeError, match="ordered sequence"):
        DatasetAuditSpec(
            target_name="unordered-components",
            target_kind="benchmark_dataset",
            task_family="free_text",
            design_stance="population_proxy",
            requested_components=unordered,
        )


def test_report_serialization_is_deterministic_complete_and_strict_json():
    details = {
        "z_key": [1, {"nested": 2.5}],
        "a_key": {"finite": 0.0},
    }
    provenance = {"source": "portable", "revision": 7}
    result = ComponentResult(
        component="b_rep",
        status="ready",
        value=0.0,
        details=details,
        assumptions=("The reference purpose is explicit.",),
        provenance=provenance,
    )
    report = DiagnosticReport(
        spec=_spec(),
        components={"b_rep": result},
        warnings=("A deterministic warning.",),
    )

    # Mutating caller-owned containers cannot mutate the frozen result.
    details["a_key"]["finite"] = 99.0
    provenance["source"] = "mutated"

    encoded = report.to_json(indent=None)
    assert encoded == report.to_json(indent=None)
    assert encoded == json.dumps(
        report.to_dict(), allow_nan=False, indent=None, sort_keys=True
    )

    def reject_nonstandard_number(token):
        raise AssertionError(f"non-standard JSON number: {token}")

    payload = json.loads(encoded, parse_constant=reject_nonstandard_number)
    assert list(payload) == sorted(payload)
    assert payload["schema_version"] == DIAGNOSTIC_SCHEMA_VERSION == "1.7"
    assert set(payload) == {
        "schema_version",
        "status",
        "spec",
        "components",
        "warnings",
        "provenance",
    }
    assert set(payload["components"]["b_rep"]) == {
        "component",
        "status",
        "value",
        "details",
        "assumptions",
        "provenance",
        "reason_code",
        "reason",
    }
    assert payload["components"]["b_rep"]["details"]["a_key"]["finite"] == 0.0
    assert payload["components"]["b_rep"]["provenance"]["source"] == "portable"
    assert payload["provenance"] == {}

    # ``to_dict`` returns fresh mutable containers, not the report internals.
    payload["components"]["b_rep"]["details"]["a_key"]["finite"] = -1
    assert report.to_dict()["components"]["b_rep"]["details"]["a_key"]["finite"] == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["details", "provenance"])
def test_component_result_rejects_nonfinite_nested_json_values(field, bad):
    with pytest.raises(ValueError, match="NaN|infinity"):
        ComponentResult(
            component="b_rep",
            status="ready",
            value=0.0,
            **{field: {"bad": bad}},
        )


@pytest.mark.parametrize("field", ["details", "provenance"])
def test_component_result_rejects_arbitrary_nested_objects(field):
    with pytest.raises(TypeError, match="JSON-compatible"):
        ComponentResult(
            component="b_rep",
            status="ready",
            value=0.0,
            **{field: {"bad": object()}},
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), object()])
def test_report_rejects_nonportable_top_level_provenance(bad):
    with pytest.raises((TypeError, ValueError)):
        DiagnosticReport(
            spec=_spec(),
            components={"b_rep": _component("b_rep", DiagnosticStatus.READY)},
            provenance={"bad": bad},
        )


def test_diagnostic_registry_is_separate_from_model_metric_registry():
    assert DIAGNOSTIC_REGISTRY is not METRIC_REGISTRY
    assert list_diagnostics() == [
        "b_diff_dep",
        "b_diff_len",
        "b_equiv",
        "b_frame",
        "b_gram",
        "b_leak",
        "b_min",
        "b_opt",
        "b_rep",
        "b_temp",
        "score_counterfactual_sensitivity",
        "score_mean_gap",
        "score_rate_gap",
        "score_wasserstein_1_gap",
    ]
    # The three backend-dependent construction slots are registered classes
    # that take an optional backend; without one they block by name.
    for slot in ("b_equiv", "b_gram", "b_diff_dep"):
        assert slot in DIAGNOSTIC_REGISTRY
        assert get_diagnostic(slot).backend is None
    assert "b_rep" in DIAGNOSTIC_REGISTRY
    assert "score_counterfactual_sensitivity" in DIAGNOSTIC_REGISTRY
    assert "score_mean_gap" in DIAGNOSTIC_REGISTRY
    assert "score_rate_gap" in DIAGNOSTIC_REGISTRY
    assert "b_rep" not in METRIC_REGISTRY
    assert "score_counterfactual_sensitivity" not in METRIC_REGISTRY
    assert "score_mean_gap" not in METRIC_REGISTRY
    assert "score_rate_gap" not in METRIC_REGISTRY
    assert "score_wasserstein_1_gap" not in METRIC_REGISTRY
    assert isinstance(get_diagnostic("b_rep"), RepresentativenessBias)
    assert isinstance(
        get_diagnostic("score_counterfactual_sensitivity"),
        ScorerCounterfactualSensitivity,
    )
    assert isinstance(get_diagnostic("score_mean_gap"), ScorerMeanGap)
    assert isinstance(get_diagnostic("score_rate_gap"), ScorerRateGap)
    assert isinstance(get_diagnostic("score_wasserstein_1_gap"), ScorerWasserstein1Gap)


def test_unknown_diagnostic_error_lists_available_alternatives():
    with pytest.raises(KeyError) as excinfo:
        get_diagnostic("not_a_diagnostic")
    assert "not_a_diagnostic" in str(excinfo.value)
    assert "b_rep" in str(excinfo.value)


def test_core_diagnostics_import_does_not_load_optional_nlp_backends(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent.parent
    script = """
import builtins
import sys

forbidden = {"spacy", "sentence_transformers", "language_tool_python"}
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise AssertionError(f"optional backend imported eagerly: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import fairLMs.diagnostics
assert forbidden.isdisjoint(sys.modules)
"""
    env = dict(os.environ)
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root) if not previous else str(repo_root) + os.pathsep + previous
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
