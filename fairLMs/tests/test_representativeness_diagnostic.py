"""Scientific and applicability tests for the generic ``b_rep`` slice."""

from __future__ import annotations

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from fairLMs.diagnostics import (
    ComponentPlan,
    DatasetAuditSpec,
    DatasetDiagnostic,
    DesignStance,
    DiagnosticStatus,
    ReferenceDistribution,
    ReferencePurpose,
    ReportStatus,
    RepresentationEvidence,
    RepresentativenessBias,
    TargetKind,
    audit_representativeness,
)

AXIS = "chromatic-family"


def _evidence(counts=None, *, axis=AXIS, provenance=None):
    return RepresentationEvidence(
        axis=axis,
        counts=counts or {"amber": 3, "teal": 1},
        source="Unregistered synthetic dataset",
        provenance=provenance or {"dataset_id": "novel-table-91"},
    )


def _reference(probabilities=None, *, axis=AXIS, purpose="population"):
    return ReferenceDistribution(
        axis=axis,
        probabilities=probabilities or {"amber": 0.5, "teal": 0.5},
        source="Explicit synthetic reference v3",
        purpose=purpose,
        population="Synthetic comparison population",
        geography="Example region",
        period="2040",
        provenance={"reference_id": "reference-3"},
    )


def _spec(
    reference=None,
    *,
    target_kind="benchmark_dataset",
    design_stance="population_proxy",
    target_name="unregistered-synthetic-benchmark",
    requested_components=("b_rep",),
):
    references = {} if reference is None else {reference.axis: reference}
    return DatasetAuditSpec(
        target_name=target_name,
        target_kind=target_kind,
        task_family="free_text",
        design_stance=design_stance,
        references=references,
        requested_components=requested_components,
    )


def _result(report):
    assert set(report.components) == {"b_rep"}
    return report.components["b_rep"]


def _audit(evidence=None, reference=None, *, smoothing_mass=0.01, **spec_kwargs):
    evidence = evidence or _evidence()
    reference = reference or _reference()
    diagnostic = RepresentativenessBias(smoothing_mass=smoothing_mass)
    return audit_representativeness(
        evidence,
        _spec(reference, **spec_kwargs),
        diagnostic=diagnostic,
    )


def test_identical_observed_and_reference_distributions_produce_zero():
    report = _audit(
        _evidence({"amber": 3, "teal": 1}),
        _reference({"amber": 0.75, "teal": 0.25}),
    )
    result = _result(report)

    assert result.status is DiagnosticStatus.READY
    assert result.value == 0.0
    assert report.status is ReportStatus.SUCCESS


def test_known_smoothed_kl_value_and_details():
    report = _audit()
    result = _result(report)

    assert result.status is DiagnosticStatus.READY
    assert result.value == pytest.approx(0.12810900111910786, rel=0, abs=1e-15)

    details = result.details
    assert details["axis"] == AXIS
    assert details["support"] == ("amber", "teal")
    assert details["sample_count"] == 4
    assert details["observed_counts"] == {"amber": 3, "teal": 1}
    assert details["observed_distribution"] == {"amber": 0.75, "teal": 0.25}
    assert details["reference_distribution"] == {"amber": 0.5, "teal": 0.5}
    assert details["smoothed_observed_distribution"] == pytest.approx(
        {
            "amber": (0.75 + 0.005) / 1.01,
            "teal": (0.25 + 0.005) / 1.01,
        }
    )
    assert details["smoothed_reference_distribution"] == pytest.approx(
        {"amber": 0.5, "teal": 0.5}
    )
    assert math.fsum(details["contributions"].values()) == pytest.approx(
        result.value, rel=0, abs=1e-15
    )
    assert details["smoothing_mass"] == 0.01
    assert details["smoothing_per_category"] == 0.005
    assert details["direction"] == "observed||reference"
    assert details["unit"] == "nats"

    assert any("descriptive" in item.lower() for item in result.assumptions)
    assert result.provenance["evidence"]["provenance"]["dataset_id"] == (
        "novel-table-91"
    )
    assert result.provenance["reference"]["source"] == (
        "Explicit synthetic reference v3"
    )
    assert result.provenance["reference"]["purpose"] == "population"


def test_class_and_functional_entry_points_agree():
    evidence = _evidence()
    reference = _reference()
    spec = _spec(reference)
    diagnostic = RepresentativenessBias(smoothing_mass=0.01)

    direct = diagnostic.compute(evidence, spec)
    reported = _result(
        audit_representativeness(evidence, spec, diagnostic=diagnostic)
    )

    assert reported.to_dict() == direct.to_dict()


def test_count_scaling_does_not_change_the_divergence():
    small = _result(_audit(_evidence({"amber": 3, "teal": 1})))
    large = _result(_audit(_evidence({"amber": 300, "teal": 100})))

    assert small.value == pytest.approx(large.value, rel=0, abs=1e-15)
    assert small.details["observed_distribution"] == large.details[
        "observed_distribution"
    ]
    assert small.details["sample_count"] == 4
    assert large.details["sample_count"] == 400


def test_mapping_insertion_order_does_not_change_value_or_numeric_details():
    forward = _result(
        _audit(
            _evidence({"amber": 3, "teal": 1}),
            _reference({"amber": 0.5, "teal": 0.5}),
        )
    )
    reverse = _result(
        _audit(
            _evidence({"teal": 1, "amber": 3}),
            _reference({"teal": 0.5, "amber": 0.5}),
        )
    )

    assert reverse.value == forward.value
    assert reverse.details == forward.details


def test_near_unit_reference_is_canonicalized_and_never_produces_negative_kl():
    reference = _reference(
        {"amber": 0.50000000025, "teal": 0.50000000025}
    )
    report = _audit(_evidence({"amber": 1, "teal": 1}), reference)
    result = _result(report)

    assert reference.input_probability_sum == pytest.approx(1.0000000005)
    assert reference.normalization_applied is True
    assert math.fsum(reference.probabilities.values()) == 1.0
    assert result.status is DiagnosticStatus.READY
    assert result.value == 0.0
    assert result.value >= 0.0
    assert result.provenance["reference"]["normalization_applied"] is True


def test_extreme_but_representable_smoothing_uses_a_finite_log_fallback():
    report = _audit(
        _evidence({"amber": 1, "teal": 0}),
        _reference({"amber": 0.0, "teal": 1.0}),
        smoothing_mass=sys.float_info.min,
    )
    result = _result(report)

    assert result.status is DiagnosticStatus.READY
    assert result.value is not None
    assert math.isfinite(result.value)
    assert result.value > 0.0


def test_wrong_target_kind_is_not_applicable_not_a_zero_score():
    evidence = _evidence()
    reference = _reference()
    report = _audit(
        evidence,
        reference,
        target_kind=TargetKind.GENERATED_OUTPUT,
    )
    result = _result(report)

    assert result.status is DiagnosticStatus.NOT_APPLICABLE
    assert result.value is None
    assert result.reason_code == "target_kind_not_supported"
    assert result.reason
    assert report.status is ReportStatus.NOT_APPLICABLE


def test_missing_reference_is_blocked_not_a_zero_score():
    evidence = _evidence()
    spec = _spec(None)
    report = audit_representativeness(evidence, spec)
    result = _result(report)

    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "missing_reference"
    assert AXIS in result.reason
    assert report.status is ReportStatus.BLOCKED


def test_reference_support_mismatch_is_blocked_not_silently_aligned():
    evidence = _evidence({"amber": 3, "teal": 1})
    reference = _reference({"amber": 0.5, "green": 0.5})
    result = _result(_audit(evidence, reference))

    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "reference_support_mismatch"
    assert "exactly the same" in result.reason


def test_unrequested_component_is_not_applicable_without_a_numeric_sentinel():
    evidence = _evidence()
    reference = _reference()
    spec = _spec(reference, requested_components=("b_leak",))
    result = _result(audit_representativeness(evidence, spec))

    assert result.status is DiagnosticStatus.NOT_APPLICABLE
    assert result.value is None
    assert result.reason_code == "component_not_requested"


def test_stress_test_stance_warns_but_does_not_change_the_value():
    evidence = _evidence()
    reference = _reference(purpose=ReferencePurpose.DESIGN_TARGET)

    population_report = _audit(
        evidence,
        reference,
        design_stance=DesignStance.POPULATION_PROXY,
    )
    stress_report = _audit(
        evidence,
        reference,
        design_stance=DesignStance.STRESS_TEST,
    )

    assert _result(stress_report).status is DiagnosticStatus.READY
    assert _result(stress_report).value == _result(population_report).value
    assert population_report.warnings == ()
    assert stress_report.warnings
    assert any("stress-test" in warning.lower() for warning in stress_report.warnings)
    assert any(
        "intentional" in warning.lower() for warning in stress_report.warnings
    )


def test_stress_test_warning_is_not_emitted_when_no_divergence_was_computed():
    report = audit_representativeness(
        _evidence(),
        _spec(None, design_stance=DesignStance.STRESS_TEST),
    )

    assert _result(report).status is DiagnosticStatus.BLOCKED
    assert report.warnings == ()


@pytest.mark.parametrize(
    "smoothing_mass",
    [
        0.0,
        -0.01,
        True,
        5e-324,
        1e-310,
        Fraction(1, 10**4000),
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_smoothing_mass_must_be_finite_positive_and_numeric(smoothing_mass):
    with pytest.raises((TypeError, ValueError)):
        RepresentativenessBias(smoothing_mass=smoothing_mass)


def test_report_runner_translates_a_ready_computation_exception_to_failed():
    class FailingDiagnostic(DatasetDiagnostic):
        name = "b_rep"

        def plan(self, evidence, spec):
            return ComponentPlan(component=self.name, status="ready")

        def compute(self, evidence, spec):
            raise OverflowError("synthetic numeric failure")

    report = audit_representativeness(
        _evidence(),
        _spec(_reference()),
        diagnostic=FailingDiagnostic(),
    )
    result = _result(report)

    assert result.status is DiagnosticStatus.FAILED
    assert result.value is None
    assert result.reason_code == "computation_failed"
    assert result.details["exception_type"] == "OverflowError"
    assert report.status is ReportStatus.FAILED


def test_trustgpt_frozen_paper_regression_fixture_is_portable_and_matches_value():
    fixture_path = (
        Path(__file__).parent
        / "data"
        / "golden"
        / "diagnostics"
        / "b_rep"
        / "trustgpt_gender_v1.json"
    )
    fixture_text = fixture_path.read_text(encoding="utf-8")
    fixture = json.loads(fixture_text)

    assert "/Users/" not in fixture_text
    assert "Downloads" not in fixture_text
    assert fixture["fixture_id"] == "trustgpt_gender_v1"
    assert fixture["evidence"]["counts"] == {"Male": 355922, "Female": 355922}
    assert fixture["reference"]["probabilities"] == {
        "Male": 0.491,
        "Female": 0.509,
    }
    assert fixture["parameters"]["smoothing_mass"] == 0.01
    assert fixture["expected"]["value"] == 0.00015883318531346828
    assert fixture["oracle"]["script"] == (
        "scripts/plot_representativeness_summary.py"
    )
    assert not Path(fixture["oracle"]["script"]).is_absolute()
    assert fixture["oracle"]["script_sha256"] == (
        "95a305c5b6c1291d6ea6a5d1c96e3b5cb42f54a86232190be79f0ea493822e39"
    )
    assert fixture["oracle"]["replayable_in_ci"] is False
    assert fixture["evidence"]["provenance"]["source_kind"] == (
        "frozen_aggregate_proxy"
    )
    assert "does not bundle or replay" in fixture["evidence"]["provenance"][
        "limitation"
    ]
    assert fixture["reference"]["source"] == (
        "Paper canonical implementation (frozen prior)"
    )
    assert fixture["reference"]["purpose"] == "proxy"
    assert fixture["reference"]["geography"] is None
    assert fixture["reference"]["period"] is None

    support = sorted(fixture["evidence"]["counts"])
    total = sum(fixture["evidence"]["counts"].values())
    mass = fixture["parameters"]["smoothing_mass"]
    alpha = mass / len(support)
    observed = {
        category: fixture["evidence"]["counts"][category] / total
        for category in support
    }
    expected_from_frozen_inputs = sum(
        ((observed[category] + alpha) / (1.0 + mass))
        * math.log(
            ((observed[category] + alpha) / (1.0 + mass))
            / (
                (fixture["reference"]["probabilities"][category] + alpha)
                / (1.0 + mass)
            )
        )
        for category in support
    )
    assert expected_from_frozen_inputs == fixture["expected"]["value"]

    evidence = RepresentationEvidence(
        axis=fixture["axis"],
        counts=fixture["evidence"]["counts"],
        source="Portable paper golden aggregate",
        provenance=fixture["evidence"]["provenance"],
    )
    reference = ReferenceDistribution(**fixture["reference"])
    spec = DatasetAuditSpec(
        target_name=fixture["dataset"],
        target_kind="benchmark_dataset",
        task_family="free_text",
        design_stance="stress_test",
        references={fixture["axis"]: reference},
        requested_components=("b_rep",),
    )
    diagnostic = RepresentativenessBias(
        smoothing_mass=fixture["parameters"]["smoothing_mass"]
    )
    report = audit_representativeness(evidence, spec, diagnostic=diagnostic)
    result = _result(report)

    assert result.status.value == fixture["expected"]["status"]
    assert result.value == pytest.approx(
        fixture["expected"]["value"], rel=0, abs=1e-18
    )
    assert result.details["observed_counts"] == {
        "Female": 355922,
        "Male": 355922,
    }
    assert result.details["reference_distribution"] == {
        "Female": 0.509,
        "Male": 0.491,
    }
    assert result.provenance["evidence"]["provenance"]["source_kind"] == (
        "frozen_aggregate_proxy"
    )
    assert result.provenance["reference"]["source"] == (
        "Paper canonical implementation (frozen prior)"
    )
