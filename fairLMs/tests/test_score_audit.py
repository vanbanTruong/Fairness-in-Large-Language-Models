"""Scientific and applicability tests for the first scorer-audit slice."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from fairLMs.datasets.diagnostics import (
    ComponentPlan,
    DatasetAuditSpec,
    DatasetDiagnostic,
    DiagnosticStatus,
    ReportStatus,
    ScoredGroups,
    ScorerMeanGap,
    TargetKind,
    audit_scores,
)

AXIS = "unfamiliar-cohort-axis"


def _evidence(*, groups=None, scores=None, score_range=(0.0, 1.0), **kwargs):
    return ScoredGroups(
        axis=AXIS,
        groups=(
            ["zeta", "alpha", "middle", "zeta", "alpha", "middle"]
            if groups is None
            else groups
        ),
        scores=([0.8, 0.1, 0.4, 1.0, 0.3, 0.6] if scores is None else scores),
        score_name="unfamiliar-score-channel",
        source="external score table v4",
        score_range=score_range,
        provenance={"table_id": "outside-catalog-44"},
        **kwargs,
    )


def _spec(
    *,
    target_kind="score_table",
    requested_components=("score_mean_gap",),
):
    return DatasetAuditSpec(
        target_name="unregistered-score-table",
        target_kind=target_kind,
        task_family="scored_rows",
        design_stance="stress_test",
        references={},
        requested_components=requested_components,
    )


def _result(report):
    assert set(report.components) == {"score_mean_gap"}
    return report.components["score_mean_gap"]


def test_known_three_group_mean_gap_matches_paper_definition():
    report = audit_scores(_evidence(), _spec())
    result = _result(report)

    assert result.status is DiagnosticStatus.READY
    assert result.value == pytest.approx(0.7, rel=0, abs=1e-15)
    assert report.status is ReportStatus.SUCCESS
    assert result.details["sample_count"] == 6
    assert result.details["support"] == ("alpha", "middle", "zeta")
    assert result.details["group_counts"] == {
        "alpha": 2,
        "middle": 2,
        "zeta": 2,
    }
    assert result.details["group_means"] == pytest.approx(
        {"alpha": 0.2, "middle": 0.5, "zeta": 0.9}
    )
    assert result.details["argmax_groups"] == ("alpha", "zeta")
    assert result.details["higher_group"] == "zeta"
    assert result.details["lower_group"] == "alpha"
    assert result.details["aggregation"] == (
        "maximum_absolute_pairwise_group_mean_difference"
    )
    assert result.details["unit"] == "score_units"
    assert len(result.details["pairwise_gaps"]) == 3
    assert [
        (pair["left_group"], pair["right_group"])
        for pair in result.details["pairwise_gaps"]
    ] == [
        ("alpha", "middle"),
        ("alpha", "zeta"),
        ("middle", "zeta"),
    ]
    assert result.provenance["evidence"]["score_name"] == ("unfamiliar-score-channel")
    assert any("not automatically comparable" in item for item in result.assumptions)


def test_equal_group_means_are_a_successful_numeric_zero():
    report = audit_scores(
        _evidence(
            groups=["amber", "amber", "teal", "teal"],
            scores=[0.0, 1.0, 0.5, 0.5],
        ),
        _spec(),
    )
    result = _result(report)

    assert result.status is DiagnosticStatus.READY
    assert result.value == 0.0
    assert result.details["argmax_groups"] == ("amber", "teal")
    assert result.details["higher_group"] is None
    assert result.details["lower_group"] is None


def test_class_and_audit_scores_entry_points_agree():
    evidence = _evidence()
    spec = _spec()
    diagnostic = ScorerMeanGap()

    direct = diagnostic.compute(evidence, spec)
    default_report = audit_scores(evidence, spec)
    singular_report = audit_scores(evidence, spec, diagnostic=diagnostic)
    plural_report = audit_scores(evidence, spec, diagnostics=[diagnostic])
    reported = _result(singular_report)

    assert reported.to_dict() == direct.to_dict()
    assert default_report.to_dict() == singular_report.to_dict()
    assert plural_report.to_dict() == singular_report.to_dict()


def test_row_permutation_does_not_change_numeric_or_aggregate_details():
    groups = ["amber", "teal", "amber", "teal", "amber"]
    scores = [0.1, 0.9, 0.3, 0.7, 0.5]
    forward = _result(audit_scores(_evidence(groups=groups, scores=scores), _spec()))
    reverse = _result(
        audit_scores(
            _evidence(groups=list(reversed(groups)), scores=list(reversed(scores))),
            _spec(),
        )
    )

    assert reverse.value == forward.value
    assert reverse.details == forward.details


def test_scaled_mean_stays_finite_for_repeated_large_finite_scores():
    result = _result(
        audit_scores(
            _evidence(
                groups=["amber", "amber", "teal", "teal"],
                scores=[1e308, 1e308, 5e307, 5e307],
                score_range=None,
            ),
            _spec(),
        )
    )

    assert result.status is DiagnosticStatus.READY
    assert result.details["group_means"] == {"amber": 1e308, "teal": 5e307}
    assert result.value == 5e307


def test_large_cancelling_scores_are_stable_under_row_permutation():
    groups = ["amber"] * 5 + ["teal"]
    scores = [1e308, 1e308, -1e308, -1e308, 1.0, 0.0]
    forward = _result(
        audit_scores(
            _evidence(groups=groups, scores=scores, score_range=None),
            _spec(),
        )
    )
    reverse = _result(
        audit_scores(
            _evidence(
                groups=list(reversed(groups)),
                scores=list(reversed(scores)),
                score_range=None,
            ),
            _spec(),
        )
    )

    assert forward.status is DiagnosticStatus.READY
    assert forward.value == pytest.approx(0.2, rel=0, abs=2e-16)
    assert reverse.value == forward.value
    assert reverse.details == forward.details


def test_unrepresentable_pairwise_gap_returns_explicit_failed_result():
    evidence = _evidence(
        groups=["amber", "teal"],
        scores=[-1e308, 1e308],
        score_range=None,
    )
    direct = ScorerMeanGap().compute(evidence, _spec())
    reported = _result(audit_scores(evidence, _spec()))

    assert direct.status is DiagnosticStatus.FAILED
    assert reported.to_dict() == direct.to_dict()
    assert reported.value is None
    assert reported.reason_code == "numeric_computation_failed"
    assert reported.details["unrepresentable_pair"] == {
        "left_group": "amber",
        "right_group": "teal",
        "left_mean": -1e308,
        "right_mean": 1e308,
    }


def test_group_renaming_and_score_translation_do_not_change_gap():
    original = _result(audit_scores(_evidence(score_range=None), _spec()))
    renamed = _result(
        audit_scores(
            _evidence(
                groups=["x", "y", "w", "x", "y", "w"],
                score_range=None,
            ),
            _spec(),
        )
    )
    shifted = _result(
        audit_scores(
            _evidence(
                scores=[10.8, 10.1, 10.4, 11.0, 10.3, 10.6],
                score_range=None,
            ),
            _spec(),
        )
    )

    assert renamed.value == original.value
    assert shifted.value == pytest.approx(original.value, rel=0, abs=2e-15)


def test_declared_score_range_validates_but_does_not_normalize_the_gap():
    result = _result(
        audit_scores(
            _evidence(
                groups=["amber", "amber", "teal", "teal"],
                scores=[0.0, 20.0, 60.0, 100.0],
                score_range=(0.0, 100.0),
            ),
            _spec(),
        )
    )

    assert result.value == 70.0
    assert result.details["score_range"] == (0.0, 100.0)


@pytest.mark.parametrize(
    "target_kind",
    [
        TargetKind.BENCHMARK_DATASET,
        TargetKind.GENERATED_OUTPUT,
        TargetKind.AGGREGATE_STATISTICS,
    ],
)
def test_wrong_target_kind_is_not_applicable_not_a_zero_gap(target_kind):
    report = audit_scores(
        _evidence(),
        _spec(target_kind=target_kind),
    )
    result = _result(report)

    assert result.status is DiagnosticStatus.NOT_APPLICABLE
    assert result.value is None
    assert result.reason_code == "target_kind_not_supported"
    assert report.status is ReportStatus.NOT_APPLICABLE


def test_unrequested_component_is_not_applicable():
    report = audit_scores(
        _evidence(),
        _spec(requested_components=("b_rep",)),
    )
    result = _result(report)

    assert result.status is DiagnosticStatus.NOT_APPLICABLE
    assert result.value is None
    assert result.reason_code == "component_not_requested"


def test_report_runner_translates_ready_compute_exception_to_failed():
    class FailingDiagnostic(DatasetDiagnostic):
        name = "score_mean_gap"

        def plan(self, evidence, spec):
            return ComponentPlan(component=self.name, status="ready")

        def compute(self, evidence, spec):
            raise ArithmeticError("synthetic scorer failure")

    report = audit_scores(
        _evidence(),
        _spec(),
        diagnostic=FailingDiagnostic(),
    )
    result = _result(report)

    assert result.status is DiagnosticStatus.FAILED
    assert result.value is None
    assert result.reason_code == "computation_failed"
    assert result.details["exception_type"] == "ArithmeticError"
    assert report.status is ReportStatus.FAILED


def test_score_report_is_strict_json_schema_1_4():
    report = audit_scores(_evidence(), _spec())
    payload = report.to_dict()

    assert payload["schema_version"] == "1.7"
    assert payload["status"] == "success"
    assert math.isfinite(payload["components"]["score_mean_gap"]["value"])
    assert report.to_json(indent=None) == report.to_json(indent=None)


def test_trustgpt_toxicity_frozen_row_level_paper_regression():
    fixture_path = (
        Path(__file__).parent
        / "data"
        / "golden"
        / "diagnostics"
        / "score_mean_gap"
        / "trustgpt_category_toxicity_v1.json"
    )
    fixture_text = fixture_path.read_text(encoding="utf-8")
    fixture = json.loads(fixture_text)

    assert "/Users/" not in fixture_text
    assert "Downloads" not in fixture_text
    assert fixture["fixture_id"] == "trustgpt_category_toxicity_v1"
    assert fixture["dataset"] == "TrustGPT"
    assert fixture["axis"] == "category"
    assert fixture["score"] == {
        "name": "toxicity",
        "range": [0.0, 1.0],
        "source": (
            "Perspective API toxicity outputs frozen by the paper scorer artifact"
        ),
    }

    evidence_fixture = fixture["evidence"]
    assert evidence_fixture["columns"] == ["category", "toxicity_score"]
    assert evidence_fixture["support"] == ["female", "male"]
    assert len(evidence_fixture["rows"]) == 200
    assert all(len(row) == 2 for row in evidence_fixture["rows"])
    assert evidence_fixture["provenance"]["retained_columns"] == [
        "category",
        "toxicity_score",
    ]
    assert evidence_fixture["provenance"]["source_kind"] == (
        "frozen_row_level_scorer_outputs"
    )
    assert "does not replay prompt preparation or call the Perspective API" in (
        evidence_fixture["provenance"]["limitation"]
    )

    oracle = fixture["oracle"]
    assert oracle["kind"] == "frozen_row_level_scorer_output_regression"
    assert oracle["source_artifact_sha256"] == (
        "02962c3eeb8568becf1233f406822d25f265378221d283d8761b99f489f1621d"
    )
    assert oracle["report_artifact_sha256"] == (
        "fe3c7ade85f21104c0e945d62f351e4c75eae88547315345c801c8c32c8d5062"
    )
    assert oracle["runner_sha256"] == (
        "934e2bd327d0e923d53097eda81b85c85aedca6f05de5cd227926f58a86169ad"
    )
    assert oracle["replayable_in_ci"] is False
    for key in ("source_artifact", "report_artifact", "runner"):
        assert not Path(oracle[key]).is_absolute()
    assert "does not execute the external paper runner" in oracle["verification_note"]
    assert "call the Perspective API" in oracle["verification_note"]

    group_field, score_field = evidence_fixture["columns"]
    records = [
        {group_field: row[0], score_field: row[1]} for row in evidence_fixture["rows"]
    ]
    evidence = ScoredGroups.from_records(
        records,
        axis=fixture["axis"],
        group_field=group_field,
        score_field=score_field,
        support=evidence_fixture["support"],
        score_name=fixture["score"]["name"],
        source=fixture["score"]["source"],
        score_range=fixture["score"]["range"],
        provenance=evidence_fixture["provenance"],
    )
    spec = DatasetAuditSpec(
        target_name=fixture["dataset"],
        target_kind="score_table",
        task_family="scored_rows",
        design_stance="stress_test",
        references={},
        requested_components=("score_mean_gap",),
    )
    result = _result(audit_scores(evidence, spec))
    expected = fixture["expected"]

    assert result.status.value == expected["status"]
    assert result.value == expected["value"]
    assert result.details["sample_count"] == expected["sample_count"]
    assert result.details["group_counts"] == expected["group_counts"]
    assert result.details["group_means"] == expected["group_means"]
    assert result.details["argmax_groups"] == tuple(expected["argmax_groups"])
    assert result.details["higher_group"] == expected["higher_group"]
    assert result.details["lower_group"] == expected["lower_group"]
    assert result.details["score_range"] == tuple(fixture["score"]["range"])
    assert len(result.details["pairwise_gaps"]) == 1
    assert (
        result.details["pairwise_gaps"][0]["signed_gap_right_minus_left"]
        == -expected["value"]
    )
    assert result.provenance["evidence"]["provenance"]["source_kind"] == (
        "frozen_row_level_scorer_outputs"
    )
