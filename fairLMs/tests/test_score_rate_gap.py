"""Contract tests for explicit thresholded scorer-rate diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from fairLMs.diagnostics import (
    ComponentPlan,
    DatasetAuditSpec,
    DatasetDiagnostic,
    DiagnosticStatus,
    ReportStatus,
    ScoredGroups,
    ScoreRateDirection,
    ScoreRateTransform,
    ScorerMeanGap,
    ScorerRateGap,
    TargetKind,
    audit_scores,
    get_diagnostic,
)

AXIS = "unfamiliar-rate-axis"
COMPONENT = "score_rate_gap"


def _evidence(*, groups=None, scores=None, score_range=(0.0, 1.0)):
    return ScoredGroups(
        axis=AXIS,
        groups=(
            ["alpha", "alpha", "middle", "middle", "zeta", "zeta"]
            if groups is None
            else groups
        ),
        scores=([0.2, 0.8, 0.6, 0.7, 0.1, 0.3] if scores is None else scores),
        score_name="unfamiliar-rate-channel",
        source="external score table v5",
        score_range=score_range,
        provenance={"table_id": "outside-catalog-rate-table-51"},
    )


def _spec(*, target_kind="score_table", requested_components=(COMPONENT,)):
    return DatasetAuditSpec(
        target_name="unregistered-rate-table",
        target_kind=target_kind,
        task_family="scored_rows",
        design_stance="stress_test",
        references={},
        requested_components=requested_components,
    )


def _transform(
    *,
    threshold=0.5,
    direction="higher",
    inclusive=True,
    provenance=None,
):
    return ScoreRateTransform(
        event_name="declared_score_event",
        threshold=threshold,
        direction=direction,
        inclusive=inclusive,
        provenance={} if provenance is None else provenance,
    )


def _rate_result(report):
    assert set(report.components) == {COMPONENT}
    return report.components[COMPONENT]


@pytest.mark.parametrize(
    (
        "direction",
        "inclusive",
        "expected_operator",
        "expected_alignment",
        "expected_rates",
    ),
    [
        (
            "higher",
            True,
            ">=",
            "paper_exact",
            {"alpha": 1.0, "teal": 0.5},
        ),
        (
            "higher",
            False,
            ">",
            "generalized_boundary_extension",
            {"alpha": 0.5, "teal": 0.0},
        ),
        (
            "lower",
            True,
            "<=",
            "generalized_lower_tail_analogue",
            {"alpha": 0.5, "teal": 1.0},
        ),
        (
            "lower",
            False,
            "<",
            "generalized_boundary_extension",
            {"alpha": 0.0, "teal": 0.5},
        ),
    ],
)
def test_rate_transform_has_explicit_direction_and_boundary_rule(
    direction,
    inclusive,
    expected_operator,
    expected_alignment,
    expected_rates,
):
    evidence = _evidence(
        groups=["alpha", "alpha", "teal", "teal"],
        scores=[0.5, 0.6, 0.4, 0.5],
    )
    result = _rate_result(
        audit_scores(
            evidence,
            _spec(),
            diagnostic=ScorerRateGap(
                transform=_transform(direction=direction, inclusive=inclusive)
            ),
        )
    )

    assert result.status is DiagnosticStatus.READY
    assert result.value == 0.5
    assert result.details["group_rates"] == expected_rates
    assert result.details["transform"]["direction"] == direction
    assert result.details["transform"]["inclusive"] is inclusive
    assert result.details["transform"]["operator"] == expected_operator
    assert result.details["transform"]["paper_alignment"] == expected_alignment


def test_rate_transform_is_strict_immutable_and_json_safe():
    provenance = {"nested": {"revision": 2}}
    transform = _transform(provenance=provenance)
    provenance["nested"]["revision"] = 99

    assert transform.direction is ScoreRateDirection.HIGHER
    assert transform.threshold == 0.5
    assert transform.inclusive is True
    assert transform.provenance["nested"]["revision"] == 2

    payload = transform.to_dict()
    assert payload["event_name"] == "declared_score_event"
    assert payload["threshold"] == 0.5
    assert payload["direction"] == "higher"
    assert payload["inclusive"] is True
    json.dumps(payload, allow_nan=False, sort_keys=True)
    payload["provenance"]["nested"]["revision"] = -1
    assert transform.provenance["nested"]["revision"] == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"event_name": ""},
        {"event_name": "   "},
        {"event_name": 7},
        {"threshold": True},
        {"threshold": "0.5"},
        {"threshold": math.nan},
        {"threshold": math.inf},
        {"direction": "sideways"},
        {"direction": True},
        {"inclusive": 1},
        {"inclusive": "true"},
        {"provenance": {"bad": object()}},
        {"provenance": {"bad": math.nan}},
    ],
)
def test_rate_transform_rejects_invalid_or_nonportable_configuration(kwargs):
    defaults = {
        "event_name": "declared_score_event",
        "threshold": 0.5,
        "direction": "higher",
        "inclusive": True,
    }
    defaults.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        ScoreRateTransform(**defaults)


def test_two_and_three_group_rate_gap_formula_and_details():
    two_group = _rate_result(
        audit_scores(
            _evidence(
                groups=["amber", "amber", "teal", "teal"],
                scores=[0.1, 0.8, 0.6, 0.9],
            ),
            _spec(),
            diagnostic=ScorerRateGap(transform=_transform()),
        )
    )
    assert two_group.value == 0.5
    assert two_group.details["event_counts"] == {"amber": 1, "teal": 2}
    assert two_group.details["group_rates"] == {"amber": 0.5, "teal": 1.0}
    assert two_group.details["argmax_groups"] == ("amber", "teal")

    three_group = _rate_result(
        audit_scores(
            _evidence(),
            _spec(),
            diagnostic=ScorerRateGap(transform=_transform()),
        )
    )
    assert three_group.value == 1.0
    assert three_group.details["sample_count"] == 6
    assert three_group.details["support"] == ("alpha", "middle", "zeta")
    assert three_group.details["group_counts"] == {
        "alpha": 2,
        "middle": 2,
        "zeta": 2,
    }
    assert three_group.details["event_counts"] == {
        "alpha": 1,
        "middle": 2,
        "zeta": 0,
    }
    assert three_group.details["group_rates"] == {
        "alpha": 0.5,
        "middle": 1.0,
        "zeta": 0.0,
    }
    assert three_group.details["aggregation"] == (
        "maximum_absolute_pairwise_group_rate_difference"
    )
    assert three_group.details["unit"] == "proportion"


def test_multigroup_ties_keep_the_first_lexicographic_pair_and_signed_order():
    result = _rate_result(
        audit_scores(
            _evidence(
                groups=["zeta", "middle", "alpha", "zeta", "middle", "alpha"],
                scores=[0.1, 0.8, 0.2, 0.3, 0.9, 0.4],
            ),
            _spec(),
            diagnostic=ScorerRateGap(transform=_transform()),
        )
    )

    assert result.value == 1.0
    assert result.details["group_rates"] == {
        "alpha": 0.0,
        "middle": 1.0,
        "zeta": 0.0,
    }
    assert result.details["argmax_groups"] == ("alpha", "middle")
    assert result.details["higher_rate_group"] == "middle"
    assert result.details["lower_rate_group"] == "alpha"
    assert result.details["pairwise_rate_gaps"] == (
        {
            "left_group": "alpha",
            "right_group": "middle",
            "left_rate": 0.0,
            "right_rate": 1.0,
            "signed_gap_right_minus_left": 1.0,
            "absolute_gap": 1.0,
        },
        {
            "left_group": "alpha",
            "right_group": "zeta",
            "left_rate": 0.0,
            "right_rate": 0.0,
            "signed_gap_right_minus_left": 0.0,
            "absolute_gap": 0.0,
        },
        {
            "left_group": "middle",
            "right_group": "zeta",
            "left_rate": 1.0,
            "right_rate": 0.0,
            "signed_gap_right_minus_left": -1.0,
            "absolute_gap": 1.0,
        },
    )


def test_equal_rates_are_a_successful_zero_and_are_distinct_from_mean_gap():
    evidence = _evidence(
        groups=["amber", "amber", "teal", "teal"],
        scores=[0.51, 0.99, 0.51, 0.52],
    )
    spec = _spec(requested_components=("score_mean_gap", COMPONENT))
    report = audit_scores(
        evidence,
        spec,
        diagnostics=(
            ScorerMeanGap(),
            ScorerRateGap(transform=_transform()),
        ),
    )

    rate = report.components[COMPONENT]
    mean = report.components["score_mean_gap"]
    assert rate.status is DiagnosticStatus.READY
    assert rate.value == 0.0
    assert rate.details["group_rates"] == {"amber": 1.0, "teal": 1.0}
    assert mean.status is DiagnosticStatus.READY
    assert mean.value == pytest.approx(0.235)
    assert report.status is ReportStatus.SUCCESS
    assert any(
        "not automatically comparable across datasets" in assumption
        for assumption in rate.assumptions
    )
    assert any("12-percentage-point" in assumption for assumption in rate.assumptions)


def test_rate_gap_is_invariant_to_row_permutation_and_group_renaming():
    groups = ["amber", "teal", "amber", "teal", "violet", "violet"]
    scores = [0.1, 0.6, 0.8, 0.7, 0.2, 0.3]
    forward = _rate_result(
        audit_scores(
            _evidence(groups=groups, scores=scores),
            _spec(),
            diagnostic=ScorerRateGap(transform=_transform()),
        )
    )
    reverse = _rate_result(
        audit_scores(
            _evidence(groups=list(reversed(groups)), scores=list(reversed(scores))),
            _spec(),
            diagnostic=ScorerRateGap(transform=_transform()),
        )
    )
    renamed = _rate_result(
        audit_scores(
            _evidence(groups=["x", "y", "x", "y", "w", "w"], scores=scores),
            _spec(),
            diagnostic=ScorerRateGap(transform=_transform()),
        )
    )

    assert reverse.value == forward.value
    assert reverse.details == forward.details
    assert renamed.value == forward.value


def test_rate_gap_runs_without_a_declared_score_range():
    result = _rate_result(
        audit_scores(
            _evidence(score_range=None),
            _spec(),
            diagnostic=ScorerRateGap(transform=_transform()),
        )
    )
    assert result.status is DiagnosticStatus.READY
    assert result.details["score_range"] is None


def test_threshold_outside_declared_range_is_blocked_not_a_degenerate_rate():
    result = _rate_result(
        audit_scores(
            _evidence(),
            _spec(),
            diagnostic=ScorerRateGap(transform=_transform(threshold=1.1)),
        )
    )
    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "threshold_outside_score_range"


def test_missing_transform_is_blocked_not_a_zero_rate():
    result = _rate_result(
        audit_scores(_evidence(), _spec(), diagnostic=ScorerRateGap())
    )
    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "missing_rate_transform"


def test_registry_constructs_an_unconfigured_rate_diagnostic_without_guessing():
    diagnostic = get_diagnostic(COMPONENT)

    assert isinstance(diagnostic, ScorerRateGap)
    assert diagnostic.transform is None
    result = _rate_result(audit_scores(_evidence(), _spec(), diagnostic=diagnostic))
    assert result.status is DiagnosticStatus.BLOCKED
    assert result.reason_code == "missing_rate_transform"


def test_wrong_target_and_unrequested_component_are_not_applicable():
    diagnostic = ScorerRateGap(transform=_transform())
    wrong_target = _rate_result(
        audit_scores(
            _evidence(),
            _spec(target_kind=TargetKind.BENCHMARK_DATASET),
            diagnostic=diagnostic,
        )
    )
    unrequested = _rate_result(
        audit_scores(
            _evidence(),
            _spec(requested_components=("score_mean_gap",)),
            diagnostic=diagnostic,
        )
    )
    for result in (wrong_target, unrequested):
        assert result.status is DiagnosticStatus.NOT_APPLICABLE
        assert result.value is None
    assert wrong_target.reason_code == "target_kind_not_supported"
    assert unrequested.reason_code == "component_not_requested"


def test_multiple_scorer_components_report_success_and_partial_explicitly():
    spec = _spec(requested_components=("score_mean_gap", COMPONENT))
    successful = audit_scores(
        _evidence(),
        spec,
        diagnostics=(
            ScorerMeanGap(),
            ScorerRateGap(transform=_transform()),
        ),
    )
    partial = audit_scores(
        _evidence(),
        spec,
        diagnostics=(ScorerMeanGap(), ScorerRateGap()),
    )

    assert set(successful.components) == {"score_mean_gap", COMPONENT}
    assert successful.status is ReportStatus.SUCCESS
    assert partial.components["score_mean_gap"].status is DiagnosticStatus.READY
    assert partial.components[COMPONENT].status is DiagnosticStatus.BLOCKED
    assert partial.status is ReportStatus.PARTIAL
    assert successful.to_dict()["provenance"] == {
        "diagnostics": ["score_mean_gap", COMPONENT]
    }


def test_runner_rejects_ambiguous_empty_duplicate_or_invalid_diagnostic_lists():
    evidence = _evidence()
    spec = _spec(requested_components=("score_mean_gap", COMPONENT))
    rate = ScorerRateGap(transform=_transform())

    with pytest.raises(TypeError, match="diagnostic.*diagnostics"):
        audit_scores(
            evidence,
            spec,
            diagnostic=ScorerMeanGap(),
            diagnostics=(rate,),
        )
    with pytest.raises(ValueError, match="diagnostics.*at least one"):
        audit_scores(evidence, spec, diagnostics=())
    with pytest.raises(ValueError, match="duplicate"):
        audit_scores(evidence, spec, diagnostics=(ScorerMeanGap(), ScorerMeanGap()))
    with pytest.raises(TypeError, match="DatasetDiagnostic"):
        audit_scores(evidence, spec, diagnostics=(object(),))


def test_runner_translates_ready_rate_exception_to_failed():
    class FailingRateDiagnostic(DatasetDiagnostic):
        name = COMPONENT

        def plan(self, evidence, spec):
            return ComponentPlan(component=self.name, status="ready")

        def compute(self, evidence, spec):
            raise ArithmeticError("synthetic rate failure")

    result = _rate_result(
        audit_scores(
            _evidence(),
            _spec(),
            diagnostic=FailingRateDiagnostic(),
        )
    )
    assert result.status is DiagnosticStatus.FAILED
    assert result.value is None
    assert result.reason_code == "computation_failed"
    assert result.details["exception_type"] == "ArithmeticError"


def test_trustgpt_toxicity_frozen_row_level_rate_regression():
    fixture_path = (
        Path(__file__).parent
        / "data"
        / "golden"
        / "diagnostics"
        / "score_mean_gap"
        / "trustgpt_category_toxicity_v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    group_field, score_field = fixture["evidence"]["columns"]
    evidence = ScoredGroups.from_records(
        [
            {group_field: row[0], score_field: row[1]}
            for row in fixture["evidence"]["rows"]
        ],
        axis=fixture["axis"],
        group_field=group_field,
        score_field=score_field,
        support=fixture["evidence"]["support"],
        score_name=fixture["score"]["name"],
        source=fixture["score"]["source"],
        score_range=fixture["score"]["range"],
        provenance=fixture["evidence"]["provenance"],
    )
    expected = fixture["expected_rate"]
    rule = expected["details"]["transform"]
    transform = ScoreRateTransform(
        event_name=rule["event_name"],
        threshold=rule["threshold"],
        direction=rule["direction"],
        inclusive=rule["inclusive"],
        provenance=rule["provenance"],
    )
    spec = DatasetAuditSpec(
        target_name=fixture["dataset"],
        target_kind="score_table",
        task_family="scored_rows",
        design_stance="stress_test",
        references={},
        requested_components=(COMPONENT,),
    )
    result = _rate_result(
        audit_scores(evidence, spec, diagnostic=ScorerRateGap(transform=transform))
    )

    assert transform.operator == rule["operator"]
    assert expected["canonical_rule"] == (
        f"score {transform.operator} {transform.threshold:g}"
    )
    assert result.status.value == expected["status"]
    assert result.value == expected["value"]
    assert result.to_dict()["details"] == expected["details"]
