"""Contract tests for empirical one-dimensional scorer distribution gaps."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from fairLMs.diagnostics import (
    DIAGNOSTIC_REGISTRY,
    DatasetAuditSpec,
    DiagnosticStatus,
    ReportStatus,
    ScoredGroups,
    ScoreRateTransform,
    ScorerMeanGap,
    ScorerRateGap,
    ScorerWasserstein1Gap,
    TargetKind,
    audit_scores,
    get_diagnostic,
    list_diagnostics,
)

AXIS = "unfamiliar-wasserstein-axis"
COMPONENT = "score_wasserstein_1_gap"


def _evidence(*, groups=None, scores=None, score_range=(0.0, 1.0)):
    return ScoredGroups(
        axis=AXIS,
        groups=(
            ["alpha", "alpha", "middle", "middle", "zeta", "zeta"]
            if groups is None
            else groups
        ),
        scores=([0.2, 0.8, 0.6, 0.7, 0.1, 0.3] if scores is None else scores),
        score_name="unfamiliar-distribution-channel",
        source="external score table v6",
        score_range=score_range,
        provenance={"table_id": "outside-catalog-wasserstein-table-61"},
    )


def _spec(*, target_kind="score_table", requested_components=(COMPONENT,)):
    return DatasetAuditSpec(
        target_name="unregistered-distribution-table",
        target_kind=target_kind,
        task_family="scored_rows",
        design_stance="stress_test",
        references={},
        requested_components=requested_components,
    )


def _result(report):
    assert set(report.components) == {COMPONENT}
    return report.components[COMPONENT]


def _frozen_json_shape(value):
    if isinstance(value, dict):
        return {key: _frozen_json_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_frozen_json_shape(item) for item in value)
    return value


def test_known_two_group_wasserstein_value_and_frozen_details_contract():
    result = _result(
        audit_scores(
            _evidence(
                groups=["amber", "amber", "teal", "teal"],
                scores=[0.0, 10.0, 4.0, 6.0],
                score_range=None,
            ),
            _spec(),
            diagnostic=ScorerWasserstein1Gap(),
        )
    )

    assert result.status is DiagnosticStatus.READY
    assert result.value == 4.0
    assert result.details == {
        "axis": AXIS,
        "score_name": "unfamiliar-distribution-channel",
        "score_range": None,
        "sample_count": 4,
        "support": ("amber", "teal"),
        "group_counts": {"amber": 2, "teal": 2},
        "pairwise_wasserstein_1": (
            {
                "left_group": "amber",
                "right_group": "teal",
                "left_sample_count": 2,
                "right_sample_count": 2,
                "wasserstein_1": 4.0,
            },
        ),
        "argmax_groups": ("amber", "teal"),
        "aggregation": (
            "maximum_pairwise_one_dimensional_empirical_wasserstein_distance"
        ),
        "estimator": "empirical_uniform_mass_per_group",
        "ground_metric": "absolute_score_difference",
        "directionality": "symmetric",
        "normalization": "none",
        "unit": "score_units",
    }
    assert result.provenance["evidence"]["score_name"] == (
        "unfamiliar-distribution-channel"
    )


def test_wasserstein_uses_each_groups_uniform_empirical_mass_with_unequal_sizes():
    result = _result(
        audit_scores(
            _evidence(
                groups=["amber", "amber", "teal"],
                scores=[0.0, 2.0, 1.0],
                score_range=None,
            ),
            _spec(),
            diagnostic=ScorerWasserstein1Gap(),
        )
    )

    assert result.status is DiagnosticStatus.READY
    assert result.value == 1.0
    assert result.details["group_counts"] == {"amber": 2, "teal": 1}
    assert result.details["pairwise_wasserstein_1"] == (
        {
            "left_group": "amber",
            "right_group": "teal",
            "left_sample_count": 2,
            "right_sample_count": 1,
            "wasserstein_1": 1.0,
        },
    )


def test_unequal_coprime_group_sizes_and_repeated_scores_have_known_w1():
    result = _result(
        audit_scores(
            _evidence(
                groups=["amber"] * 3 + ["teal"] * 5,
                scores=[0.0, 0.0, 3.0, 1.0, 1.0, 1.0, 4.0, 4.0],
                score_range=None,
            ),
            _spec(),
            diagnostic=ScorerWasserstein1Gap(),
        )
    )

    # ECDF areas: 2/3 on [0, 1], 1/15 on [1, 3], and 2/5 on [3, 4].
    assert result.status is DiagnosticStatus.READY
    assert result.value == pytest.approx(1.2)
    assert result.details["group_counts"] == {"amber": 3, "teal": 5}


def test_three_groups_reports_all_sorted_pairs_and_the_maximum_distance():
    result = _result(
        audit_scores(
            _evidence(
                groups=["zeta", "alpha", "middle", "zeta", "alpha", "middle"],
                scores=[4.0, 0.0, 1.0, 4.0, 0.0, 1.0],
                score_range=None,
            ),
            _spec(),
            diagnostic=ScorerWasserstein1Gap(),
        )
    )

    assert result.value == 4.0
    assert result.details["support"] == ("alpha", "middle", "zeta")
    assert result.details["argmax_groups"] == ("alpha", "zeta")
    assert result.details["pairwise_wasserstein_1"] == (
        {
            "left_group": "alpha",
            "right_group": "middle",
            "left_sample_count": 2,
            "right_sample_count": 2,
            "wasserstein_1": 1.0,
        },
        {
            "left_group": "alpha",
            "right_group": "zeta",
            "left_sample_count": 2,
            "right_sample_count": 2,
            "wasserstein_1": 4.0,
        },
        {
            "left_group": "middle",
            "right_group": "zeta",
            "left_sample_count": 2,
            "right_sample_count": 2,
            "wasserstein_1": 3.0,
        },
    )


def test_multigroup_ties_keep_the_first_lexicographic_pair_without_direction():
    result = _result(
        audit_scores(
            _evidence(
                groups=["zeta", "middle", "alpha", "zeta", "middle", "alpha"],
                scores=[0.0, 2.0, 0.0, 0.0, 2.0, 0.0],
                score_range=None,
            ),
            _spec(),
            diagnostic=ScorerWasserstein1Gap(),
        )
    )

    assert result.value == 2.0
    assert result.details["argmax_groups"] == ("alpha", "middle")
    assert result.details["directionality"] == "symmetric"
    assert "higher_group" not in result.details
    assert "lower_group" not in result.details
    assert all(
        "signed_gap_right_minus_left" not in pair
        for pair in result.details["pairwise_wasserstein_1"]
    )


def test_same_mean_and_threshold_rate_can_have_nonzero_wasserstein_distance():
    evidence = _evidence(
        groups=["amber", "amber", "teal", "teal", "teal", "teal"],
        scores=[0.0, 1.0, 0.25, 0.25, 0.75, 0.75],
    )
    spec = _spec(
        requested_components=(
            "score_mean_gap",
            "score_rate_gap",
            COMPONENT,
        )
    )
    report = audit_scores(
        evidence,
        spec,
        diagnostics=(
            ScorerMeanGap(),
            ScorerRateGap(
                transform=ScoreRateTransform(
                    event_name="score_at_or_above_half",
                    threshold=0.5,
                    direction="higher",
                    inclusive=True,
                )
            ),
            ScorerWasserstein1Gap(),
        ),
    )

    assert report.components["score_mean_gap"].value == 0.0
    assert report.components["score_rate_gap"].value == 0.0
    assert report.components[COMPONENT].value == 0.25
    assert report.status is ReportStatus.SUCCESS


def test_wasserstein_is_invariant_to_row_order_group_renaming_and_translation():
    groups = ["amber", "teal", "amber", "teal", "violet", "violet"]
    scores = [0.1, 0.6, 0.8, 0.7, 0.2, 0.3]
    diagnostic = ScorerWasserstein1Gap()
    forward = _result(
        audit_scores(
            _evidence(groups=groups, scores=scores, score_range=None),
            _spec(),
            diagnostic=diagnostic,
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
            diagnostic=diagnostic,
        )
    )
    renamed = _result(
        audit_scores(
            _evidence(
                groups=["x", "y", "x", "y", "w", "w"],
                scores=scores,
                score_range=None,
            ),
            _spec(),
            diagnostic=diagnostic,
        )
    )
    translated = _result(
        audit_scores(
            _evidence(
                groups=groups,
                scores=[score + 10.0 for score in scores],
                score_range=None,
            ),
            _spec(),
            diagnostic=diagnostic,
        )
    )

    assert reverse.value == forward.value
    assert reverse.details == forward.details
    assert renamed.value == forward.value
    assert translated.value == pytest.approx(forward.value, rel=0, abs=1e-15)


def test_wasserstein_scales_in_native_score_units_without_range_normalization():
    groups = ["amber", "amber", "teal", "teal"]
    scores = [0.0, 2.0, 1.0, 3.0]
    original = _result(
        audit_scores(
            _evidence(groups=groups, scores=scores, score_range=None),
            _spec(),
            diagnostic=ScorerWasserstein1Gap(),
        )
    )
    scaled = _result(
        audit_scores(
            _evidence(
                groups=groups,
                scores=[7.0 * score for score in scores],
                score_range=None,
            ),
            _spec(),
            diagnostic=ScorerWasserstein1Gap(),
        )
    )

    assert scaled.value == pytest.approx(7.0 * original.value)
    assert scaled.details["normalization"] == "none"
    assert scaled.details["unit"] == "score_units"


def test_identical_group_distributions_are_a_successful_zero_with_stable_pair():
    result = _result(
        audit_scores(
            _evidence(
                groups=["amber", "amber", "teal", "teal"],
                scores=[0.2, 0.8, 0.2, 0.8],
            ),
            _spec(),
            diagnostic=ScorerWasserstein1Gap(),
        )
    )

    assert result.status is DiagnosticStatus.READY
    assert result.value == 0.0
    assert result.details["argmax_groups"] == ("amber", "teal")


def test_wasserstein_runs_without_score_range_and_handles_representable_extremes():
    result = _result(
        audit_scores(
            _evidence(
                groups=["amber", "amber", "teal", "teal"],
                scores=[1e308, 1e308, 5e307, 5e307],
                score_range=None,
            ),
            _spec(),
            diagnostic=ScorerWasserstein1Gap(),
        )
    )

    assert result.status is DiagnosticStatus.READY
    assert result.value == 5e307
    assert result.details["score_range"] is None


def test_representable_cross_sign_extreme_avoids_interval_width_overflow():
    result = _result(
        audit_scores(
            _evidence(
                groups=["amber", "amber", "teal", "teal"],
                scores=[-1e308, 1e308, 1e308, 1e308],
                score_range=None,
            ),
            _spec(),
            diagnostic=ScorerWasserstein1Gap(),
        )
    )

    assert result.status is DiagnosticStatus.READY
    assert result.value == 1e308


def test_subnormal_intervals_are_combined_before_the_final_float_rounding():
    smallest = math.ulp(0.0)
    result = _result(
        audit_scores(
            _evidence(
                groups=["amber", "amber", "amber", "teal", "teal"],
                scores=[0.0, smallest, 0.0, -smallest, 0.0],
                score_range=None,
            ),
            _spec(),
            diagnostic=ScorerWasserstein1Gap(),
        )
    )

    # The exact value is (5 / 6) * smallest, which rounds once to smallest.
    assert result.status is DiagnosticStatus.READY
    assert result.value == smallest


def test_positive_distance_too_small_for_a_nonzero_float_fails_explicitly():
    smallest = math.ulp(0.0)
    result = _result(
        audit_scores(
            _evidence(
                groups=["amber", "amber", "amber", "teal", "teal", "teal"],
                scores=[0.0, 0.0, 0.0, 0.0, 0.0, smallest],
                score_range=None,
            ),
            _spec(),
            diagnostic=ScorerWasserstein1Gap(),
        )
    )

    assert result.status is DiagnosticStatus.FAILED
    assert result.value is None
    assert result.reason_code == "numeric_computation_failed"


def test_unrepresentable_extreme_distance_returns_explicit_failed_result():
    evidence = _evidence(
        groups=["amber", "teal"],
        scores=[-1e308, 1e308],
        score_range=None,
    )
    direct = ScorerWasserstein1Gap().compute(evidence, _spec())
    reported = _result(
        audit_scores(evidence, _spec(), diagnostic=ScorerWasserstein1Gap())
    )

    assert direct.status is DiagnosticStatus.FAILED
    assert reported.to_dict() == direct.to_dict()
    assert reported.value is None
    assert reported.reason_code == "numeric_computation_failed"


def test_wrong_target_and_unrequested_component_are_not_applicable():
    diagnostic = ScorerWasserstein1Gap()
    wrong_target = _result(
        audit_scores(
            _evidence(),
            _spec(target_kind=TargetKind.BENCHMARK_DATASET),
            diagnostic=diagnostic,
        )
    )
    unrequested = _result(
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
    evidence = _evidence()
    spec = _spec(requested_components=("score_mean_gap", "score_rate_gap", COMPONENT))
    successful = audit_scores(
        evidence,
        spec,
        diagnostics=(
            ScorerMeanGap(),
            ScorerRateGap(
                transform=ScoreRateTransform(
                    event_name="score_at_or_above_half",
                    threshold=0.5,
                    direction="higher",
                    inclusive=True,
                )
            ),
            ScorerWasserstein1Gap(),
        ),
    )
    partial = audit_scores(
        evidence,
        spec,
        diagnostics=(ScorerMeanGap(), ScorerRateGap(), ScorerWasserstein1Gap()),
    )

    assert set(successful.components) == {
        "score_mean_gap",
        "score_rate_gap",
        COMPONENT,
    }
    assert successful.status is ReportStatus.SUCCESS
    assert partial.components["score_mean_gap"].status is DiagnosticStatus.READY
    assert partial.components["score_rate_gap"].status is DiagnosticStatus.BLOCKED
    assert partial.components[COMPONENT].status is DiagnosticStatus.READY
    assert partial.status is ReportStatus.PARTIAL
    assert successful.to_dict()["provenance"] == {
        "diagnostics": ["score_mean_gap", "score_rate_gap", COMPONENT]
    }


def test_registry_exports_default_wasserstein_component_and_schema_is_strict_json():
    diagnostic = get_diagnostic(COMPONENT)
    report = audit_scores(_evidence(), _spec(), diagnostic=diagnostic)
    payload = report.to_dict()

    assert COMPONENT in DIAGNOSTIC_REGISTRY
    assert COMPONENT in list_diagnostics()
    assert isinstance(diagnostic, ScorerWasserstein1Gap)
    assert diagnostic == ScorerWasserstein1Gap()
    assert payload["schema_version"] == "1.7"
    assert math.isfinite(payload["components"][COMPONENT]["value"])
    assert report.to_json(indent=None) == report.to_json(indent=None)
    json.dumps(payload, allow_nan=False, sort_keys=True)


def test_trustgpt_toxicity_frozen_row_level_wasserstein_regression():
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
    spec = DatasetAuditSpec(
        target_name=fixture["dataset"],
        target_kind="score_table",
        task_family="scored_rows",
        design_stance="stress_test",
        references={},
        requested_components=(COMPONENT,),
    )
    result = _result(audit_scores(evidence, spec, diagnostic=ScorerWasserstein1Gap()))
    expected = fixture["expected_wasserstein_1"]

    assert result.status.value == expected["status"]
    assert result.value == expected["value"]
    assert result.details == _frozen_json_shape(expected["details"])
