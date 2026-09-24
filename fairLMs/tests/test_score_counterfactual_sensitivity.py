"""Contracts for paired counterfactual scorer sensitivity."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from fairLMs.datasets.diagnostics import (
    DIAGNOSTIC_REGISTRY,
    DatasetAuditSpec,
    DiagnosticStatus,
    PairedScores,
    ReportStatus,
    ScoredGroups,
    ScorerCounterfactualSensitivity,
    ScorerMeanGap,
    TargetKind,
    audit_scores,
    get_diagnostic,
    list_diagnostics,
)

AXIS = "unfamiliar-paired-intervention-axis"
COMPONENT = "score_counterfactual_sensitivity"


def _evidence(
    *,
    pair_ids=None,
    conditions=None,
    scores=None,
    condition_roles=("baseline", "intervention"),
    score_range=None,
):
    return PairedScores(
        axis=AXIS,
        pair_ids=(["p1", "p1", "p2", "p2"] if pair_ids is None else pair_ids),
        conditions=(
            ["baseline", "intervention", "baseline", "intervention"]
            if conditions is None
            else conditions
        ),
        scores=([1.0, 4.0, 10.0, 4.0] if scores is None else scores),
        condition_roles=condition_roles,
        score_name="unfamiliar-paired-score-channel",
        source="external paired score table v1",
        pairing_basis="Reviewed minimal identity-token substitutions, protocol v2",
        score_range=score_range,
        provenance={"table_id": "outside-catalog-paired-table-19"},
    )


def _spec(*, target_kind="score_table", requested_components=(COMPONENT,)):
    return DatasetAuditSpec(
        target_name="unregistered-paired-score-table",
        target_kind=target_kind,
        task_family="paired_sentences",
        design_stance="stress_test",
        references={},
        requested_components=requested_components,
    )


def _result(report):
    assert set(report.components) == {COMPONENT}
    return report.components[COMPONENT]


def test_known_value_and_frozen_detail_contract():
    result = _result(
        audit_scores(
            _evidence(),
            _spec(),
            diagnostic=ScorerCounterfactualSensitivity(),
        )
    )

    assert result.status is DiagnosticStatus.READY
    assert result.value == 4.5
    assert result.details == {
        "axis": AXIS,
        "score_name": "unfamiliar-paired-score-channel",
        "score_range": None,
        "sample_count": 4,
        "pair_count": 2,
        "condition_roles": ("baseline", "intervention"),
        "condition_counts": {"baseline": 2, "intervention": 2},
        "pairing_basis": ("Reviewed minimal identity-token substitutions, protocol v2"),
        "mean_absolute_difference": 4.5,
        "mean_signed_difference_second_minus_first": -1.5,
        "minimum_absolute_difference": 3.0,
        "maximum_absolute_difference": 6.0,
        "first_role_higher_pair_count": 1,
        "second_role_higher_pair_count": 1,
        "equal_score_pair_count": 0,
        "aggregation": "mean_absolute_within_pair_score_difference",
        "estimator": "empirical_uniform_mass_per_pair",
        "signed_difference_orientation": (
            "condition_roles[1]_minus_condition_roles[0]"
        ),
        "directionality": "symmetric_absolute_primary_value",
        "unit": "score_units",
    }
    assert result.provenance["evidence"]["pair_count"] == 2
    assert result.provenance["evidence"]["pairing_basis"].startswith("Reviewed")
    assert any("differ solely" in assumption for assumption in result.assumptions)
    assert any("cannot verify" in assumption for assumption in result.assumptions)
    assert any("point estimate" in assumption for assumption in result.assumptions)


def test_equal_scores_are_a_successful_zero_not_missing_evidence():
    report = audit_scores(
        _evidence(scores=[0.2, 0.2, -3.0, -3.0]),
        _spec(),
        diagnostic=ScorerCounterfactualSensitivity(),
    )
    result = _result(report)

    assert result.status is DiagnosticStatus.READY
    assert result.value == 0.0
    assert result.details["mean_signed_difference_second_minus_first"] == 0.0
    assert result.details["equal_score_pair_count"] == 2
    assert report.status is ReportStatus.SUCCESS


def test_each_complete_pair_receives_equal_mass():
    evidence = _evidence(
        pair_ids=["p1", "p1", "p2", "p2", "p3", "p3"],
        conditions=["baseline", "intervention"] * 3,
        scores=[0.0, 0.0, 5.0, 5.0, 0.0, 9.0],
    )
    result = _result(
        audit_scores(
            evidence,
            _spec(),
            diagnostic=ScorerCounterfactualSensitivity(),
        )
    )

    assert result.value == 3.0
    assert result.details["pair_count"] == 3
    assert result.details["estimator"] == "empirical_uniform_mass_per_pair"


def test_row_permutation_and_pair_id_renaming_do_not_change_the_result():
    baseline = _result(
        audit_scores(
            _evidence(),
            _spec(),
            diagnostic=ScorerCounterfactualSensitivity(),
        )
    )
    permuted_and_renamed = _result(
        audit_scores(
            _evidence(
                pair_ids=["renamed-z", "renamed-a", "renamed-z", "renamed-a"],
                conditions=["intervention", "baseline", "baseline", "intervention"],
                scores=[4.0, 10.0, 1.0, 4.0],
            ),
            _spec(),
            diagnostic=ScorerCounterfactualSensitivity(),
        )
    )

    assert permuted_and_renamed.value == baseline.value == 4.5
    assert (
        permuted_and_renamed.details["mean_signed_difference_second_minus_first"]
        == -1.5
    )


def test_global_role_order_swap_preserves_primary_value_and_flips_signed_detail():
    forward = _result(
        audit_scores(
            _evidence(),
            _spec(),
            diagnostic=ScorerCounterfactualSensitivity(),
        )
    )
    reversed_roles = _result(
        audit_scores(
            _evidence(condition_roles=("intervention", "baseline")),
            _spec(),
            diagnostic=ScorerCounterfactualSensitivity(),
        )
    )

    assert reversed_roles.value == forward.value == 4.5
    assert reversed_roles.details["mean_signed_difference_second_minus_first"] == 1.5
    assert reversed_roles.details["directionality"] == (
        "symmetric_absolute_primary_value"
    )


def test_translation_invariance_and_native_unit_scaling():
    base = _result(
        audit_scores(
            _evidence(),
            _spec(),
            diagnostic=ScorerCounterfactualSensitivity(),
        )
    )
    translated = _result(
        audit_scores(
            _evidence(scores=[101.0, 104.0, 110.0, 104.0]),
            _spec(),
            diagnostic=ScorerCounterfactualSensitivity(),
        )
    )
    scaled = _result(
        audit_scores(
            _evidence(scores=[2.5, 10.0, 25.0, 10.0]),
            _spec(),
            diagnostic=ScorerCounterfactualSensitivity(),
        )
    )

    assert translated.value == base.value == 4.5
    assert scaled.value == 2.5 * base.value == 11.25
    assert scaled.details["unit"] == "score_units"


def test_pair_alignment_matters_even_when_condition_marginals_are_identical():
    aligned = _result(
        audit_scores(
            _evidence(scores=[0.0, 0.0, 1.0, 1.0]),
            _spec(),
            diagnostic=ScorerCounterfactualSensitivity(),
        )
    )
    crossed = _result(
        audit_scores(
            _evidence(scores=[0.0, 1.0, 1.0, 0.0]),
            _spec(),
            diagnostic=ScorerCounterfactualSensitivity(),
        )
    )

    # Both condition roles have the same marginal multiset {0, 1}.
    assert aligned.value == 0.0
    assert crossed.value == 1.0


def test_unrequested_and_wrong_target_are_not_applicable():
    diagnostic = ScorerCounterfactualSensitivity()
    unrequested = _result(
        audit_scores(
            _evidence(),
            _spec(requested_components=("score_mean_gap",)),
            diagnostic=diagnostic,
        )
    )
    wrong_target = _result(
        audit_scores(
            _evidence(),
            _spec(target_kind=TargetKind.BENCHMARK_DATASET),
            diagnostic=diagnostic,
        )
    )

    assert unrequested.status is DiagnosticStatus.NOT_APPLICABLE
    assert unrequested.reason_code == "component_not_requested"
    assert wrong_target.status is DiagnosticStatus.NOT_APPLICABLE
    assert wrong_target.reason_code == "target_kind_not_supported"
    assert unrequested.value is wrong_target.value is None


def test_paired_evidence_requires_explicit_compatible_diagnostic_selection():
    with pytest.raises(TypeError, match="explicit paired diagnostic"):
        audit_scores(_evidence(), _spec())
    with pytest.raises(TypeError, match="PairedScores"):
        audit_scores(_evidence(), _spec(), diagnostic=ScorerMeanGap())

    grouped = ScoredGroups(
        axis="cohort",
        groups=("a", "b"),
        scores=(0.1, 0.2),
        score_name="score",
        source="grouped table",
    )
    with pytest.raises(TypeError, match="PairedScores"):
        audit_scores(
            grouped,
            _spec(),
            diagnostic=ScorerCounterfactualSensitivity(),
        )


def test_direct_class_and_report_entry_point_are_equivalent():
    diagnostic = ScorerCounterfactualSensitivity()
    direct = diagnostic.compute(_evidence(), _spec())
    through_report = _result(audit_scores(_evidence(), _spec(), diagnostic=diagnostic))

    assert direct == through_report
    assert diagnostic.plan(_evidence(), _spec()).status is DiagnosticStatus.READY


def test_unrepresentable_within_pair_difference_is_explicit_failure():
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    result = _result(
        audit_scores(
            _evidence(
                pair_ids=["p1", "p1"],
                conditions=["baseline", "intervention"],
                scores=[-maximum, maximum],
            ),
            _spec(),
            diagnostic=ScorerCounterfactualSensitivity(),
        )
    )

    assert result.status is DiagnosticStatus.FAILED
    assert result.value is None
    assert result.reason_code == "numeric_computation_failed"
    assert result.details["unrepresentable_pair"]["pair_id"] == "p1"


def test_positive_mean_too_small_for_a_nonzero_float_is_explicit_failure():
    minimum_subnormal = float.fromhex("0x0.0000000000001p-1022")
    result = _result(
        audit_scores(
            _evidence(scores=[0.0, minimum_subnormal, 0.0, 0.0]),
            _spec(),
            diagnostic=ScorerCounterfactualSensitivity(),
        )
    )

    assert result.status is DiagnosticStatus.FAILED
    assert result.value is None
    assert result.reason_code == "numeric_computation_failed"
    assert "too small" in result.reason


def test_registry_exports_default_component_and_schema_is_strict_json_1_4():
    diagnostic = get_diagnostic(COMPONENT)
    report = audit_scores(_evidence(), _spec(), diagnostic=diagnostic)
    payload = report.to_dict()

    assert COMPONENT in DIAGNOSTIC_REGISTRY
    assert COMPONENT in list_diagnostics()
    assert isinstance(diagnostic, ScorerCounterfactualSensitivity)
    assert diagnostic == ScorerCounterfactualSensitivity()
    assert payload["schema_version"] == "1.7"
    assert math.isfinite(payload["components"][COMPONENT]["value"])
    assert report.to_json(indent=None) == report.to_json(indent=None)
    json.dumps(payload, allow_nan=False, sort_keys=True)


def test_bbq_age_toxicity_frozen_paper_counterfactual_regression():
    fixture_path = (
        Path(__file__).parent
        / "data"
        / "golden"
        / "diagnostics"
        / "score_counterfactual_sensitivity"
        / "bbq_age_toxicity_v1.json"
    )
    fixture_text = fixture_path.read_text(encoding="utf-8")
    fixture = json.loads(fixture_text)
    config = fixture["input"]
    records = [dict(zip(config["columns"], row, strict=True)) for row in config["rows"]]

    evidence = PairedScores.from_records(
        records,
        axis=config["axis"],
        pair_id_field=config["pair_id_field"],
        condition_field=config["condition_field"],
        score_field=config["score_field"],
        condition_roles=config["condition_roles"],
        score_name=config["score_name"],
        source=config["source"],
        pairing_basis=config["pairing_basis"],
        score_range=config["score_range"],
        provenance=config["provenance"],
    )
    spec = DatasetAuditSpec(**fixture["spec"])
    report = audit_scores(
        evidence,
        spec,
        diagnostic=ScorerCounterfactualSensitivity(),
    )
    result = _result(report)
    expected = fixture["expected"]

    assert evidence.pair_count == expected["pair_count"] == 100
    assert result.status.value == expected["status"] == "ready"
    assert result.value == expected["value"] == 0.0052965244199999996
    assert result.details["mean_absolute_difference"] == expected["value"]
    assert result.details["mean_signed_difference_second_minus_first"] == (
        expected["mean_signed_difference_second_minus_first"]
    )
    assert result.details["minimum_absolute_difference"] == (
        expected["minimum_absolute_difference"]
    )
    assert result.details["maximum_absolute_difference"] == (
        expected["maximum_absolute_difference"]
    )
    assert result.details["first_role_higher_pair_count"] == 94
    assert result.details["second_role_higher_pair_count"] == 6
    assert result.details["equal_score_pair_count"] == 0
    assert fixture["oracle"]["replayable_in_ci"] is False
    assert fixture_text.count("/Users/") == 0
