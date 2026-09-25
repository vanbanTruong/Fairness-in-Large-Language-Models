"""Paired scorer sensitivity for an unfamiliar result-table schema."""

from fairLMs.datasets.diagnostics import (
    DatasetAuditSpec,
    PairedScores,
    ScorerCounterfactualSensitivity,
    audit_scores,
)


def main() -> None:
    rows = [
        {"match__key_41": "case-2", "variant__code_8": "swap", "score__23": 0.4},
        {"match__key_41": "case-1", "variant__code_8": "base", "score__23": 0.2},
        {"match__key_41": "case-3", "variant__code_8": "swap", "score__23": 0.1},
        {"match__key_41": "case-1", "variant__code_8": "swap", "score__23": 0.5},
        {"match__key_41": "case-2", "variant__code_8": "base", "score__23": 0.8},
        {"match__key_41": "case-3", "variant__code_8": "base", "score__23": 0.1},
    ]
    evidence = PairedScores.from_records(
        rows,
        axis="declared-identity-intervention",
        pair_id_field="match__key_41",
        condition_field="variant__code_8",
        score_field="score__23",
        condition_roles=("baseline", "identity_swap"),
        score_name="external-safety-channel",
        source="Unregistered paired score export v3",
        pairing_basis=(
            "Human-reviewed template pairs differing only in the declared "
            "identity token"
        ),
        condition_map={"base": "baseline", "swap": "identity_swap"},
        score_range=(0.0, 1.0),
        provenance={
            "table_id": "outside-catalog-paired-table-19",
            "unit_of_analysis": "one scored text variant",
            "pair_review_protocol": "minimal-contrast-review-v2",
            "score_source": "pre-existing score__23 field",
        },
    )
    spec = DatasetAuditSpec(
        target_name="unregistered-paired-score-table",
        target_kind="score_table",
        task_family="paired_sentences",
        design_stance="stress_test",
        references={},
        requested_components=("score_counterfactual_sensitivity",),
    )

    report = audit_scores(
        evidence,
        spec,
        diagnostic=ScorerCounterfactualSensitivity(),
    )
    result = report.components["score_counterfactual_sensitivity"]
    if result.status.value == "ready":
        print(f"status=ready value={result.value:.12f} {result.details['unit']}")
        print("pair_count=", result.details["pair_count"])
        print("condition_roles=", tuple(result.details["condition_roles"]))
        print(
            "signed_second_minus_first=",
            result.details["mean_signed_difference_second_minus_first"],
        )
        print("pairing_basis=", result.details["pairing_basis"])
    else:
        print(result.status.value, result.reason_code, result.reason)
    print(report.to_json())


if __name__ == "__main__":
    main()
