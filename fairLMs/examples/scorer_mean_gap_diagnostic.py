"""Scoring-instrument audit for an unfamiliar row-level score schema."""

from fairLMs.diagnostics import DatasetAuditSpec, ScoredGroups, audit_scores


def main() -> None:
    # Both the group codes and scores already exist. The adapter only validates
    # and maps these explicitly named fields; it does not derive either value.
    rows = [
        {"segment__code_v9": "external/A7", "opaque_score__17": 0.1},
        {"segment__code_v9": "external/T9", "opaque_score__17": 0.4},
        {"segment__code_v9": "external/V2", "opaque_score__17": 0.8},
        {"segment__code_v9": "external/A7", "opaque_score__17": 0.3},
        {"segment__code_v9": "external/T9", "opaque_score__17": 0.6},
        {"segment__code_v9": "external/V2", "opaque_score__17": 1.0},
    ]

    evidence = ScoredGroups.from_records(
        rows,
        axis="unfamiliar-cohort-axis",
        group_field="segment__code_v9",
        score_field="opaque_score__17",
        support=("amber", "teal", "violet"),
        score_name="external-safety-channel",
        source="Unregistered row-level score export v4",
        value_map={
            "external/A7": "amber",
            "external/T9": "teal",
            "external/V2": "violet",
        },
        score_range=(0.0, 1.0),
        provenance={
            "table_id": "outside-catalog-score-table-44",
            "unit_of_analysis": "one scored benchmark row",
            "group_label_source": "pre-existing segment__code_v9 field",
            "score_source": "pre-existing opaque_score__17 field",
        },
    )
    spec = DatasetAuditSpec(
        target_name="unregistered-score-table",
        target_kind="score_table",
        task_family="scored_rows",
        design_stance="stress_test",
        references={},
        requested_components=("score_mean_gap",),
    )

    report = audit_scores(evidence, spec)
    result = report.components["score_mean_gap"]
    if result.status.value == "ready":
        print(f"status=ready value={result.value:.12f} {result.details['unit']}")
        print("group_means=", dict(result.details["group_means"]))
        print("argmax_groups=", tuple(result.details["argmax_groups"]))
    else:
        print(result.status.value, result.reason_code, result.reason)
    print(report.to_json())


if __name__ == "__main__":
    main()
