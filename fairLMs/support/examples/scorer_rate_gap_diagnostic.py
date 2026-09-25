"""Rate-gap audit for an unfamiliar row-level score schema."""

from fairLMs.datasets.diagnostics import (
    DatasetAuditSpec,
    ScoredGroups,
    ScoreRateTransform,
    ScorerRateGap,
    audit_scores,
)


def main() -> None:
    # Group codes and scores already exist. The adapter validates and maps these
    # fields; it does not infer a group from text or generate a score.
    rows = [
        {"segment__code_v9": "external/A7", "opaque_score__17": 0.1},
        {"segment__code_v9": "external/T9", "opaque_score__17": 0.4},
        {"segment__code_v9": "external/V2", "opaque_score__17": 0.8},
        {"segment__code_v9": "external/A7", "opaque_score__17": 0.3},
        {"segment__code_v9": "external/T9", "opaque_score__17": 0.5},
        {"segment__code_v9": "external/V2", "opaque_score__17": 1.0},
    ]
    evidence = ScoredGroups.from_records(
        rows,
        axis="unfamiliar-cohort-axis",
        group_field="segment__code_v9",
        score_field="opaque_score__17",
        support=("amber", "teal", "violet"),
        score_name="external-safety-channel",
        source="Unregistered row-level score export v5",
        value_map={
            "external/A7": "amber",
            "external/T9": "teal",
            "external/V2": "violet",
        },
        # This validates scores. It does not select or imply the threshold.
        score_range=(0.0, 1.0),
        provenance={
            "table_id": "outside-catalog-score-table-51",
            "unit_of_analysis": "one scored benchmark row",
            "group_label_source": "pre-existing segment__code_v9 field",
            "score_source": "pre-existing opaque_score__17 field",
        },
    )
    transform = ScoreRateTransform(
        event_name="score_at_or_above_policy_threshold",
        threshold=0.5,
        direction="higher",
        inclusive=True,
        provenance={
            "rule_source": "Example evaluation policy v1",
            "selection_timing": "fixed before comparing groups",
            "interpretation": "exactly 0.5 is included in the event",
        },
    )
    spec = DatasetAuditSpec(
        target_name="unregistered-score-table",
        target_kind="score_table",
        task_family="scored_rows",
        design_stance="stress_test",
        references={},
        requested_components=("score_rate_gap",),
    )

    report = audit_scores(
        evidence,
        spec,
        diagnostic=ScorerRateGap(transform=transform),
    )
    result = report.components["score_rate_gap"]
    if result.status.value == "ready":
        print(f"status=ready value={result.value:.12f} {result.details['unit']}")
        print("operator=", result.details["transform"]["operator"])
        print("paper_alignment=", result.details["transform"]["paper_alignment"])
        print("group_counts=", dict(result.details["group_counts"]))
        print("event_counts=", dict(result.details["event_counts"]))
        print("group_rates=", dict(result.details["group_rates"]))
        print("argmax_groups=", tuple(result.details["argmax_groups"]))
    else:
        print(result.status.value, result.reason_code, result.reason)
    print(report.to_json())


if __name__ == "__main__":
    main()
