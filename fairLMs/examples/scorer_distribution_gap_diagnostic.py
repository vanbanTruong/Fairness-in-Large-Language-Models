"""Wasserstein-1 audit for an unfamiliar row-level score schema."""

from fairLMs.datasets.diagnostics import (
    DatasetAuditSpec,
    ScoredGroups,
    ScorerWasserstein1Gap,
    audit_scores,
)


def main() -> None:
    # Both groups have mean 0.5 and a 0.5 rate under score >= 0.5. Their
    # empirical distributions still differ by W1 = 0.25 score units.
    rows = [
        {"segment__code_v11": "external/A7", "distribution_score__29": 0.0},
        {"segment__code_v11": "external/T9", "distribution_score__29": 0.25},
        {"segment__code_v11": "external/A7", "distribution_score__29": 1.0},
        {"segment__code_v11": "external/T9", "distribution_score__29": 0.75},
    ]
    evidence = ScoredGroups.from_records(
        rows,
        axis="unfamiliar-cohort-axis",
        group_field="segment__code_v11",
        score_field="distribution_score__29",
        support=("amber", "teal"),
        score_name="external-distribution-channel",
        source="Unregistered row-level score export v6",
        value_map={
            "external/A7": "amber",
            "external/T9": "teal",
        },
        # The range validates scores; W1 remains unnormalized native-score distance.
        score_range=(0.0, 1.0),
        provenance={
            "table_id": "outside-catalog-score-table-63",
            "unit_of_analysis": "one scored benchmark row",
            "group_label_source": "pre-existing segment__code_v11 field",
            "score_source": "pre-existing distribution_score__29 field",
        },
    )
    spec = DatasetAuditSpec(
        target_name="unregistered-score-table",
        target_kind="score_table",
        task_family="scored_rows",
        design_stance="stress_test",
        references={},
        requested_components=("score_wasserstein_1_gap",),
    )

    report = audit_scores(
        evidence,
        spec,
        diagnostic=ScorerWasserstein1Gap(),
    )
    result = report.components["score_wasserstein_1_gap"]
    if result.status.value == "ready":
        pairwise = [dict(pair) for pair in result.details["pairwise_wasserstein_1"]]
        print(f"status=ready value={result.value:.12f} {result.details['unit']}")
        print("group_counts=", dict(result.details["group_counts"]))
        print("pairwise_wasserstein_1=", pairwise)
        print("argmax_groups=", tuple(result.details["argmax_groups"]))
        print("estimator=", result.details["estimator"])
        print("ground_metric=", result.details["ground_metric"])
        print("directionality=", result.details["directionality"])
        print("normalization=", result.details["normalization"])
    else:
        print(result.status.value, result.reason_code, result.reason)
    print(report.to_json())


if __name__ == "__main__":
    main()
