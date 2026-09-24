"""Representativeness diagnostic for an unregistered synthetic benchmark."""

from fairLMs.datasets.diagnostics import (
    DatasetAuditSpec,
    ReferenceDistribution,
    RepresentationEvidence,
    audit_representativeness,
)


def main():
    evidence = RepresentationEvidence(
        axis="community",
        counts={"amber": 3, "teal": 1},
        source="Synthetic example benchmark",
    )
    reference = ReferenceDistribution(
        axis="community",
        probabilities={"amber": 0.5, "teal": 0.5},
        source="Synthetic design specification",
        purpose="design_target",
        population="Intended synthetic benchmark composition",
    )
    spec = DatasetAuditSpec(
        target_name="outside-catalog-example",
        target_kind="benchmark_dataset",
        task_family="free_text",
        design_stance="stress_test",
        references={"community": reference},
    )

    report = audit_representativeness(evidence, spec)
    result = report.components["b_rep"]
    print(f"status={result.status.value} value={result.value:.12f}")
    print(report.to_json())


if __name__ == "__main__":
    main()
