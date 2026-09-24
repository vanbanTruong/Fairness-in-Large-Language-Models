"""Offline comparison of explicit predictions with preserved metric config."""

from fairLMs.definitions import EqualOpportunityGap, GroupPredictions
from fairLMs.mitigation import MetricEvaluation, compare_before_after

before = GroupPredictions([1, 1, 1, 1], [1, 1, 1, 0], ["A", "A", "B", "B"])
after = GroupPredictions([1, 1, 1, 1], [1, 1, 1, 1], ["A", "A", "B", "B"])
report = compare_before_after(
    None,
    None,
    metrics={
        "equal_opportunity": MetricEvaluation(
            metric=EqualOpportunityGap(g1="A", g2="B", positive_label=1),
            data=before,
            after_data=after,
        ),
    },
)
print(report.to_json())
