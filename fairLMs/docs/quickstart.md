# Quickstart

Three things worth seeing in order: a metric on a shipped benchmark, the same
metric on data you built yourself, and a dataset audit that refuses to
manufacture a number.

## 1. A shipped benchmark

```python
from fairLMs.datasets import CrowSPairs
from fairLMs.metrics import CrowSPairsScore
from fairLMs.models import HuggingFaceModel

model = HuggingFaceModel("bert-base-uncased", task="mlm")
data = CrowSPairs(n_max=50)
metric = CrowSPairsScore()

result = metric.compute(model, data)
print(result.score)         # percentage of pairs preferring the stereotype
print(result.details)       # {'accuracy': ..., 'n_pairs': 50}
print(result.by_category)   # per-bias-type breakdown
```

`task="mlm"` selects the masked-LM head this metric needs. See
[Models](api/models.md) for the five task labels.

Metrics can also be looked up by registry name:

```python
from fairLMs.metrics import get_metric, list_metrics

list_metrics()                      # all 33 registry names
metric = get_metric("crows_pairs_score")
```

## 2. Your own data, same call

Pair metrics accept any sequence of mappings with `stereotype` and
`anti_stereotype` keys, so a list of dicts works with no adapter code:

```python
pairs = [
    {
        "stereotype": "The nurse said she was tired.",
        "anti_stereotype": "The nurse said he was tired.",
        "bias_type": "gender",
    }
]

result = metric.compute(model, pairs)      # identical call
```

For metrics that need more structure than a flat sequence, build a validated
container from [`fairLMs.metrics.data`](api/containers.md). Containers check
their own shape at construction, so malformed evidence fails immediately
instead of deep inside a metric:

```python
from fairLMs.data import weat_c1          # bundled WEAT stimulus set
from fairLMs.metrics import WEAT

encoder = HuggingFaceModel("bert-base-uncased", task="encoder")
WEAT().compute(encoder, weat_c1)
```

## 3. Metrics that need no model

Five metrics score predictions you already have, and also exist as plain
functions in the style of `sklearn.metrics`:

```python
from fairLMs.metrics import equal_opportunity_gap, accuracy_disparity

equal_opportunity_gap(y_true, y_pred, groups, g1="A", g2="B")   # -> float
accuracy_disparity(scores_stereotype, scores_counter)           # -> float
```

Use the class form (`EqualOpportunityGap`, …) when you want the full
`MetricResult` rather than a bare float.

## 4. Unavailable is not zero

Dataset diagnostics are a separate, dataset-first API: they consume explicit
evidence and an audit spec rather than a model, and report applicability as a
first-class outcome.

```python
from fairLMs.diagnostics import (
    DatasetAuditSpec,
    ScoredGroups,
    ScorerMeanGap,
    ScorerRateGap,
    audit_scores,
)

evidence = ScoredGroups(
    axis="cohort",
    groups=("amber", "amber", "teal", "teal"),
    scores=(0.1, 0.3, 0.8, 1.0),
    score_name="example_safety_score",
    source="Existing row-level score export v1",
    score_range=(0.0, 1.0),
)
spec = DatasetAuditSpec(
    target_name="example-score-table",
    target_kind="score_table",
    task_family="scored_rows",
    design_stance="stress_test",
    references={},
    requested_components=("score_mean_gap", "score_rate_gap"),
)

report = audit_scores(evidence, spec, diagnostics=(ScorerMeanGap(), ScorerRateGap()))

print(report.components["score_mean_gap"].status.value)   # ready
print(report.components["score_mean_gap"].value)          # 0.7

# score_rate_gap needs an explicit score-to-event rule, which was not supplied:
print(report.components["score_rate_gap"].status.value)   # blocked
print(report.components["score_rate_gap"].value)          # None, not 0.0
```

A missing threshold rule is `blocked`, not a gap of zero. That distinction is
the defining commitment of the diagnostics layer.

## Next

- [Bring your own data](guides/own-data.md)
- [Auditing a dataset](guides/dataset-audit.md)
- [Writing a metric](guides/custom-metric.md)
