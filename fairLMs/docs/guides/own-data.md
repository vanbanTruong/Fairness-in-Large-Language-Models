# Bring your own data

Evidence you prepare yourself is indistinguishable from evidence a shipped
loader produces: both arrive as the `data` argument of `compute`, and there is
no adapter layer to write.

Two input styles work:

1. **A sequence of mappings**, what the loaders return. Pair metrics want dicts
   with `stereotype` and `anti_stereotype` keys; a plain list of dicts is
   accepted directly.
2. **A validated container** from `fairLMs.definitions.data`, for metrics that need
   more structure than a flat sequence, such as the four role sets WEAT
   requires.

## Choosing a container

| Container | Fields | Metrics that accept it |
|---|---|---|
| `WordSets` | `target_1`, `target_2`, `attribute_1`, `attribute_2` | WEAT, SEAT, CEAT |
| `ContextSets` | same four roles, as sentence contexts | SEAT, CEAT |
| `VectorSets` | same four roles, as precomputed vectors | WEAT, SEAT, CEAT |
| `SentenceTriples` | `stereotype`, `anti_stereotype`, `unrelated` | `context_association_test` |
| `StereotypeLabelled` | two `LabeledSentences` (stereotype / anti-stereotype) | `stereotypical_divergence` |
| `ContrastSpec` | `group_terms`, `contrast_pairs`, `templates` | `contrast_based_score` |
| `GroupWordPairs` | `group_1`, `group_2` | `discovery_of_correlations` |
| `GroupPredictions` | `y_true`, `y_pred`, `groups` | `equal_opportunity_gap`, `fair_inference_score`, `context_based_disparity` |
| `ScorePair` | `stereotype`, `counter_stereotype` | `accuracy_disparity` |
| `GroupProperties` | `groups`, `properties`, `ab_template`, `rb_template` | `bias_amplifier` |
| `QuerySpec` | `queries`, `neutral_prompt_fn`, `group_prompt_fn`, `group_values` | `sensitive_name_similarity` |
| `DemographicPrompts` | `prompts`, `stereotype_words`, `counter_words`, `neutral_words` | demographic-representation family |
| `PromptPairs` | `factual`, `counterfactual` | counterfactual-fairness family |
| `ProbeSet` | `probes` | `gradient_based_bias_estimation`, `natural_indirect_effect` |
| `LabeledSentences` | `sentences`, `labels`, `pair_ids` | encoder-decoder extrinsic family |
| `OccupationTriples` | `triples` | `stereotypical_log_likelihood` |
| `ConceptSpec` | `concepts`, `prompt_template`, `group_terms` | `cooccurrence_association` |

Not every metric wants a container. `log_probability_bias_score`, for instance,
takes a plain sequence of attribute words (professions) and carries its
templates and group tokens as constructor configuration instead. The
authoritative contract for each metric is its own docstring in the
[API reference](../api/metrics.md); the generated
[Metrics registry](../registry/metrics.md) lists every registry name.

## Validation happens at construction

Containers validate their own structure when you build them, so malformed
evidence fails at the point you created it rather than deep inside a metric:

```python
from fairLMs.definitions.data import WordSets

WordSets(
    target_1=["nurse", "teacher"],
    target_2=["engineer", "lawyer"],
    attribute_1=["she", "her"],
    attribute_2=["he", "him"],
)
```

Metrics additionally reject keyword arguments they do not understand, so a
misspelled parameter raises `TypeError` instead of silently using a default.

## Worked example: pairs from a CSV

```python
import pandas as pd
from fairLMs.definitions import CrowSPairsScore
from fairLMs.definitions.models import HuggingFaceModel

df = pd.read_csv("my_pairs.csv")     # columns: stereotype, anti_stereotype, bias_type

pairs = df.to_dict("records")

model = HuggingFaceModel("bert-base-uncased", task="mlm")
result = CrowSPairsScore().compute(model, pairs)
print(result.score, result.by_category)
```

## Worked example: predictions you already have

Five metrics need no model at all, so pass the predictions directly:

```python
from fairLMs.definitions import EqualOpportunityGap
from fairLMs.definitions.data import GroupPredictions

data = GroupPredictions(
    y_true=[1, 0, 1, 1],
    y_pred=[1, 0, 0, 1],
    groups=["A", "A", "B", "B"],
)
EqualOpportunityGap().compute(None, data)
```
