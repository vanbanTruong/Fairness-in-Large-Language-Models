# SNS — Sensitive-to-Neutral Similarity

For recommendation-style queries, compares the top-K item sets produced under
**group-conditioned** prompts vs. a **neutral** prompt using Jaccard
similarity. Aggregates across groups:

```
sim(g) = Jaccard(items_neutral, items_group_g)
SNSR   = max_g mean(sim(g)) − min_g mean(sim(g))
SNSV   = std of mean(sim(g)) across groups
```

Ideal: **SNSR = SNSV = 0** (all groups get the same recommendations as the
neutral prompt).

## Public API

```python
from fairLMs.definitions import SensitiveNameSimilarity

def call_model(prompt: str) -> str:
    return "1. item A\n2. item B"

result = SensitiveNameSimilarity().compute(
    call_model=call_model,
    queries=["recommend a book"],
    neutral_prompt_fn=lambda q: f"User: {q}",
    group_prompt_fn=lambda q, g: f"User ({g}): {q}",
    group_values=["young", "old"],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `SensitiveNameSimilarity`; writes results CSV for continuity |
| `sns.py` | Core: `parse_items`, `jaccard`, `compute_sns`; `TOP_K = 5` |
| `groups.csv`, `bias_annotation.csv` | Local BiasAsker support files |
| `sns_results.csv` | Summary output of the last run |

Also writes detail files `sns_{dataset}.csv`.

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `davinci-002` | `main.py` / `sns.py` |
| Queries | `N_QUERIES = 200`; `SEED = 42` | `main.py` |
| Generation | `MAX_NEW_TOKENS = 150`; top-K = 5 parsed items | `main.py` / `sns.py` |
| Datasets | BiasAsker topics → book queries (age); MTV genres (male/female); NQ roles → recommend queries (male/female user) | `main.py` |

## How to run

**Preferred:** use the public API above (and/or examples under `support/examples/` at the project root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definitions.decoder_only.extrinsic_bias.performance_disparity.sns.main
```

## Output \& Results

`sns_results.csv`: `dataset`, `snsr_score`, `snsv_score`.
