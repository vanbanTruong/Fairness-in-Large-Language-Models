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

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: builds queries, calls the Completions API, writes summary + detail CSVs |
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

```bash
pip install openai pandas numpy datasets
export OPENAI_API_KEY=...
cd <this directory>
python main.py
```

Optional: `gender-guesser` for MTV base-rate logging.

## Output \& Results

`sns_results.csv`: `dataset`, `snsr_score`, `snsv_score`.
