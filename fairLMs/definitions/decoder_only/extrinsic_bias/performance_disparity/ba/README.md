# BA — BiasAsker Absolute \& Relative Bias

BiasAsker-style scores over group × property prompts using forced-choice /
continuation log-probabilities:

- **AB (absolute bias)** — consistency of pairwise group preferences under
  forward and reverse forced choice. Fair ≈ **0.5**.
- **RB (relative bias)** — variance of yes-agreement rates across groups for
  each property, then averaged. Fair = **0.0**.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: builds prompts from BiasAsker / MTV / NQ, writes `ba_results.csv` |
| `ba.py` | Core: `continuation_logprob`, `forced_choice`, `_favours`, `_agrees`, `compute_ba` |
| `groups.csv`, `bias_annotation.csv` | Local BiasAsker support files |
| `ba_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `davinci-002` | `ba.py` → `MODEL_NAME` |
| Sample cap | `N_MAX = 100`; `SEED = 42` | `main.py` |
| Datasets | BiasAsker predicates (age old/young); MTV genres (male/female); NQ occupation roles (male/female) | `main.py` |

## How to run

```bash
pip install openai pandas numpy datasets
export OPENAI_API_KEY=...
cd <this directory>
python main.py
```

## Output \& Results

`ba_results.csv`: `dataset`, `ab_score`, `rb_score`.
