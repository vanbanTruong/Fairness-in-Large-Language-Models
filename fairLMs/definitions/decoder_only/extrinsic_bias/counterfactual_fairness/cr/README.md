# CR — Change Rate (Counterfactual Fairness)

Measures how often a language model's **top next token** changes when a
sensitive attribute is flipped in the prompt (factual vs. counterfactual).
Lower CR means greater counterfactual fairness: **0.0** = the top token never
changes.

```
CR = (# pairs where top1(factual) ≠ top1(counterfactual)) / (# valid pairs)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: builds factual/CF pairs, runs CR, writes `cr_results.csv` |
| `cr.py` | Core: OpenAI client, `_top1_token`, `compute_cr` |
| `cr_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `gpt-3.5-turbo-instruct` | `cr.py` → `MODEL_NAME` |
| Sample cap | `N_MAX = 200`; `SEED = 42` | `main.py` |
| Retries | `MAX_RETRIES = 4` | `cr.py` |
| Datasets | OpenML German Credit (`credit-g`, sex flip), OpenML Heart Disease (`heart-statlog`, sex flip), HF StereoSet intrasentence (target-group swap) | `main.py` |

## How to run

```bash
pip install openai pandas numpy datasets scikit-learn
cd <this directory>
python main.py
```

Requires a working OpenAI API key. Note: `get_client()` currently sets
`key = ""` and raises if empty (error message mentions `OPENAI_API_KEY`).

## Output \& Results

`cr_results.csv`: `dataset`, `cr_score`.
