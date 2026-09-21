# CTF — Counterfactual Token Fairness

Measures the mean **total variation distance (TVD)** between next-token
distributions on factual vs. counterfactual prompts. **0** = counterfactually
fair (identical distributions); higher = larger distributional shift when the
sensitive attribute flips.

```
TVD(p, q) = min(1, ½ Σ_k |p_k − q_k| + residual mass)
CTF       = mean TVD over valid pairs
```

Distributions use the Completions API top logprobs (`TOP_LOGPROBS = 5`) plus
residual probability mass for unseen tokens.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: builds factual/CF pairs, runs CTF, writes `ctf_results.csv` |
| `ctf.py` | Core: `next_token_dist`, `tvd`, `compute_ctf` |
| `ctf_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `gpt-3.5-turbo-instruct` | `ctf.py` → `MODEL_NAME` |
| Top logprobs | 5 | `ctf.py` → `TOP_LOGPROBS` |
| Sample cap | `N_MAX = 200`; `SEED = 42` | `main.py` |
| Datasets | OpenML German Credit (sex flip), OpenML Heart Disease (sex flip), HF StereoSet intrasentence (target-group swap) | `main.py` |

## How to run

```bash
pip install openai pandas numpy datasets scikit-learn
cd <this directory>
python main.py
```

`get_client()` currently reads `HF_TOKEN` from the environment while the error
message says to set `OPENAI_API_KEY`.

## Output \& Results

`ctf_results.csv`: `dataset`, `ctf_score`.

## Caveat

For Heart Disease, sex can be a real clinical risk factor — a likelihood shift
may reflect correct medical dependence, not unfairness.
