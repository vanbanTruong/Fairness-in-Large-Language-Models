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

## Public API

```python
from fairLMs.definitions import CounterfactualFairnessScore
from fairLMs.definitions.models import OpenAIModel

result = CounterfactualFairnessScore().compute(
    model=OpenAIModel(),
    factual_prompts=["a male, age: 30, job: clerk"],
    counterfactual_prompts=["a female, age: 30, job: clerk"],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `CounterfactualFairnessScore`; writes results CSV for continuity |
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

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definitions.decoder_only.extrinsic_bias.counterfactual_fairness.ctf.main
```

## Output \& Results

`ctf_results.csv`: `dataset`, `ctf_score`.

## Caveat

For Heart Disease, sex can be a real clinical risk factor — a likelihood shift
may reflect correct medical dependence, not unfairness.
