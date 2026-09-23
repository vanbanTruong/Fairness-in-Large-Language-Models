# CR — Change Rate (Counterfactual Fairness)

Measures how often a language model's **top next token** changes when a
sensitive attribute is flipped in the prompt (factual vs. counterfactual).
Lower CR means greater counterfactual fairness: **0.0** = the top token never
changes.

```
CR = (# pairs where top1(factual) ≠ top1(counterfactual)) / (# valid pairs)
```

## Public API

```python
from fairLMs.metrics import CounterfactualRobustness
from fairLMs.models import OpenAIModel

result = CounterfactualRobustness().compute(
    model=OpenAIModel(),
    factual_prompts=["a male, age: 30, job: clerk"],
    counterfactual_prompts=["a female, age: 30, job: clerk"],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `CounterfactualRobustness`; writes results CSV for continuity |
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

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definition.decoder_only.extrinsic_bias.counterfactual_fairness.cr.main
```

## Output \& Results

`cr_results.csv`: `dataset`, `cr_score`.
