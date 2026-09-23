# GBE — Gradient-based Bias Estimation

Estimates how much each GPT-2 attention **Value head** contributes to a SEAT
effect size by backpropagating `|d|` into per-head Value masks. Reports the
**mass** of positive gradients (normalized absolute mass on positive entries).
Chance baseline for the related sign-proportion statistic is **0.5**; a
permutation null tests significance. The CSV stores the mass statistic as
`gbe_score`.

```
d        = SEAT Cohen's d on sentence embeddings
loss     = |d|
GBE_mass = Σ max(m,0) / Σ |m|     over head-mask gradients m
```

## Public API

```python
from fairLMs.metrics import GradientBasedBiasEstimation
from fairLMs.models import HuggingFaceModel

result = GradientBasedBiasEstimation().compute(
    model=HuggingFaceModel("gpt2", task="causal"),
    X=["John"],
    Y=["Mary"],
    A=["career"],
    B=["family"],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `GradientBasedBiasEstimation`; writes results CSV for continuity |
| `gbe.py` | Core: `install_head_masks`, `seat_effect_size`, `compute_gbe_matrix`, `compute_gbe`, `compute_gbe_mass`, `gbe_permutation_null` |
| `red_pill_corpus.csv` | Optional Red Pill vocabulary source |
| `gbe_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `gpt2-medium` (`attn_implementation="eager"`, `model.double()`) | `main.py` |
| Seed / null | `SEED = 42`; `N_PERM = 50`; `PERMUTE_MODE = "attributes"` | `main.py` |
| Red Pill | `REDPILL_NROWS = 200000`, `REDPILL_MIN_FREQ = 100`, `REDPILL_TOP_GENDER = 8` | `main.py` |
| Datasets | Hardcoded StereoSet / Winogender word lists; Red Pill vocab mined from corpus `body` column | `main.py` |

## How to run

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definition.decoder_only.intrinsic_bias.attention_head_based_disparity.gbe.main
```

## Output \& Results

`gbe_results.csv`: `dataset`, `gbe_score` (observed gradient mass).
