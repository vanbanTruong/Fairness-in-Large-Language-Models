# NIE — Natural Indirect Effect (Attention-Head Mediation)

Causal mediation via attention-head activation patching on GPT-2. For each
head, average the relative change in the stereo/anti odds ratio when that
head's activations are intervened:

```
y     = exp(ℓ_anti − ℓ_stereo)
NIEₕ  = mean( (y_int / y_base) − 1 ) over prompts
score = fraction of heads with |NIEₕ| > threshold
```

Default threshold is `NIE_THRESHOLD = 0.003` (fixed; not a top-decile cut).

## Public API

```python
from fairLMs.metrics import NaturalIndirectEffect
from fairLMs.models import HuggingFaceModel

result = NaturalIndirectEffect().compute(
    model=HuggingFaceModel("gpt2", task="causal"),
    probes=[("The nurse said", "he", "she")],
    N_LAYERS=12,
    N_HEADS=12,
    HEAD_DIM=64,
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `NaturalIndirectEffect`; writes results CSV for continuity |
| `nie.py` | Core: `_capture_acts`, `compute_nie_matrix`, `compute_nie` |
| `red_pill_corpus.csv` | Red Pill comment corpus |
| `nie_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `gpt2-medium` | `main.py` → `MODEL_NAME` |
| Sample cap | `N_MAX = 200`; `SEED = 42` | `main.py` |
| Threshold | `NIE_THRESHOLD = 0.003` | `main.py` |
| Red Pill | `REDPILL_NROWS = 150000` | `main.py` |
| Datasets | StereoSet profession (target → “person” CF); Winogender (occupation → anti-stereo noun); Red Pill comments with occupation cues | `main.py` |

Hooks target GPT-2 `transformer.h[*].attn.c_proj`.

## How to run

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definition.decoder_only.intrinsic_bias.attention_head_based_disparity.nie.main
```

## Output \& Results

`nie_results.csv`: `dataset`, `nie_score`.
