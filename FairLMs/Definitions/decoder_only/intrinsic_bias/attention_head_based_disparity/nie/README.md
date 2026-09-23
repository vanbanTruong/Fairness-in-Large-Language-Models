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

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: builds StereoSet / Winogender / Red Pill prompts, writes `nie_results.csv` |
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

```bash
pip install torch transformers datasets pandas numpy
cd <this directory>
python main.py
```

## Output \& Results

`nie_results.csv`: `dataset`, `nie_score`.
