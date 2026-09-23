# AD — Accuracy Disparity

Performance-disparity metric: absolute gap in mean accuracy between two
demographic groups.

```
AD = |Acc_s − Acc_s'|
```

Lower is more equitable. Scoring differs by dataset (stereotype “No”
preference, forced-choice genre accuracy, or token F1 on QA).

## Public API

```python
from fairLMs.metrics import AccuracyDisparity

result = AccuracyDisparity().compute(
    scores_s=[0.9, 0.8, 0.7],
    scores_sp=[0.6, 0.5, 0.4],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `AccuracyDisparity`; writes results CSV for continuity |
| `ad.py` | Core metric math (called by the public API): OpenAI helpers + `compute_ad` |
| `groups.csv`, `bias_annotation.csv` | Local BiasAsker support files |
| `ad_summary.csv` | Output written by the current runner |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `davinci-002` | `ad.py` → `MODEL_NAME` |
| Sample cap | `N_MAX = 1000`; `SEED = 42` | `main.py` |
| Age cutoffs | young ≤ 30, old ≥ 65 | `main.py` → `YOUNG_MAX`, `OLD_MIN` |
| Datasets | Local BiasAsker (age); MTV Music Artists gist CSV (gender via `gender-guesser`); HF Natural Questions (gender of answer entity; token F1) | `main.py` |

## How to run

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definition.decoder_only.extrinsic_bias.performance_disparity.ad.main
```

## Output \& Results

`ad_summary.csv`: `dataset`, `ad_score`.
