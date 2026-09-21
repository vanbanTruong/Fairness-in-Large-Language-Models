# AD — Accuracy Disparity

Performance-disparity metric: absolute gap in mean accuracy between two
demographic groups.

```
AD = |Acc_s − Acc_s'|
```

Lower is more equitable. Scoring differs by dataset (stereotype “No”
preference, forced-choice genre accuracy, or token F1 on QA).

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: BiasAsker / MTV / NQ runners, writes `ad_summary.csv` |
| `ad.py` | OpenAI helpers + `compute_ad` |
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

```bash
pip install openai pandas numpy datasets gender-guesser
export OPENAI_API_KEY=...
cd <this directory>
python main.py
```

MTV loading requires network access to the gist URL.

## Output \& Results

`ad_summary.csv`: `dataset`, `ad_score`.
