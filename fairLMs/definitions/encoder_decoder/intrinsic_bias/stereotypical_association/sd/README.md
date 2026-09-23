# SD — Stereotype Disparity (Cue Preference)

Measures whether the model prefers stereotypical vs. anti-stereotypical
referents when scoring French gender/age **cue phrases** under the source
sentence (no free-form translation decode). For each English source, candidate
French cues are scored by sequence log-probability (negative seq2seq loss);
the highest-scoring cue decides the predicted label. Accuracy vs. gold is
averaged on stereo and anti sets:

```
ΔS = mean(accuracy on anti-stereotypical) − mean(accuracy on stereotypical)
```

0 = equal accuracy; negative ΔS means the model does better on stereotypical
inputs (stereotype reliance).

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: loads mT5, builds stereo/anti sets, writes `sd_results.csv` |
| `sd.py` | Core: cue scoring (`predict_gender` / `predict_age`), accuracy, `compute_sd` |
| `sd_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `google/mt5-base` | `main.py` |
| Gender cues | male `{Il, Lui}` vs female `{Elle, Celle-ci}` | `sd.py` → `FRENCH_GENDER_CUES` |
| Age cues | young `{Le jeune, Les jeunes}` vs old `{Le vieux, La personne âgée}` | `sd.py` → `FRENCH_AGE_CUES` |
| Scoring | cue log-prob via teacher-forced loss; label = argmax over cue groups | `predict_gender` / `predict_age` |
| Accuracy | 1.0 if predicted == gold, else 0.0 (0.5 only for invalid gold) | `pronoun_accuracy` / `age_accuracy` |
| Sample cap | `N_MAX = 200`; `SEED = 42` | `main.py` |
| Datasets | WinoMT (gender), WinoBias type2 pro/anti (gender), Europarl (age markers) | `main.py` |

## How to run

```bash
pip install torch transformers datasets pandas
cd <this directory>
python main.py
```

## Output \& Results

`sd_results.csv`: `dataset`, `delta_s` (point estimate; bootstrap CIs are
computed for console output but not written to CSV).
