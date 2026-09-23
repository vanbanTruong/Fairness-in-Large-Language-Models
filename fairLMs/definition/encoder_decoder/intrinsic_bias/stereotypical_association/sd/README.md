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

## Public API

```python
from fairLMs.metrics import StereotypicalDivergence
from fairLMs.models import HuggingFaceModel

result = StereotypicalDivergence().compute(
    model=HuggingFaceModel("t5-small", task="seq2seq"),
    stereo_sentences=["translate English to French: The doctor is busy."],
    stereo_labels=["Le médecin est occupé."],
    anti_sentences=["translate English to French: The nurse is busy."],
    anti_labels=["L'infirmière est occupée."],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `StereotypicalDivergence`; writes results CSV for continuity |
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

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definition.encoder_decoder.intrinsic_bias.stereotypical_association.sd.main
```

## Output \& Results

`sd_results.csv`: `dataset`, `delta_s` (point estimate; bootstrap CIs are
computed for console output but not written to CSV).
