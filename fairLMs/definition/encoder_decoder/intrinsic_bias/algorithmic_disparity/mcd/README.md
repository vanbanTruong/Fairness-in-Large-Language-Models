# MCD — Morphological Complexity Disparity (Translation)

Measures morphological richness of the model's French translations. All
translated words (length ≥ 3) are stemmed with the French Snowball stemmer;
for every stem with ≥2 surface forms, the distribution over observed
word-forms yields Shannon entropy **H** (higher = more diverse inflection)
and Simpson concentration **D** (higher = fewer forms dominate). Means of H
and D are reported per dataset on a **pooled** sentence set (no demographic
group split in the current runner).

## Public API

```python
from fairLMs.metrics import MorphologicalChoiceDivergence
from fairLMs.models import HuggingFaceModel

result = MorphologicalChoiceDivergence().compute(
    model=HuggingFaceModel("t5-small", task="seq2seq"),
    sentences=["translate English to French: The doctor is busy."],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `MorphologicalChoiceDivergence`; writes results CSV for continuity |
| `mcd.py` | Core: translation, stemming, per-stem entropy/Simpson (`compute_mcd`) |
| `mcd_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `google-t5/t5-base`, prefix `"translate English to French: "` | `main.py` / `mcd.py` |
| Decoding | beam search, `num_beams=4`, `max_new_tokens=128` | `mcd.py` |
| Stemmer | NLTK `SnowballStemmer("french")`; words shorter than 3 chars skipped; stems with &lt;2 occurrences skipped | `mcd.py` |
| Sample cap | `N_MAX = 1000`; `SEED = 42` | `main.py` |
| Datasets | WinoMT (GitHub URL), XNLI (HF premises), Europarl (HF en–fr) | `main.py` |

## How to run

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definition.encoder_decoder.intrinsic_bias.algorithmic_disparity.mcd.main
```

## Output \& Results

`mcd_results.csv`: `dataset`, `h`, `d` (point estimates; bootstrap CIs are
computed for console output but not written to CSV).
