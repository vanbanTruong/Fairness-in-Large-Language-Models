# NPD — Normalized Position Disparity (Summarization)

Position-based bias metric for summarization: does the model draw its summary
content from the same **positions** of the source document as the reference?
The article is split into sentences and divided into `K = 10` equal segments.
Each summary sentence is mapped to its most similar article sentence (TF-IDF
cosine argmax), producing a distribution over the 10 position segments. NPD is
the earth-mover's (Wasserstein) distance between the model summary's position
distribution and the reference distribution (the gold summary's, or uniform if
no gold exists), normalized by **`(K − 1)`**:

```
NPD = W₁(p_ref, p_model) / (K − 1)
```

0 = same positional profile; larger = the model systematically favors
different parts of the document (e.g., lead bias).

## Public API

```python
from fairLMs.definitions import NormalizedPositionDistance
from fairLMs.definitions.models import HuggingFaceModel

result = NormalizedPositionDistance().compute(
    model=HuggingFaceModel("facebook/mbart-large-50-many-to-many-mmt", task="seq2seq"),
    articles=["summarize: A short news article about an event."],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `NormalizedPositionDistance`; writes results CSV for continuity |
| `npd.py` | Core: sentence splitting, TF-IDF position mapping, EMD (`compute_npd`, `generate_summary`) |
| `npd_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `facebook/mbart-large-50-many-to-many-mmt`, prefix `"summarize: "` | `main.py` / `npd.py` |
| Decoding | beam search, `num_beams=4`, `length_penalty=2.0`, `max_new_tokens=128` | `generate_summary()` |
| Segments | `K = 10` | `compute_npd()` |
| Position mapping | TF-IDF (fit on article + summary sentences), cosine argmax | `_segment_distribution()` |
| Sample cap | `N_MAX = 200`; `SEED = 42` | `main.py` |
| Datasets | XSum (HF, with gold summaries), WinoMT (GitHub URL), XNLI (HF) | `main.py` |

WinoMT and XNLI have no gold summaries → uniform reference distribution.

## How to run

**Preferred:** use the public API above (and/or examples under `support/examples/` at the project root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definitions.encoder_decoder.extrinsic_bias.position_based.main
```

## Output \& Results

`npd_results.csv`: `dataset`, `npd_score`.
