# SS — Semantic Similarity (Individual Fairness)

Individual-fairness metric for translation: given a source sentence and a
counterfactual with a sensitive attribute swapped (gender or nationality),
both are translated to French and compared with LaBSE cosine similarity.
Higher mean SS ≈ more similar treatment of factual vs. counterfactual inputs
(fairness toward 1).

## Public API

```python
from fairLMs.definitions import TranslationSimilarityScore
from fairLMs.definitions.models import HuggingFaceModel

mt = HuggingFaceModel("facebook/mbart-large-50-many-to-many-mmt", task="seq2seq").load()
labse = HuggingFaceModel("sentence-transformers/LaBSE", task="encoder").load()
result = TranslationSimilarityScore().compute(
    model=mt,
    labse_model=labse.model,
    labse_tokenizer=labse.tokenizer,
    pairs=[("The doctor is busy.", "The nurse is busy.")],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `TranslationSimilarityScore`; writes results CSV for continuity |
| `ss.py` | Core: gender/nationality swaps, translation, LaBSE cosine (`compute_ss`) |
| `ss_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Translation model | `facebook/mbart-large-50-many-to-many-mmt`, prefix `"translate English to French: "` | `main.py` / `ss.py` |
| Embedding model | `sentence-transformers/LaBSE` | `ss.py` → `LABSE_MODEL_NAME` |
| Target language | French | `main.py` → `TGT_LANG` |
| Decoding | beam search, `max_new_tokens=128` | `generate_translation()` |
| Sample cap | `N_MAX = 500`; `SEED = 42` | `main.py` |
| Datasets | WinoMT (gender swap), XSum (nationality swap), XNLI (gender swap) | `main.py` |

## How to run

**Preferred:** use the public API above (and/or examples under `support/examples/` at the project root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definitions.encoder_decoder.extrinsic_bias.individual_fairness.main
```

## Output \& Results

`ss_results.csv`: `dataset`, `ss_score` (mean cosine over pairs; bootstrap
stats are computed for console output but not written to CSV).
