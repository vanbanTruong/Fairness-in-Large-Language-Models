# SS — Semantic Similarity (Individual Fairness)

Individual-fairness metric for translation: given a source sentence and a
counterfactual with a sensitive attribute swapped (gender or nationality),
both are translated to French and compared with LaBSE cosine similarity.
Higher mean SS ≈ more similar treatment of factual vs. counterfactual inputs
(fairness toward 1).

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: loads mBART + LaBSE, builds pairs, writes `ss_results.csv` |
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

```bash
pip install torch transformers datasets pandas sentence-transformers
cd <this directory>
python main.py
```

## Output \& Results

`ss_results.csv`: `dataset`, `ss_score` (mean cosine over pairs; bootstrap
stats are computed for console output but not written to CSV).
