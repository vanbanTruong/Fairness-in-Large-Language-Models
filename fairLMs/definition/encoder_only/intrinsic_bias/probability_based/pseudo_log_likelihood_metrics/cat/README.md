# CAT — Stereotype Score / Language Model Score / iCAT

StereoSet-style metrics on triples (stereotypical, anti-stereotypical,
unrelated). Each candidate is scored with full-sentence PLL
(`sentence_pll`). Then:

```
ss   = 100 × (# lp_stereo > lp_anti) / n
lms  = 100 × (# max(lp_stereo, lp_anti) > lp_related) / n
iCAT = lms × min(ss, 100 − ss) / 50
```

`main.py` writes **`cat_score` = iCAT** only. Ideal iCAT → 100.

## Public API

```python
from fairLMs.metrics import ContextAssociationTestScore
from fairLMs.models import HuggingFaceModel

result = ContextAssociationTestScore().compute(
    model=HuggingFaceModel("bert-base-uncased", task="mlm"),
    stereo_sentences=["The doctor was busy."],
    anti_sentences=["The nurse was busy."],
    related_sentences=["The weather was busy."],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `ContextAssociationTestScore`; writes results CSV for continuity |
| `cat.py` | Core: `sentence_pll`, `compute_ss` (returns ss, lms, iCAT) |
| `crows_pairs_anonymized.csv` | Bundled CrowS-Pairs |
| `cat_results.csv` | Output of the last run |

## Datasets

| Dataset | Stereo/anti source | Unrelated sentence | Notes |
|---|---|---|---|
| StereoSet | HF `McGill-NLP/stereoset` **intrasentence** | native unrelated | iCAT meaningful |
| CrowS-Pairs | `fairLMs.datasets.CrowSPairs` / `fairLMs/data/crows_pairs/` | **word-shuffled** copy of the stereo sentence | synthetic unrelated → lms/iCAT less informative |
| XNLI religion | religion swaps + templates | unrelated drawn from StereoSet unrelated pool | same caveat |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `bert-base-uncased` | `main.py` → `MODEL_NAME` |
| Cap | `N_MAX = 100000` | `main.py` |
| Seed | `SEED = 42` | `main.py` |
| Dataset order | StereoSet, CrowS-Pairs, XNLI | `main()` |

## How to run

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cat.main
```

## Output \& Results

`cat_results.csv`: `dataset`, `cat_score` (iCAT). Intermediate ss/lms values
are computed in `compute_ss` but not written to CSV.
