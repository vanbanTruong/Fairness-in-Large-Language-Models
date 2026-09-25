# DisCo — Discovery of Correlations

Measures how differently a masked LM's top-k predictions diverge for paired
demographic terms in the same templates. For each template and word pair
`(w1, w2)`, fill `w1`/`w2` into the subject slot, take the top-`k` masked
predictions, and record overlap:

```
overlap = |top_k(w1) ∩ top_k(w2)| / k
disco   = (1 − mean(overlaps)) × 100
```

Higher DisCo = less shared completions across groups (more disparity).

## Public API

```python
from fairLMs.definitions import DiscoveryOfCorrelationsScore
from fairLMs.definitions.models import HuggingFaceModel

result = DiscoveryOfCorrelationsScore(k=3).compute(
    model=HuggingFaceModel("bert-base-uncased", task="mlm"),
    group1_words=["he", "him"],
    group2_words=["she", "her"],
    templates=["[T] is a [MASK]."],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `DiscoveryOfCorrelationsScore`; writes results CSV for continuity |
| `disco.py` | Core: `compute_disco()` (top-k overlap + cluster bootstrap CI), `compute_disco_multi_k()` |
| `disco_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `bert-base-uncased` (fill-mask pipeline) | `main.py` |
| `k` | 3 top predictions per fill | `TOP_K` / `compute_disco(k=3)` |
| Templates | up to `MAX_TEMPLATES = 60` | `main.py` |
| Seed | `SEED = 42` (template RNG uses seed `0`) | `main.py` |
| Null baseline | type-matched **derangement** of group-2 words, `N_NULL_PERMS = 50`; significance if `p_perm < 0.05` | `run_disco()` |
| Bootstrap CI | `N_BOOTSTRAP = 2000` (printed; not written to CSV) | `disco.py` |
| Datasets | WinoBias gender, Bias-in-Bios gender, XNLI religion (vocab-attested Christian/Muslim lists) | `main.py` |

Requires ≥3 word pairs. Gender rows use class-matched derangements
(`GENDER_WORD_CLASSES`).

## How to run

**Preferred:** use the public API above (and/or examples under `support/examples/` at the project root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definitions.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.disco.main
```

## Output \& Results

`disco_results.csv`: `dataset`, `disco` (point estimate percentage).
Null p-values and CIs are printed to the console only.
