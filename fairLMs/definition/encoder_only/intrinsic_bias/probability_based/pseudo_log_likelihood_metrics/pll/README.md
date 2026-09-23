# PLL — Pseudo Log-Likelihood Score

Full-sentence pseudo-log-likelihood comparison of stereotypical vs.
anti-stereotypical sentences. Each sentence is scored by masking every
interior token in turn and summing `log P(w_i | rest)`. A pair counts as
biased if the stereotype scores higher. The reported score is the percentage
of pairs preferring the stereotype — **50% ≈ unbiased**.

Unlike CPS (`../cps/`), PLL scores **all** interior tokens (including the
differing demographic words), not only the shared span.

## Public API

```python
from fairLMs.metrics import PseudoLogLikelihoodScore
from fairLMs.datasets import CrowSPairs
from fairLMs.models import HuggingFaceModel

result = PseudoLogLikelihoodScore().compute(
    model=HuggingFaceModel("bert-base-uncased", task="mlm"),
    dataset=CrowSPairs(n_max=32),
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `PseudoLogLikelihoodScore`; writes results CSV for continuity |
| `pll.py` | Core: `score_sentence_pll()`, `compute_pll()` |
| `crows_pairs_anonymized.csv` | Prefer `fairLMs.datasets.CrowSPairs` / bundled `fairLMs/data/crows_pairs/` (legacy leaf CSV may remain) |
| `pll_results.csv` | Output of the last run |

## Datasets

| Dataset | Source |
|---|---|
| CrowS-Pairs (pooled) | `fairLMs.datasets.CrowSPairs` / `fairLMs/data/crows_pairs/` |
| StereoSet (intersentence, validation) | HF `stereoset` |
| XNLI religion | religion-term swaps + templates (`n_max=100000`) |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `bert-base-uncased` (`BertForMaskedLM`) | `main.py` |
| Sentence score | sum of masked interior-token log-probs | `score_sentence_pll()` |
| Pair decision | stereo wins if stereo score &gt; anti score | `compute_pll()` |
| Protocol | single full-data pass | `main.py` |
| Randomness | none — fully deterministic | — |

## How to run

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.pll.main
```

## Output \& Results

`pll_results.csv`: `dataset`, `pll_score`.
