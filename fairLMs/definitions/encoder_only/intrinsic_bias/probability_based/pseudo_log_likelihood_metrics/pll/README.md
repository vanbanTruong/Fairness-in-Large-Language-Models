# PLL — Pseudo Log-Likelihood Score

Full-sentence pseudo-log-likelihood comparison of stereotypical vs.
anti-stereotypical sentences. Each sentence is scored by masking every
interior token in turn and summing `log P(w_i | rest)`. A pair counts as
biased if the stereotype scores higher. The reported score is the percentage
of pairs preferring the stereotype — **50% ≈ unbiased**.

Unlike CPS (`../cps/`), PLL scores **all** interior tokens (including the
differing demographic words), not only the shared span.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: loads BERT, loads three datasets, runs PLL, writes `pll_results.csv` |
| `pll.py` | Core: `score_sentence_pll()`, `compute_pll()` |
| `crows_pairs_anonymized.csv` | Bundled CrowS-Pairs dataset |
| `pll_results.csv` | Output of the last run |

## Datasets

| Dataset | Source |
|---|---|
| CrowS-Pairs (pooled) | bundled CSV |
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

From the **repository root**:

```bash
python -m encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.pll.main
```

## Output \& Results

`pll_results.csv`: `dataset`, `pll_score`.
