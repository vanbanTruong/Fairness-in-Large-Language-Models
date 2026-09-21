# AUL — All Unmasked Likelihood

All Unmasked Likelihood (Kaneko & Bollegala, 2022): for each
(stereotypical, anti-stereotypical) sentence pair, score each sentence as the
**mean log-probability of its interior tokens with no masking**
(`use_attention=False`). A pair counts as biased if the stereotype scores
higher. The reported score is the percentage of pairs preferring the
stereotype — **50% ≈ unbiased**.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: loads BERT, loads three datasets, runs 20×80% subsample protocol, writes `aul_results.csv` |
| `aul.py` | Core: `compute_aul()` |
| `crows_pairs_anonymized.csv` | Bundled CrowS-Pairs dataset |
| `aul_results.csv` | Output of the last run |

Token scoring uses `score_sentence` in `fairLLMs/definition/encoder_only/utils.py`.

## Datasets

| Dataset | Source |
|---|---|
| CrowS-Pairs (all bias types, pooled) | bundled CSV |
| StereoSet (intersentence, validation) | HF `stereoset` |
| XNLI religion | religion-term swaps + templates (`n_max=100000`) |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `bert-base-uncased` (`BertForMaskedLM`) | `main.py` |
| Scoring | mean unmasked interior-token log-prob | `compute_aul` / utils |
| Protocol | `N_RUNS = 20` seeds × `SUBSAMPLE = 0.8`; report mean over runs | `main.py` |
| Seeds | fixed list starting with 42, 137, … | `main.py` → `SEEDS` |

`aggregate()` reports mean, std, 95% percentile CI, and a significance flag
(`*` when the CI excludes 50). Both AUL and token-rank accuracy are written.

## How to run

From the **repository root**:

```bash
python -m fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.aul.main
```

## Output \& Results

`aul_results.csv`: `dataset`, `n_pairs`, `aul_score`, `aul_std`,
`aul_ci_low`, `aul_ci_high`, `aul_sig`, `acc_mean`, `acc_std`,
`acc_ci_low`, `acc_ci_high`, `acc_sig`.

## References

- Kaneko, M., & Bollegala, D. (2022). *Unmasking the Mask — Evaluating Social
  Biases in Masked Language Models.* AAAI 2022.
