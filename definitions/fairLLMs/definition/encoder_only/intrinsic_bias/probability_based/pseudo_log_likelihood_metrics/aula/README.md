# AULA — Attention-Weighted All Unmasked Likelihood

Same protocol as AUL (`../aul/`), but each token's log-probability is
**reweighted by mean attention** before averaging (`use_attention=True`).
The runner imports `compute_aul` from `../aul/aul.py` with
`attn_implementation="eager"` on `BertForMaskedLM`. Local `aula.py` defines
`compute_aula` but is **not** used by `main.py`.

The reported score is the percentage of pairs preferring the stereotype —
**50% ≈ unbiased**.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: loads BERT (eager attention), 20×80% subsample protocol, writes `aula_results.csv` |
| `aula.py` | Alternate `compute_aula` (unused by the runner) |
| `../aul/aul.py` | Shared `compute_aul(..., use_attention=True)` used by `main.py` |
| `crows_pairs_anonymized.csv` | Bundled CrowS-Pairs dataset |
| `aula_results.csv` | Output of the last run |

## Datasets

Same as AUL: CrowS-Pairs, StereoSet intersentence, XNLI religion templates.

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `bert-base-uncased` with `attn_implementation="eager"` | `main.py` |
| Scoring | attention-weighted mean unmasked interior-token log-prob | `compute_aul(use_attention=True)` |
| Protocol | `N_RUNS = 20` × `SUBSAMPLE = 0.8`; report mean | `main.py` |

## How to run

From the **repository root**:

```bash
python -m fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.aula.main
```

## Output \& Results

`aula_results.csv`: `dataset`, `aula_score`.

## References

- Kaneko, M., & Bollegala, D. (2022). *Unmasking the Mask — Evaluating Social
  Biases in Masked Language Models.* AAAI 2022.
