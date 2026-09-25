# AULA — Attention-Weighted All Unmasked Likelihood

Same protocol as AUL (`../aul/`), but each token's log-probability is
**reweighted by mean attention** before averaging (`use_attention=True`).
The runner imports `compute_aul` from `../aul/aul.py` with
`attn_implementation="eager"` on `BertForMaskedLM`. Local `aula.py` defines
`compute_aula` but is **not** used by `main.py`.

The reported score is the percentage of pairs preferring the stereotype —
**50% ≈ unbiased**.

## Public API

```python
from fairLMs.definitions import AllUnmaskedLikelihoodAttentionScore
from fairLMs.datasets import CrowSPairs
from fairLMs.definitions.models import HuggingFaceModel

result = AllUnmaskedLikelihoodAttentionScore().compute(
    model=HuggingFaceModel("bert-base-uncased", task="mlm"),
    dataset=CrowSPairs(n_max=32),
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `AllUnmaskedLikelihoodAttentionScore`; writes results CSV for continuity |
| `aula.py` | Core metric math (called by the public API): Alternate `compute_aula` (unused by the runner) |
| `../aul/aul.py` | Shared `compute_aul(..., use_attention=True)` used by `main.py` |
| `crows_pairs_anonymized.csv` | Prefer `fairLMs.datasets.CrowSPairs`; its small canonical CSV is bundled under `fairLMs/datasets/resources/crows_pairs/` |
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

**Preferred:** use the public API above (and/or examples under `support/examples/` at the project root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definitions.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.aula.main
```

## Output \& Results

`aula_results.csv`: `dataset`, `aula_score`.

## References

- Kaneko, M., & Bollegala, D. (2022). *Unmasking the Mask — Evaluating Social
  Biases in Masked Language Models.* AAAI 2022.
