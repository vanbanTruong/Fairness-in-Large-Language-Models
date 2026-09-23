# Equal Opportunity — Gap in True Positive Rates

Equal-opportunity gap under forced-choice NLI entailment:

```
Gap_{g,y} = TPR_{g1} − TPR_{g2}
```

where TPR for a group is P(ŷ = y | y_true = y, group = g). The runner builds
entailment probes per dataset, scores them with an MNLI model, and writes one
gap value per dataset row. For BBQ, the reported value is the **mean absolute
gap** across categories.

## Public API

```python
from fairLMs.metrics import EqualOpportunityGap

result = EqualOpportunityGap().compute(
    y_true=[1, 1, 0, 0],
    y_pred=[1, 0, 0, 0],
    groups=["A", "B", "A", "B"],
    g1="A",
    g2="B",
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `EqualOpportunityGap`; writes results CSV for continuity |
| `equal_opportunity.py` | Core: `gap_g_y()` — vectorized per-group TPR and gap, NaN-safe |
| `data/` | Bundled BBQ / WinoBias support files |
| `gap_g_y_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `textattack/roberta-base-MNLI` by default; overridable via `--model` | `main.py` |
| Batch size / max length | `BATCH_SIZE = 32`, `MAX_LENGTH = 512` | `main.py` |
| Sample cap | `--max-samples` default **1000** (WinoBias loader uses all when unset in its path) | CLI |
| Bootstrap | `N_BOOTSTRAP = 1000`, seed 42 (used for console diagnostics; CSV stores the point / mean-abs value only) | `main.py` |
| Groups | Bias-in-Bios: male/female; WinoBias: **pro/anti**; BBQ: per-category then mean \|gap\| | `main.py` |

## How to run

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definition.encoder_only.extrinsic_bias.equal_opportunity.main
```

## Output \& Results

`gap_g_y_results.csv`: `dataset`, `gap_g_y`.

For Bias-in-Bios / WinoBias this is the signed gap; for BBQ it is
`mean_abs_gap` across categories stored under the same column name.
