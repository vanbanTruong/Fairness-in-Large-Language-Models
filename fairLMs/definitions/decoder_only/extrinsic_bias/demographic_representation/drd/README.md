# DRD — Demographic Representation Disparity

After generating short continuations, counts stereo vs. counter demographic
word mentions and measures how far their shares sit from equal representation:

```
p_s  = n_s  / (n_s + n_s')
p_s' = n_s' / (n_s + n_s')
DRD  = ½ |p_s − 0.5| + ½ |p_s' − 0.5|
```

**DRD = 0** is fair (equal share). If no stereo/counter mentions appear, both
shares default to 0.5 and DRD = 0. Neutral word lists in `AXES` are not used by
`compute_drd`.

## Public API

```python
from fairLMs.definitions import DemographicRepresentationDivergence
from fairLMs.definitions.models import HuggingFaceModel

# Prompts often come from fairLMs.datasets.BBQ / CrowSPairs.
result = DemographicRepresentationDivergence().compute(
    model=HuggingFaceModel("gpt2", task="causal"),
    prompts=["The person who"],
    stereo_words=["he", "him"],
    counter_words=["she", "her"],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `DemographicRepresentationDivergence`; writes results CSV for continuity |
| `drd.py` | Core: `generate`, `_count_mentions`, `compute_drd` |
| `data/` | Prefer `fairLMs.datasets.BBQ` / `CrowSPairs`; only CrowS-Pairs is bundled |
| `drd_results.csv` | Summary output of the last run |

Also writes per-run `drd_{dataset}_by_axis.csv` and `drd_{dataset}_rows.csv`.

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `meta-llama/Llama-2-7b-hf` | `main.py` → `MODEL_NAME` |
| Generation | greedy (`do_sample=False`), `MAX_NEW_TOKENS = 30` | `drd.py` / `main.py` |
| Sample cap | `N_MAX = 200`; `SEED = 42`; BBQ `DRD_PER_CAT_MAX = 500` | `main.py` |
| Datasets | BBQ ambig, CrowS-Pairs, Natural Questions (gender axis) | `main.py` |

## How to run

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definitions.decoder_only.extrinsic_bias.demographic_representation.drd.main
```

## Output \& Results

`drd_results.csv`: `dataset`, `drd_score`.
