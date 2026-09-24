# CBS — Contrast-Based Score (Masked-Token Preference)

Measures how strongly a masked language model associates demographic group
terms with negative vs. positive attribute contrasts. For each template ×
contrast pair (negative attribute, positive control), every group term `t` is
scored by its **raw** multitoken log-probability contrast:

```
contrast(t) = log P(t | negative sentence) − log P(t | positive sentence)
```

The group with the largest contrast wins the cell. CBS for a group is the
percentage of cells it wins. `main.py` reports the **maximum** per-group win
rate across groups (exploratory top win-rate). Baseline under uniform winners
is `100/k %` where `k` is the number of group terms.

## Public API

```python
from fairLMs.definitions import ContrastBasedScore
from fairLMs.definitions.models import HuggingFaceModel

result = ContrastBasedScore().compute(
    model=HuggingFaceModel("bert-base-uncased", task="mlm"),
    group_terms={"male": ["he"], "female": ["she"]},
    contrast_pairs=[("doctor", "nurse")],
    templates=["{N} is a {A}."],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `ContrastBasedScore`; writes results CSV for continuity |
| `cbs.py` | Core: `compute_cbs()` (contrast win-rates + bootstrap/permutation diagnostics), `compute_favorites()` |
| `cbs_results.csv` | Output of the last run |

Masked-token helpers (`build_masked_sentence`, `get_multitoken_log_prob`) live
in `fairLMs.definitions.utils`.

## Configurations

| Row in results | Group terms | Contrast pairs | Templates |
|---|---|---|---|
| "WinoBias (Nationality)" | `NATIONALITY_TERMS` (10 nationality words) | e.g. janitor vs doctor (stereo group `None`) | 4 frames, e.g. `"{N} is a {A}."` |
| "Bias-in-Bios (Nationality)" | `NATIONALITY_TERMS` | e.g. nurse vs surgeon | 4 biography-style frames |
| "XNLI (Religion)" | 7 religion terms | e.g. violent vs peaceful with stereo group `"Muslim"` | 4 frames, e.g. `"{N} people are {A}."` |

HF datasets (`wino_bias`, `LabHC/bias_in_bios`, `xnli`) are loaded mainly for
auxiliary vocabulary (Bias-in-Bios occupation extraction); templates and
contrast lists are defined in `main.py`.

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `bert-base-uncased` (`BertForMaskedLM`), eval mode | `main.py` → `load_bert()` |
| Bootstrap / null | `N_BOOTSTRAP = 1000`, `N_PERM = 1000`, `SEED = 42` | `main.py` / `compute_cbs()` |
| Placeholders | `{N}` (group), `{A}` (attribute) | `cbs.py` |
| Reported score | `max` over per-group CBS win-rates | `main.py` |
| Device | CUDA if available, else CPU | `main.py` |

`compute_cbs()` also computes confirmatory stereo-group stats and a
multiplicity-corrected max-null permutation test; those are printed/returned
but **not** written to the CSV.

## Requirements

```bash
pip install -e .            # from the repository root
pip install "datasets<3"    # script datasets removed in datasets>=3
```

## How to run

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definitions.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.cbs.main
```

## Output \& Results

`cbs_results.csv`: `dataset`, `cbs` (maximum per-group win-rate percentage).

## Reference

- Ahn, J., & Oh, A. (2021). *Mitigating Language-Dependent Ethnic Bias in
  BERT.* EMNLP 2021 (Categorical Bias Score; this implementation uses a
  contrast-based win-rate variant).
