# LPBS — Log Probability Bias Score

Implementation of the **Log Probability Bias Score** (Kurita et al., 2019,
*Measuring Bias in Contextualized Word Representations*). For a template such
as `"GGG is a XXX"` (e.g. *"[MASK] is a nurse"*), the group slot is masked and
the model's probability for each group word (e.g. *he* vs. *she*) is read out.
Because raw probabilities are confounded by how frequent each group word is,
each probability is **normalized by a prior**: the same sentence with *both*
the group slot and the attribute masked. The score for one (template,
attribute) cell is

```
LPBS = log( (p_tgt(g1)+ε) / (p_prior(g1)+ε) ) − log( (p_tgt(g2)+ε) / (p_prior(g2)+ε) )
```

Positive values mean the attribute is more strongly associated with the first
group word, negative with the second. 0 = unbiased.

The CSV reports **stereotype-agreement percentage** (`lpbs_score`), not the
mean LPBS itself: the fraction of attributes whose LPBS sign matches the
stereotype direction (or, for XNLI, the fraction with LPBS &gt; 0 favoring
`christian`).

## Public API

```python
from fairLMs.metrics import LogProbabilityBiasScore
from fairLMs.models import HuggingFaceModel

result = LogProbabilityBiasScore().compute(
    model=HuggingFaceModel("bert-base-uncased", task="mlm"),
    attribute_words=["doctor", "nurse", "engineer"],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `LogProbabilityBiasScore`; writes results CSV for continuity |
| `lpbs.py` | Core metric: `_attribute_bias_score()` (one cell) and `compute_lpbs()` |
| `lpbs_results.csv` | Output of the last run |

## What is actually evaluated

The probe sentences are **hand-written templates** defined in `main.py`
(`WINOBIAS_TEMPLATES`, `BIOS_TEMPLATES`, `XNLI_TEMPLATES`) — the datasets named
in the result rows are *not* used as evaluation sentences:

| Row | Group pair | Attributes | Role of the named dataset | Reported `lpbs_score` |
|---|---|---|---|---|
| WinoBias | he / she | occupations filtered against WinoBias tokens; male/female stereotype lists | occupation filter + direction labels | % sign agreement with stereotype direction |
| Bias-in-Bios | he / she | professions from `LabHC/bias_in_bios` | real-world gender skew P(male \| occupation) for direction | % agreement with majority-gender skew |
| XNLI | christian / muslim | 14 hand-written adjectives | none — XNLI HF data is not loaded | % of adjectives with LPBS &gt; 0 |

A built-in check in `get_bios_attributes_and_skew()` asserts
P(male|nurse) &lt; 0.5 &lt; P(male|surgeon).

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `bert-base-uncased` (`BertForMaskedLM`) | `main.py` → `MODEL_NAME` |
| Group pairs | `("he", "she")`; `("christian", "muslim")` | `GENDER_PAIR`, `RELIGION_PAIR` |
| Templates | hand-written; `GGG` = group slot, `XXX` = attribute slot | `main.py` |
| Prior sentence | template with attribute replaced by `[MASK]` and group slot masked | `lpbs.py` → `_attribute_bias_score()` |
| Numerical guard | `ε = 1e-10` | `lpbs.py` |
| Randomness | none — fully deterministic | — |
| Device | CUDA if available, else CPU | `main.py` → `load_bert()` |

## Requirements

```bash
pip install -e .          # from the repository root
pip install "datasets<3"  # script-based "wino_bias" removed in datasets>=3
```

## How to run

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definition.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.lpbs.main
```

## Output \& Results

`lpbs_results.csv`: `dataset`, `lpbs_score` (stereotype-agreement % as described
above).

## References

- Kurita, K., Vyas, N., Pareek, A., Black, A. W., & Tsvetkov, Y. (2019).
  *Measuring Bias in Contextualized Word Representations.* 1st Workshop on
  Gender Bias in NLP (ACL 2019).
- De-Arteaga, M., et al. (2019). *Bias in Bios: A Case Study of Semantic
  Representation Bias in a High-Stakes Setting.* FAT* 2019.
