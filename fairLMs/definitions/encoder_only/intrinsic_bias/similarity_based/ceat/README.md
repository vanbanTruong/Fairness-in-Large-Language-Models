# CEAT — Contextualized Embedding Association Test

Implementation of CEAT (Guo & Caliskan, 2021, *Detecting Emergent
Intersectional Biases: Contextualized Word Embeddings Contain a Distribution
of Human-like Biases*), adapted to `bert-base-uncased`. Because a contextual
model gives every word a different embedding in every sentence, bias is not a
single number but a **distribution over contexts**: each trial samples one
context embedding per word, computes a WEAT effect size (Cohen's d) on that
sample, and the per-trial effect sizes are pooled with a **random-effects
meta-analysis (DerSimonian–Laird)** into a Combined Effect Size (CES).

This version follows the original protocol: **one context per word per trial**
(`SAMPLE_SIZE = 1`) and full DL pooling — Q statistic, between-trial
heterogeneity τ², weights 1/(v+τ²), and a z-test on the pooled CES.

## Public API

```python
from fairLMs.definitions import CEAT
from fairLMs.definitions.models import HuggingFaceModel

# Each *_contexts value is a list of contextualized sentences per target/attribute word.
result = CEAT(sample_size=1, n_trials=10).compute(
    model=HuggingFaceModel("bert-base-uncased", task="encoder"),
    T1_contexts=[["John is here."], ["Paul is here."]],
    T2_contexts=[["Amy is here."], ["Joan is here."]],
    A1_contexts=[["This is about career."], ["This is about salary."]],
    A2_contexts=[["This is about family."], ["This is about home."]],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `generate_embeddings.py` | Step 1: embeds up to 1,000 Reddit context sentences per word with BERT, writes `bert_weat{3,6,9,10}.pickle` |
| `main.py` | Short public-API demo using `CEAT`; writes results CSV for continuity |
| `ceat.py` | Core metric math (called by the public API): Standalone `compute_ceat()` + `dersimonian_laird()` (main.py has its own copy of both) |
| `sen_dic_1.pickle` | Reddit sentence dictionary used by `generate_embeddings.py` (present in this directory) |
| `bert_weat{3,6,9,10}.pickle` | Precomputed context embedding pools |
| `ceat_results.csv` | Output of the last run |

A local `data.py` may exist but word lists used for embedding generation live
in `generate_embeddings.py` (`GUO_CANDIDATES`).

## Test cases

WEAT 3 / 6 / 9 / 10, labeled C1–C4 (race, gender, disease, age). Word lists
are hard-coded in `generate_embeddings.py` (`GUO_CANDIDATES`) — they are
pre-trimmed subsets of the full WEAT stimuli, filtered against the sentence
corpus and equalized in length by `filter_and_equalize()`.

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `bert-base-uncased` (note: the original paper used `bert-base-cased`) | `generate_embeddings.py` → `MODEL` |
| Word embedding | mean over the word's subword tokens, last hidden layer, `max_length=128`, `torch.no_grad`, eval mode | `generate_embeddings.py` → `embed_word()` |
| Contexts per word | ≤ 1,000 sentences (`MAX_SENTENCES`) | `generate_embeddings.py` |
| Trials | `N_TRIALS = 10000` | `main.py` |
| Sampling | `SAMPLE_SIZE = 1` — one random context embedding per word per trial | `main.py` → `_sample_mean()` |
| Per-trial variance | variance of the pooled associations (+1e-10) | `main.py` |
| Pooling | DerSimonian–Laird: Q, τ² = max(0, (Q−(k−1))/c), w* = 1/(v+τ²); CES, se, two-sided z-test p | `dersimonian_laird()` |
| Seeds | 20 fixed seeds are run, but **only seed 42's result is written** to the CSV | `main.py` → `SEEDS`, `run_multi_seed()` |

## Requirements and data

1. Install dependencies from the repo-root `pyproject.toml` (`transformers`,
   `torch`, `numpy`, `scipy`, `pandas`).
2. Ensure `sen_dic_1.pickle` is in this directory (shipped here; originally from
   the CEAT release at github.com/weiguowilliam/CEAT).
3. Generate embedding pools (if needed), then run the metric:

```bash
pip install -e .
# Prefer the Public API above; optional legacy:
python -m fairLMs.definitions.encoder_only.intrinsic_bias.similarity_based.ceat.main
```

Both scripts use local paths, so run them from inside this directory (the
package must also be importable; shared helpers live in `fairLMs.definitions.utils`).

## How to run

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definitions.encoder_only.intrinsic_bias.similarity_based.ceat.main
```

## Output \& Results

`ceat_results.csv`: `test` (C1–C4), `ceat_score` (pooled CES), `se`.
|CES| ≈ 0.2 / 0.5 / 0.8 reads as small / medium / large.

## References

- Guo, W., & Caliskan, A. (2021). *Detecting Emergent Intersectional Biases:
  Contextualized Word Embeddings Contain a Distribution of Human-like Biases.*
  AIES 2021.
- DerSimonian, R., & Laird, N. (1986). *Meta-analysis in clinical trials.*
  Controlled Clinical Trials.
