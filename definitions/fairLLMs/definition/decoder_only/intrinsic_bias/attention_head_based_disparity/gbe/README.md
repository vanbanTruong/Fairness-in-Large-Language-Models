# GBE — Gradient-based Bias Estimation

Estimates how much each GPT-2 attention **Value head** contributes to a SEAT
effect size by backpropagating `|d|` into per-head Value masks. Reports the
**mass** of positive gradients (normalized absolute mass on positive entries).
Chance baseline for the related sign-proportion statistic is **0.5**; a
permutation null tests significance. The CSV stores the mass statistic as
`gbe_score`.

```
d        = SEAT Cohen's d on sentence embeddings
loss     = |d|
GBE_mass = Σ max(m,0) / Σ |m|     over head-mask gradients m
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: loads GPT-2, builds word lists, runs GBE + null, writes `gbe_results.csv` |
| `gbe.py` | Core: `install_head_masks`, `seat_effect_size`, `compute_gbe_matrix`, `compute_gbe`, `compute_gbe_mass`, `gbe_permutation_null` |
| `red_pill_corpus.csv` | Optional Red Pill vocabulary source |
| `gbe_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `gpt2-medium` (`attn_implementation="eager"`, `model.double()`) | `main.py` |
| Seed / null | `SEED = 42`; `N_PERM = 50`; `PERMUTE_MODE = "attributes"` | `main.py` |
| Red Pill | `REDPILL_NROWS = 200000`, `REDPILL_MIN_FREQ = 100`, `REDPILL_TOP_GENDER = 8` | `main.py` |
| Datasets | Hardcoded StereoSet / Winogender word lists; Red Pill vocab mined from corpus `body` column | `main.py` |

## How to run

```bash
pip install torch transformers pandas numpy
cd <this directory>
python main.py
```

**Path caveat:** `_MAIN_DIR` is hard-coded to `/content/drive/MyDrive` (Colab).
For local runs, point it at this directory so `red_pill_corpus.csv` and
`gbe_results.csv` resolve correctly.

## Output \& Results

`gbe_results.csv`: `dataset`, `gbe_score` (observed gradient mass).
