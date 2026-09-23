# AUC — Counterfactual Fairness (Masked Attribute Recoverability)

Counterfactual-fairness probe: sensitive attributes in the source text are
masked/neutralized, the encoder produces a mean-pooled embedding, and a
logistic regression tries to predict the original demographic group from that
embedding. **Lower AUC** ≈ group identity is less recoverable after masking ≈
fairer representations (chance = 0.5).

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: loads mBART, builds masked sentences + labels, writes `auc_results.csv` |
| `auc.py` | Core: encoder mean-pool embeddings, multi-seed LR AUC (`compute_auc`) |
| `auc_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `facebook/mbart-large-50-many-to-many-mmt` (encoder used; float32; eval mode) | `main.py` |
| Classifier | sklearn logistic regression; AUC averaged over `N_SEEDS = 10` train/test splits | `auc.py` |
| Test ratio | 0.2 | `compute_auc` |
| Sample cap | `N_MAX = 200`; `SEED = 42` | `main.py` |
| Datasets | WinoMT (gender → they), XSum (nationality strip), XNLI (gender mask) | `main.py` |

## How to run

```bash
pip install torch transformers datasets pandas scikit-learn
cd <this directory>
python main.py
```

## Output \& Results

`auc_results.csv`: `dataset`, `auc_score`.
