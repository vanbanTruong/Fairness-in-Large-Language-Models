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

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: loads prompts, generates, writes summary + detail CSVs |
| `drd.py` | Core: `generate`, `_count_mentions`, `compute_drd` |
| `data/` | Bundled BBQ jsonl + CrowS-Pairs CSV |
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

```bash
pip install torch transformers datasets pandas numpy
export HF_TOKEN=...
cd <this directory>
python main.py
```

## Output \& Results

`drd_results.csv`: `dataset`, `drd_score`.
