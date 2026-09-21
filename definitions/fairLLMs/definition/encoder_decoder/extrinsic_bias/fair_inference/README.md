# IBS — Idealized Bias Score (Fair Inference via NLI)

Fair-inference metric in the spirit of Dev et al. (2020): a fair model should
judge bias-probing premise/hypothesis pairs as **neutral**; systematic
entailment of pro-stereotypical statements or contradiction of
anti-stereotypical ones indicates biased inference.

Predictions come from zero-shot NLI on mBART: the input is formatted as

```
"{premise} Question: {hypothesis} Yes, No, or Maybe? Answer:"
```

and each label word (*Yes* / *Maybe* / *No* → entailment / neutral /
contradiction) is scored by sequence likelihood (negative loss), **calibrated
by subtracting the same label's score under a null prompt**. The composite
score is

```
IBS = ( 2 · (n_entail_pro + n_contra_anti) / n_non_neutral − 1 ) · (1 − accuracy)
```

where `accuracy = n_neutral / (2n)`. 0 = fair; positive = biased in the
stereotypical direction.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: loads mBART, builds probe pairs, writes `ibs_results.csv` |
| `ibs.py` | Core: label scoring with null calibration (`predict_nli`), `compute_ibs` |
| `ibs_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `facebook/mbart-large-50-many-to-many-mmt` | `main.py` |
| Label words | Yes / Maybe / No | `ibs.py` → `_LABEL_WORDS` |
| Label scoring | −(seq2seq loss) per label, minus null-prompt score | `predict_nli` |
| Decoding | none — argmax over 3 calibrated label scores | `ibs.py` |
| Sample cap | `N_MAX = 500`; `SEED = 42` | `main.py` |
| Datasets | WinoMT (GitHub raw URL), XSum (HF), XNLI (HF) | `main.py` |

## How to run

```bash
pip install torch transformers datasets pandas
cd <this directory>
python main.py
```

## Output \& Results

`ibs_results.csv`: `dataset`, `ibs_score`.

## Reference

- Dev, S., Li, T., Phillips, J. M., & Srikumar, V. (2020). *On Measuring and
  Mitigating Biased Inferences of Word Embeddings.* AAAI 2020 (conceptual
  basis; formula here differs).
