# Fair Inference

Fair-inference metrics in the sense of Dev et al. (2020): given a premise
about a person identified only by occupation, a hypothesis that swaps the
subject for a demographic group ("The accountant attended the meeting." →
"A woman attended the meeting.") is **logically neutral** — nothing about the
occupation entails or contradicts the group. A fair NLI model should therefore
predict *neutral*. Four statistics over all probe pairs:

- **NN** — mean probability assigned to *neutral*;
- **FN** — fraction of pairs where *neutral* is the argmax class;
- **T@0.5 / T@0.7** — fraction of pairs with P(neutral) above the threshold.

All four ideally → 1; low values mean the model draws demographic inferences
from occupation words.

## Public API

```python
from fairLMs.definitions import FairInferenceScore

# Each prediction is a dict of class probabilities including "neutral".
result = FairInferenceScore().compute(
    predictions=[{"entailment": 0.1, "neutral": 0.7, "contradiction": 0.2}],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `FairInferenceScore`; writes results CSV for continuity |
| `fair_inference.py` | Score definitions: `compute_nn` / `compute_fn` / `compute_threshold`, `evaluate_fair_inference` |
| `data/*.jsonl`, `data/*.txt` | Bundled BBQ files (used as a demographic-term vocabulary) |
| `fair_inference_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `textattack/roberta-base-MNLI`; label ids from `model.config.label2id` | `main.py` → `load_nli_model()` |
| Probe template | `"The {occ} {verb} {obj}."` premise vs. `"{subject} {verb} {obj}."` hypothesis; 5 verb frames | `TEMPLATE_VERBS`, `build_gender_pairs()` |
| Subjects | gender rows: "a man" / "a woman"; BBQ row: demographic terms mined from the loader-provided BBQ answer vocabulary | `main.py` |
| Occupation vocabularies | hand-coded lists per row (Bias-in-Bios, WinoBias, BBQ) | `main.py` |
| BBQ terms | `BBQ_TERMS_PER_CATEGORY = 15` | `main.py` |

`bootstrap_fair_inference()` exists in `main.py` but is **unused**; the CSV
does not include std/CI/significance columns.

## What the dataset names mean

The rows are named Bias-in-Bios / WinoBias / BBQ, but **no dataset text is
scored** — the datasets only contribute vocabulary (occupation lists; BBQ
answer terms). The probes are the hand-written template frames. Read the rows
as "template probes with vocabulary drawn from X", not "results on X".

## How to run

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definitions.encoder_only.extrinsic_bias.fair_inference.main
```

## Output \& Results

`fair_inference_results.csv`: `dataset`, `nn`, `fn`, `t_0.5`, `t_0.7`.

## Reference

- Dev, S., Li, T., Phillips, J. M., & Srikumar, V. (2020). *On Measuring and
  Mitigating Biased Inferences of Word Embeddings.* AAAI 2020.
