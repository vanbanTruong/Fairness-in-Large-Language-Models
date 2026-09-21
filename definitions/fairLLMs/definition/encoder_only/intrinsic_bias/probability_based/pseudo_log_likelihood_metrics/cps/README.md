# CPS — CrowS-Pairs Score

Implementation of the original **CrowS-Pairs metric** (Nangia et al., 2020).
For each (stereotypical, anti-stereotypical) sentence pair, the two sentences
differ only in a few words (e.g. *black* ↔ *white*). CPS masks the **shared
(unmodified) tokens** one at a time — conditioning on the differing words — and
sums the log-probabilities of the true tokens. A pair counts as "biased" if the
stereotypical sentence scores higher. The score is the percentage of pairs
preferring the stereotype — **50% = unbiased**, higher = stereotypical bias.

Masking only the shared tokens is what distinguishes CPS from the naive
full-sentence PLL comparison (see `../pll/`): it avoids directly scoring the
swapped words themselves, whose corpus frequencies differ.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: loads BERT, loads the three datasets, runs CPS, writes `cps_results.csv` |
| `cps.py` | Core metric: shared-token alignment + `score_sentence_cps()` + `compute_cps()` |
| `crows_pairs_anonymized.csv` | Bundled CrowS-Pairs dataset (1,508 pairs) |
| `cps_results.csv` | Output of the last run |

The token alignment helper `get_span()` (difflib over the two token-id
sequences) lives in `fairLLMs/definition/encoder_only/utils.py`.

## Datasets

| Dataset | Source |
|---|---|
| CrowS-Pairs (all 9 bias categories, pooled) | bundled CSV in this directory |
| StereoSet (intersentence, validation, pooled) | HuggingFace `stereoset` / `intersentence` |
| XNLI religion | religion-term swaps + templates (`n_max=100000`) |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `bert-base-uncased` (`BertForMaskedLM`) | `main.py` → `load_bert()` |
| Token alignment | `difflib.SequenceMatcher` over token ids; positions marked `equal` are the shared span; `[CLS]`/`[SEP]` excluded | `cps.py`, `utils.py` → `get_span()` |
| Sentence score | Sum of log P(shared token) with each shared token masked one at a time | `score_sentence_cps()` |
| Pair decision | stereo counts iff `pro_score > anti_score` (scores rounded to 3 decimals; ties count as non-stereo) | `cps.py` |
| Protocol | single full-data pass (no multi-seed subsample) | `main.py` |
| Randomness | none — fully deterministic | — |

## Requirements

```bash
pip install -e .          # from the repository root
pip install "datasets<3"  # for script datasets where needed
```

## How to run

From the **repository root**:

```bash
python -m fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cps.main
```

## Output \& Results

`cps_results.csv`: `dataset`, `cps_score`.

For reference, the original CrowS-Pairs paper reports ≈60.5 for
BERT-base-uncased on CrowS-Pairs; this implementation obtains a similar
order of magnitude (tokenizer/version and alignment details differ).

## References

- Nangia, N., Vania, C., Bhalerao, R., & Bowman, S. R. (2020). *CrowS-Pairs: A
  Challenge Dataset for Measuring Social Biases in Masked Language Models.*
  EMNLP 2020.
