# SEAT — Sentence Encoder Association Test

Implementation of SEAT (May et al., 2019, *On Measuring Social Biases in
Sentence Encoders*), a sentence-level extension of WEAT: each word is inserted
into semantically bleached templates ("This is {}.", "{} is here.", …), the
resulting sentences are encoded with BERT, and the WEAT statistic (Cohen's d +
permutation test) is computed on the sentence embeddings.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: loads BERT, runs the four tests, writes `seat_results.csv` |
| `seat.py` | Core: template instantiation, encoding, per-term averaging, `compute_seat()` |
| `seat_results.csv` | Output of the last run |

**Word lists** come from `../weat/data.py` → `ALL_TESTS` (imported in
`main.py`). A local `data.py` may exist in this directory but is **not** used
by the runner.

Sentence encoding lives in `fairLLMs/definition/encoder_only/utils.py` →
`encode_sentence()`. `compute_seat` currently calls it without an explicit
`device`, so encoding defaults to CPU inside `encode_sentence`.

## Test cases

C1–C4 (race / gender / disease / age), word lists hard-coded in
`weat/data.py` — same sizes as WEAT (see that README).

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `bert-base-uncased` (`BertModel`) | `main.py` → `load_bert()` |
| Templates | the 5 defined inside `compute_seat()`: "This is {}.", "This is a {}.", "{} is here.", "This will {}.", "{} are things." (the set in `apply_templates()` defaults is unused when `compute_seat` supplies its own) | `seat.py` |
| Pooling | `"mean"` — mask-aware mean over tokens | `main.py` / `utils.py` |
| Per-term vector | the 5 template sentence embeddings are **averaged into one vector per word** | `seat.py` |
| Effect size | Cohen's d, pooled std (ddof=1) | `utils.py` → `cohens_d()` |
| p-value | one-sided permutation test, `n_samples = 10000` | `utils.py` → `permutation_pval()` |
| Seed | `np.random.seed(43)` before every test | `main.py` → `run_seat()` |

## How to run

```bash
pip install -e .        # from the repository root
python -m fairLLMs.definition.encoder_only.intrinsic_bias.similarity_based.seat.main
```

## Output \& Results

`seat_results.csv`: `test`, `effect_size_d`, `p_value` per test case.

## References

- May, C., Wang, A., Bordia, S., Bowman, S. R., & Rudinger, R. (2019). *On
  Measuring Social Biases in Sentence Encoders.* NAACL 2019.
- Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). *Semantics derived
  automatically from language corpora contain human-like biases.* Science.
