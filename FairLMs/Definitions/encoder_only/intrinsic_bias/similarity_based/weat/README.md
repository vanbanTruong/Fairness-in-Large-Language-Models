# WEAT — Word Embedding Association Test (on BERT-Base-Uncased)

Implementation of the Word Embedding Association Test (Caliskan et al., 2017,
*Semantics derived automatically from language corpora contain human-like biases*).

WEAT measures whether two sets of **target** words (e.g. European-American vs.
African-American names) are differentially associated with two sets of
**attribute** words (e.g. pleasant vs. unpleasant), using cosine similarity.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: loads BERT, builds word embeddings, runs all tests, writes `weat_results.csv` |
| `weat.py` | Core metric: `compute_weat(T1_vecs, T2_vecs, A_vecs, B_vecs)` → effect size `d`, p-value `p` |
| `data.py` | Word lists for the four test cases (`ALL_TESTS`) |
| `weat_results.csv` | Output of the last run |

Shared statistical helpers (`association_vectorized`, `cohens_d`,
`permutation_pval`) live in `encoder_only/utils.py`.

## Test cases

All word lists are hard-coded in `data.py` — no external data files are needed.

| Test | Bias type | Targets (T1 vs. T2) | Attributes (A1 vs. A2) |
|---|---|---|---|
| C1 | Race | European-American vs. African-American names (20/20) | Pleasant vs. Unpleasant (24/22) |
| C2 | Gender | Male vs. Female names (8/8) | Career vs. Family (8/8) |
| C3 | Disease | Mental vs. Physical illness terms (6/6) | Temporary vs. Permanent (7/7) |
| C4 | Age | Young vs. Old names (7/7) | Pleasant vs. Unpleasant (15/14) |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `bert-base-uncased` (HuggingFace pretrained checkpoint, inference only) | `main.py` → `load_bert()` |
| Word embedding | Encode the bare word (`[CLS] word [SEP]`), take the **last hidden layer**, drop `[CLS]`/`[SEP]`, **mean over subword tokens**; word is lowercased | `main.py` → `get_embedding()` |
| Effect size | Cohen's *d*: (mean(s(T1)) − mean(s(T2))) / pooled std (ddof=1) of all associations | `utils.py` → `cohens_d()` |
| Association s(w, A, B) | mean cosine(w, A) − mean cosine(w, B) | `utils.py` → `association_vectorized()` |
| p-value | One-sided permutation test over equal-size re-partitions of T1 ∪ T2. Exact enumeration when C(2n, n) ≤ `n_samples`, otherwise `n_samples` random permutations | `utils.py` → `permutation_pval()` |
| `n_samples` | 10,000 | `weat.py` → `compute_weat()` |
| Random seed | `np.random.seed(43)` | `main.py` → `run_weat()` |
| Device | CUDA if available, else CPU | `main.py` → `load_bert()` |

Because the pretrained checkpoint is fixed, embeddings are deterministic, and the
permutation seed is fixed, **repeated runs produce identical results**
(up to negligible floating-point differences across hardware/library versions).

## Requirements

Python ≥ 3.9 and the dependencies declared in the repo-root `pyproject.toml`:

- `numpy >= 1.24`
- `scipy >= 1.11`
- `torch >= 2.0`
- `transformers >= 4.38`

## How to run

From the **repository root**:

```bash
pip install -e .
python -m encoder_only.intrinsic_bias.similarity_based.weat.main
```

## Output \& Results

Results are written to `weat_results.csv` with columns:

- `test` — test-case name (C1–C4)
- `effect_size_d` — Cohen's d
- `p_value` — permutation p-value

## References

- Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). *Semantics derived
  automatically from language corpora contain human-like biases.* Science.
