# Bundled data

## Corpus files

Canonical copies of datasets that several metrics share.

- `crows_pairs/crows_pairs_anonymized.csv` — CrowS-Pairs
- `bbq/*.jsonl` — BBQ category files

Loaders in `fairLMs.datasets` resolve these paths first, then fall back to
legacy locations for externally maintained checkouts. Byte-identical duplicate
copies of these corpora were removed from `fairLMs/definition/`, leaving this
directory as the single source; the [changelog](../../docs/changelog.md) records
the change.
See [NOTICE.md](NOTICE.md) for attribution and [checksums.json](checksums.json)
for the exact bundled bytes.

## Word sets (`__init__.py`)

Association tests take four labelled term lists rather than a corpus, so they
ship as importable constants instead of loader classes. `__init__.py` re-exports
the stimuli under
`fairLMs/definition/encoder_only/intrinsic_bias/similarity_based/` as validated
`WordSets` objects, purely to shorten the import and drop the unpacking at the
call site:

```python
from fairLMs.data import weat_c1
from fairLMs.metrics import WEAT

WEAT().compute(model, weat_c1)
```

Available: `weat_c1`–`weat_c4` (Caliskan et al., 2017) and `seat_c1`–`seat_c4`
(May et al., 2019, expanded name lists), plus the `WORD_SETS` registry,
`WORD_SET_LABELS`, `list_word_sets()`, and `get_word_set()`.

The `definition/` files remain the source of truth — nothing is duplicated here,
and `tests/test_bundled_word_sets.py` asserts the re-exports match them term for
term. Because this directory is now a package, `[tool.setuptools.package-data]`
in `pyproject.toml` is keyed on `fairLMs.data`.
