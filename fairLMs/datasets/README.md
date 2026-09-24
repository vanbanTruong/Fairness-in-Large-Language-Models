# Fairness datasets and diagnostics

This directory is the public data layer of FairLMs. It contains reusable
dataset loaders, dataset-first diagnostics, and the small resources that are
safe and useful to distribute with the package.

## Load a dataset

All loaders implement `FairnessDataset.load()` and return a sequence of plain
records:

```python
from fairLMs.datasets import BBQ, CrowSPairs, StereoSet

crows = CrowSPairs(n_max=100).load()
bbq_age = BBQ(categories=["Age"], n_max=100).load()
stereoset = StereoSet(split="validation", n_max=100).load()
```

CrowS-Pairs is bundled because its CSV is small. BBQ and other large datasets
are downloaded into the normal Hugging Face cache on first use. Loaders that
cannot rely on a maintained Hub source accept `root=` or another explicit local
path; the generated [loader registry](../docs/registry/loaders.md) records the
source policy for every class.

## Audit a dataset

Dataset construction, leakage, representativeness, and score-table diagnostics
live under `fairLMs.datasets.diagnostics` and are re-exported from
`fairLMs.datasets`:

```python
from fairLMs.datasets import DatasetAuditSpec, audit_dataset

report = audit_dataset(
    rows,
    spec=DatasetAuditSpec(target_kind="text", axes=["gender"]),
)
```

Evidence is explicit and diagnostics report `ready`, `blocked`,
`not_applicable`, or `failed`; missing inputs never become a fabricated score.

## Layout

```text
datasets/
├── diagnostics/              # Dataset and result-table audits
├── resources/                # Small redistributed files and attribution
├── base.py                   # FairnessDataset contract
├── _sources.py               # Hub and local-source helpers
├── _paths.py                 # Bundled-resource resolution
└── *.py                      # Individual dataset loaders
```

Large corpora, generated outputs, charts, and model caches are deliberately not
stored in this package. Tests use small synthetic fixtures under `tests/`.

## Dataset survey

The loaders cover the constrained-form and open-ended benchmarks discussed in
*Datasets for Fairness in Language Models: An In-Depth Survey*, including BBQ,
CrowS-Pairs, StereoSet, WinoBias, Winogender, GAP, BOLD, HONEST,
RealToxicityPrompts, HolisticBias, EEC, Bias-NLI, RedditBias, Grep-BiasIR,
UnQover, TrustGPT, Bias in Bios, and XNLI-derived counterfactual pairs.

When using a dataset, cite its original publication and follow the license and
delivery information in [`resources/NOTICE.md`](resources/NOTICE.md) and the
loader registry.
