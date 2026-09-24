# Fairness definitions

This directory is the single implementation and public API for FairLMs'
model-level fairness definitions. It combines the tested FairLMs metric API
with the architecture taxonomy used by the accompanying paper.

## Public API

Install the repository package once from the repository root:

```bash
python -m pip install -e ./fairLMs
```

Import metric classes and evidence containers from `fairLMs.definitions`:

```python
from fairLMs.definitions import WEAT, WordSets, list_metrics

metric = WEAT()
result = metric.compute(model, word_sets)
print(result.score)
```

All 33 registered definitions use the same interface:

```python
metric.compute(model, data)
```

Use `list_metrics()` to inspect registry names and `get_metric(name)` to create
a metric dynamically.

## Taxonomy and low-level implementations

The three architecture directories contain low-level computation functions,
small stimuli, and runnable examples. Shared contracts, model adapters,
numerical helpers, and reusable word sets also live here:

```text
definitions/
├── core/
├── models/
├── utils/
├── resources/
├── encoder_only/
│   ├── intrinsic_bias/
│   └── extrinsic_bias/
├── decoder_only/
│   ├── intrinsic_bias/
│   └── extrinsic_bias/
└── encoder_decoder/
    ├── intrinsic_bias/
    └── extrinsic_bias/
```

Advanced users may import a low-level function directly, for example:

```python
from fairLMs.definitions.encoder_only.intrinsic_bias.similarity_based.weat.weat import (
    compute_weat,
)
```

The public classes in this directory validate inputs, resolve model adapters,
and call these low-level functions. Applications should prefer the public API
so definitions can be combined consistently with dataset loaders, diagnostics,
and mitigation workflows.

## Data policy

Large benchmark corpora are not duplicated inside individual metric folders.
Use `fairLMs.datasets` to download or load them from their maintained sources.
Only small reusable stimuli and package test resources are stored locally.

Each leaf directory has a README describing the definition, inputs, references,
and a runnable `python -m fairLMs.definitions...main` example.
