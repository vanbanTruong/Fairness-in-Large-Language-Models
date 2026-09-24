# Writing a metric

A metric subclasses `FairnessMetric` and follows scikit-learn's estimator
conventions: configuration in `__init__` (stored verbatim, no validation, no
computation), data at `compute`, and a `MetricResult` back.

```python
from typing import Any

from fairLMs.definitions import FairnessMetric, MetricResult
from fairLMs.definitions.data import SentenceTriples
from fairLMs.definitions.resolve import get_tokenizer_model


class MyPairGap(FairnessMetric):
    """One-line summary. ``data`` is a SentenceTriples."""

    name = "my_pair_gap"
    bias_type = "intrinsic"                 # "intrinsic" | "extrinsic"
    architectures = ("encoder_only",)       # declaration, for documentation
    required_task = "mlm"                   # enforced when a model is resolved

    def __init__(self, *, n_samples: int = 1000):
        self.n_samples = n_samples           # store verbatim: nothing else

    def compute(self, model: Any = None, data: Any = None, **legacy: Any) -> MetricResult:
        self._reject_unknown_kwargs(legacy)
        tokenizer, hf_model, device = get_tokenizer_model(model, metric=self)
        ...
        return MetricResult(
            score=float(value),
            details={"n_pairs": len(data.stereotype)},
            by_category={"gender": ...},
        )
```

## The four class attributes

| Attribute | Purpose | Enforced? |
|---|---|---|
| `name` | the registry key, snake_case | n/a |
| `bias_type` | `"intrinsic"` or `"extrinsic"` | no |
| `architectures` | `"encoder_only"`, `"decoder_only"`, `"encoder_decoder"` | no |
| `required_task` | which head the checkpoint must be loaded with | **yes** |

`bias_type` and `architectures` document where a metric sits in the
[taxonomy](../taxonomy.md) and drive the generated
[registry table](../registry/metrics.md).

`required_task` is different: it is checked every time your metric resolves a
model. Declare the task whose head exposes the quantity you read: `"mlm"` for
vocabulary logits, `"encoder"` for bare hidden states, `"causal"` for
next-token distributions, `"seq2seq"` for generation,
`"sequence_classification"` for label logits. A caller who passes the wrong one
gets a `TypeError` naming both tasks and the fix, instead of an
`AttributeError` from inside your numerics, or, worse, plausible numbers from
an untrained head.

Leave it `None` if your metric scores precomputed predictions, calls an API, or
genuinely works with any head. The check is also skipped for raw
`(tokenizer, model)` tuples, which carry no task to compare against.

## What you get for free

Because configuration lives in `__init__` under its own name,
`FairnessMetric._param_names` introspects the signature and `get_params` /
`set_params` work without any extra code, which is what makes
`sklearn.base.clone` and parameter sweeps possible:

```python
m = MyPairGap(n_samples=500)
m.get_params()                    # {'n_samples': 500}
type(m)(**m.get_params())         # equivalent metric
```

Call `self._reject_unknown_kwargs(legacy, *allowed)` so a misspelled keyword
raises `TypeError` instead of silently falling back to a default.

## Resolving the model

Never load a checkpoint inside a metric. `get_tokenizer_model` normalizes every
accepted input (a `HuggingFaceModel` adapter, a `LoadedModel`, a
`(tokenizer, model)` tuple, or `model=` plus `tokenizer=`) into one
`(tokenizer, model, device)` triple. Metrics that score precomputed predictions
accept `model=None` and skip it entirely.

Pass `metric=self` so the resolver can check your `required_task` against the
task the model was loaded with; omitting it silently disables that check.

## Registering

Add the class to `METRIC_REGISTRY` in `fairLMs/definitions/__init__.py`, keyed on its
`name`, and export it in `__all__`. Registration is what enrols it in the
contract suite that runs over every registered metric:

```bash
pytest                    # contract suite covers the whole registry
pytest -k my_pair_gap     # just yours
```

## Returning results

`MetricResult` has three fields: `score` (the headline float), `details` (a dict
for secondary quantities: sample counts, sub-scores, units) and `by_category`
(an optional per-group breakdown). `float(result)` returns `score`, so a metric
can be dropped into numeric code directly.
