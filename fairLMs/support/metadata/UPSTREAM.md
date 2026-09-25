# Upstream source

The executable library code in this directory was synchronized from
[`michaellarionov/JMLR_Library`](https://github.com/michaellarionov/JMLR_Library)
at commit `e1ed4c5` (`Release 0.5.0`) on 2026-09-22.

The upstream MIT license is preserved in [`LICENSE`](../../LICENSE). The imported
package namespace was changed mechanically from `fairlms` to `fairLMs` so it
can be called from this repository as:

```python
from fairLMs.mitigation import list_mitigators
from fairLMs.datasets.diagnostics import audit_dataset
from fairLMs.datasets import BBQ
```

The upstream runtime tree was merged into `definitions/`, which is now the only
definition package and the public metric API. The earlier research snapshot
from commit `1eb0c9d` was removed after comparison because it duplicated the
same 33 metrics, was not part of the installed `fairLMs` package, and contained
hundreds of megabytes of repeated dataset files. Dataset access now remains the
responsibility of `fairLMs.datasets` and the small shared resources bundled by
the package.

The synchronized runtime is organized into three implementation packages:
`datasets` (including diagnostics), `definitions` (including shared contracts,
model adapters, numerical helpers, and word sets), and `mitigation`. Shared
project documentation, tests, examples, maintenance scripts, and metadata live
under `support/` and are excluded from the wheel.
