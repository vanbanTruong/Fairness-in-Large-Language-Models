# Upstream source

The executable library code in this directory was synchronized from
[`michaellarionov/JMLR_Library`](https://github.com/michaellarionov/JMLR_Library)
at commit `e1ed4c5` (`Release 0.5.0`) on 2026-09-22.

The upstream MIT license is preserved in [`LICENSE`](LICENSE). The imported
package namespace was changed mechanically from `fairlms` to `fairLMs` so it
can be called from this repository as:

```python
from fairLMs.mitigation import list_mitigators
from fairLMs.diagnostics import audit_dataset
from fairLMs.datasets import BBQ
```

The upstream runtime directory `definition/` is kept separate from the existing
`definitions/` research snapshot. This prevents synchronization from
overwriting the definitions currently under review.
