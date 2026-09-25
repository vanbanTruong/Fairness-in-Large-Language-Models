# utils

Shared numerical helpers used by fairness definitions: pseudo-log-likelihood
scoring, embedding and pooling, association statistics, masking, and summary
statistics. Custom definitions can reuse them instead of reimplementing the
same scoring logic.

::: fairLMs.definitions.utils
    options:
      show_root_heading: true
      show_source: true
      members_order: source
      filters: ["!^_"]

CSV result output is separate from numerical helpers and always writes outside
the installed package:

::: fairLMs.definitions.io
    options:
      show_root_heading: true
      show_source: true
      members_order: source
      filters: ["!^_"]
