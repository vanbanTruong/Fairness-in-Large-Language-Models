# utils

Shared numerical and I/O helpers. These are the building blocks the metrics are assembled from: pseudo-log-likelihood scoring, embedding and pooling, association statistics, masking, bundled-data paths and CSV output. They are public and stable, so a custom metric can reuse them rather than reimplementing the same pooling or permutation test.

::: fairLMs.utils
    options:
      show_root_heading: true
      show_source: true
      members_order: source
      filters: ["!^_"]
