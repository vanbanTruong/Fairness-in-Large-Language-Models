# FairLMs

A Python library for measuring bias and fairness in language models.

Thirty-three published bias metrics, re-implemented behind one interface. Every
metric exposes `compute(model, data) -> MetricResult`, follows scikit-learn's
estimator conventions (configuration in `__init__`, data at `compute`, working
`get_params` / `set_params`), and accepts any model through the same adapter
layer, so the same metric runs across checkpoints and architectures without
per-metric loading code.

Alongside the metrics, a separate dataset-first diagnostics layer audits
evaluation data and score tables directly. Evidence that cannot support a
measurement is reported as `blocked` or `not_applicable`, **never as a measured
zero**. The distinction between "no disparity" and "you have not given me what
I need to say anything" is preserved in the report rather than collapsed into a
number.

The metrics hold to the same rule where they can. Each declares the model head
it reads, so a checkpoint loaded for the wrong task is refused rather than
scored; and inputs on which a metric cannot estimate anything are rejected
instead of returning a zero that reads as a finding. See
[Divergences](registry/divergences.md).

<div class="grid cards" markdown>

- **[Install](install.md)**: `pip install fairLMs`
- **[Quickstart](quickstart.md)**: first metric in ten lines
- **[Tour](notebooks/tour.ipynb)**: the whole library in one runnable notebook
- **[Taxonomy](taxonomy.md)**: how the 33 metrics are organised
- **[Registry](registry/metrics.md)**: every metric, loader and diagnostic
- **[API reference](api/metrics.md)**: full generated reference
- **[Citing FairLMs](citation.md)**: and what to record for reproducibility

</div>

## Coverage at a glance

| | Encoder-only | Decoder-only | Encoder-decoder |
|---|---|---|---|
| **Intrinsic** | 11 metrics | 4 | 4 |
| **Extrinsic** | 3 | 7 | 4 |

Eighteen benchmark loaders (CrowS-Pairs, BBQ, StereoSet, Bias in Bios,
WinoBias, XNLI, BOLD, HONEST, RealToxicityPrompts, HolisticBias, EEC, GAP,
Winogender, Bias-NLI, RedditBias, Grep-BiasIR, UnQover, TrustGPT) fill the same
`data` argument you can pass by hand, plus eight bundled WEAT / SEAT word sets.

Eleven dataset diagnostics: axis representativeness (`b_rep`), stereotype
leakage (`b_leak`), five of the eight construction-vector slots (`b_min`,
`b_diff_len`, `b_frame`, `b_opt`, `b_temp`) and four scoring-instrument audits
(mean gap, rate gap, Wasserstein-1 gap, counterfactual sensitivity). The three
remaining construction slots read a quantity that needs an optional backend; `fairLMs.datasets.diagnostics.backends` ships a reference embedding, grammar-checker and parser backend, and without one such a slot is `blocked` by name once it is requested and its required evidence view is present, and `not_applicable` before that.

!!! note "Status"
    MIT licensed. Declared support: Python 3.10–3.13. CI runs the test suite on
    Python 3.10 and 3.13 on Linux; see the repository CI for the current test
    count and coverage.
