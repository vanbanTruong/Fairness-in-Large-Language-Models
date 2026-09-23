# Examples

Short scripts that use the **public** fairLMs API:

```python
from fairLMs.metrics import CrowSPairsScore
```

Leaf demos under `fairLMs/definition/**/main.py` also call this API and remain
runnable via `python -m fairLMs.definition....main`. Prefer these examples
(or the Public API sections in each metric README) for new code.

Dataset-first diagnostics use the separate `fairLMs.diagnostics` API because
their applicability states and multi-part reports are not scalar model metrics.

## Quick start

```bash
pip install -e .
python examples/equal_opportunity_gap.py
python examples/representativeness_diagnostic.py
python examples/scorer_mean_gap_diagnostic.py
python examples/scorer_rate_gap_diagnostic.py
python examples/scorer_distribution_gap_diagnostic.py
python examples/scorer_counterfactual_sensitivity_diagnostic.py
python examples/crows_pairs_score.py
python examples/reproduce_crows_pairs.py        # published CrowS-Pairs values, ~4 min on CPU
python examples/reproduce_seat_biasbench.py     # published SEAT effect sizes, downloads six sentence files
```

## Scripts

| Script | Metric(s) |
|--------|-----------|
| `crows_pairs_score.py` | CrowSPairsScore (CPS) |
| `log_probability_bias.py` | LogProbabilityBiasScore (LPBS) |
| `weat.py` | WEAT |
| `equal_opportunity_gap.py` | EqualOpportunityGap |
| `accuracy_disparity.py` | AccuracyDisparity |
| `representativeness_diagnostic.py` | Dataset representativeness (`b_rep`) |
| `scorer_mean_gap_diagnostic.py` | Scoring-instrument group mean gap (`score_mean_gap`) |
| `scorer_rate_gap_diagnostic.py` | Explicit-threshold group event-rate gap (`score_rate_gap`) |
| `scorer_distribution_gap_diagnostic.py` | Empirical Wasserstein-1 score-distribution gap (`score_wasserstein_1_gap`) |
| `scorer_counterfactual_sensitivity_diagnostic.py` | Complete-pair scorer sensitivity (`score_counterfactual_sensitivity`) |

Heavier metrics (seq2seq, OpenAI, full BBQ) are documented in their leaf
READMEs; use `list_metrics()` to discover class names.
