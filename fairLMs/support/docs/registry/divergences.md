# Divergences from released runners

Where the original reference implementations and FairLMs disagree, the
difference is deliberate and recorded here.

## Unavailable evidence is not a zero

The systematic divergence is in the diagnostics layer. Several released bias
runners return a numeric result, usually `0.0`, when a required input is
absent: no reference distribution, no threshold rule, an incomplete
counterfactual pair. FairLMs refuses.

`ComponentResult` enforces this structurally: a non-`ready` component must carry
`value=None`, and constructing one with a numeric sentinel raises `ValueError`.
A report therefore distinguishes "the groups do not differ" (`ready`, value
`0.0`) from "you have not supplied what this component needs" (`blocked`,
value `None`).

| Input condition | Typical released behaviour | FairLMs |
|---|---|---|
| No score-to-event rule supplied to a rate gap | returns `0.0` | `blocked`, `value=None` |
| No reference distribution for an axis | infers a uniform or corpus prior | `blocked`, `value=None` |
| Incomplete counterfactual pair | drops the row silently | pair-completeness validation, then `blocked` |
| Reference probabilities not summing to 1 | renormalizes silently | rejected outside a `1e-9` tolerance; canonicalization recorded in the report |

## Known metric-level differences

| Metric | Difference | Rationale |
|---|---|---|
| AUL / AULA | attention weights are read with the attention implementation forced to `eager` | SDPA and flash-attention backends silently return no attentions even when `output_attentions=True`, which would otherwise yield AUL scores mislabelled as AULA |
| WEAT / SEAT / CEAT | the permutation-test seed is explicit constructor configuration (`seed=`, default `None`) | it appears in `get_params()`, so a reported p-value can be pinned and reproduced rather than depending on ambient global RNG state |
| all metrics reading a specific head | `required_task` is checked against the task the checkpoint was loaded with | the same checkpoint under a different head is a different measurement; a mismatch used to fail as an `AttributeError`, or to return numbers read from a randomly initialized head |
| `counterfactual_auc` | string labels, a single-member class, more than two classes, and an unscorable `test_ratio` are refused | each would otherwise report `0.0`, which as a score is not "no signal" but a perfectly anti-recoverable attribute |
| `stereotypical_divergence` | a label vocabulary the scorer cannot grade is refused | the scorers return 0.5 for unrecognised labels, so a wrong vocabulary produced `m_stereo == m_anti == 0.5` and a divergence of exactly 0.0, a non-result shaped exactly like a finding of parity |

The last three follow the same rule as the diagnostics layer: a number is
returned only when the inputs can support it. Where the underlying
implementation short-circuits to a numeric sentinel for diagnostic purposes
(`compute_auc` returning `0.0` with its rows attached), that behaviour is
preserved for direct callers and refused at the metric boundary.

## Status

This page is not yet exhaustive. Each row should eventually link to a golden
fixture pinning both behaviours so that a change in either implementation fails
CI. The diagnostics rows and the last three metric rows are covered by the test
suite today; the AUL/AULA and seed rows are not.
