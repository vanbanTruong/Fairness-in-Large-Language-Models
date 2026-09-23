# Preparing audit evidence

The evidence layer is the boundary between a user's source artifact and the
validated structures that `fairLMs` can audit. It is similar to data
preprocessing, but it also records the measurement decisions that give the
numbers meaning.

```text
source dataset or result table
        |
        |  segment, label, score, select, and document upstream
        v
explicit records, counts, or scored rows
        |
        |  map named fields and validate with fairLMs
        v
typed evidence
        |
        |  apply only compatible diagnostics
        v
structured diagnostic report
```

## What the evidence layer does

The current adapters:

- read fields or columns that the user names explicitly;
- apply an explicit raw-to-canonical category map when one is supplied;
- reject undeclared missing values, unmapped categories, non-finite scores, and
  other malformed input;
- preserve the mapping and other JSON-safe provenance; and
- freeze the validated values so later mutation of the source objects cannot
  change an audit silently.

The current adapters do not:

- split a document or conversation into sentences;
- infer gender, race, religion, or another category from text or names;
- guess which column is a protected-group field;
- run a model or external scorer to create scores;
- choose a reference population from a dataset name; or
- silently coerce bad values or drop incomplete rows.

Segmentation, annotation, classification, and scorer inference may be valid
upstream steps, but their rules, versions, coverage, and known uncertainty
must be documented. They are not neutral format conversion.

## Choose the evidence type by question

| Audit question | Typed evidence | Required input | Current diagnostic |
|---|---|---|---|
| Does a benchmark's composition differ from an explicit population or design target? | `RepresentationEvidence` | Category counts, or already-labelled rows plus complete support | `b_rep` through `audit_representativeness(...)` |
| Do existing scorer/model scores have different group means? | `ScoredGroups` | One declared group and one finite score per row | `score_mean_gap` through `audit_scores(...)` |
| How often does one declared score-derived event occur in each group? | `ScoredGroups` plus `ScoreRateTransform` | Existing group/score rows plus an explicit threshold, direction, and boundary rule | `score_rate_gap` through `audit_scores(...)` |
| How different are the complete one-dimensional group score distributions? | `ScoredGroups` | Existing row-level scores whose numeric distances have a shared meaning across groups | `score_wasserstein_1_gap` through `audit_scores(...)` |
| How much does one scorer change within declared two-condition pairs? | `PairedScores` | One pair ID, one declared condition role, and one finite score per row; exactly one row per role in every pair | `score_counterfactual_sensitivity` through explicit `audit_scores(...)` selection |

These types are intentionally separate. A score table is not dataset
construction evidence, and a dataset composition is not a scorer audit. Future
leakage and construction diagnostics will add evidence geometries for matrices,
texts, aligned pairs, options, and templates instead of forcing every source
into one universal table.

## Minimum preparation checklist

Before adapting a source, write down:

1. **Target and question.** State what artifact is being audited and which
   diagnostic claim the evidence can support.
2. **Unit of analysis.** Decide whether one row means a prompt, response,
   sentence, person, annotation, membership, or scored prediction.
3. **Axis and categories.** Define the axis, every category, complete support,
   multi-label policy, and unknown-value policy.
4. **Label or score source.** Name the dataset field, annotation protocol,
   classifier, model, API, or transformation that produced each required value.
5. **Coverage.** Reconcile source, included, and excluded units and record the
   exclusion reasons. The package cannot detect rows it never receives.
6. **Comparison semantics.** For representativeness, declare a compatible
   reference distribution. For scores, declare the score channel and its valid
   range when known. For a rate, also name the event and declare the threshold,
   direction, inclusive/exclusive boundary, and source of that rule. For W1,
   confirm that score ordering and numeric distances have the same substantive
   meaning for every group. For paired sensitivity, declare exactly two ordered
   condition roles, the pair-construction protocol, and why rows sharing a pair
   ID are valid contrasts.
7. **Provenance.** Record versions, splits, mappings, thresholds, hashes, and
   other JSON-safe information needed to reconstruct the evidence.

Then use a direct constructor when the source is already in the canonical
shape, or `from_records(...)` / `from_dataframe(...)` when external field names
and category codes need an explicit adapter.

## Choose scorer estimands explicitly

The scorer components answer different questions. Mean, rate, and W1 use
unpaired `ScoredGroups`; paired sensitivity uses the separate `PairedScores`
geometry because marginal group rows cannot reconstruct pair alignment.

| Component | Question | Information used | Unit |
|---|---|---|---|
| `score_mean_gap` | How far apart are average scores? | Every score's numeric magnitude | Native `score_units` |
| `score_rate_gap` | How far apart are the frequencies of this declared event? | A binary event produced by one threshold rule | `proportion` |
| `score_wasserstein_1_gap` | How far apart are complete group score distributions? | The location, spread, tails, and shape of each empirical distribution | Native `score_units` |
| `score_counterfactual_sensitivity` | How much does the score change within the declared pairs? | Absolute score difference inside every complete two-role pair | Native `score_units` |

### Declare score-to-event rules explicitly

A rate transform is executable measurement configuration, not a note added
after computation. Declare all of it:

- `event_name`: what crossing the threshold operationally means;
- `threshold`: a finite value in the scorer's native units;
- `direction`: `"higher"` for scores above the threshold or `"lower"` for
  scores below it;
- `inclusive`: whether equality satisfies the event; and
- `provenance`: the policy, protocol, or analysis version that selected the
  rule.

The transform's exact operator is therefore one of `>=`, `>`, `<=`, or `<`.
`fairLMs` applies the same serialized rule to every group. It never guesses a
threshold, score direction, or equality boundary. The formal paper definition
is `higher` with `inclusive=True` (`score >= threshold`). The paper also
documents a lower-inclusive tail analogue; exclusive boundaries are generalized
package extensions. The serialized transform records this distinction in
`paper_alignment`.

The following complete example deliberately has equal group means but different
event rates:

```python
from fairLMs.diagnostics import (
    DatasetAuditSpec,
    ScoreRateTransform,
    ScoredGroups,
    ScorerMeanGap,
    ScorerRateGap,
    audit_scores,
)

evidence = ScoredGroups(
    axis="cohort",
    groups=("alpha", "alpha", "teal", "teal"),
    scores=(0.0, 1.0, 0.5, 0.5),
    score_name="existing_score_channel",
    source="Existing row-level score export v1",
    score_range=(0.0, 1.0),
)
transform = ScoreRateTransform(
    event_name="score_at_or_above_half",
    threshold=0.5,
    direction="higher",
    inclusive=True,
    provenance={
        "rule_source": "Evaluation protocol v1, fixed before group comparison",
    },
)
spec = DatasetAuditSpec(
    target_name="example-score-table",
    target_kind="score_table",
    task_family="scored_rows",
    design_stance="stress_test",
    references={},
    requested_components=("score_mean_gap", "score_rate_gap"),
)
report = audit_scores(
    evidence,
    spec,
    diagnostics=(
        ScorerMeanGap(),
        ScorerRateGap(transform=transform),
    ),
)

mean_result = report.components["score_mean_gap"]
rate_result = report.components["score_rate_gap"]
assert mean_result.value == 0.0
assert rate_result.value == 0.5
assert rate_result.details["group_counts"] == {"alpha": 2, "teal": 2}
assert rate_result.details["event_counts"] == {"alpha": 1, "teal": 2}
assert rate_result.details["group_rates"] == {"alpha": 0.5, "teal": 1.0}
```

For each group `g`, the diagnostic computes
`rate_g = event_count_g / group_count_g`, then returns the maximum absolute
pairwise difference between group rates. Equality matters: changing the example
from `inclusive=True` (`>= 0.5`) to `inclusive=False` (`> 0.5`) changes which
group has the higher rate. Conversely, two groups can have different means but
the same event rate. A rate gap is not a normalized mean gap, and neither
estimand substitutes for the other.

`score_range` is optional validation metadata. It does **not** imply a threshold
at zero, `0.5`, the midpoint, the median, or any other value. If a declared range
exists and the threshold falls outside it, `score_rate_gap` is `blocked`. If no
transform is supplied, it is `blocked` with `missing_rate_transform`; missing
configuration is never reported as a zero gap. Calling `audit_scores(...)`
without an explicit diagnostic selection still defaults to `score_mean_gap`, so
pass `ScorerRateGap(transform=...)` or an explicit `diagnostics=(...)` sequence.

### Compare complete empirical score distributions

`score_wasserstein_1_gap` needs no event threshold and no reference
distribution. For every group pair, it computes the one-dimensional
Wasserstein-1 distance
`W1(P, Q) = inf_gamma E_gamma[abs(X - Y)]` between the two empirical score
distributions, then returns the maximum pairwise distance. Each observed row has
uniform mass within its group. The report records this as
`estimator="empirical_uniform_mass_per_group"`, uses absolute score difference as
the `ground_metric="absolute_score_difference"`, and sets
`directionality="symmetric"` and `normalization="none"`. Publishing this
additive component advances the diagnostic report schema to `1.3`.

This hand-checkable example has equal group means and equal event rates under
the rule `score >= 0.5`, yet its distributions differ:

```python
from fairLMs.diagnostics import (
    DatasetAuditSpec,
    ScoredGroups,
    ScorerWasserstein1Gap,
    audit_scores,
)

evidence = ScoredGroups(
    axis="cohort",
    groups=("amber", "amber", "teal", "teal"),
    scores=(0.0, 1.0, 0.25, 0.75),
    score_name="existing_score_channel",
    source="Existing row-level score export v2",
    score_range=(0.0, 1.0),
)
spec = DatasetAuditSpec(
    target_name="example-score-table",
    target_kind="score_table",
    task_family="scored_rows",
    design_stance="stress_test",
    references={},
    requested_components=("score_wasserstein_1_gap",),
)
report = audit_scores(
    evidence,
    spec,
    diagnostic=ScorerWasserstein1Gap(),
)
result = report.components["score_wasserstein_1_gap"]
assert result.value == 0.25
assert result.details["group_counts"] == {"amber": 2, "teal": 2}
assert result.details["unit"] == "score_units"
assert result.details["directionality"] == "symmetric"
assert result.details["normalization"] == "none"
```

Both means are `0.5`, and each group has one of two rows at or above `0.5`.
After sorting, the optimal one-dimensional transport moves `0.0` to `0.25` and
`1.0` to `0.75`, for an average absolute movement of `0.25`. Thus equal means or
equal threshold rates do not imply equal distributions. Conversely, W1 combines
location, spread, tail, and shape differences; it is not a pure shape-only
statistic. A zero empirical W1 means the compared observed distributions
coincide; it does not establish equality in an unobserved population.

The unit is the scorer's native numeric unit. `score_range` validates input but
does not rescale an arbitrary range into `[0, 1]` or make unrelated score
channels comparable. Calling `audit_scores(...)` without an explicit diagnostic
selection still defaults to `score_mean_gap`, so select
`ScorerWasserstein1Gap()` directly or include it in `diagnostics=(...)`.

### Prepare complete pairs before computing sensitivity

The paper's counterfactual scorer component is

`Delta_cf = (1 / number_of_pairs) * sum(abs(score_second - score_first))`.

Every validated pair receives equal mass. The primary value is absolute and
therefore unchanged if the two condition roles are globally reversed. The
report also exposes
`mean_signed_difference_second_minus_first`, but this is secondary directional
detail whose orientation is determined only by the declared
`condition_roles` order.

`PairedScores` deliberately has a stricter shape than `ScoredGroups`:

- `pair_ids` identifies which two scored rows form one comparison;
- `conditions` assigns each row to one of exactly two declared roles;
- every pair must contain exactly one row for each role;
- `scores` must already be finite numbers in one named scorer channel;
- `pairing_basis` records why the pairing is substantively justified; and
- `score_range`, when supplied, validates inputs but does not normalize the
  result.

The records and DataFrame adapters require three explicit field or column
names. They do not guess pair IDs or condition roles, coerce score strings,
drop missing rows, or retain only whatever pair IDs happen to have two rows.
Unknown conditions, duplicate roles, incomplete pairs, non-finite scores, and
out-of-range scores are evidence errors rather than zero sensitivity.

```python
from fairLMs.diagnostics import (
    DatasetAuditSpec,
    PairedScores,
    ScorerCounterfactualSensitivity,
    audit_scores,
)

rows = [
    {"match": "p2", "variant": "swap", "score": 0.0},
    {"match": "p1", "variant": "base", "score": 0.0},
    {"match": "p2", "variant": "base", "score": 1.0},
    {"match": "p1", "variant": "swap", "score": 1.0},
]
evidence = PairedScores.from_records(
    rows,
    axis="declared_identity_intervention",
    pair_id_field="match",
    condition_field="variant",
    score_field="score",
    condition_roles=("baseline", "identity_swap"),
    condition_map={"base": "baseline", "swap": "identity_swap"},
    score_name="existing_score_channel",
    source="Paired score export v1",
    pairing_basis="Reviewed minimal identity-token substitutions",
    score_range=(0.0, 1.0),
)
spec = DatasetAuditSpec(
    target_name="paired-score-table",
    target_kind="score_table",
    task_family="paired_sentences",
    design_stance="stress_test",
    references={},
    requested_components=("score_counterfactual_sensitivity",),
)
report = audit_scores(
    evidence,
    spec,
    diagnostic=ScorerCounterfactualSensitivity(),
)
result = report.components["score_counterfactual_sensitivity"]
assert result.value == 1.0
assert result.details["pair_count"] == 2
assert result.details["unit"] == "score_units"
```

This component measures sensitivity to the declared paired intervention. It
does not verify the intervention from scores alone. A shared pair ID does not
prove that two texts differ only in identity, preserve meaning and fluency, or
avoid other edits. Those claims require an upstream construction/review
protocol recorded in `pairing_basis` and provenance. Without that condition,
the value remains a paired difference but should not be interpreted as an
identity-isolated causal effect.

Pairing is information, not formatting. For example, baseline scores `{0, 1}`
and intervention scores `{0, 1}` have identical marginals. Aligned as `0->0`
and `1->1`, sensitivity is zero; crossed as `0->1` and `1->0`, it is one.
Neither group means nor W1 can recover which pairing was intended.

### Interpretation and misuse warnings

- Apply one event rule to every group. Group-specific thresholds define
  different events and are not this estimand.
- Choose the threshold from a policy or protocol before inspecting group gaps.
  Searching thresholds to maximize a disparity is exploratory model selection;
  disclose the search and do not present the selected gap as pre-specified.
- Every scalar here is descriptive, not a causal effect or a fairness pass/fail
  rule. A zero gap does not establish fairness, and a non-zero gap does not
  establish discrimination. In particular, group-level W1 is not a
  counterfactual identity effect and can reflect topical, stylistic, or sample
  composition differences.
- The paired component supports an identity-isolated interpretation only when
  the externally declared pairs are valid minimal interventions. The package
  validates pair geometry, not semantic equivalence or the truth of the
  pairing claim.
- Without ground-truth outcomes and explicit conditioning, an event-rate gap is
  not a false-positive rate, false-negative rate, calibration measure, equalized
  odds, or equal opportunity metric. It only compares the declared score event.
- W1 is symmetric and has no sign or higher/lower group. Inspect
  `pairwise_wasserstein_1` and `argmax_groups`; the maximum hides all other group
  pairs and cannot support a directional claim.
- W1 assumes that absolute distance in the native score space is meaningful.
  Do not apply it to category IDs or arbitrarily spaced ordinal codes. Rescaling,
  clipping, nonlinear transforms, filtering, or changing scorer versions can
  change the value; `normalization="none"` means no cross-scorer correction was
  applied.
- W1 requires the row-level score distributions. Group means, threshold rates,
  selected quantiles, or other incomplete aggregates cannot reconstruct it; do
  not manufacture pseudo-rows from aggregates.
- Paired sensitivity requires row-level pair membership and both condition
  scores. Two marginal score lists or aggregate group summaries cannot
  reconstruct it, and arbitrary row-order matching is not valid evidence.
- Missing or invalid row-level evidence is not a successful zero gap; it must
  remain a validation or applicability failure.
- Inspect `group_counts`, `event_counts`, `group_rates`, every rate gap, and every
  W1 pair that applies. Uniform empirical mass within each group does not remove
  sampling uncertainty or make small and large groups equally precise.
- These are point estimates. They provide no confidence interval, hypothesis
  test, or guarantee of practical significance; small groups, outliers, heavy
  tails, repeated observations, and differential coverage can affect them.
- Treat the output as a per-dataset diagnostic. The paper does not support
  ranking datasets by this value or treating it as cross-dataset comparable;
  even matching scorer versions, native score geometry, event rules where
  applicable, populations, filtering, and coverage would be necessary context
  rather than a proof of comparability.
- W1 is a one-dimensional marginal comparison, not a multivariate distribution
  audit and not a matching or paired counterfactual analysis.
- Scorer inference and group assignment remain upstream measurement steps.
  Their error or differential coverage is not corrected by either thresholding
  or distributional comparison.

## Raw-text example

Suppose the source is one document containing many sentences and the intended
unit is a sentence. `fairLMs` will not split or classify that document. An
upstream process must produce rows such as:

```python
prepared_rows = [
    {"sentence_id": "s1", "declared_group": "group_a"},
    {"sentence_id": "s2", "declared_group": "group_b"},
]
```

The evidence adapter can validate and count `declared_group`, but the report's
provenance should identify the sentence splitter, label source, review policy,
and coverage. If those decisions change, the evidence and potentially the
scientific conclusion change too.

## Next examples

- See [Preparing representativeness evidence](preparing_representativeness_evidence.md)
  for counts, records, DataFrames, unknown values, multi-label policies,
  references, and a complete `b_rep` audit.
- See [`scorer_mean_gap_diagnostic.py`](https://github.com/michaellarionov/FairLMs/blob/main/examples/scorer_mean_gap_diagnostic.py)
  for an unfamiliar result schema with explicit group and score mappings.
- See [`scorer_rate_gap_diagnostic.py`](https://github.com/michaellarionov/FairLMs/blob/main/examples/scorer_rate_gap_diagnostic.py)
  for an explicit score-to-event rule, per-group denominators and event counts,
  and a complete `score_rate_gap` report.
- See [`scorer_distribution_gap_diagnostic.py`](https://github.com/michaellarionov/FairLMs/blob/main/examples/scorer_distribution_gap_diagnostic.py)
  for equal group means and threshold rates but a non-zero empirical W1 gap.
- See [`scorer_counterfactual_sensitivity_diagnostic.py`](https://github.com/michaellarionov/FairLMs/blob/main/examples/scorer_counterfactual_sensitivity_diagnostic.py)
  for explicit pair/condition/score mappings and a declared minimal-contrast
  basis on an unfamiliar result schema.
