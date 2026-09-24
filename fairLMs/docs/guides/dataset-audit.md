# Auditing a dataset

Diagnostics audit the evaluation data and the scoring instrument directly,
rather than inferring their quality from the scores a model happens to produce.
They take explicit evidence plus a `DatasetAuditSpec`, never a model, and
return a `DiagnosticReport` whose components carry a status alongside any value.

## The four states

| State | Meaning |
|---|---|
| `ready` | the component was computed |
| `blocked` | a required input (evidence, reference or rule) was not supplied |
| `not_applicable` | the evidence cannot support this component |
| `failed` | the computation raised |

`blocked` and `not_applicable` carry `value=None`, never `0.0`. The report's own
status aggregates its components into `success`, `partial`, `blocked`,
`not_applicable` or `failed`.

## Representativeness

`b_rep` is smoothed `KL(observed ‖ reference)` in nats over one axis. It
requires an explicit reference distribution *and* its provenance; the package
will not infer a population prior from a dataset's name.

```python
from fairLMs.datasets.diagnostics import (
    DatasetAuditSpec,
    ReferenceDistribution,
    RepresentationEvidence,
    audit_representativeness,
)

evidence = RepresentationEvidence(
    axis="community",
    counts={"amber": 3, "teal": 1},
    source="My benchmark, evaluation split",
)
reference = ReferenceDistribution(
    axis="community",
    probabilities={"amber": 0.5, "teal": 0.5},
    source="Benchmark design specification v1",
    purpose="design_target",
    population="Intended benchmark composition",
)
spec = DatasetAuditSpec(
    target_name="my-benchmark",
    target_kind="benchmark_dataset",
    task_family="free_text",
    design_stance="stress_test",
    references={"community": reference},
)

report = audit_representativeness(evidence, spec)
result = report.components["b_rep"]
print(result.status.value, result.value)
print(report.to_json())
```

`RepresentationEvidence.from_records(...)` and `.from_dataframe(...)` accept
unfamiliar schemas with explicit field names, support and optional value
mapping; the mapping is preserved in provenance so a recoding is reproducible.

## Design stance matters

`population_proxy` and `stress_test` use the same mathematics and *not* the same
interpretation. A stress test may deliberately over-sample a category, so the
report warns that divergence is descriptive evidence rather than an automatic
fairness failure. The spec records the stance and each reference's purpose, and
the report serializes both.

## Stereotype leakage

`b_leak` is smoothed normalized mutual information between declared group terms
and declared trait terms, over the complete `|G| x |T|` pair space. The paper
preset -- add-one smoothing, base-2 logs, the complete pair space -- is the
default, and both are recorded in the result.

Turning text into counts is a *separate* stage with its own immutable
configuration, so a count matrix and the corpus it came from can never disagree
about the lexicon, the window or the tokenization:

```python
from fairLMs.datasets.diagnostics import (
    DatasetAuditSpec,
    LeakageExtractionConfig,
    SurfaceCooccurrenceExtractor,
    TextEvidence,
    audit_leakage,
)

extraction = LeakageExtractionConfig(
    group_lexicon=("she", "her", "he", "him"),
    trait_lexicon=("nurse", "engineer"),
    window=5,
)
texts = TextEvidence(
    axis="gender",
    texts=("she is a nurse", "he is an engineer", "she is an engineer"),
    source="My benchmark, evaluation split",
)
spec = DatasetAuditSpec(
    target_name="my-benchmark",
    target_kind="benchmark_dataset",
    task_family="free_text",
    design_stance="stress_test",
    protected_axes=("gender",),
    requested_components=("b_leak",),
    leakage_extraction=extraction,
)

report = audit_leakage(texts, spec)
print(report.components["b_leak"].value)

# The same fixture through the supplied-count-matrix path gives the same value.
counts = SurfaceCooccurrenceExtractor(config=extraction).extract(texts)
assert audit_leakage(counts, spec).components["b_leak"].value == (
    report.components["b_leak"].value
)
```

An extraction that runs and matches nothing may report exactly `0.0`, but only
with `details["zero_lexical_hits"]` and a warning saying the zero reflects an
absence of lexicon matches rather than an absence of association. An all-zero
matrix with no extraction record is `blocked`
(`zero_counts_without_extraction_record`), because malformed evidence must not
be indistinguishable from a measured zero.

## The construction vector

`b_constr` is not one number. It is a vector of eight independent components in
a fixed order, and the package deliberately publishes no aggregate construction
score:

| Slot | Question |
|---|---|
| `b_min` | how far the two sides of a pair differ once declared identity terms are masked |
| `b_equiv` | embedding-space equivalence of the two sides |
| `b_gram` | grammaticality difference between the two sides |
| `b_diff_len` | largest pairwise group mean-length gap, over the sample-weighted pooled mean |
| `b_diff_dep` | dependency-structure difference across groups |
| `b_frame` | largest pairwise gap in the rate of a declared framing |
| `b_opt` | signed option-length difference between two declared option roles |
| `b_temp` | template-count imbalance across declared groups, with its coverage ratio |

`b_equiv`, `b_gram` and `b_diff_dep` read a quantity that needs an optional
backend: a sentence embedding, a grammatical-error count and a dependency-tree
depth. Each is a registered class (`SemanticEquivalence`, `GrammarConsistency`,
`DependencyDepthDisparity`) that takes `backend=`, and
`fairLMs.datasets.diagnostics.backends` ships one reference implementation per protocol:

```python
from fairLMs.datasets.diagnostics import (
    DependencyDepthDisparity, GrammarConsistency, IdentityMaskConfig, SemanticEquivalence,
)
from fairLMs.datasets.diagnostics.backends import (
    HuggingFaceEmbeddingBackend,   # any Hugging Face encoder; core dependencies only
    LanguageToolGrammarBackend,    # pip install "fairLMs[grammar]" (needs Java)
    SpacyDependencyBackend,        # pip install "fairLMs[parse]"; python -m spacy download en_core_web_sm
)

diagnostics = (
    SemanticEquivalence(
        identity_mask=IdentityMaskConfig(identity_terms=("he", "she", "his", "her")),
        backend=HuggingFaceEmbeddingBackend(),   # sentence-transformers/all-MiniLM-L6-v2
    ),
    GrammarConsistency(backend=LanguageToolGrammarBackend()),
    DependencyDepthDisparity(backend=SpacyDependencyBackend()),
)
report = audit_construction(evidence, spec, axis="gender", diagnostics=diagnostics)
```

The backend's `revision` (model, version, pooling rule) is written into the
component's provenance, because a similarity or a depth is comparable only
against values from the same backend. Any object with a `revision` string and
the protocol's method (`encode`, `count_errors` or `depths`) is accepted, so a
different embedding model or parser plugs in without subclassing.

The backend is checked last, so a slot without one is `blocked` with a reason
code naming the missing backend only when it was requested *and* the evidence
view it needs is present for the audited axis -- `paired_texts` for `b_equiv`
and `b_gram`, `grouped_texts` for `b_diff_dep`. Otherwise it is `not_applicable`
for the ordinary reason: `component_not_requested`, `target_kind_not_supported`,
or `evidence_view_not_supplied`. The snippet below supplies only `grouped_texts`
and no backends, so it prints `b_equiv` and `b_gram` as `not_applicable` and
`b_diff_dep` as `blocked`. To find the slots that need a backend, read
`BACKEND_CONSTRUCTION_SLOTS` and `CONSTRUCTION_BACKEND_REQUIREMENTS` rather than
filtering on status. A missing backend blocks only its own slot; every other
slot is unaffected.

Two rules are worth stating outright. Option roles are always supplied by name:
`stereotype` and `anti_stereotype` are never inferred from the order the options
appear in. And a declared group with no rows, or any normalized component with a
zero denominator, is `blocked` rather than reported as `0.0`.

```python
from fairLMs.datasets.diagnostics import (
    CONSTRUCTION_SLOTS,
    DatasetEvidence,
    GroupedTexts,
    LengthDisparity,
    audit_construction,
    construction_vector,
)

grouped = GroupedTexts(
    axis="gender",
    groups=("female", "male", "female"),
    texts=("she is a nurse", "he is an engineer", "she is an engineer"),
    declared_groups=("female", "male"),
    source="My benchmark, evaluation split",
)
evidence = DatasetEvidence(
    target_name="my-benchmark",
    texts={"gender": texts},
    grouped_texts={"gender": grouped},
)
construction_spec = DatasetAuditSpec(
    target_name="my-benchmark",
    target_kind="benchmark_dataset",
    task_family="free_text",
    design_stance="stress_test",
    protected_axes=("gender",),
    requested_components=CONSTRUCTION_SLOTS,
)

construction_report = audit_construction(
    evidence, construction_spec, axis="gender", diagnostics=(LengthDisparity(),)
)
for result in construction_vector(construction_report):
    print(result.component, result.status.value, result.value)
```

`DiagnosticReport` alphabetizes its components, so `CONSTRUCTION_SLOTS`,
`construction_vector(report)` and `provenance["slot_order"]` are the only
authorities for the declared order.

## One audit over several evidence views

`audit_dataset` runs several components over one axis of one dataset and returns
a single report. It plans and runs exactly what `spec.requested_components`
asks for: registry membership never causes anything to run, and no component is
ever chosen from a dataset's name.

```python
from fairLMs.datasets.diagnostics import audit_dataset

audit_spec = DatasetAuditSpec(
    target_name="my-benchmark",
    target_kind="benchmark_dataset",
    task_family="free_text",
    design_stance="stress_test",
    protected_axes=("gender",),
    requested_components=("b_leak", *CONSTRUCTION_SLOTS),
    leakage_extraction=extraction,
)

audit = audit_dataset(
    evidence, audit_spec, axis="gender", diagnostics=(LengthDisparity(),)
)
plan = audit.plan()          # applicability only; no arithmetic has run
report = audit.run()
```

Each evidence view is built by its own container's adapter and passed in by
axis; `DatasetEvidence` re-models nothing and never converts one view into
another. A requested component whose view is absent is reported, not skipped:
`b_min` without `paired_texts` is `not_applicable`
(`evidence_view_not_supplied`), and `b_leak` with neither a count matrix nor
text is `blocked` (`missing_association_evidence`). Requesting any construction
slot puts all eight in the report, so the vector is never silently short.

The audit is single-axis on purpose. A three-axis benchmark is three calls and
three reports, because a rollup across axes is a decision the caller should make
explicitly. Row-level scorer results keep their own entry point: a `score_*`
name inside `audit_dataset` is reported `not_applicable`
(`component_requires_score_evidence`) and pointed at `audit_scores`.

## Auditing scores

Four components describe one scoring instrument, at row level. None of them runs
a model or infers groups from raw text.

| Component | Question | Unit |
|---|---|---|
| `score_mean_gap` | largest difference in group means | native score units |
| `score_rate_gap` | largest difference in group event rates, after one explicit score-to-event rule | proportion |
| `score_wasserstein_1_gap` | largest pairwise W₁ between full empirical distributions | native score units |
| `score_counterfactual_sensitivity` | mean absolute score change within complete declared pairs | native score units |

```python
from fairLMs.datasets.diagnostics import (
    DatasetAuditSpec,
    ScoreRateTransform,
    ScoredGroups,
    ScorerMeanGap,
    ScorerRateGap,
    ScorerWasserstein1Gap,
    audit_scores,
)

evidence = ScoredGroups(
    axis="cohort",
    groups=("amber", "amber", "teal", "teal"),
    scores=(0.1, 0.3, 0.8, 1.0),
    score_name="example_safety_score",
    source="Existing row-level score export v1",
    score_range=(0.0, 1.0),
)
rate_transform = ScoreRateTransform(
    event_name="score_at_or_above_policy_threshold",
    threshold=0.5,
    direction="higher",
    inclusive=True,
    provenance={"rule_source": "Example policy v1"},
)
spec = DatasetAuditSpec(
    target_name="example-score-table",
    target_kind="score_table",
    task_family="scored_rows",
    design_stance="stress_test",
    references={},
    requested_components=(
        "score_mean_gap",
        "score_rate_gap",
        "score_wasserstein_1_gap",
    ),
)

report = audit_scores(
    evidence,
    spec,
    diagnostics=(ScorerMeanGap(), ScorerRateGap(transform=rate_transform), ScorerWasserstein1Gap()),
)
for name, component in report.components.items():
    print(name, component.status.value, component.value, component.details.get("unit"))
```

Paired sensitivity needs its own evidence type, because group marginals do not
record which rows are counterparts:

```python
from fairLMs.datasets.diagnostics import PairedScores, ScorerCounterfactualSensitivity

paired = PairedScores(
    axis="declared_identity_intervention",
    pair_ids=("p1", "p1", "p2", "p2"),
    conditions=("baseline", "swap", "baseline", "swap"),
    scores=(0.1, 0.4, 0.8, 0.3),
    condition_roles=("baseline", "swap"),
    score_name="example_safety_score",
    source="Existing paired score export v1",
    pairing_basis="Reviewed minimal identity-token substitutions",
    score_range=(0.0, 1.0),
)
paired_spec = DatasetAuditSpec(
    target_name="example-paired-score-table",
    target_kind="score_table",
    task_family="paired_sentences",
    design_stance="stress_test",
    references={},
    requested_components=("score_counterfactual_sensitivity",),
)
paired_report = audit_scores(
    paired, paired_spec, diagnostic=ScorerCounterfactualSensitivity()
)
print(paired_report.components["score_counterfactual_sensitivity"].value)   # 0.4
```

## What these numbers are not

There is no implicit threshold, direction or boundary rule anywhere.
`score_range` validates scores; it is neither a threshold nor a W₁
normalization. Mean, rate and W₁ gaps answer different questions, and none is a
causal claim, a pass/fail rule or an error-rate metric. Paired sensitivity
supports identity-isolated causal language only if the pairs really differ
solely in the declared intervention. The package validates pair completeness
but cannot verify that semantic claim from scores alone.

Treat each value as a per-dataset, per-scorer diagnostic (and rate gaps
additionally as per-rule) rather than as a way to rank datasets or unrelated
scorer scales.

Further reading: [Preparing audit evidence](../preparing_audit_evidence.md) and
the [representativeness guide](../preparing_representativeness_evidence.md).
