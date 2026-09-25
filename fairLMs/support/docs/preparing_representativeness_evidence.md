# Preparing representativeness evidence

`b_rep` compares the observed composition of one explicitly named axis with an
explicit reference distribution. The package can count already-labelled rows,
but it cannot decide what a row represents or infer an axis category from raw
text. Those decisions are part of the measurement contract and should be made
before calling `fairLMs`.

This guide covers the three supported evidence paths:

- an existing category-to-count mapping;
- an iterable of mapping records; and
- one explicitly named column in a pandas `DataFrame`.

All three paths produce one `RepresentationEvidence` object. The audit itself
accepts that object, not raw records or a `DataFrame`.

## Write the measurement contract first

A representativeness value is only interpretable when the observed and
reference distributions count compatible things. Record at least the following
decisions before preparing the evidence.

| Decision | What to specify | Why it matters |
|---|---|---|
| `unit_of_analysis` | One prompt, response, conversation, sentence, person, annotation, or label membership | Every submitted record increments one count. Changing the unit changes the distribution. |
| Axis definition | The property being counted and the meaning of every category | Category names alone rarely define the construct. |
| `label_source` | Dataset field, human annotation protocol, external classifier, or another declared source | `fairLMs` consumes labels; it does not create them or validate their substantive correctness. |
| Support | The complete set of possible categories, including any retained `unknown` category | Observed and reference support must match exactly. |
| Multi-label policy | Collapse to one label, count memberships, or run separate binary audits | The current adapters assign one category to each submitted row. |
| Unknown policy | Retain as a category, exclude before adaptation, or resolve by a documented rule | Missing values are not silently dropped. Exclusion changes coverage and the denominator. |
| Coverage | Source units, included units, excluded units, exclusion reasons, and coverage rate | The package only sees submitted evidence and cannot recover omitted units. |
| Reference scope | Population/design target, geography, period, category definitions, and denominator | A mathematically valid but incompatible reference is not meaningful. |
| Provenance | Dataset version, split, query or script version, label protocol/model, hashes, and dates | The result should be reproducible and auditable. |

`unit_of_analysis`, `label_source`, and the coverage fields are recommended
provenance conventions, not separately validated fields in the current API.
Store them as JSON-safe provenance, for example:

```python
measurement_contract = {
    "unit_of_analysis": "one benchmark prompt",
    "axis_definition": "author-declared region for the prompt source",
    "label_source": "benchmark metadata field region_code, schema v2",
    "multi_label_policy": "one mutually exclusive label per prompt",
    "unknown_policy": "retain missing metadata as the unknown category",
    "source_unit_count": 8,
    "included_unit_count": 8,
    "excluded_unit_count": 0,
    "coverage_rate": 1.0,
}
```

Provenance values may contain JSON-compatible dictionaries, lists, strings,
booleans, finite numbers, and `null`/`None`. Do not put paths, exceptions,
datetime objects, sets, NaN, or infinity into provenance without first
converting them to a portable representation.

## Raw text is not automatically prepared

The representativeness API does **not**:

- split documents or conversations into sentences;
- classify text into demographic or other categories;
- infer which field contains a category;
- infer a category mapping or the complete support from observed values; or
- infer a population prior from a dataset name.

For example, a list of strings is not valid record evidence:

```python
raw_documents = ["First document.", "Second document."]

# Not supported: each item is a string, not a mapping row with an existing label.
# RepresentationEvidence.from_records(raw_documents, ...)
```

If the unit is a sentence, split the text before using `fairLMs` and record the
splitter and its version. If labels come from a classifier, run it before using
`fairLMs` and record the model or prompt version, thresholds, calibration,
manual-review policy, and coverage. Classification and segmentation error are
then part of the measurement uncertainty; `b_rep` does not estimate them.

Extra text fields are allowed in record rows, but the adapter ignores them. It
reads only the explicitly named group field and counts the already-assigned
category.

## Path 1: use existing aggregate counts

Use the direct constructor when counts were produced by a database query,
annotation report, or other upstream process. Include zero-count cells so the
evidence records the complete support.

```python
from fairLMs.datasets.diagnostics import RepresentationEvidence

evidence = RepresentationEvidence(
    axis="source_region",
    counts={"north": 5, "south": 3, "unknown": 0},
    source="Example benchmark metadata export, evaluation split",
    provenance={
        "unit_of_analysis": "one benchmark prompt",
        "label_source": "metadata export schema v2",
        "multi_label_policy": "one mutually exclusive label per prompt",
        "unknown_policy": "retain as an explicit category",
        "source_unit_count": 8,
        "included_unit_count": 8,
        "excluded_unit_count": 0,
        "coverage_rate": 1.0,
        "query_id": "region-counts-v2",
    },
)

assert evidence.total == 8
assert evidence.support == ("north", "south", "unknown")
assert evidence.to_dict()["counts"] == {
    "north": 5,
    "south": 3,
    "unknown": 0,
}
```

Counts must be integers, not percentages, probabilities, fractional weights, or
booleans. They must be non-negative, contain at least two categories, and have a
positive total.

## Path 2: count labelled records

`from_records` accepts an iterable whose items are mappings. It reads one
explicit field per row. Use `value_map` when raw codes differ from the canonical
support labels.

```python
from fairLMs.datasets.diagnostics import RepresentationEvidence

records = [
    {"prompt_id": "p1", "region_code": "N", "text": "Example one"},
    {"prompt_id": "p2", "region_code": "S", "text": "Example two"},
    {"prompt_id": "p3", "region_code": "N", "text": "Example three"},
    {"prompt_id": "p4", "region_code": None, "text": "Example four"},
]

evidence = RepresentationEvidence.from_records(
    records,
    axis="source_region",
    group_field="region_code",
    support=("north", "south", "unknown"),
    source="Example benchmark records, evaluation split",
    value_map={"N": "north", "S": "south", None: "unknown"},
    provenance={
        "dataset_id": "example-benchmark-v2",
        "unit_of_analysis": "one benchmark prompt",
        "label_source": "record field region_code",
        "multi_label_policy": "one mutually exclusive label per prompt",
        "unknown_policy": "map null to unknown",
        "source_unit_count": 4,
        "included_unit_count": 4,
        "excluded_unit_count": 0,
        "coverage_rate": 1.0,
    },
)

assert dict(evidence.counts) == {
    "north": 2,
    "south": 1,
    "unknown": 1,
}
assert evidence.provenance["field_mapping"] == {"group": "region_code"}
assert evidence.provenance["value_map_used"] is True
```

The adapter preserves the complete portable `value_map` in provenance, including
entries not observed in these rows. Raw mapping keys may be `None`, booleans,
integers, finite floats, or strings. Canonical values must be non-empty strings
in the declared support.

Without `value_map`, every group-field value must already be a canonical support
string. Values are matched exactly; case, whitespace, and synonyms are not
normalized.

## Path 3: count one DataFrame column

`from_dataframe` applies the same contract to one explicitly named pandas
column. Other columns are ignored.

```python
import pandas as pd

from fairLMs.datasets.diagnostics import RepresentationEvidence

frame = pd.DataFrame(
    {
        "prompt_id": ["p1", "p2", "p3", "p4"],
        "region_code": ["N", "S", "S", None],
        "text": ["One", "Two", "Three", "Four"],
    }
)

# pandas may materialize a missing value as NaN or pd.NA. Apply the declared
# unknown policy before handing the column to fairLMs.
frame["region_code"] = frame["region_code"].fillna("UNSPECIFIED")

evidence = RepresentationEvidence.from_dataframe(
    frame,
    axis="source_region",
    group_column="region_code",
    support=("north", "south", "unknown"),
    source="Example benchmark DataFrame, evaluation split",
    value_map={"N": "north", "S": "south", "UNSPECIFIED": "unknown"},
    provenance={
        "dataset_id": "example-frame-v2",
        "unit_of_analysis": "one DataFrame row / benchmark prompt",
        "label_source": "column region_code",
        "multi_label_policy": "one mutually exclusive label per prompt",
        "unknown_policy": "map null to unknown",
        "source_unit_count": 4,
        "included_unit_count": 4,
        "excluded_unit_count": 0,
        "coverage_rate": 1.0,
    },
)

assert dict(evidence.counts) == {
    "north": 1,
    "south": 2,
    "unknown": 1,
}
assert evidence.provenance["adapter"] == "dataframe"
```

The selected column name must exist and be unique. Missing values are not
dropped. If pandas represents missing data as `NaN` or `pd.NA`, replace it with
a portable explicit string such as `"UNSPECIFIED"` before adapting, as shown
above, then map that marker according to the declared unknown policy.

## Multi-label data

The current record and DataFrame adapters assign exactly one category to every
submitted row. They do not accept a list of labels in the group field, and direct
counts do not accept fractional weights.

Choose and document one of these preprocessing strategies:

1. **One primary label per unit.** Collapse labels with a deterministic,
   scientifically justified rule. The unit remains the original record.
2. **Membership counts.** Expand a record into one row per label. The unit is now
   a `(record, label membership)` pair, so the observed total can exceed the
   number of records. The reference must use the same membership denominator.
3. **Separate binary audits.** For each label, build `present`/`absent` counts and
   a compatible binary reference. This answers several distinct questions; it
   is not one multinomial audit.

Do not duplicate rows to handle multi-label data while continuing to describe
the unit as “one record.” That makes the measurement contract and denominator
inconsistent.

## Unknown values and coverage

There is no universal unknown policy. Two common defensible choices are:

- retain `unknown` as an explicit category in both observed and reference
  support; or
- exclude unresolved units before adaptation and report the resulting coverage,
  exclusion rule, and exclusion counts in provenance.

The adapter deliberately rejects missing or unmapped values instead of silently
dropping them. Mapping `None` to `unknown`, as shown above, is explicit retention.
If units are excluded upstream, `RepresentationEvidence.total` and the reported
`sample_count` describe only included units. `fairLMs` cannot determine the
source-universe denominator or verify the reported coverage.

Pay special attention to differential missingness. A high overall coverage rate
can still hide systematic omissions from one category, source, language, or time
period.

## Complete `b_rep` audit

The following script creates evidence, declares a population reference, runs the
diagnostic, checks applicability, and serializes the full report.

```python
from fairLMs.datasets.diagnostics import (
    DatasetAuditSpec,
    ReferenceDistribution,
    RepresentationEvidence,
    audit_representativeness,
)

evidence = RepresentationEvidence(
    axis="source_region",
    counts={"north": 48, "south": 32, "unknown": 0},
    source="Example benchmark metadata export, evaluation split",
    provenance={
        "dataset_version": "2.1",
        "unit_of_analysis": "one benchmark prompt",
        "label_source": "author-declared region_code, schema v2",
        "multi_label_policy": "one mutually exclusive label per prompt",
        "unknown_policy": "retain unknown as an explicit category",
        "source_unit_count": 80,
        "included_unit_count": 80,
        "excluded_unit_count": 0,
        "coverage_rate": 1.0,
        "extraction_script_sha256": "example-not-a-production-hash",
    },
)

reference = ReferenceDistribution(
    axis="source_region",
    probabilities={"north": 0.45, "south": 0.45, "unknown": 0.10},
    source="Example population table, release 2025",
    purpose="population",
    population="Eligible prompt-source population",
    geography="Example region",
    period="2025",
    provenance={
        "table_id": "population-table-2025-region",
        "category_definition_version": "region-schema-v2",
        "denominator": "all eligible source records",
    },
)

spec = DatasetAuditSpec(
    target_name="example-benchmark-v2.1",
    target_kind="benchmark_dataset",
    task_family="free_text",
    design_stance="population_proxy",
    references={"source_region": reference},
    requested_components=("b_rep",),
)

report = audit_representativeness(evidence, spec)
result = report.components["b_rep"]

if result.status.value == "ready":
    print(f"B_rep={result.value:.12f} {result.details['unit']}")
    print("observed:", dict(result.details["observed_distribution"]))
    print("reference:", dict(result.details["reference_distribution"]))
else:
    print(result.status.value, result.reason_code, result.reason)

print(report.to_json())
```

With the inputs above, the first line is approximately:

```text
B_rep=0.112748699602 nats
```

For support size `K`, smoothing mass `epsilon` (default `0.01`), normalized
observed proportions `p_i`, and reference probabilities `q_i`, the diagnostic
computes:

```text
alpha = epsilon / K
p'_i = (p_i + alpha) / (1 + epsilon)
q'_i = (q_i + alpha) / (1 + epsilon)
B_rep = sum_i p'_i * ln(p'_i / q'_i)
```

Both distributions are smoothed, the direction is `observed || reference`, and
the logarithm is natural, so the unit is nats. Smoothing is applied to normalized
probabilities, not as pseudocounts added to raw counts. Scaling every observed
count by the same factor therefore does not change `B_rep`.

`B_rep` is descriptive divergence, not a fairness pass/fail rule. A value has no
universal acceptable threshold. `population_proxy` and `stress_test` use the
same mathematics; a ready stress-test report adds a warning because divergence
may be intentional.

## Validation errors versus diagnostic states

Malformed inputs fail immediately with `TypeError` or `ValueError`. Examples
include non-integer counts, an empty input, missing record fields, unmapped
values, a non-unique DataFrame column, non-portable provenance, and reference
probabilities that do not sum to `1.0` within the allowed absolute tolerance.

Once valid evidence and a valid audit specification exist, scientific
applicability is represented in the result rather than with a numeric sentinel:

| Condition | Component status | `reason_code` | Value |
|---|---|---|---|
| `b_rep` was not requested | `not_applicable` | `component_not_requested` | `None` |
| Target is not `benchmark_dataset` | `not_applicable` | `target_kind_not_supported` | `None` |
| No reference for the evidence axis | `blocked` | `missing_reference` | `None` |
| Observed and reference support differ | `blocked` | `reference_support_mismatch` | `None` |
| Inputs are applicable and computation succeeds | `ready` | `None` | finite float |

Always inspect `status` before formatting `value` as a number.

## Common mistakes

### Passing text and expecting automatic labels

```python
# Wrong shape: rows are strings, and no label source has been declared.
rows = ["A document about the north.", "A document about the south."]
```

Segment and label upstream. Then pass mapping rows such as
`{"text": ..., "region_label": "north"}` and name `region_label` explicitly.

### Omitting zero-count support cells

```python
# Incomplete if "unknown" is part of the declared taxonomy.
counts = {"north": 5, "south": 3}
```

Use `{"north": 5, "south": 3, "unknown": 0}` and include `unknown` in the
reference as well. The diagnostic does not silently align mismatched support.

### Treating missing values as an implicit filter

The adapters raise on missing values. Either map a normalized missing marker to
an explicit category or filter upstream and record coverage and exclusions.

### Supplying fractions or sample weights as counts

```python
# Invalid: direct evidence requires integer counts.
counts = {"north": 2.5, "south": 1.5}
```

Choose a unit that can be counted with integers. Fractional multi-label weights
are not supported by the current evidence contract.

### Using incompatible reference semantics

Matching category strings are necessary but not sufficient. A sentence-level
observed distribution should not be compared with a person-level reference, and
membership counts should not be compared with a record-level reference. Align
unit, taxonomy, population, geography, period, coverage, and denominator.

### Using probabilities that do not sum to one

Reference probabilities outside an absolute `1e-9` sum tolerance are rejected;
they are not silently normalized. Values within that tolerance are
canonicalized onto the probability simplex, and the report records the original
sum and whether canonicalization occurred.

### Storing process-local objects in provenance

Convert paths and dates to strings, arrays to lists, and library-specific scalar
types to ordinary finite Python numbers. Provenance mappings require string
keys. The adapter reserves `adapter`, `field_mapping`, `value_map`, and
`value_map_used` for its own reproducibility metadata.

## Reproducibility checklist

Before publishing or comparing a result, verify that:

- the unit of analysis is explicit and unchanged across observed/reference data;
- the axis and every category are operationally defined;
- the label source and version are recorded;
- sentence splitting or classification, if any, happened upstream and is
  documented;
- the multi-label and unknown policies are explicit;
- source, included, and excluded unit counts reconcile with the coverage rate;
- zero-count categories are retained when they belong to the support;
- observed and reference support match exactly;
- reference purpose, population, geography, period, and denominator are stated;
- the design stance is justified; and
- the full JSON report and enough provenance to rebuild the evidence are saved.
