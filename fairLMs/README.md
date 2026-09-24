# FairLMs

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An installable local library for fairness definitions, dataset loaders,
dataset diagnostics, bias metrics, and mitigation methods for language models.
The runtime library was synchronized from JMLR Library release 0.5.0; see
[`UPSTREAM.md`](UPSTREAM.md) for the exact source commit and integration notes.

The long-term goal is a **stable, sklearn-style API**: import a metric, call `compute(...)`, and get a result — without caring where the implementation lives.

## Install

```bash
git clone https://github.com/vanbanTruong/Fairness-in-Large-Language-Models.git
cd "Fairness-in-Large-Language-Models/fairLMs"
pip install -e .
```

The public import name in this repository is case-sensitive: `fairLMs`.

### Optional extras

```bash
pip install -e ".[openai]"   # API-backed decoder metrics (CR, CTF, BA)
pip install -e ".[dev]"      # pytest
pip install -e ".[all]"      # openai + common extras
```

An editable install means your edits take effect immediately, with no reinstall.

### Requirements

Python ≥ 3.10 — the floor for `torch`, `transformers` and `datasets`.

Installed automatically: `torch`, `transformers`, `datasets`, `numpy`, `pandas`,
`scipy`, `scikit-learn`, `wordfreq`, `nltk`. The last three are needed at import
time by `counterfactual_auc` / `normalized_position_distance`,
`lexical_frequency_proportion`, and `morphological_choice_divergence`
respectively.

## Versioning

The synchronized upstream version is single-sourced in
[`_version.py`](_version.py). The source commit is recorded separately in
[`UPSTREAM.md`](UPSTREAM.md).

## Quick start

```python
from fairLMs.definitions import CrowSPairsScore, list_metrics
from fairLMs.datasets import CrowSPairs
from fairLMs.definitions.models import HuggingFaceModel

model = HuggingFaceModel("bert-base-uncased", task="mlm")
result = CrowSPairsScore().compute(model, CrowSPairs(n_max=50))
print(result.score, result.by_category)

# Discover metrics
print(list_metrics())
```

Mitigation, diagnostics, and loaders are available from the same local package:

```python
from fairLMs.mitigation import (
    CorpusWithLexicon,
    CounterfactualDataAugmentation,
    SwapLexicon,
    TextRecords,
)
from fairLMs.datasets import BBQ

lexicon = SwapLexicon(axis="gender", pairs=[("he", "she")], source="example")
records = TextRecords(texts=["He is a nurse."], source="example")
result = CounterfactualDataAugmentation().apply(
    None, CorpusWithLexicon(records=records, lexicon=lexicon)
)
print(result.result.texts)

rows = BBQ(n_max=10).load()
```

See [`examples/representativeness_diagnostic.py`](examples/representativeness_diagnostic.py)
for a complete dataset diagnostic call.

### The contract

Every metric follows scikit-learn's estimator conventions:

```python
Metric(**config).compute(model, data) -> MetricResult
```

* **Configuration** goes in the constructor, keyword-only, and is introspectable
  via `get_params()` / `set_params()` — so metrics can be cloned or swept.
* **Data** is the second positional argument: a `FairnessDataset`, a plain
  sequence, or a typed container from `fairLMs.definitions.data` for metrics that
  need several labelled sets.
* **Unknown keywords raise `TypeError`** instead of silently using a default.

```python
from fairLMs.definitions import WEAT, SEAT, WordSets

words = WordSets(target_1=t1, target_2=t2, attribute_1=a1, attribute_2=a2)
WEAT(n_samples=10_000).compute(model, words)
SEAT(pooling="cls").compute(model, words)      # same data, different metric

WEAT(pooling="cls").get_params()               # {'n_samples': 10000, 'pooling': 'cls'}
```

The published association tests ship pre-wrapped in `fairLMs.definitions.resources`, so a
standard run needs no term lists at all — and swapping the checkpoint does not
change the call:

```python
from fairLMs.definitions.resources import weat_c1, list_word_sets
from fairLMs.definitions import SEAT
from fairLMs.definitions.models import HuggingFaceModel

metric = SEAT(n_samples=1_000, seed=0)
for ckpt in ("bert-base-uncased", "roberta-base"):
    result = metric.compute(HuggingFaceModel(ckpt, task="encoder"), weat_c1)
    print(ckpt, result.score, result.details["p_value"])

list_word_sets()   # ['weat_c1', …, 'seat_c1', …]
```

`seed` is ordinary constructor config, so a reported p-value is reproducible
from the metric's own parameters — no `np.random.seed` at the call site, and
nothing about the run recorded outside `get_params()`:

```python
SEAT(n_samples=1_000, seed=0).get_params()
# {'n_samples': 1000, 'pooling': 'mean', 'seed': 0, 'templates': None}
```

`weat_c1` *is* a `WordSets`, so user-supplied evidence goes through the exact
same call with no adapter code:

```python
mine = WordSets(target_1=names_a, target_2=names_b,
                attribute_1=pleasant, attribute_2=unpleasant)
metric.compute(model, mine)
```

Containers validate at construction, so mistakes fail immediately:

```python
>>> ContextSets(["a sentence"], ...)
TypeError: target_1 must be a mapping of term -> list of context sentences, got
list. (A flat list of sentences is not accepted; CEAT samples contexts per term,
so terms must be keyed.)
```

### Metrics that need no model

Five metrics score predictions you already have. They also exist as plain
functions, mirroring `sklearn.metrics`:

```python
from fairLMs.definitions import equal_opportunity_gap, accuracy_disparity

equal_opportunity_gap(y_true, y_pred, groups, g1="A", g2="B")   # -> float
accuracy_disparity(scores_stereotype, scores_counter)           # -> float
```

Also available: `inference_bias_score`, `fair_inference_score`,
`context_based_disparity`. Use the class form
(`EqualOpportunityGap`, …) when you want the full `MetricResult` with
diagnostics.

### Dataset diagnostics

Dataset diagnostics are a separate, dataset-first API. They consume explicit
evidence and audit intent rather than a model, and return a structured report
whose components can be `ready`, `blocked`, or `not_applicable`. Missing
evidence is never represented as a score of zero.

The first diagnostic is axis-level representativeness (`b_rep`): smoothed
`KL(observed || reference)` in nats. It requires an explicit reference and its
provenance; the package does not infer a population prior from a dataset name.

Alongside it, `b_leak` measures group-trait association as smoothed normalized
mutual information over a declared lexicon pair space, and `b_constr` is
published as a vector of eight independent construction slots (`b_min`,
`b_equiv`, `b_gram`, `b_diff_len`, `b_diff_dep`, `b_frame`, `b_opt`, `b_temp`)
with no aggregate score. Three of the slots read a quantity that needs an
optional backend -- a sentence embedding (`b_equiv`), a grammatical-error count
(`b_gram`) or a dependency-tree depth (`b_diff_dep`). Each is a real class that
takes `backend=`; `fairLMs.datasets.diagnostics.backends` ships a reference backend for
each, and `pip install "fairLMs[nlp]"` adds the grammar-checker and parser
dependencies. Without a backend such a slot is `blocked` with a reason code
naming the missing backend once it was requested and its required evidence view
is present, and otherwise `not_applicable` for the ordinary reason (not
requested, unsupported target kind, or view not supplied) --
`BACKEND_CONSTRUCTION_SLOTS` and `CONSTRUCTION_BACKEND_REQUIREMENTS`, not the
status, name the slots that need a backend. `audit_dataset(evidence, spec, axis=...)` runs the requested
components over one axis of a `DatasetEvidence` and returns one report, while
`audit_scores` remains the separate entry point for row-level scores. See the
[dataset audit guide](docs/guides/dataset-audit.md).

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
    target_name="my-unregistered-benchmark",
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

`population_proxy` and `stress_test` use the same mathematics but not the same
interpretation. A stress test may deliberately over-sample a category; the
report therefore warns that divergence is descriptive evidence rather than an
automatic fairness failure. Use `RepresentationEvidence.from_records(...)` or
`.from_dataframe(...)` with explicit field names, support, and optional value
mapping for unfamiliar schemas. Adapters preserve the complete JSON-safe value
mapping in provenance so a recoding can be reproduced. Reference probabilities
outside an absolute `1e-9` sum tolerance are rejected; values inside that
tolerance are canonicalized onto the probability simplex and the report records
both the input sum and whether canonicalization occurred.

Start with [Preparing audit evidence](docs/preparing_audit_evidence.md) for the
evidence-layer boundary and input checklist. The detailed
[representativeness guide](docs/preparing_representativeness_evidence.md)
covers supported input paths, raw text, coverage, references, and a complete
`b_rep` example.

#### Scoring instrument audits

Scorer diagnostics audit scores that already exist at row level. They do not
infer groups from raw text or run a model/scorer to generate scores.
`score_mean_gap` is the descriptive maximum absolute group-mean difference in
the scorer's native score units. `score_rate_gap` first applies one explicit,
serialized score-to-event rule to every group, then reports the maximum absolute
group event-rate difference as a proportion. `score_wasserstein_1_gap` compares
the complete one-dimensional empirical score distributions and reports their
maximum pairwise Wasserstein-1 distance in the scorer's native score units.
`score_counterfactual_sensitivity` separately averages absolute score changes
inside complete, explicitly declared two-condition pairs.

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
    diagnostics=(
        ScorerMeanGap(),
        ScorerRateGap(transform=rate_transform),
        ScorerWasserstein1Gap(),
    ),
)
mean_result = report.components["score_mean_gap"]
rate_result = report.components["score_rate_gap"]
w1_result = report.components["score_wasserstein_1_gap"]
print(mean_result.value, mean_result.details["unit"])  # 0.7 score_units
print(rate_result.value, rate_result.details["unit"])  # 1.0 proportion
print(w1_result.value, w1_result.details["unit"])  # 0.7 score_units
```

Paired sensitivity uses independent evidence because group marginals do not
preserve which rows are counterparts:

```python
from fairLMs.datasets.diagnostics import (
    PairedScores,
    ScorerCounterfactualSensitivity,
)

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
    paired,
    paired_spec,
    diagnostic=ScorerCounterfactualSensitivity(),
)
print(paired_report.components["score_counterfactual_sensitivity"].value)  # 0.4
```

There is no implicit threshold, direction, or boundary rule. `score_range`
validates scores; it is neither a threshold nor a Wasserstein normalization. A
missing rate transform is `blocked`, not a zero gap. Mean, rate, and W1 gaps
answer different questions, and none is a causal claim, fairness pass/fail rule,
or error-rate metric. Paired sensitivity supports identity-isolated causal
language only if the pairs really differ solely in the declared intervention;
the package validates pair completeness but cannot verify that semantic claim
from scores. W1 and paired sensitivity are symmetric primary values and carry
no higher/lower direction.
`higher` plus `inclusive=True` is the paper-exact `score >= threshold`
definition; the report marks lower-tail and exclusive-boundary variants
separately. Treat each value as a per-dataset, per-scorer diagnostic, and rate
gaps additionally as per-rule, rather than using them to rank datasets or
unrelated scorer scales. See
[Preparing audit evidence](docs/preparing_audit_evidence.md) and the runnable
[`score_rate_gap`](examples/scorer_rate_gap_diagnostic.py) and
[`score_wasserstein_1_gap`](examples/scorer_distribution_gap_diagnostic.py)
and
[`score_counterfactual_sensitivity`](examples/scorer_counterfactual_sensitivity_diagnostic.py)
examples.

Shared loaders:

```python
from fairLMs.datasets import CrowSPairs, BBQ, StereoSet
from fairLMs.definitions.models import load_masked_lm

crows = CrowSPairs().load()
bbq = BBQ(categories=["Age"]).load()
loaded = load_masked_lm("bert-base-uncased")
```

Leaf runners under `definitions/` are short demos of the same public API:

```bash
python -m fairLMs.definitions.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cps.main
```

See also the local [`examples/`](examples/) directory.

## Package layout

```
fairLMs/
├── datasets/       # Loaders, dataset diagnostics, and small resources
│   ├── diagnostics/
│   └── resources/
├── definitions/    # Public API and the 33 architecture-organized definitions
│   ├── data.py     # Validated input containers (WordSets, ProbeSet, …)
│   ├── functional.py # sklearn.metrics-style functions (model-free metrics)
│   ├── core/       # Applicability, parameters, and provenance
│   ├── models/     # Hugging Face and OpenAI adapters
│   ├── utils/      # Numerical metric helpers
│   ├── resources/  # WEAT/SEAT word sets
│   ├── encoder_only/
│   ├── decoder_only/
│   └── encoder_decoder/
├── mitigation/     # Pre-, in-, intra-, and post-processing mitigators
├── docs/           # Project documentation
├── tests/          # Unit, packaging, and workflow tests
├── examples/       # Runnable examples
├── scripts/        # Documentation and release tooling
├── __init__.py
└── _version.py
```

Repo-root `examples/` has additional runnable snippets.

Every metric exposes the same method: `compute(...)`.

## Datasets

Eighteen loaders share one interface. Only the small CrowS-Pairs CSV is
distributed; larger corpora are fetched or pointed at. See [Loaders](docs/registry/loaders.md) for
the generated table with every constructor argument.

| Class | Source | Notes |
|-------|--------|--------|
| `CrowSPairs` | Bundled CSV under `datasets/resources/crows_pairs/` | Stereotype / anti pairs |
| `BBQ` | Hugging Face `heegyu/bbq`, cached at first use | Optional `context_condition` filter |
| `StereoSet` | Hugging Face (`stereoset` / `McGill-NLP/stereoset`) | Pairs or triples |
| `BiasInBios` | Hugging Face `LabHC/bias_in_bios` | Profession / gender helpers |
| `WinoBias` | Hugging Face `wino_bias` | Occupation direction helpers |
| `XNLIReligionPairs` | Hugging Face XNLI + templates | Religion swap pairs |
| `BOLD` | Hugging Face `AmazonScience/bold` | Generation prompts, five domains |
| `HONEST` | Hugging Face `MilaNLProc/honest` | Masked templates, `binary` / `queer_nonqueer` |
| `RealToxicityPrompts` | Hugging Face `allenai/real-toxicity-prompts` | `challenging_only`, `min_toxicity` filters |
| `HolisticBias` | Hugging Face `fairnlp/holistic-bias` | `sentences` or `nouns`, per-axis counts |
| `EquityEvaluationCorpus` (`EEC`) | Hugging Face `peixian/equity_evaluation_corpus` | Matched gender / race templates |
| `GAP` | Hugging Face `google-research-datasets/gap` | Balanced masculine / feminine pronouns |
| `Winogender` | Hugging Face `oskarvanderwal/winogender` | Occupation skew helpers need a local `root=` |
| `BiasNLI` | Local `root=` | The retained three-column release, verbatim |
| `RedditBias` | Local `root=` | `comments` / `pairs` / `phrases`, five axes |
| `GrepBiasIR` | Local `root=` | Query/document pairs, gendered writings |
| `UnQover` | Local `root=` | Streams the multi-GB slotmaps |
| `TrustGPT` | Bundled templates + your norms | No data release; prompts are built |

The four `root=` benchmarks are distributed only from their own project pages and never
download anything; the loader raises with the project URL if you have not
pointed it at a copy.

### Bundled word sets

Association tests need four labelled term lists rather than a corpus, so they
are importable constants instead of loader classes. Each one is an
already-validated `WordSets`, ready to pass straight to `compute`:

| Name | Test | Terms |
|------|------|-------|
| `weat_c1` … `weat_c4` | Caliskan et al. (2017) C1–C4 | Race, gender, disease, age |
| `seat_c1` … `seat_c4` | May et al. (2019), expanded name lists | Same four axes |

```python
from fairLMs.definitions.resources import weat_c2, get_word_set, WORD_SET_LABELS

get_word_set("seat_c1")                # same objects, by name
WORD_SET_LABELS["weat_c2"]             # 'C2 – Gender (Male/Female names × Career/Family)'
```

Both families work with `WEAT` and `SEAT`; the `seat_*` lists are the larger
name sets the sentence templates were sized for. All eight have balanced target
lists, which `SEAT` requires. CEAT is not included — it consumes `ContextSets`
(terms keyed to context sentences), not four flat lists.

## Models

```python
from fairLMs.definitions.models import (
    HuggingFaceModel,
    load_masked_lm,      # task="mlm"
    load_encoder,        # task="encoder"
    load_sequence_classifier,
    load_seq2seq,
    load_causal_lm,
    OpenAIModel,
)

HuggingFaceModel("roberta-base", task="sequence_classification").load()
OpenAIModel("davinci-002").load()  # needs OPENAI_API_KEY; pip install fairLMs[openai]
```

Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for gated models such as Llama-2.

## Design principles

1. **One verb for metrics** — `compute` (sklearn’s `fit` / `predict` analogue).
2. **Separate metrics from datasets** — reuse the same metric on CrowS-Pairs, StereoSet, or custom data.
3. **Stable public surface** — internals under `definitions/` can change without breaking user code.
4. **Book-aligned taxonomy** — `definitions/{encoder_only,encoder_decoder,decoder_only}/{intrinsic_bias,extrinsic_bias}/…` mirrors the conceptual organization of the accompanying textbook.
5. **Applicability before computation** — dataset diagnostics report missing or
   incompatible evidence instead of manufacturing a numeric result.

## Development

```bash
pip install -e ".[dev]"
pytest                  # contract suite over every metric in the registry
# Run a leaf metric script:
python -m fairLMs.definitions.encoder_only.intrinsic_bias.similarity_based.weat.main
```

`tests/test_common.py` is the analogue of scikit-learn's `check_estimator`: it
runs the parameter/repr/keyword contract across `METRIC_REGISTRY`, so a new
metric that breaks the shape fails there rather than surprising a user. It needs
no network or model weights.

Tests that do need model weights skip cleanly when the weights aren't cached, so
the suite is hermetic:

```bash
HF_HUB_OFFLINE=1 pytest -q     # what CI runs; a few seconds, no downloads
```

The packaging test catches a class of problem that an editable development
environment can hide: `fairLMs.definitions` eagerly imports every metric family, so any
third-party module imported at module scope under `fairLMs/definitions/` is a
hard requirement of `import fairLMs`. If such a dependency is only listed in an
extra, a clean installation cannot import the package. The local
[`tests/test_packaging.py`](tests/test_packaging.py) guard walks the source AST
and names the offending file.

`fairLMs.definitions.io.results_to_csv` writes to a caller-supplied directory,
or to `./fairlms-results` by default. Run artifacts never go into the package.

## License

MIT
