# Changelog

## 0.5.0

Corrected StereoSet roles/context and CAT LMS comparisons; connected public
SelfDebiasing generation to probability damping; required projection layer and
pair identity; measured INLP after the final projection; preserved API model
selection and stopped successful retries. XNLI now produces deduplicated
counterfactual evidence. Added configured before/after evaluation, JSON-safe
undefined scores, run provenance, revision controls, real offline model tests,
source archive validation, and corpus checksums/attribution. No upstream release
or paper update is implied by this development version.

**Twelve benchmark loaders added.** The registry goes from 6 loaders to 18,
covering the remaining datasets studied in the accompanying survey: `BOLD`,
`HONEST`, `RealToxicityPrompts`, `HolisticBias`, `EquityEvaluationCorpus`
(alias `EEC`), `GAP`, `Winogender`, `BiasNLI`, `RedditBias`, `GrepBiasIR`,
`UnQover` and `TrustGPT`. Each subclasses `FairnessDataset` and returns
`load()` examples a metric's `data` argument accepts.

Where the bytes come from is now declared rather than inferred: every loader
carries a `data_origin` string, and the generated
[Loaders](registry/loaders.md) table groups by it. Seven of the new loaders
download published data files from the Hugging Face Hub at first use, via
`huggingface_hub.hf_hub_download` rather than `datasets.load_dataset`, because
several of those repositories still ship a loading script that `datasets>=3`
refuses to run. Four benchmarks are distributed only from their own project
page and take a `root=` instead of downloading anything; `TrustGPT` has no data
release at all, so its loader reproduces the three published prompt templates
and takes the social norms from the caller. Nothing new is vendored into the
wheel.

Two details worth knowing before you use them. The archive published under the
Bias-NLI name is a three-column SNLI-shaped release, not the template-expanded
inference set of Dev et al. (2020), and there is no deterministic join between
the two, so `BiasNLI` returns the release verbatim and asserts nothing about
how individual rows were built. `UnQover` streams its slotmaps incrementally,
because the released files reach 2.2 GB and `json.load` on the largest would
need tens of gigabytes before `n_max` could take effect.

**Construction vector complete.** `b_equiv`, `b_gram` and `b_diff_dep` are
implemented as `SemanticEquivalence`, `GrammarConsistency` and
`DependencyDepthDisparity`, each taking an optional `backend=` that satisfies
`EmbeddingBackend`, `GrammarCheckerBackend` or `DependencyParserBackend`.
`fairLMs.datasets.diagnostics.backends` ships `HuggingFaceEmbeddingBackend` (core
dependencies), `LanguageToolGrammarBackend` (`fairLMs[grammar]`) and
`SpacyDependencyBackend` (`fairLMs[parse]`); `fairLMs[nlp]` installs both extras.
The backend revision is recorded in component provenance. Without a backend the
three slots block by name exactly as before, so existing reports are unchanged;
the registry now holds 14 diagnostics.

**CrowS-Pairs orientation fixed.** The loader had swapped `sent_more` and
`sent_less` for the 218 anti-stereotype rows. In CrowS-Pairs `sent_more` is the
more stereotyping sentence in every row, and the published metric counts a
preference for it regardless of the `stereo_antistereo` label. With the fix,
`CrowSPairsScore` on `bert-base-uncased` reproduces Nangia et al. (2020) to one
decimal: 60.5 overall, 61.1 on stereotype pairs, 56.9 on anti-stereotype pairs
(previously 58.5 / 61.1 / 43.1). Scores reported with earlier development
versions on anti-stereotype rows, and therefore the gender category, were
inverted.

**Bundled corpora consolidated.** 84 files under `fairLMs/definitions/` that were
byte-identical to a copy under `fairLMs/datasets/resources/` were deleted, removing
359,582,594 bytes. `fairLMs/datasets/resources/` is now the single source, and
`fairLMs/datasets/resources/checksums.json` pins the bytes that remain. Loaders resolve
`fairLMs/datasets/resources/` first and still fall back to legacy locations, so externally
maintained checkouts are unaffected. Historical MCD outputs from earlier
development are no longer distributed; they were never validation results for
this version.


Versions follow the scheme described in the
[README](https://github.com/vanbanTruong/Fairness-in-Large-Language-Models/tree/main/fairLMs#versioning): while the
package is pre-1.0, the minor version moves on behaviour changes and the patch
version on additions and fixes. Metric definitions can change between minor
versions, so pin a version when reporting a score. See
[Citing FairLMs](citation.md).

## Unreleased

**Runtime layout consolidated.** The installed implementation now has three
feature packages: `fairLMs.datasets` (including diagnostics),
`fairLMs.definitions` (including shared contracts, model adapters, numerical
helpers, and word sets), and `fairLMs.mitigation`. The former root-level
implementation and compatibility modules were removed. Documentation, tests,
examples, and maintenance scripts remain source-only project directories.

**Large dataset snapshots removed.** CrowS-Pairs remains bundled because its
CSV is small. BBQ now downloads requested JSONL categories from its Hugging
Face mirror into the standard cache, while still accepting `data_dir=` for an
existing local copy. Generated outputs, charts, duplicate corpora, and package
artifact directories are no longer distributed.

**Mismatched model heads are refused.** Every metric now declares
`required_task`, and `fairLMs.definitions.resolve.check_task` compares it against
the `task` a checkpoint was loaded with. Passing a `task="encoder"` model to
`crows_pairs_score` previously failed with an `AttributeError` on a missing
`.logits`, and the reverse mismatch could return numbers read from a randomly
initialized head; both now raise a `TypeError` that names the required task,
the actual task and the fix. Raw `(tokenizer, model)` tuples carry no task and
are unaffected. The 9 metrics that score precomputed predictions or call an API
declare `required_task = None`.

**`stereotypical_divergence` accepts a real custom scorer.** `metric_fn` was
dispatched by function `__name__` through a two-entry table, so any callable
other than `pronoun_accuracy` or `age_accuracy` failed with a bare `KeyError`.
It now takes an accompanying `predict_fn`, and an unpaired custom scorer raises
a `ValueError` naming both the built-ins and the escape hatch.

**`stereotypical_divergence` validates its label vocabulary.** Its scorers
return 0.5 for a gold label they do not recognise, so a wholly wrong vocabulary
(`["negative"]` where `["male", "female"]` was meant) produced
`m_stereo == m_anti == 0.5` and a divergence of exactly 0.0, indistinguishable
from a real finding of parity. A complete mismatch is now an error; individual
unknown labels still score as chance.

**`counterfactual_auc` refuses inputs it cannot estimate.** The underlying
`compute_auc` returns `0.0` when it cannot fit a probe, but as a score `0.0` is
the most extreme possible finding. String labels (which count as neither class),
a class with fewer than two members, and a `test_ratio` too small to hold both
classes are now all rejected up front with named errors. `compute_auc` itself is
unchanged, so callers using it directly keep the diagnostic short-circuit.

**Documentation.** A Material for MkDocs site at
<https://michaellarionov.github.io/FairLMs/>, with registry tables generated
from the live registries and checked in CI.

## 0.4.0

The WEAT/SEAT/CEAT permutation seed became explicit constructor configuration
(`seed=`, default `None`). It appears in `get_params()`, so a reported p-value
can be pinned and reproduced instead of depending on ambient RNG state.

## 0.3.1

Added `fairLMs.definitions.resources`: the bundled WEAT and SEAT word sets (`weat_c1`–`weat_c4`,
`seat_c1`–`seat_c4`) as validated `WordSets` containers, with a `WORD_SETS`
registry and `get_word_set` / `list_word_sets` accessors.

## 0.3.0

Completed the rename to `fairLMs`. The package, its imports and the repository
all moved from the previous name; there is no compatibility shim.

## 0.2.0

Reusable dataset and score-table diagnostics: the `fairLMs.datasets.diagnostics` package,
with axis representativeness (`b_rep`) and the four scoring-instrument audits,
the `ready` / `blocked` / `not_applicable` / `failed` applicability model, and
versioned JSON reports.

## 0.1.x

Made the package installable: declared previously undeclared dependencies,
added release metadata, single-sourced the version through
`fairLMs/_version.py`, and read it via AST rather than a regex.
