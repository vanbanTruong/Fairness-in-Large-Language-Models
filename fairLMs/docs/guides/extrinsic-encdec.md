# Extrinsic × encoder-decoder

Four metrics measure extrinsic bias in encoder-decoder models: bias in the
output of a generation or classification task rather than in internal
representations.

| Metric | Question | Evidence |
|---|---|---|
| `counterfactual_auc` | is the protected attribute recoverable from the encoder? | `LabeledSentences` |
| `inference_bias_score` | does the model err asymmetrically on NLI? | `(label, prediction)` tuples |
| `translation_similarity_score` | do counterfactual inputs translate to the same meaning? | `(original, counterfactual)` pairs |
| `normalized_position_distance` | does summarization favour a position in the source? | article strings |

Two of them (`counterfactual_auc`, `normalized_position_distance`) need
`task="seq2seq"`; `inference_bias_score` needs no model at all; and
`translation_similarity_score` needs two models: a translator and a sentence
encoder.

## End to end

```python
from fairLMs.definitions import (
    CounterfactualAucScore,
    InferenceBiasScore,
    NormalizedPositionDistance,
)
from fairLMs.definitions.data import LabeledSentences
from fairLMs.definitions.models import HuggingFaceModel

t5 = HuggingFaceModel("t5-small", task="seq2seq")

# --- IBS: no model: audit (label, prediction) pairs you already have -------
ibs = InferenceBiasScore().compute(None, [
    ("entailment", "entailment"),
    ("entailment", "neutral"),
    ("neutral", "neutral"),
    ("contradiction", "neutral"),
])
print(ibs.score)                          # 0.0
print(ibs.details["counts"]["accuracy"])  # 0.5
print(ibs.details["counts"]["n_pairs"])   # 4

# --- AUC: can a linear probe recover the attribute from the encoder? --------
sents = LabeledSentences(
    sentences=[
        "The nurse said she was tired.", "The nurse said he was tired.",
        "The doctor said she was busy.", "The doctor said he was busy.",
        "The teacher said she was late.", "The teacher said he was late.",
        "The engineer said she was done.", "The engineer said he was done.",
    ],
    labels=[1, 0, 1, 0, 1, 0, 1, 0],      # must be integer class ids
)

auc = CounterfactualAucScore(test_ratio=0.5, seed=42, n_seeds=3).compute(t5, sents)
print(auc.score)                     # 1.0
print(auc.details["auc_std"])        # 0.0
print(auc.details["n_class_0"])      # 4
print(auc.details["n_class_1"])      # 4

# --- NPD: which part of the source does the summary come from? -------------
npd = NormalizedPositionDistance(max_new_tokens=24, K=4).compute(t5, [
    "The council met on Monday. Costs rose sharply. "
    "Residents objected loudly. A vote was deferred.",
    "Rain fell all week. The river rose. Two roads closed. "
    "Crews worked overnight.",
])
print(npd.score)                     # 0.1667
print(npd.details["n_articles"])     # 2
```

Real output on `t5-small`. `auc.score == 1.0` says the gender attribute is
perfectly linearly recoverable from the encoder representation on these eight
sentences; 0.5 would mean not recoverable at all. With four items per class and
`n_seeds=3` that is a demonstration of the mechanics. Use hundreds of pairs and
the default `n_seeds=10` before reporting, and always report `auc_std`.

!!! note "`counterfactual_auc` labels must be integers 0 and 1"
    The probe is a binary classifier. String labels are refused rather than
    counted as neither class:

    ```python
    CounterfactualAucScore().compute(t5, LabeledSentences(sents, ["female", "male", ...]))
    # TypeError: CounterfactualAucScore labels must be integer class ids, got
    # 'female', 'male'. ... Encode the attribute first, e.g.
    # [0 if g == 'male' else 1 for g in groups].
    ```

    Three other inputs are refused for the same reason; each would otherwise
    report a score of `0.0`, which is not "no signal" but the *most extreme
    possible finding*: a class with fewer than two members, more than two
    distinct classes, and a `test_ratio` so small that no held-out set can
    contain both classes. All four checks read the labels only, so they never
    intercept a genuine AUC.

    The lower-level `compute_auc` still returns `0.0` with the rows attached in
    these cases, by design; it is a diagnostic short-circuit for callers who
    want to inspect why.

## Translation similarity needs a second model

`translation_similarity_score` translates each member of a counterfactual pair
and measures whether the two translations mean the same thing, so it needs a
sentence encoder in addition to the translator. There is no default:

```python
from fairLMs.definitions import TranslationSimilarityScore
from fairLMs.definitions.models import HuggingFaceModel

labse = HuggingFaceModel("sentence-transformers/LaBSE", task="encoder").load()

tss = TranslationSimilarityScore(
    labse_model=labse.model,
    labse_tokenizer=labse.tokenizer,
    tgt_lang="French",
    max_new_tokens=64,
).compute(t5, [
    ("The nurse said she was tired.", "The nurse said he was tired."),
    ("The engineer said he was done.", "The engineer said she was done."),
])
print(tss.score)
```

LaBSE is a ~1.8 GB download on first use and is not bundled, so this is the one
metric in the group that cannot run offline from a clean cache.

## Position bias is not a group fairness metric

`normalized_position_distance` measures whether generated summaries are drawn
disproportionately from one region of the source document. It has no protected
attribute and no groups. It belongs in this family because it is a disparity in
task output, but do not read it as a demographic fairness claim.

Mechanically: the article is split into sentences and binned into `K` positional
slots, the summary's sentences are matched back to those slots to give a
distribution, and the score is the Wasserstein-1 distance between that
distribution and a reference, divided by `K − 1` so it lands in 0…1.

The reference is **uniform by default**, that is, "a summary should draw evenly
from the whole document", which is an assumption, not a fact. Pass
`gold_summaries=` to compare against the positional profile of real reference
summaries instead:

```python
NormalizedPositionDistance(K=4, gold_summaries=[...]).compute(t5, articles)
```

`details["rows"]` records `"(uniform)"` or the gold summary used per article, so
which baseline produced a number is always visible in the result.

## Choosing between them

`inference_bias_score` is free: if you already have NLI predictions, run it.
`counterfactual_auc` is the strongest evidence in this group that a model
*encodes* an attribute rather than merely correlating with it, and it is cheap
because the probe trains on frozen representations. `translation_similarity_score`
is the only one that measures a user-visible harm (a mistranslated pronoun) but
it is the most expensive to set up. `normalized_position_distance` answers a
different question from the other three; run it when summarization is the actual
task.

## Data sources

None of these has a bundled dataset. The leaf runners under
`fairLMs/definitions/encoder_decoder/extrinsic_bias/` pull articles from XSum and
sentence pairs from Europarl via `datasets.load_dataset` at runtime, so they need
network access on first use.
