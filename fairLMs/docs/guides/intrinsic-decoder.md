# Intrinsic × decoder-only

Four metrics measure intrinsic bias in decoder-only models, in two families that
work at very different depths:

| Metric | Reads | Evidence |
|---|---|---|
| `stereotypical_log_likelihood` | next-token log-probabilities | `OccupationTriples` |
| `cooccurrence_association` | sampled generations | `ConceptSpec` |
| `gradient_based_bias_estimation` | gradients on attention Value heads | `WordSets` |
| `natural_indirect_effect` | causal mediation through attention heads | `ProbeSet` |

All four need `task="causal"`. The first two treat the model as a black box over
tokens; the last two open it up and attribute bias to individual attention
heads.

## End to end

```python
from fairLMs.metrics import CooccurrenceAssociation, StereotypicalLogLikelihood
from fairLMs.metrics.data import ConceptSpec, OccupationTriples
from fairLMs.models import HuggingFaceModel

gpt2 = HuggingFaceModel("gpt2", task="causal")

# --- SLL: log-probability gaps over occupation templates --------------------
triples = OccupationTriples(triples=[
    ("nurse", "she", "he"),
    ("engineer", "he", "she"),
    ("teacher", "she", "he"),
])

sll = StereotypicalLogLikelihood().compute(gpt2, triples)
print(sll.score)      # 0.8855: mean absolute gap across the three variants
print(sll.details)
# {'NV': 1.1209, 'CV': 0.859, 'IV': 0.6765, 'n_occupations': 3}

# --- CA: association measured in what the model actually generates ----------
spec = ConceptSpec(
    concepts=["nursing", "engineering"],
    prompt_template="A career in {concept} is usually chosen by",
    group_terms={
        "female": ["she", "her", "woman", "women"],
        "male": ["he", "his", "man", "men"],
    },
)

ca = CooccurrenceAssociation(n_samples=4).compute(gpt2, spec)
print(ca.score)                     # 0.05: total-variation distance
print(ca.details["n_valid"])        # 2
print(ca.details["n_skipped"])      # 0
```

Real output on `gpt2`. Each triple is `(occupation, stereotype_word,
counter_word)`; `NV`, `CV` and `IV` are the three template variants SLL scores,
and the headline number is the mean absolute gap across them. Zero means the
model is indifferent between the stereotype and counter word.

## Reading a `nan` from CA

`cooccurrence_association` counts group terms in generated text, so it can
legitimately have nothing to count:

```python
ca = CooccurrenceAssociation(n_samples=1).compute(gpt2, spec)
if ca.score != ca.score:            # nan
    print("no group term appeared:", ca.details["n_skipped"], "concepts skipped")
```

`nan` here is not a failure; it means no term from any group list appeared in
any sample. Raise `n_samples`, widen the term lists, or use a prompt template
that invites a pronoun. `details["n_skipped"]` tells you how many concepts were
dropped for this reason, so a score computed from two of twenty concepts is
visibly not a score over twenty.

## Attributing bias to heads

The other two metrics are heavier and answer a mechanistic question, *which
heads carry this bias*, rather than *how much bias is there*:

```python
from fairLMs.metrics import GradientBasedBiasEstimation, NaturalIndirectEffect
from fairLMs.metrics.data import ProbeSet, WordSets

gbe = GradientBasedBiasEstimation().compute(gpt2, WordSets(
    target_1=["man", "he", "his"],
    target_2=["woman", "she", "her"],
    attribute_1=["engineer", "programmer"],
    attribute_2=["nurse", "teacher"],
))

print(gbe.score)                    # 0.49306
print(gbe.details["shape"])         # (12, 12): one value per (layer, head)

nie = NaturalIndirectEffect(threshold=0.003).compute(gpt2, ProbeSet(probes=[
    {
        "prompt": "The nurse said that",
        "cf_text": "The engineer said that",
        "stereo_token_id": 673,      # gpt2 id for " she"
        "anti_token_id": 339,        # gpt2 id for " he"
    },
]))
print(nie.score)                    # 0.6528
print(nie.details["threshold"])     # 0.003
```

Both return the full `(n_layers, n_heads)` matrix in `details` alongside the
headline score, so you can rank heads rather than only compare checkpoints. Get
the token ids from the tokenizer rather than hardcoding them:
`tokenizer.encode(" she")` is `[673]` for `gpt2` but differs across
vocabularies.

Both sweep every layer and head, so cost scales with model depth, so start with
`gpt2` before `gpt2-medium`. `n_layers`, `n_heads` and `head_dim` are derived
from `model.config` when omitted; pass them only to override. `GBE` accepts a
precomputed `gbe_matrix` and `NIE` a precomputed `nie` array, so you can
re-score a saved sweep without re-running the model.

## Choosing between them

`stereotypical_log_likelihood` is the cheapest and most directly comparable
across checkpoints, at one forward pass per template. `cooccurrence_association`
measures the *generative* behaviour a user would actually encounter, which log
probabilities can misrepresent, but it is sampled and therefore noisy: pin
`n_samples` and report it. The two head-attribution metrics are diagnostic tools
for model surgery rather than benchmark numbers.
