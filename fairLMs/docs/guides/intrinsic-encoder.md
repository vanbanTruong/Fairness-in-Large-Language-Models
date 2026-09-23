# Intrinsic × encoder-only

Eleven metrics measure intrinsic bias in encoder-only models: bias read off the
model's own probabilities or representations, with no downstream task attached.

They split into three families by what they read:

| Family | Metrics | Reads |
|---|---|---|
| Pseudo-log-likelihood | `crows_pairs_score`, `pseudo_log_likelihood_score`, `all_unmasked_likelihood_score`, `all_unmasked_likelihood_attention_score`, `context_association_test` | sentence probabilities from an MLM head |
| Masked token | `log_probability_bias_score`, `discovery_of_correlations`, `contrast_based_score` | fill probabilities at a `[MASK]` position |
| Similarity | `weat`, `seat`, `ceat` | embedding geometry, no head at all |

The first two families need `task="mlm"`; the similarity family needs
`task="encoder"`. That is the only place the distinction matters. See
[Models](../api/models.md).

## End to end

```python
from fairLMs.data import weat_c1
from fairLMs.datasets import CrowSPairs
from fairLMs.metrics import AllUnmaskedLikelihoodScore, CrowSPairsScore, WEAT
from fairLMs.models import HuggingFaceModel

# --- pair-preference metrics: masked-LM head, bundled CrowS-Pairs ------------
mlm = HuggingFaceModel("bert-base-uncased", task="mlm")
pairs = CrowSPairs(n_max=20)

cps = CrowSPairsScore().compute(mlm, pairs)
print(cps.score)                      # 55.0 : % of pairs preferring the stereotype
print(cps.details["accuracy"])         # 65.43859649122807
print(cps.by_category)
# {'race-color': 50.0, 'socioeconomic': 0.0, 'gender': 66.7, 'disability': 100.0,
#  'nationality': 50.0, 'sexual-orientation': 100.0, 'physical-appearance': 100.0}

# The same evidence, a different scoring rule: no reloading, no reshaping:
aul = AllUnmaskedLikelihoodScore().compute(mlm, pairs)
print(aul.score)                      # 30.0

# --- similarity metrics: bare encoder, bundled WEAT stimuli ------------------
enc = HuggingFaceModel("bert-base-uncased", task="encoder")
weat = WEAT(seed=0).compute(enc, weat_c1)
print(weat.score)                     # 0.76  : Cohen's d effect size
print(weat.details["p_value"])         # 0.0065: permutation test
print(weat.details["seed"])           # 0     : pinned, so this is reproducible
```

Values above are the real output of this snippet on `bert-base-uncased` with 20
CrowS-Pairs rows. Raise `n_max` (or drop it for the full 1508 pairs) for
anything you intend to report.

## Reading the numbers

`crows_pairs_score` and the AUL family both report *the percentage of pairs
where the stereotypical sentence scores higher*, so 50 is parity and above 50
favours the stereotype. They disagree here (55.0 vs 30.0) because they score
sentences differently: CPS masks and scores the shared tokens, leaving the differing group
tokens visible in each sentence, AUL scores every token unmasked in one pass. That disagreement is the
point of implementing both; neither is a corrected version of the other.

`weat` reports Cohen's *d*, where 0 is no differential association and the
conventional large-effect threshold is 0.8. Always report `seed` alongside
`p_value`. The p-value comes from a 10,000-sample permutation test, and with
`seed=None` (the default) it is not reproducible.

## Swapping the checkpoint

Because the adapter is the only thing that knows about loading, sweeping models
is a loop:

```python
for ckpt in ("bert-base-uncased", "roberta-base", "distilbert-base-uncased"):
    model = HuggingFaceModel(ckpt, task="mlm")
    print(ckpt, CrowSPairsScore().compute(model, pairs).score)
```

!!! note "AULA requires attention tensors"
    The model must return attention tensors when called with
    `output_attentions=True`. Where supported, scoring temporarily selects eager
    attention and restores the original backend afterward. Missing attention
    tensors raise an explicit error. BERT and DistilBERT are covered by offline
    integration tests using real transformers classes and random weights.

## Other data sources

`CrowSPairs` is bundled, so the example above runs offline. `StereoSet`
(`as_triples=True`) supplies the `SentenceTriples` that
`context_association_test` wants, and any list of dicts with `stereotype` /
`anti_stereotype` keys works directly. See
[Bring your own data](own-data.md).

StereoSet intersentence rows preserve a `scoring_context`. It stays visible and
only candidate tokens are scored. CAT reports candidate masked-PLL scores and
micro aggregation across triples; it is not the full official StereoSet
per-target macro benchmark. LMS uses both meaningful-vs-unrelated comparisons.

XNLI religion substitutions are returned as counterfactual pairs with source
metadata; they do not establish human stereotype/anti-stereotype labels.
