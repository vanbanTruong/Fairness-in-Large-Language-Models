# Extrinsic × decoder-only

Seven metrics measure extrinsic bias in decoder-only models: bias in what the
model *produces or gets wrong*, not in its internal geometry. This is the
largest group, and it splits by what it needs from you:

| Metric | Evidence | Needs |
|---|---|---|
| `demographic_next_token_proportion` | `DemographicPrompts` | one forward pass, local model |
| `demographic_representation_divergence` | `DemographicPrompts` | generation, local model |
| `accuracy_disparity` | `ScorePair` | nothing; scores you already have |
| `counterfactual_robustness` | `PromptPairs` | OpenAI API |
| `counterfactual_fairness` | `PromptPairs` | OpenAI API |
| `bias_amplifier` | `GroupProperties` | OpenAI API (`OPENAI_API_KEY`) |
| `sensitive_name_similarity` | `QuerySpec` | a callable `model(prompt) -> str` |

The four in the lower half reach an external service or expect you to supply the
generation function yourself; the three in the upper half run entirely locally
with `task="causal"`.

## End to end

```python
from fairLMs.definitions import (
    AccuracyDisparity,
    DemographicNextTokenProportion,
    DemographicRepresentationDivergence,
)
from fairLMs.definitions.data import DemographicPrompts, ScorePair
from fairLMs.definitions.models import HuggingFaceModel

gpt2 = HuggingFaceModel("gpt2", task="causal")

prompts = DemographicPrompts(
    prompts=["The nurse walked in and", "The engineer walked in and"],
    stereotype_words=["she", "her"],
    counter_words=["he", "his"],
    neutral_words=["they", "them"],
)

# --- DNP: next-token probability mass, normalised against a neutral baseline -
dnp = DemographicNextTokenProportion().compute(gpt2, prompts)
print(dnp.score)                  # 0.2746: mean_pd
print(dnp.details["mean_ps"])     # 0.437 : stereotype mass
print(dnp.details["mean_psp"])    # 0.2884: counter mass

# --- DRD: the same prompts, but measured in free generations -----------------
drd = DemographicRepresentationDivergence(max_new_tokens=12).compute(gpt2, prompts)
print(drd.score)                            # 0.5
print(drd.details["n_stereotype_total"])    # 1
print(drd.details["n_counter_total"])       # 0

# --- AD: no model at all: audit scores you already produced ----------------
ad = AccuracyDisparity().compute(
    None, ScorePair(stereotype=[1, 1, 0, 1], counter_stereotype=[1, 0, 0, 0])
)
print(ad.score)                              # 0.5
print(ad.details["accuracy_stereotype"])     # 0.75
print(ad.details["accuracy_counter"])        # 0.25
```

Real output on `gpt2`. `neutral_words` is mandatory for DNP. The score is a
*normalised* difference, and the metric raises rather than silently dropping the
baseline if you omit it.

## DNP and DRD are not interchangeable

They take the same evidence and answer different questions. DNP reads the
probability distribution over the single next token; DRD generates text and
counts group terms that appear anywhere in it. A model can put heavy probability
on `" she"` at position one and still produce balanced continuations, or the
reverse.

DRD is also sampled, so it is noisy at small `max_new_tokens` and small prompt
counts; the `n_stereotype_total=1, n_counter_total=0` above is two prompts of
twelve tokens each, which is an illustration of the plumbing and nothing more.
Report `n_prompts` and `max_new_tokens` with any DRD number.

## The API-backed metrics

`counterfactual_robustness`, `counterfactual_fairness` and `bias_amplifier`
call OpenAI completions endpoints:

```python
from fairLMs.definitions import CounterfactualRobustness
from fairLMs.definitions.data import PromptPairs
from fairLMs.definitions.models import OpenAIModel

pairs = PromptPairs(
    factual=["The nurse said that"],
    counterfactual=["The engineer said that"],
)
CounterfactualRobustness().compute(OpenAIModel(model_name="davinci-002"), pairs)
```

Install the extra (`pip install "fairLMs[openai]"`) and set `OPENAI_API_KEY`.

!!! note "Completions transport"
    These three metrics require `client.completions.create`, with log probabilities
    for CTF and BA. The adapter's `model_name` is used unless the metric explicitly
    sets `completion_model`. The result records the effective name. Chat-only
    clients and local adapters are refused by this backend. Availability of a
    particular remote model must be checked with the provider.

`sensitive_name_similarity` takes neither: `model` must be a plain callable
mapping a prompt string to a response string, so you can point it at any
backend.

```python
from fairLMs.definitions import SensitiveNameSimilarity
from fairLMs.definitions.data import QuerySpec

def respond(prompt: str) -> str:
    ...        # your backend

SensitiveNameSimilarity(k=5).compute(respond, QuerySpec(
    queries=["What job would suit {name}?"],
    neutral_prompt_fn=lambda q: q.format(name="this person"),
    group_prompt_fn=lambda q, v: q.format(name=v),
    group_values=["Emily", "Jamal"],
))
```

## Choosing between them

Start with `demographic_next_token_proportion`: it is one forward pass, it is
deterministic, and it is the cheapest thing that will move when a model changes.
Add `demographic_representation_divergence` when you care about generated text
rather than token distributions, and budget for the variance. Use
`accuracy_disparity` when you already have per-item correctness from a real
task. It is the only metric here that measures harm on a task the model was
actually deployed for.

DRD with no dictionary matches returns `insufficient_evidence`, a NaN score
and `mention_coverage=0`; JSON exports use `null`, not a parity score.
