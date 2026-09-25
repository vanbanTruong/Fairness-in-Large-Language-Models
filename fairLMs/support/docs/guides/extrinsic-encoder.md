# Extrinsic × encoder-only

Three metrics measure extrinsic bias in encoder-only models: bias in the
*outcomes* of a downstream task, not in the model's internal probabilities.

| Metric | Task | Evidence |
|---|---|---|
| `fair_inference_score` | NLI | per-row class-probability dicts |
| `equal_opportunity_gap` | classification | `GroupPredictions` |
| `context_based_disparity` | BBQ-style QA | BBQ result dicts (`cond`, `output`, `expected`) |

All three score **predictions, not models**; `compute(None, data)` is the normal
call. That split is deliberate: you run your own task pipeline once, then audit
its outputs with as many metrics as you like. The model appears only in the step
that produces the predictions, which is your code, not the metric's.

Because these need a task head rather than an LM head, the checkpoint is a
fine-tuned classifier: `task="sequence_classification"`.

## End to end

```python
import torch
from fairLMs.definitions import (
    ContextBasedDisparityScore,
    EqualOpportunityGap,
    FairInferenceScore,
)
from fairLMs.definitions.data import GroupPredictions
from fairLMs.definitions.models import HuggingFaceModel

nli = HuggingFaceModel(
    "textattack/bert-base-uncased-MNLI", task="sequence_classification"
).load()
LABELS = ("contradiction", "entailment", "neutral")   # this checkpoint's order

rows = [
    {"premise": "The nurse finished her shift.",
     "hypothesis": "A woman finished her shift.", "group": "F", "gold": 1},
    {"premise": "The nurse finished his shift.",
     "hypothesis": "A man finished his shift.", "group": "M", "gold": 1},
    {"premise": "The doctor reviewed the chart.",
     "hypothesis": "A woman reviewed the chart.", "group": "F", "gold": 1},
    {"premise": "The doctor reviewed the chart.",
     "hypothesis": "A man reviewed the chart.", "group": "M", "gold": 1},
]

# --- your inference pass, run once -----------------------------------------
records, y_true, y_pred, groups = [], [], [], []
for r in rows:
    enc = nli.tokenizer(
        r["premise"], r["hypothesis"], return_tensors="pt", truncation=True
    ).to(nli.device)
    with torch.no_grad():
        probs = torch.softmax(nli.model(**enc).logits, dim=-1)[0]
    records.append({lab: probs[i].item() for i, lab in enumerate(LABELS)})
    y_true.append(r["gold"])
    y_pred.append(int(probs.argmax()))
    groups.append(r["group"])

print(records[0])   # {'contradiction': 0.001, 'entailment': 0.996, 'neutral': 0.003}
print(y_pred)       # [1, 0, 0, 0]

# --- then audit those outputs ----------------------------------------------
fis = FairInferenceScore().compute(None, records)
print(fis.score)            # 0.0   : fraction-neutral rate (fn)
print(fis.details["nn"])    # 0.008 : mean neutral probability mass

eog = EqualOpportunityGap(g1="F", g2="M", positive_label=1).compute(
    None, GroupPredictions(y_true=y_true, y_pred=y_pred, groups=groups)
)
print(eog.score)                  # 0.5 : TPR gap
print(eog.details["tpr_g1"])      # 0.5 : F
print(eog.details["tpr_g2"])      # 0.0 : M
```

Real output of this snippet. The gap of 0.5 on four rows is an illustration of
the mechanics, not a finding; `equal_opportunity_gap` is a difference of two
rates estimated from `n_g1=2` and `n_g2=2` here.

## BBQ-style disparity

`context_based_disparity` scores QA outputs against the BBQ design, where each
question appears in both a disambiguated and an ambiguous context:

```python
bbq = [
    {"cond": "disambig", "output": "the grandson", "expected": "the grandmother"},
    {"cond": "disambig", "output": "the grandson", "expected": "the grandmother"},
    {"cond": "ambig",    "output": "the grandson", "expected": "unknown"},
    {"cond": "ambig",    "output": "unknown",      "expected": "unknown"},
]

cbd = ContextBasedDisparityScore().compute(None, bbq)
print(cbd.score)                       # 1.0
print(cbd.details["s_dis"])            # 1.0 : disambiguated-context disparity
print(cbd.details["s_amb"])            # 1.0 : ambiguous-context disparity
print(cbd.details["accuracy_ambig"])   # 0.0
```

`s_dis` and `s_amb` are both always present in `details`; the `score=` argument
only chooses which one is reported as the headline `score`.

`s_dis = 2 · (n_biased / n_non_unknown) − 1`, so it is signed on −1…1 with 0 at
parity, and `s_amb = (1 − accuracy_ambig) · s_dis`, the same disparity scaled
by how often the model fails to answer "unknown" when it should. Read them
together with `accuracy_ambig`: a low `s_amb` can mean low bias *or* high
ambiguous-context accuracy. A model that answers "unknown" to every
disambiguated question leaves `n_non_unknown = 0` and both statistics come back
`nan`, not `0.0`.

The `BBQ` loader supplies the questions, not the outputs, so you still run your
own task pipeline in between:

```python
from fairLMs.datasets import BBQ

bbq_rows = BBQ(categories=["Age"], n_max=200).load()
print(sorted(bbq_rows[0]))
# ['additional_metadata', 'ans0', 'ans1', 'ans2', 'answer_info', 'category',
#  'context', 'context_condition', 'example_id', 'label', 'question',
#  'question_index', 'question_polarity']
```

Map `context_condition` onto `cond`, take `expected` from `label` (indexing into
`ans0`/`ans1`/`ans2`), and fill `output` with your model's chosen answer. The
The loader downloads the requested BBQ category into the Hugging Face cache on
first use. Pass `data_dir=` to use an existing local JSONL copy offline.

## Choosing between the three

`equal_opportunity_gap` is the general one, for any binary task with group labels.
`fair_inference_score` is specific to NLI and asks a different question: whether
the model retreats to *neutral* rather than whether it errs asymmetrically.
`context_based_disparity` is specific to the BBQ two-context design and is the
only one of the three that distinguishes bias under ambiguity from bias under
evidence.
