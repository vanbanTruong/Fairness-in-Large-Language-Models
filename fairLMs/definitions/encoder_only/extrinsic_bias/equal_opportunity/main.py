"""Demo of ``EqualOpportunityGap`` on BBQ disambig rows with MNLI."""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F

from fairLMs.datasets import BBQ
from fairLMs.definitions.encoder_only.extrinsic_bias.equal_opportunity.stimuli import (
    BIOS_OCCUPATIONS,
    bbq_disambig_examples,
)
from fairLMs.definitions import EqualOpportunityGap
from fairLMs.definitions.models import HuggingFaceModel


def _forced_choice(model, examples, batch_size=8):
    loaded = model.load()
    tokenizer, hf_model, device = loaded.tokenizer, loaded.model, loaded.device
    label2id = {k.lower(): v for k, v in hf_model.config.label2id.items()}
    id_ent = label2id.get("entailment", label2id.get("entail", 0))
    premises = [ex["premise"] for ex in examples]
    gold = [ex["gold_hyp"] for ex in examples]
    dist = [ex["distractor_hyp"] for ex in examples]
    groups = [ex["group"] for ex in examples]

    def entail_probs(hyps):
        probs = []
        hf_model.eval()
        for start in range(0, len(premises), batch_size):
            enc = tokenizer(
                premises[start : start + batch_size],
                hyps[start : start + batch_size],
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                p = F.softmax(hf_model(**enc).logits, dim=-1)[:, id_ent].cpu().tolist()
            probs.extend(p)
        return probs

    gp, dp = entail_probs(gold), entail_probs(dist)
    y_pred = [1 if g > d else 0 for g, d in zip(gp, dp)]
    y_true = [1] * len(examples)
    return y_true, y_pred, groups


def main():
    print(f"BIOS occupations available: {len(BIOS_OCCUPATIONS)}")
    rows = list(BBQ(n_max=400).load())
    examples = bbq_disambig_examples(rows, n_max=48)
    print(f"BBQ rows loaded: {len(rows)}; disambig evaluation examples: {len(examples)}")
    if not examples:
        print("No BBQ disambig examples found — check the loader source or data_dir")
        return

    counts = Counter(ex["group"] for ex in examples)
    print(f"Group tag counts (top): {counts.most_common(6)}")
    # Choose the two most frequent tags as g1/g2
    (g1, _), (g2, _) = counts.most_common(2)

    model = HuggingFaceModel("textattack/bert-base-uncased-MNLI", task="sequence_classification")
    y_true, y_pred, groups = _forced_choice(model, examples)
    result = EqualOpportunityGap().compute(
        y_true=y_true, y_pred=y_pred, groups=groups, g1=g1, g2=g2, y=1
    )
    print(result)
    print(f"score={result.score}  (g1={g1!r}, g2={g2!r})")


if __name__ == "__main__":
    main()
