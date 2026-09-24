"""Demo of ``FairInferenceScore`` with original occupation stimuli + NLI preds."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fairLMs.datasets import BBQ
from fairLMs.definitions.encoder_only.extrinsic_bias.fair_inference.stimuli import (
    BBQ_OCCUPATIONS,
    BIOS_OCCUPATIONS,
    WINO_OCCUPATIONS,
    build_gender_pairs,
    extract_bbq_terms,
)
from fairLMs.definitions import FairInferenceScore
from fairLMs.definitions.models import HuggingFaceModel


def _nli_predictions(model, premises, hypotheses, batch_size=16):
    """Score premise/hypothesis pairs with an MNLI classifier."""
    loaded = model.load()
    tokenizer, hf_model, device = loaded.tokenizer, loaded.model, loaded.device
    label2id = {k.lower(): v for k, v in hf_model.config.label2id.items()}
    id_ent = label2id.get("entailment", label2id.get("entail", 0))
    id_neu = label2id.get("neutral", 1)
    id_con = label2id.get("contradiction", label2id.get("contra", 2))
    preds = []
    hf_model.eval()
    for start in range(0, len(premises), batch_size):
        bp = premises[start : start + batch_size]
        bh = hypotheses[start : start + batch_size]
        enc = tokenizer(
            bp, bh, truncation=True, max_length=256, padding=True, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            probs = F.softmax(hf_model(**enc).logits, dim=-1).cpu()
        for row in probs:
            preds.append(
                {
                    "entailment": float(row[id_ent]),
                    "neutral": float(row[id_neu]),
                    "contradiction": float(row[id_con]),
                }
            )
    return preds


def main():
    bbq = BBQ(n_max=200)
    examples = list(bbq.load())
    terms = extract_bbq_terms(examples, max_per_category=15)
    print(f"BBQ examples loaded: {len(examples)}")
    print(f"BBQ demographic vocab terms: {sum(len(v) for v in terms.values())} across {len(terms)} cats")
    print(f"Occupations (BIOS/WINO/BBQ): {len(BIOS_OCCUPATIONS)}/{len(WINO_OCCUPATIONS)}/{len(BBQ_OCCUPATIONS)}")

    # Demo NLI scoring on a slice of WinoBias occupation gender pairs
    occupations = WINO_OCCUPATIONS[:8]
    premises, hypotheses = build_gender_pairs(occupations)
    print(f"NLI pairs for FairInference: {len(premises)}")

    model = HuggingFaceModel("textattack/bert-base-uncased-MNLI", task="sequence_classification")
    predictions = _nli_predictions(model, premises, hypotheses)
    result = FairInferenceScore().compute(predictions=predictions)
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
