"""Demo of ``AccuracyDisparity`` using BiasAsker CSVs + gpt2 forced choice."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from fairLMs.definition.decoder_only.extrinsic_bias.performance_disparity.ad.stimuli import (
    biasasker_ad_prompts,
    load_groups_and_props,
)
from fairLMs.metrics import AccuracyDisparity
from fairLMs.models import HuggingFaceModel

_LEAF = Path(__file__).resolve().parent


def _forced_choice_yes_no(model, prompts):
    loaded = model.load()
    tok, hf, device = loaded.tokenizer, loaded.model, loaded.device
    yes_id = tok.encode(" Yes", add_special_tokens=False)[0]
    no_id = tok.encode(" No", add_special_tokens=False)[0]
    scores = []
    hf.eval()
    for q in prompts:
        ids = tok(q, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = hf(**ids).logits[0, -1]
            lp = F.log_softmax(logits, dim=-1)
        # "No" = rejecting stereotype claim (treated as correct / fair answer)
        scores.append(1.0 if lp[no_id] > lp[yes_id] else 0.0)
    return scores


def main():
    buckets, props = load_groups_and_props(_LEAF, n_props=20, n_groups_per=10)
    by_group = biasasker_ad_prompts(buckets, props, n_max=24)
    print(
        f"BiasAsker age groups: old={len(buckets['old'])} young={len(buckets['young'])} "
        f"props={len(props)} prompts_old={len(by_group['old'])} prompts_young={len(by_group['young'])}"
    )
    model = HuggingFaceModel("gpt2", task="causal")
    scores_s = _forced_choice_yes_no(model, by_group["old"])
    scores_sp = _forced_choice_yes_no(model, by_group["young"])
    result = AccuracyDisparity().compute(scores_s=scores_s, scores_sp=scores_sp)
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
