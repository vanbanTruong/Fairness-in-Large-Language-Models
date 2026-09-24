"""Demo of ``ContextBasedDisparityScore`` on BBQ JSONL with MNLI."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fairLMs.datasets import BBQ
from fairLMs.definitions.encoder_only.extrinsic_bias.context_based_disparity.stimuli import (
    bbq_rows_to_protocol,
)
from fairLMs.definitions import ContextBasedDisparityScore
from fairLMs.definitions.models import HuggingFaceModel


def _score_bbq_with_nli(model, rows, n_max=40):
    """Predict BBQ answers via MNLI: context+question entails each candidate."""
    loaded = model.load()
    tok, hf, device = loaded.tokenizer, loaded.model, loaded.device
    label2id = {k.lower(): v for k, v in hf.config.label2id.items()}
    id_ent = label2id.get("entailment", label2id.get("entail", 0))
    hf.eval()

    # Start from gold protocol for expected labels, then overwrite output
    protocol = bbq_rows_to_protocol(rows[:n_max])
    scored = []
    for row, proto in zip(rows[:n_max], protocol):
        premise = f"{row.get('context', '').strip()} {row.get('question', '').strip()}"
        cands = [str(row.get(f"ans{i}", "")).strip() for i in (0, 1, 2)]
        enc = tok(
            [premise] * 3,
            cands,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            probs = F.softmax(hf(**enc).logits, dim=-1)[:, id_ent].cpu()
        best = int(torch.argmax(probs).item())
        info = row.get("answer_info", {})
        meta = info.get(f"ans{best}")
        tag = str(meta[1]).lower() if isinstance(meta, (list, tuple)) and len(meta) > 1 else ""
        stereo = {
            str(g).strip().lower()
            for g in row.get("additional_metadata", {}).get("stereotyped_groups", [])
        }
        if "unknown" in tag:
            output = "UNKNOWN"
        elif tag in stereo:
            output = "target"
        else:
            output = "nontarget"
        scored.append({**proto, "output": output})
    return scored


def main():
    rows = list(BBQ(n_max=80).load())
    print(f"BBQ rows loaded: {len(rows)}")
    model = HuggingFaceModel("textattack/bert-base-uncased-MNLI", task="sequence_classification")
    outputs = _score_bbq_with_nli(model, rows, n_max=min(40, len(rows)))
    n_dis = sum(1 for o in outputs if o["cond"] == "disambig")
    n_amb = sum(1 for o in outputs if o["cond"] == "ambig")
    print(f"Protocol outputs: {len(outputs)} (disambig={n_dis}, ambig={n_amb})")

    result = ContextBasedDisparityScore(score="s_dis").compute(
        outputs=outputs, also_s_amb=True
    )
    print(result)
    print(f"score={result.score}")
    print(f"details={result.details}")


if __name__ == "__main__":
    main()
