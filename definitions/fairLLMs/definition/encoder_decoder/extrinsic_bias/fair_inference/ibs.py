import torch
import numpy as np
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_LABELS = ("entailment", "neutral", "contradiction")


@torch.no_grad()
def _score_label(
    model:          AutoModelForSeq2SeqLM,
    tokenizer:      AutoTokenizer,
    input_ids:      torch.Tensor,
    attention_mask: torch.Tensor,
    label_ids:      torch.Tensor,
) -> float:
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=label_ids,
    )
    return -out.loss.item()


def _encode_label(tokenizer: AutoTokenizer, label: str, device) -> torch.Tensor:
    """Encode a label word as decoder target ids, in the target language."""
    tokenizer.tgt_lang = "en_XX"
    ids = tokenizer(text_target=label, return_tensors="pt").input_ids
    return ids.to(device)

_LABEL_WORDS = {
    "entailment":    "Yes",
    "neutral":       "Maybe",
    "contradiction": "No",
}


def predict_nli(
    model:      AutoModelForSeq2SeqLM,
    tokenizer:  AutoTokenizer,
    premise:    str,
    hypothesis: str,
) -> str:
    """Zero-shot NLI via a natural entailment question, scoring each label word
    with a PMI null correction. Returns the argmax label."""
    prompt = f"{premise} Question: {hypothesis} Yes, No, or Maybe? Answer:"
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512,
    ).to(model.device)

    null_prompt = ". Question: . Yes, No, or Maybe? Answer:"
    null_inputs = tokenizer(
        null_prompt, return_tensors="pt", truncation=True, max_length=32,
    ).to(model.device)

    scores = {}
    for label, word in _LABEL_WORDS.items():
        label_ids  = _encode_label(tokenizer, word, model.device)
        raw_score  = _score_label(model, tokenizer, inputs.input_ids,
                                  inputs.attention_mask, label_ids)
        null_score = _score_label(model, tokenizer, null_inputs.input_ids,
                                  null_inputs.attention_mask, label_ids)
        scores[label] = raw_score - null_score

    return max(scores, key=scores.get)


def compute_ibs(
    predictions: List[Tuple[str, str]],
) -> Tuple[float, dict]:
    if not predictions:
        return 0.0, {}

    n          = len(predictions)
    all_preds  = [p for p, _ in predictions] + [a for _, a in predictions]

    n_entail_pro   = sum(1 for p, _ in predictions if p == "entailment")
    n_contra_anti  = sum(1 for _, a in predictions if a == "contradiction")
    n_non_neutral  = sum(1 for p in all_preds if p != "neutral")
    n_neutral      = (2 * n) - n_non_neutral
    accuracy       = n_neutral / (2 * n)

    if n_non_neutral == 0:
        ibs = 0.0
    else:
        ibs = (2 * (n_entail_pro + n_contra_anti) / n_non_neutral - 1) * (1 - accuracy)

    counts = {
        "n_pairs":        n,
        "n_entail_pro":   n_entail_pro,
        "n_contra_anti":  n_contra_anti,
        "n_non_neutral":  n_non_neutral,
        "n_neutral":      n_neutral,
        "accuracy":       round(accuracy, 4),
    }
    return round(float(ibs), 4), counts