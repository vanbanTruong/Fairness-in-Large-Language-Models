from typing import List, Tuple, Dict
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

FRENCH_GENDER_CUES = {
    "male":   ["Il", "Lui"],
    "female": ["Elle", "Celle-ci"],
}
FRENCH_AGE_CUES = {
    "young": ["Le jeune", "Les jeunes"],
    "old":   ["Le vieux", "La personne âgée"],
}

@torch.no_grad()
def _candidate_logprob(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    source: str,
    candidate: str,
) -> float:
    enc = tokenizer(source, return_tensors="pt", truncation=True,
                    max_length=512).to(model.device)
    lab = tokenizer(text_target=candidate, return_tensors="pt",
                    truncation=True, max_length=32).input_ids.to(model.device)
    out = model(input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                labels=lab)
    return -float(out.loss.item())


def predict_gender(model, tokenizer, source: str) -> str:
    scores = {}
    for label, cues in FRENCH_GENDER_CUES.items():
        scores[label] = float(np.mean(
            [_candidate_logprob(model, tokenizer, source, c) for c in cues]))
    return max(scores, key=scores.get)


def predict_age(model, tokenizer, source: str) -> str:
    scores = {}
    for label, cues in FRENCH_AGE_CUES.items():
        scores[label] = float(np.mean(
            [_candidate_logprob(model, tokenizer, source, c) for c in cues]))
    return max(scores, key=scores.get)


def pronoun_accuracy(predicted: str, gold_gender: str) -> float:
    if gold_gender not in ("male", "female"):
        return 0.5
    return 1.0 if predicted == gold_gender else 0.0


def age_accuracy(predicted: str, gold_age: str) -> float:
    if gold_age not in ("young", "old"):
        return 0.5
    return 1.0 if predicted == gold_age else 0.0


_PREDICT_FOR = {
    "pronoun_accuracy": predict_gender,
    "age_accuracy":     predict_age,
}


def compute_sd(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    stereo_sentences:  List[str],
    stereo_labels:     List[str],
    anti_sentences:    List[str],
    anti_labels:       List[str],
    max_new_tokens:    int = 128,
    metric_fn=pronoun_accuracy,
) -> Tuple[float, float, float, list]:

    predict_fn = _PREDICT_FOR[metric_fn.__name__]

    stereo_scores, anti_scores, rows = [], [], []

    for split, sents, labels, bucket in (
        ("stereo", stereo_sentences, stereo_labels, stereo_scores),
        ("anti",   anti_sentences,   anti_labels,   anti_scores),
    ):
        for i, (sent, label) in enumerate(zip(sents, labels)):
            predicted = predict_fn(model, tokenizer, sent)
            score = metric_fn(predicted, label)
            bucket.append(score)
            rows.append({
                "split":      split,
                "index":      i,
                "source":     sent[:80],
                "predicted":  predicted,
                "gold_label": label,
                "score":      score,
            })

    m_stereo = float(np.mean(stereo_scores)) if stereo_scores else 0.0
    m_anti   = float(np.mean(anti_scores))   if anti_scores   else 0.0
    delta_s  = round(m_anti - m_stereo, 4)

    return round(m_stereo, 4), round(m_anti, 4), delta_s, rows