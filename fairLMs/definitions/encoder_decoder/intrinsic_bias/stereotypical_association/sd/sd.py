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

#: Label domain each built-in scorer can actually score. A gold label outside
#: its domain scores 0.5 (chance) rather than raising, so callers need this to
#: tell "the model was undecided" from "I passed the wrong label vocabulary".
LABEL_DOMAIN_FOR = {
    "pronoun_accuracy": ("male", "female"),
    "age_accuracy":     ("young", "old"),
}


def resolve_predict_fn(metric_fn, predict_fn=None):
    """Pair a scorer with the prediction routine that produces its labels.

    Each built-in scorer is meaningful only alongside a specific prediction
    routine — ``pronoun_accuracy`` grades French gender cues, ``age_accuracy``
    grades French age cues — so the two are looked up together. A custom
    ``metric_fn`` is therefore accepted only with an explicit ``predict_fn``,
    rather than silently mis-paired with a built-in predictor.
    """
    if predict_fn is not None:
        return predict_fn
    try:
        return _PREDICT_FOR[metric_fn.__name__]
    except (AttributeError, KeyError):
        raise ValueError(
            f"No prediction routine is paired with metric_fn="
            f"{getattr(metric_fn, '__name__', metric_fn)!r}. Either use a "
            f"built-in scorer ({', '.join(sorted(_PREDICT_FOR))}), or pass "
            f"predict_fn=<callable(model, tokenizer, sentence) -> str> "
            f"alongside your own metric_fn."
        ) from None


def compute_sd(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    stereo_sentences:  List[str],
    stereo_labels:     List[str],
    anti_sentences:    List[str],
    anti_labels:       List[str],
    max_new_tokens:    int = 128,
    metric_fn=pronoun_accuracy,
    predict_fn=None,
) -> Tuple[float, float, float, list]:

    predict_fn = resolve_predict_fn(metric_fn, predict_fn)

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