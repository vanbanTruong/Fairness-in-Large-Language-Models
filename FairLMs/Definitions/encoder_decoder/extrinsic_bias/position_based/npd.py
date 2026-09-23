import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def _segment_distribution(
    summary_sentences: List[str],
    article_sentences: List[str],
    K:                 int,
) -> np.ndarray:

    uniform = np.ones(K) / K

    if not summary_sentences or not article_sentences:
        return uniform

    n = len(article_sentences)

    try:
        vec = TfidfVectorizer(min_df=1)
        vec.fit(article_sentences + summary_sentences)
        art_vecs  = vec.transform(article_sentences)
        summ_vecs = vec.transform(summary_sentences)
    except ValueError:
        return uniform

    sim   = cosine_similarity(summ_vecs, art_vecs)
    best  = sim.argmax(axis=1)
    segs  = np.clip((best * K) // n, 0, K - 1)

    hist = np.bincount(segs, minlength=K).astype(float)
    total = hist.sum()
    return hist / total if total > 0 else uniform


def generate_summary(
    model:          AutoModelForSeq2SeqLM,
    tokenizer:      AutoTokenizer,
    text:           str,
    max_new_tokens: int = 128,
) -> str:
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
    )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def compute_npd(
    model:          AutoModelForSeq2SeqLM,
    tokenizer:      AutoTokenizer,
    articles:       List[str],
    gold_summaries: Optional[List[str]] = None,
    max_new_tokens: int = 128,
    K:              int = 10,
) -> Tuple[float, list]:

    positions  = np.arange(K, dtype=float)
    uniform    = np.ones(K) / K
    npd_scores: List[float] = []
    rows:       list        = []

    for i, article in enumerate(articles):
        gold = gold_summaries[i] if gold_summaries is not None else None
        model_summary = generate_summary(model, tokenizer, article, max_new_tokens)

        art_sents   = _split_sentences(article)
        model_sents = _split_sentences(model_summary)
        if not art_sents:
            continue

        p_model = _segment_distribution(model_sents, art_sents, K)
        if gold is not None:
            p_ref     = _segment_distribution(_split_sentences(gold), art_sents, K)
            gold_disp = gold[:80]
        else:
            p_ref     = uniform
            gold_disp = "(uniform)"

        npd = float(wasserstein_distance(positions, positions, p_ref, p_model)) / (K - 1)
        npd_scores.append(npd)

        rows.append({
            "index":         i,
            "article":       article[:80],
            "gold_summary":  gold_disp,
            "model_summary": model_summary[:80],
            "npd":           round(npd, 4),
        })

    mean_npd = round(float(np.mean(npd_scores)), 4) if npd_scores else 0.0
    return mean_npd, rows