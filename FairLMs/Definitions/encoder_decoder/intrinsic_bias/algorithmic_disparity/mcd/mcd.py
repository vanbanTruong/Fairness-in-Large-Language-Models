import math
import re
from collections import defaultdict
from typing import List, Tuple, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from nltk.stem.snowball import SnowballStemmer
    _STEMMER = SnowballStemmer("french")
except ImportError:
    raise ImportError("pip install nltk")


def _stem(word: str) -> str:
    """Return Snowball stem of a French word."""
    return _STEMMER.stem(word.lower())


def _tokenize_words(text: str) -> List[str]:
    
    return re.findall(r"[^\W\d_]+", text)


def generate_translation(model, tokenizer, text, max_new_tokens=128):
    prompt = f"translate English to French: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def compute_mcd(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    sentences: List[str],
    max_new_tokens: int = 128,
) -> Tuple[float, float, list]:
    
    stem_wordforms: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rows = []

    for i, sentence in enumerate(sentences):
        generated = generate_translation(model, tokenizer, sentence, max_new_tokens)
        words = _tokenize_words(generated)
        for w in words:
            if len(w) < 3:
                continue
            stem = _stem(w)
            stem_wordforms[stem][w.lower()] += 1
        rows.append({
            "index":     i,
            "input":     sentence[:80],
            "generated": generated[:80],
            "n_words":   len(words),
        })

    h_scores = []
    d_scores = []
    for stem, wf_counts in stem_wordforms.items():
        total = sum(wf_counts.values())
        if total < 2:
            continue
        p_vals = [c / total for c in wf_counts.values()]
        
        h = -sum(p * math.log(p) for p in p_vals if p > 0)
        
        d = sum(p ** 2 for p in p_vals)
        h_scores.append(h)
        d_scores.append(d)

    mean_h = round(sum(h_scores) / len(h_scores), 4) if h_scores else 0.0
    mean_d = round(sum(d_scores) / len(d_scores), 4) if d_scores else 0.0
    return mean_h, mean_d, rows