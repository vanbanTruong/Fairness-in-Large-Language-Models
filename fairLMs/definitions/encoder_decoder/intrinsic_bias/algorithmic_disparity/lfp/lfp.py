import re
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from wordfreq import top_n_list, zipf_frequency
    _TOP1000 = set(top_n_list("fr", 1000))
    _TOP2000 = set(top_n_list("fr", 2000))
except ImportError:
    raise ImportError("pip install wordfreq")


def _classify_word(word: str) -> int:
    """Return 1, 2, or 3 for the frequency band of a word."""
    w = word.lower().strip()
    if not w or not w.isalpha():
        return 1  
    if w in _TOP1000:
        return 1
    if w in _TOP2000:
        return 2
    return 3


def _tokenize_words(text: str) -> List[str]:
    
    return re.findall(r"[^\W\d_]+", text)


def generate_translation(model, tokenizer, text, max_new_tokens=128):
    prompt = f"translate English to French: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def compute_lfp(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    sentences: List[str],
    max_new_tokens: int = 128,
) -> Tuple[float, float, float, list]:
    
    b1_total = b2_total = b3_total = n_total = 0
    rows = []

    for i, sentence in enumerate(sentences):
        generated = generate_translation(
            model, tokenizer, sentence, max_new_tokens
        )
        words  = _tokenize_words(generated)
        counts = {1: 0, 2: 0, 3: 0}
        for w in words:
            counts[_classify_word(w)] += 1
        n = len(words)
        b1_total += counts[1]
        b2_total += counts[2]
        b3_total += counts[3]
        n_total  += n

        rows.append({
            "index":     i,
            "input":     sentence[:80],
            "generated": generated[:80],
            "n_words":   n,
            "b1":        counts[1],
            "b2":        counts[2],
            "b3":        counts[3],
        })

    if n_total == 0:
        return 0.0, 0.0, 0.0, rows

    pb1 = round(b1_total / n_total, 4)
    pb2 = round(b2_total / n_total, 4)
    pb3 = round(b3_total / n_total, 4)
    return pb1, pb2, pb3, rows