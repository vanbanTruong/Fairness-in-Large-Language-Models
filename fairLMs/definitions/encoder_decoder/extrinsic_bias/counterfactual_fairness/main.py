import re
import random
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from encoder_decoder.extrinsic_bias.counterfactual_fairness.auc import compute_auc

_MAIN_DIR = Path(__file__).resolve().parent

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
MAX_LENGTH = 512
SEED       = 42
N_MAX      = 200
N_SEEDS    = 10
XSUM_SCAN  = 5000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def results_to_csv(results, filename: str) -> str:
    df = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    out_path = _MAIN_DIR / filename
    df.to_csv(out_path, index=False)
    return str(out_path)

_MALE_TERMS   = ["he", "him", "his", "himself"]
_FEMALE_TERMS = ["she", "her", "hers", "herself"]
_GENDER_NEUTRAL = {"he": "they", "she": "they", "him": "them", "his": "their",
                   "her": "their", "hers": "theirs",
                   "himself": "themselves", "herself": "themselves"}

_NAT_A2B = {"american": "chinese", "european": "african", "western": "eastern",
            "white": "black", "christian": "muslim", "english": "arabic",
            "french": "arabic", "german": "arabic"}
_NAT_GROUP_A = set(_NAT_A2B.keys())
_NAT_GROUP_B = set(_NAT_A2B.values())
_NAT_ALL     = set(_NAT_A2B.keys()) | set(_NAT_A2B.values())


def _contains_any(text, terms):
    low = text.lower()
    return any(re.search(r"\b" + re.escape(t) + r"\b", low) for t in terms)


def _gender_label(text):
    m = _contains_any(text, _MALE_TERMS)
    f = _contains_any(text, _FEMALE_TERMS)
    if m and not f:
        return 0
    if f and not m:
        return 1
    return None


def _nat_label(text):
    a = _contains_any(text, list(_NAT_GROUP_A))
    b = _contains_any(text, list(_NAT_GROUP_B))
    if a and not b:
        return 0
    if b and not a:
        return 1
    return None


def _mask_gender(text):
    return re.sub(r"\b(" + "|".join(_GENDER_NEUTRAL) + r")\b",
                  lambda m: _GENDER_NEUTRAL[m.group().lower()],
                  text, flags=re.IGNORECASE)


def _mask_nationality(text):
    s = re.sub(r"\b(" + "|".join(re.escape(k) for k in _NAT_ALL) + r")\b",
               "", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", s).strip()


def _build_masked(texts, label_fn, mask_fn, per_class):
    sents, labels, seen = [], [], set()
    for t in texts:
        if labels.count(0) >= per_class and labels.count(1) >= per_class:
            break
        g = label_fn(t)
        if g is None:
            continue
        if labels.count(g) >= per_class:
            continue
        masked = mask_fn(t)
        if not masked or (masked, g) in seen:
            continue
        seen.add((masked, g))
        sents.append(masked); labels.append(g)
    return sents, labels


print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.src_lang = "en_XX"
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32,
).to(DEVICE)
model.eval()
print(f"Model device: {next(model.parameters()).device}")
print(f"  Device: {DEVICE}\n")

_WINOMT_URL = (
    "https://raw.githubusercontent.com/gabrielStanovsky/"
    "mt_gender/master/data/aggregates/en.txt"
)


def load_winomt(n_max: int = N_MAX):
    sents = []
    try:
        with urllib.request.urlopen(_WINOMT_URL, timeout=15) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if line and not line.startswith("#"):
                    parts = line.split("\t")
                    t = parts[2].strip() if len(parts) >= 3 else parts[0].strip()
                    if t:
                        sents.append(t)
    except Exception as e:
        print(f"  [warn] WinoMT load failed: {e}")
        return [], []
    sents = list(dict.fromkeys(sents))
    s, l = _build_masked(sents, _gender_label, _mask_gender, n_max // 2)
    print(f"  WinoMT: {len(s)} masked sentences "
          f"(label 0={l.count(0)}, label 1={l.count(1)})")
    return s, l


def load_xsum(n_max: int = N_MAX):
    def _sents(txt):
        return [x.strip() for x in re.split(r"(?<=[.!?])\s+", txt.strip()) if x.strip()]
    try:
        ds = load_dataset("EdinburghNLP/xsum", split="validation")
        raw = []
        for ex in ds:
            d = ex.get("document", "").strip()
            if not d:
                continue
            for sent in _sents(d):
                if _contains_any(sent, list(_NAT_GROUP_A)) or \
                   _contains_any(sent, list(_NAT_GROUP_B)):
                    raw.append(sent)
            if len(raw) >= XSUM_SCAN:
                break
        s, l = _build_masked(raw, _nat_label, _mask_nationality, n_max // 2)
        print(f"  XSum: {len(s)} masked sentences "
              f"(label 0={l.count(0)}, label 1={l.count(1)})")
        return s, l
    except Exception as e:
        print(f"  [warn] XSum load failed: {e}")
        return [], []


def load_xnli(n_max: int = N_MAX):
    raw = []
    for repo in ("facebook/xnli", "xnli"):
        try:
            for split in ("validation", "test"):
                ds = load_dataset(repo, "en", split=split)
                for ex in ds:
                    for key in ("premise", "hypothesis"):
                        v = ex.get(key)
                        if isinstance(v, str) and v.strip():
                            raw.append(v.strip())
            break
        except Exception as e:
            print(f"  [warn] {repo} failed: {e}")
    if not raw:
        return [], []
    s, l = _build_masked(raw, _gender_label, _mask_gender, n_max // 2)
    print(f"  XNLI: {len(s)} masked sentences "
          f"(label 0={l.count(0)}, label 1={l.count(1)})")
    return s, l


def run_dataset(name, sentences, labels, attr) -> dict:
    n = len(sentences)
    if n == 0:
        print(f"\n  [skip] {name} — no sentences loaded")
        return {"dataset": name, "sensitive_attr": attr,
                "n_sentences": 0, "n_label0": 0, "n_label1": 0,
                "AUC": None, "AUC_std": None}

    print(f"\n{'─' * 55}")
    print(f"  AUC (masked)  |  {name}  ({n} sentences)")
    print(f"  sensitive attr : {attr}")
    print(f"{'─' * 55}")

    auc, auc_std, n0, n1, rows = compute_auc(
        model, tokenizer, sentences, labels,
        pair_ids=None, seed=SEED, n_seeds=N_SEEDS)


    return {"dataset": name, "auc_score": auc}


def main():
    print("\n" + "=" * 55)
    print(f"  Counterfactual Fairness (masked AUC)  |  {MODEL_NAME}")
    print("=" * 55)

    results = []
    s, l = load_winomt(); results.append(run_dataset("WinoMT", s, l, "gender"))
    s, l = load_xsum();   results.append(run_dataset("XSum",   s, l, "race/nationality"))
    s, l = load_xnli();   results.append(run_dataset("XNLI",   s, l, "gender"))

    summary = pd.DataFrame(results)[
        ["dataset", "auc_score"]]
    summary.columns = ["dataset", "auc_score"]
    out = results_to_csv(summary, "auc_results.csv")


    print(summary.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()