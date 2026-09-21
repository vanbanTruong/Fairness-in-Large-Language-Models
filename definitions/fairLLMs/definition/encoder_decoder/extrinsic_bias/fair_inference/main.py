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

from fairLLMs.definition.encoder_decoder.extrinsic_bias.fair_inference.ibs import predict_nli, compute_ibs

_MAIN_DIR = Path(__file__).resolve().parent

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
SEED       = 42
N_MAX      = 500
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
_M2F = {"he": "she", "him": "her", "his": "her", "himself": "herself"}
_F2M = {"she": "he", "her": "him", "hers": "his", "herself": "himself"}
_GENDER_NEUTRAL = {"he": "they", "she": "they", "him": "them", "his": "their",
                   "her": "their", "hers": "theirs",
                   "himself": "themselves", "herself": "themselves"}

_NAT_A2B = {"american": "chinese", "european": "african", "western": "eastern",
            "white": "black", "christian": "muslim", "english": "arabic",
            "french": "arabic", "german": "arabic"}
_NAT_B2A = {v: k for k, v in _NAT_A2B.items()}
_NAT_ALL = {**_NAT_A2B, **_NAT_B2A}
_NAT_GROUP_A = set(_NAT_A2B.keys())


def _contains_any(text, terms):
    low = text.lower()
    return any(re.search(r"\b" + re.escape(t) + r"\b", low) for t in terms)


def _swap(text, table):
    has_a = _contains_any(text, list(table))
    if not has_a:
        return text, False
    s = re.sub(r"\b(" + "|".join(re.escape(k) for k in table) + r")\b",
               lambda m: table[m.group().lower()], text, flags=re.IGNORECASE)
    return s, s != text


def _swap_gender(text):
    has_m = _contains_any(text, _MALE_TERMS)
    has_f = _contains_any(text, _FEMALE_TERMS)
    if has_m and not has_f:
        return _swap(text, _M2F)
    if has_f and not has_m:
        return _swap(text, _F2M)
    return text, False


def _swap_nationality(text):
    has_a = _contains_any(text, list(_NAT_GROUP_A))
    has_b = _contains_any(text, list(set(_NAT_A2B.values())))
    if has_a == has_b:
        return text, False
    return _swap(text, _NAT_ALL)


def _neutralize_gender(text):
    return re.sub(r"\b(" + "|".join(_GENDER_NEUTRAL) + r")\b",
                  lambda m: _GENDER_NEUTRAL[m.group().lower()],
                  text, flags=re.IGNORECASE)


def _neutralize_nationality(text):
    s = re.sub(r"\b(" + "|".join(re.escape(k) for k in _NAT_ALL) + r")\b",
               "", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", s).strip()


def _build_triples(texts, swap_fn, group0_fn, neutralize_fn, n_max):
    triples, seen = [], set()
    for t in texts:
        if len(triples) >= n_max:
            break
        sw, changed = swap_fn(t)
        if not changed:
            continue
        is_a = group0_fn(t)
        a_sent, b_sent = (t, sw) if is_a else (sw, t)
        premise = neutralize_fn(t)
        if not premise or premise in seen or premise == a_sent or premise == b_sent:
            continue
        seen.add(premise)
        triples.append({"premise": premise, "pro_stereo": a_sent,
                        "anti_stereo": b_sent})
    return triples


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


def load_winomt(n_max=N_MAX):
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
        return []
    sents = list(dict.fromkeys(sents))
    triples = _build_triples(sents, _swap_gender,
                             lambda t: _contains_any(t, _MALE_TERMS),
                             _neutralize_gender, n_max)
    print(f"  WinoMT: {len(triples)} triples (gender, directional pro=male)")
    return triples


def load_xsum(n_max=N_MAX):
    def _sents(txt):
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt.strip()) if s.strip()]
    try:
        ds = load_dataset("EdinburghNLP/xsum", split="validation")
        raw = []
        for ex in ds:
            d = ex.get("document", "").strip()
            if not d:
                continue
            for s in _sents(d):
                if _contains_any(s, list(_NAT_GROUP_A)) or \
                   _contains_any(s, list(set(_NAT_A2B.values()))):
                    raw.append(s)
            if len(raw) >= XSUM_SCAN:
                break
        triples = _build_triples(raw, _swap_nationality,
                                 lambda t: _contains_any(t, list(_NAT_GROUP_A)),
                                 _neutralize_nationality, n_max)
        print(f"  XSum: {len(triples)} triples (nationality, directional pro=group A)")
        return triples
    except Exception as e:
        print(f"  [warn] XSum load failed: {e}")
        return []


def load_xnli(n_max=N_MAX):
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
        return []
    triples = _build_triples(raw, _swap_gender,
                             lambda t: _contains_any(t, _MALE_TERMS),
                             _neutralize_gender, n_max)
    print(f"  XNLI: {len(triples)} triples (gender, directional pro=male)")
    return triples


def run_dataset(name, pairs, attr) -> dict:
    n = len(pairs)
    if n == 0:
        print(f"\n  [skip] {name} — no pairs loaded")
        return {"dataset": name, "sensitive_attr": attr, "n_pairs": 0,
                "n_entail_pro": None, "n_contra_anti": None,
                "accuracy": None, "IBS": None}

    print(f"\n{'─' * 55}")
    print(f"  IBS  |  {name}  ({n} triples)")
    print(f"  sensitive attr : {attr}")
    print(f"{'─' * 55}")

    predictions, rows = [], []
    for i, pair in enumerate(pairs):
        pred_pro  = predict_nli(model, tokenizer, pair["premise"], pair["pro_stereo"])
        pred_anti = predict_nli(model, tokenizer, pair["premise"], pair["anti_stereo"])
        predictions.append((pred_pro, pred_anti))
        rows.append({"index": i, "premise": pair["premise"][:80],
                     "pro_stereo": pair["pro_stereo"][:60],
                     "anti_stereo": pair["anti_stereo"][:60],
                     "pred_pro": pred_pro, "pred_anti": pred_anti})

    ibs, counts = compute_ibs(predictions)

    return {"dataset": name, "ibs_score": ibs}


def main():
    print("\n" + "=" * 55)
    print(f"  Fair Inference (IBS)  |  {MODEL_NAME}")
    print("=" * 55)

    results = []
    results.append(run_dataset("WinoMT", load_winomt(), "gender"))
    results.append(run_dataset("XSum",   load_xsum(),   "race/nationality"))
    results.append(run_dataset("XNLI",   load_xnli(),   "gender"))

    summary = pd.DataFrame(results)[
        ["dataset", "ibs_score"]]
    summary.columns = ["dataset", "ibs_score"]
    out = results_to_csv(summary, "ibs_results.csv")

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()