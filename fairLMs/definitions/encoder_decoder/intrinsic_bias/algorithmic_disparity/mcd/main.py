import math
import random
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from encoder_decoder.intrinsic_bias.algorithmic_disparity.mcd.mcd import compute_mcd, generate_translation, _tokenize_words, _stem

_MAIN_DIR      = Path(__file__).resolve().parent
MODEL_NAME     = "google-t5/t5-base"
MAX_NEW_TOKENS = 128
SEED           = 42
N_MAX          = 1000
N_BOOTSTRAP    = 1000
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32,
).to(DEVICE)
model.eval()
print(f"Model device: {next(model.parameters()).device}")
print(f"  Device: {DEVICE}\n")


_WINOMT_URL = (
    "https://raw.githubusercontent.com/gabrielStanovsky/mt_gender/master/"
    "data/aggregates/en.txt"
)


def load_winomt(n_max: int = N_MAX) -> List[str]:
    sentences = []
    try:
        with urllib.request.urlopen(_WINOMT_URL, timeout=15) as resp:
            for line in resp:
                line  = line.decode("utf-8").strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    sentences.append(parts[2].strip())
                if len(sentences) >= n_max:
                    break
    except Exception as e:
        print(f"  [warn] WinoMT load failed: {e}")
    print(f"  WinoMT: {len(sentences)} sentences loaded")
    return sentences


def load_europarl(n_max: int = N_MAX) -> List[str]:
    try:
        ds        = load_dataset("Helsinki-NLP/europarl", "en-fr",
                                 split="train", streaming=True,
                                 trust_remote_code=True)
        sentences = []
        for ex in ds:
            text = ex.get("translation", {}).get("en", "").strip()
            if text:
                sentences.append(text)
            if len(sentences) >= n_max:
                break
        print(f"  Europarl: {len(sentences)} sentences loaded")
        return sentences
    except Exception as e:
        print(f"  [warn] Europarl load failed: {e}")
        return []


def load_xnli(n_max: int = N_MAX) -> List[str]:
    try:
        ds = load_dataset("xnli", "en", split="validation")
        seen, sentences = set(), []
        for ex in ds:
            text = (ex.get("premise") or "").strip()
            if text and text not in seen:
                seen.add(text)
                sentences.append(text)
            if len(sentences) >= n_max:
                break
        print(f"  XNLI: {len(sentences)} premises loaded")
        return sentences
    except Exception as e:
        print(f"  [warn] XNLI load failed: {e}")
        return []

def _compute_hd_from_translations(translations: List[str]):
    stem_wordforms = defaultdict(lambda: defaultdict(int))
    for generated in translations:
        words = _tokenize_words(generated)
        for w in words:
            if len(w) < 3:
                continue
            stem = _stem(w)
            stem_wordforms[stem][w.lower()] += 1

    h_scores, d_scores = [], []
    for stem, wf_counts in stem_wordforms.items():
        total = sum(wf_counts.values())
        if total < 2:
            continue
        p_vals = [c / total for c in wf_counts.values()]
        h_scores.append(-sum(p * math.log(p) for p in p_vals if p > 0))
        d_scores.append(sum(p ** 2 for p in p_vals))

    mean_h = sum(h_scores) / len(h_scores) if h_scores else 0.0
    mean_d = sum(d_scores) / len(d_scores) if d_scores else 0.0
    return mean_h, mean_d


def bootstrap_mcd(translations: List[str],
                  n_bootstrap: int = N_BOOTSTRAP,
                  seed: int = SEED) -> dict:
    rng   = np.random.default_rng(seed)
    trans = np.array(translations)
    n     = len(trans)

    h_scores, d_scores = [], []

    for _ in range(n_bootstrap):
        idx      = rng.choice(n, size=n, replace=True)
        resample = list(trans[idx])
        try:
            h, d = _compute_hd_from_translations(resample)
            if np.isfinite(h) and np.isfinite(d):
                h_scores.append(h)
                d_scores.append(d)
        except Exception:
            continue

    print(f"    Bootstrap valid samples: {len(h_scores)}/{n_bootstrap}")

    def agg(scores, point_estimate):
        if len(scores) < 10:
            return {"point":   round(point_estimate, 4),
                    "std":     None,
                    "ci_low":  None,
                    "ci_high": None,
                    "sig":     ""}
        arr     = np.array(scores)
        ci_low  = float(np.percentile(arr, 2.5))
        ci_high = float(np.percentile(arr, 97.5))
        std     = float(np.std(arr, ddof=1))
        sig     = "*" if ci_low > 0 or ci_high < 0 else ""
        return {
            "point":   round(point_estimate, 4),
            "std":     round(std,            4),
            "ci_low":  round(ci_low,         4),
            "ci_high": round(ci_high,        4),
            "sig":     sig,
        }

    h_pt, d_pt = _compute_hd_from_translations(translations)
    return {
        "h": agg(h_scores, h_pt),
        "d": agg(d_scores, d_pt),
    }


def run_dataset(name: str, sentences: List[str], attr: str) -> dict:
    n = min(len(sentences), N_MAX)
    if n == 0:
        print(f"\n  [skip] {name} — no sentences loaded")
        return {"dataset": name, "sensitive_attr": attr, "n_sentences": 0,
                "h": None, "h_std": None, "h_ci_low": None,
                "h_ci_high": None, "h_sig": "",
                "d": None, "d_std": None, "d_ci_low": None,
                "d_ci_high": None, "d_sig": ""}

    print(f"\n{'─' * 60}")
    print(f"  MCD  |  {name}  ({n} sentences)")
    print(f"  sensitive attr : {attr}")
    print(f"{'─' * 60}")

    
    print(f"    Generating {n} translations ...")
    translations = []
    for i, sent in enumerate(sentences[:n]):
        translations.append(
            generate_translation(model, tokenizer, sent, MAX_NEW_TOKENS)
        )
        if (i + 1) % 50 == 0:
            print(f"    ... {i + 1}/{n}", end="\r")
    print()

    mean_h, mean_d = _compute_hd_from_translations(translations)

    aggs = bootstrap_mcd(translations)

    def fmt(agg):
        if agg["ci_low"] is None:
            return f"{agg['point']:.4f}   N/A              N/A"
        return (f"{agg['point']:.4f} ±{agg['std']:.4f} "
                f"[{agg['ci_low']:.4f}, {agg['ci_high']:.4f}]{agg['sig']}")

    def flat(key, agg):
        return {
            f"{key}":         agg["point"]
        }

    row = {"dataset": name}
    row.update(flat("h", aggs["h"]))
    row.update(flat("d", aggs["d"]))
    return row


def main():
    HF_TOKEN = None

    print("\n" + "=" * 65)
    print(f"  Morphological Complexity Disparity (MCD)  |  {MODEL_NAME}")
    print(f"  Significance: Bootstrap CI ({N_BOOTSTRAP} resamples)")
    print("  H = Shannon Entropy  (higher = more diverse)")
    print("  D = Simpson Index    (higher = more homogeneous)")
    print("=" * 65)

    results = []
    results.append(run_dataset("WinoMT",
                               load_winomt(), "gender"))
    results.append(run_dataset("XNLI",
                               load_xnli(),
                               "religion"))
    results.append(run_dataset("Europarl",
                               load_europarl(), "linguistic-complexity"))

    pd.DataFrame(results).to_csv(_MAIN_DIR / "mcd_results.csv", index=False)
    print(f"\n[INFO] Saved: mcd_results.csv")


if __name__ == "__main__":
    main()