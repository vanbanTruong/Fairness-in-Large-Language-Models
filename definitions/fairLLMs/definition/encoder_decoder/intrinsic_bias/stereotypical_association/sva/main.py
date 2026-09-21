import random
import re
import urllib.request
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from fairLLMs.definition.encoder_decoder.intrinsic_bias.stereotypical_association.sva.sva import (
    compute_stereotype_direction,
    compute_sva,
    _GENDER_STEREO_TEXTS,
    _GENDER_ANTI_TEXTS,
    _AGE_STEREO_TEXTS,
    _AGE_ANTI_TEXTS,
)

warnings.filterwarnings("ignore")

_MAIN_DIR = Path(__file__).resolve().parent

MODEL_NAME    = "google/mt5-base"
SEED          = 42
N_MAX         = 100
N_MC_SAMPLES  = 50
TOP_PCT       = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def results_to_csv(results, filename: str) -> str:
    df = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    out_path = _MAIN_DIR / filename
    df.to_csv(out_path, index=False)
    return str(out_path)


print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
).to(DEVICE)
model.eval()

N_LAYERS = model.config.num_layers
N_HEADS  = model.config.num_heads
N_TOTAL  = N_LAYERS * N_HEADS

print(f"Model device  : {next(model.parameters()).device}")
print(f"  Device      : {DEVICE}")
print(f"  Enc layers  : {N_LAYERS}")
print(f"  Heads/layer : {N_HEADS}")
print(f"  Total heads : {N_TOTAL}\n")

_WINOMT_URL = (
    "https://raw.githubusercontent.com/gabrielStanovsky/"
    "mt_gender/master/data/aggregates/en.txt"
)

def load_winomt(n_max: int = N_MAX) -> Tuple[List[str], List[str]]:
    try:
        ds_pro  = load_dataset("uclanlp/wino_bias", "type1_pro",  split="test")
        ds_anti = load_dataset("uclanlp/wino_bias", "type1_anti", split="test")
        stereo, anti = [], []
        for rp, ra in zip(ds_pro, ds_anti):
            s = " ".join(rp["tokens"])
            a = " ".join(ra["tokens"])
            n_diff = sum(1 for x, y in zip(rp["tokens"], ra["tokens"]) if x != y)
            if s and a and 1 <= n_diff <= 3:
                stereo.append(s)
                anti.append(a)
            if len(stereo) >= n_max:
                break
        print(f"  WinoMT: {len(stereo)} pairs")
        return stereo, anti
    except Exception as e:
        print(f"  [warn] WinoMT load failed: {e}")
        return [], []


def load_winobias(n_max: int = N_MAX) -> Tuple[List[str], List[str]]:
    for split in ("validation", "test"):
        try:
            ds_pro  = load_dataset("uclanlp/wino_bias", "type1_pro",  split=split)
            ds_anti = load_dataset("uclanlp/wino_bias", "type1_anti", split=split)
            stereo, anti = [], []
            for rp, ra in zip(ds_pro, ds_anti):
                s = " ".join(rp["tokens"])
                a = " ".join(ra["tokens"])
                n_diff = sum(1 for x, y in zip(rp["tokens"], ra["tokens"]) if x != y)
                if s and a and 1 <= n_diff <= 3:
                    stereo.append(s)
                    anti.append(a)
                if len(stereo) >= n_max:
                    break
            print(f"  WinoBias: {len(stereo)} pairs")
            return stereo, anti
        except Exception:
            continue
    print("  [warn] WinoBias load failed")
    return [], []

_AGE_PATTERNS = [
    (r"\belderly\b",        "adult"),
    (r"\bold people\b",     "people"),
    (r"\bold man\b",        "man"),
    (r"\bold woman\b",      "woman"),
    (r"\bold men\b",        "men"),
    (r"\bold women\b",      "women"),
    (r"\bsenior citizen\b", "citizen"),
    (r"\bpensioner\b",      "person"),
    (r"\bretiree\b",        "person"),
    (r"\bthe aged\b",       "people"),
    (r"\byoung people\b",   "people"),
    (r"\byoungster\b",      "person"),
    (r"\bthe young\b",      "people"),
]

def load_europarl(n_max: int = N_MAX) -> Tuple[List[str], List[str]]:
    try:
        ds = load_dataset(
            "Helsinki-NLP/europarl", "en-fr",
            split="train", streaming=True,
        )
        stereo, anti = [], []
        for item in ds:
            text = (item.get("translation") or {}).get("en", "") or item.get("text", "")
            if not text:
                continue
            for pattern, replacement in _AGE_PATTERNS:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    swapped = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                    if swapped != text and len(swapped) > 20:
                        stereo.append(text)
                        anti.append(swapped)
                        break
            if len(stereo) >= n_max:
                break
        print(f"  Europarl: {len(stereo)} pairs")
        return stereo, anti
    except Exception as e:
        print(f"  [warn] Europarl load failed: {e}")
        return [], []


def run_dataset(
    name:          str,
    stereo_sents:  List[str],
    anti_sents:    List[str],
    attr:          str,
    stereo_dir_texts: List[str],
    anti_dir_texts:   List[str],
) -> dict:
    n = min(len(stereo_sents), len(anti_sents), N_MAX)
    if n < 10:
        print(f"\n  [skip] {name} — only {n} pairs (need ≥ 10)")
        return {"dataset": name, "sensitive_attr": attr,
                "n_pairs": n, "SVA": None}

    print(f"\n{'─' * 55}")
    print(f"  SVA  |  {name}  ({n} pairs)")
    print(f"  sensitive attr : {attr}")
    print(f"{'─' * 55}")

    print("  Computing stereotype direction …")
    direction = compute_stereotype_direction(
        model, tokenizer, stereo_dir_texts, anti_dir_texts,
    )

    sva, phi = compute_sva(
        model, tokenizer,
        stereo_sents[:n], anti_sents[:n],
        direction,
        n_layers=N_LAYERS, n_heads=N_HEADS,
        n_samples=N_MC_SAMPLES,
        top_pct=TOP_PCT,
    )

    top5_idx = np.argsort(np.abs(phi))[::-1][:5]
    top5     = [(int(i), round(float(phi[i]), 4)) for i in top5_idx]

    print(f"\n  SVA φ (top {int(TOP_PCT*100)}% heads) : {sva:.4f}")
    print(f"  Top-5 heads (idx, φ)         : {top5}")
    print(f"  (higher φ = head encodes more stereotypical bias)")

    rows = [
        {"head_idx": int(i), "layer": int(i) // N_HEADS,
         "head": int(i) % N_HEADS, "phi": round(float(phi[i]), 6)}
        for i in range(N_TOTAL)
    ]

    return {
        "dataset":        name,
        "sva_score":            round(sva, 4),
    }

def main():
    print("\n" + "=" * 55)
    print(f"  SVA (Shapley-Value Attribution)  |  {MODEL_NAME}")
    print("=" * 55)

    results = []

    winomt_s, winomt_a = load_winomt()
    results.append(run_dataset(
        "WinoMT", winomt_s, winomt_a, "gender",
        _GENDER_STEREO_TEXTS, _GENDER_ANTI_TEXTS,
    ))

    winob_s, winob_a = load_winobias()
    results.append(run_dataset(
        "WinoBias", winob_s, winob_a, "gender",
        _GENDER_STEREO_TEXTS, _GENDER_ANTI_TEXTS,
    ))

    euro_s, euro_a = load_europarl()
    results.append(run_dataset(
        "Europarl", euro_s, euro_a, "age",
        _AGE_STEREO_TEXTS, _AGE_ANTI_TEXTS,
    ))

    summary = pd.DataFrame(
        [r for r in results if r.get("sva_score") is not None]
    )[["dataset", "sva_score"]]
    summary.columns = ["dataset","sva_score"]
    out = results_to_csv(summary, "sva_results.csv")

    print("\n" + "=" * 55)
    print("  SVA Summary  (higher φ = more stereotypical head contribution)")
    print("=" * 55)
    print(summary.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()