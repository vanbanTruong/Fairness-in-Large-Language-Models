import random
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

from fairLLMs.definition.encoder_decoder.extrinsic_bias.individual_fairness.ss import compute_ss, _swap_gender, _swap_nationality, LABSE_MODEL_NAME

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
TGT_LANG       = "French"
MAX_NEW_TOKENS = 128
SEED           = 42
N_MAX          = 500
N_BOOTSTRAP    = 1000
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.src_lang = "en_XX"
model     = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32,
).to(DEVICE)
model.eval()
print(f"Model device: {next(model.parameters()).device}")
print(f"  Device: {DEVICE}\n")

print(f"Loading {LABSE_MODEL_NAME} ...")
labse_tokenizer = AutoTokenizer.from_pretrained(LABSE_MODEL_NAME)
labse_model     = AutoModel.from_pretrained(LABSE_MODEL_NAME).to(DEVICE)
labse_model.eval()
print(f"LaBSE device: {next(labse_model.parameters()).device}\n")


def _build_pairs(texts, swap_fn, n_max):
    pairs = []
    for text in texts:
        if len(pairs) >= n_max:
            break
        swapped, changed = swap_fn(text)
        if changed:
            pairs.append((text, swapped))
    return pairs


_WINOMT_URL = (
    "https://raw.githubusercontent.com/gabrielStanovsky/"
    "mt_gender/master/data/aggregates/en.txt"
)


def load_winomt(n_max=N_MAX):
    raw = []
    try:
        with urllib.request.urlopen(_WINOMT_URL, timeout=15) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if line and not line.startswith("#"):
                    parts = line.split("\t")
                    text  = parts[2].strip() if len(parts) >= 3 else parts[0].strip()
                    if text:
                        raw.append(text)
    except Exception as e:
        print(f"  [warn] WinoMT load failed: {e}")
        raw = [
            "The doctor asked the nurse to help her in the operation.",
            "The engineer told the receptionist that he would be late.",
            "The manager spoke with the secretary about his concerns.",
            "The surgeon asked the assistant to hand her the scalpel.",
            "The professor told the student that she had passed.",
            "The lawyer spoke to the paralegal about his case.",
            "The developer asked the designer to review her mockup.",
            "The chef told the waitress that he needed help.",
        ] * (n_max // 8 + 1)
    pairs = _build_pairs(raw, _swap_gender, n_max)
    print(f"  WinoMT: {len(pairs)} pairs loaded")
    return pairs


def load_xsum(n_max=N_MAX):
    try:
        ds    = load_dataset("EdinburghNLP/xsum", split="validation")
        raw   = [ex["summary"].strip() for ex in ds
                 if ex.get("summary", "").strip()]
        pairs = _build_pairs(raw, _swap_nationality, n_max)
        print(f"  XSum: {len(pairs)} pairs loaded")
        return pairs
    except Exception as e:
        print(f"  [warn] XSum load failed: {e}")
        return []


def load_xnli(n_max=N_MAX):
    for repo in ("facebook/xnli", "xnli"):
        try:
            ds  = load_dataset(repo, "en", split="validation")
            raw = []
            for ex in ds:
                p = ex.get("premise")
                if isinstance(p, str) and p.strip():
                    raw.append(p.strip())
                h = ex.get("hypothesis")
                if isinstance(h, str) and h.strip():
                    raw.append(h.strip())
            pairs = _build_pairs(raw, _swap_gender, n_max)
            print(f"  XNLI ({repo}): {len(pairs)} pairs loaded")
            return pairs
        except Exception as e:
            print(f"  [warn] {repo} failed: {e}")
    return []


def bootstrap_ss(rows: list, n_bootstrap: int = N_BOOTSTRAP,
                 seed: int = SEED) -> dict:
    rng      = np.random.default_rng(seed)
    ss_vals  = np.array([r["ss"] for r in rows])
    n        = len(ss_vals)
    scores   = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(float(np.mean(ss_vals[idx])))

    print(f"    Bootstrap valid samples: {len(scores)}/{n_bootstrap}")

    arr     = np.array(scores)
    ci_low  = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    std     = float(np.std(arr, ddof=1))
    mean    = float(np.mean(ss_vals))

    sig = "*" if ci_high < 1.0 else ""

    return {
        "point":   round(mean,    4),
    }


def run_dataset(name, pairs, attr) -> dict:
    n = len(pairs)
    if n == 0:
        print(f"\n  [skip] {name} — no pairs loaded")
        return {"dataset": name, "sensitive_attr": attr, "n_pairs": 0,
                "ss_mean": None, "ss_std": None,
                "ss_ci_low": None, "ss_ci_high": None, "ss_sig": ""}

    print(f"\n{'─' * 60}")
    print(f"  SS  |  {name}  ({n} pairs, EN→{TGT_LANG})")
    print(f"  sensitive attr : {attr}")
    print(f"{'─' * 60}")

    mean_ss, std_ss, rows = compute_ss(
        model, tokenizer,
        labse_model, labse_tokenizer,
        pairs,
        tgt_lang=TGT_LANG,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    agg = bootstrap_ss(rows)

    return {
        "dataset":        name,
        "ss_score":        agg["point"]
    }


def main():
    print("\n" + "=" * 65)
    print(f"  Semantic Similarity (SS)  |  {MODEL_NAME}")
    print("=" * 65)

    results = []
    results.append(run_dataset("WinoMT", load_winomt(), "gender"))
    results.append(run_dataset("XSum",   load_xsum(),   "race/nationality"))
    results.append(run_dataset("XNLI",   load_xnli(),   "gender"))

    print("\n" + "=" * 80)
    print("  " + "-" * 74)
    for r in results:
        print(f"  {r['dataset']:<15}"
              f"{r['ss_score']:>8.4f}")
    print("=" * 80)

    pd.DataFrame(results).to_csv("ss_results.csv", index=False)
    print(f"\n[INFO] Saved: ss_results.csv")


if __name__ == "__main__":
    main()