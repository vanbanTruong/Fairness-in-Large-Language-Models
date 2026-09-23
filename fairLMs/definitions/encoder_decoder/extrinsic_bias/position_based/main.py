import random
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from encoder_decoder.extrinsic_bias.position_based.npd import compute_npd

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
MAX_NEW_TOKENS = 128
K_SEGMENTS     = 10
SENTS_PER_DOC  = 20
SEED           = 42
N_MAX          = 200
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


def _chunk_into_docs(sentences: List[str], per_doc: int, n_max: int) -> List[str]:
    random.shuffle(sentences)
    docs = []
    for i in range(0, len(sentences) - per_doc + 1, per_doc):
        docs.append(" ".join(sentences[i:i + per_doc]))
        if len(docs) >= n_max:
            break
    return docs


def load_xsum(n_max=N_MAX):
    try:
        ds = load_dataset("EdinburghNLP/xsum", split="validation")
        articles, summaries = [], []
        for ex in ds:
            if len(articles) >= n_max:
                break
            doc  = ex.get("document", "").strip()
            summ = ex.get("summary",  "").strip()
            if doc and summ:
                articles.append(doc)
                summaries.append(summ)
        return articles, summaries
    except Exception as e:
        print(f"  [warn] XSum load failed: {e}")
        return [], None

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
                    text  = parts[2].strip() if len(parts) >= 3 else parts[0].strip()
                    if text:
                        sents.append(text)
    except Exception as e:
        print(f"  [warn] WinoMT load failed: {e}")
        return [], None
    sents = list(dict.fromkeys(sents))
    docs  = _chunk_into_docs(sents, SENTS_PER_DOC, n_max)
    return docs, None


def load_xnli(n_max=N_MAX):
    sents = []
    for repo in ("facebook/xnli", "xnli"):
        try:
            ds = load_dataset(repo, "en", split="validation")
            for ex in ds:
                for key in ("premise", "hypothesis"):
                    v = ex.get(key)
                    if isinstance(v, str) and v.strip():
                        sents.append(v.strip())
            break
        except Exception as e:
            print(f"  [warn] {repo} failed: {e}")
    if not sents:
        return [], None
    sents = list(dict.fromkeys(sents))
    docs  = _chunk_into_docs(sents, SENTS_PER_DOC, n_max)
    return docs, None


def bootstrap_npd(rows: list, n_bootstrap: int = N_BOOTSTRAP,
                  seed: int = SEED) -> dict:
    rng      = np.random.default_rng(seed)
    npd_vals = np.array([r["npd"] for r in rows])
    n        = len(npd_vals)
    scores   = [float(np.mean(npd_vals[rng.choice(n, size=n, replace=True)]))
                for _ in range(n_bootstrap)]
    print(f"    Bootstrap valid samples: {len(scores)}/{n_bootstrap}")
    arr     = np.array(scores)
    ci_low  = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    return {"point":  round(float(np.mean(npd_vals)), 4)}


def run_dataset(name, articles, summaries, attr) -> dict:
    n = len(articles)
    reference = "gold" if summaries is not None else "uniform"
    if n == 0:
        print(f"\n  [skip] {name} — no articles")
        return {"dataset": name, "npd_score": None}

    print(f"\n{'─' * 60}")
    print(f"  NPD  |  {name}  ({n} articles, K={K_SEGMENTS}, ref={reference})")
    print(f"{'─' * 60}")

    npd, rows = compute_npd(
        model, tokenizer, articles, summaries,
        max_new_tokens=MAX_NEW_TOKENS, K=K_SEGMENTS)

    agg = bootstrap_npd(rows)
    return {"dataset": name, "npd_score": agg["point"]}


def main():
    print("\n" + "=" * 65)
    print(f"  Position-based Disparity (NPD)  |  {MODEL_NAME}")
    print("=" * 65)

    results = []
    a, s = load_winomt(); results.append(run_dataset("WinoMT", a, s, "position"))
    a, s = load_xsum();   results.append(run_dataset("XSum",   a, s, "position"))
    a, s = load_xnli();   results.append(run_dataset("XNLI",   a, s, "position"))

    print("\n" + "=" * 84)
    print(f"  {'Dataset':<12} {'Ref':<9} {'Attr':<10} {'NPD':>8} {'Std':>6} {'95% CI':>22}")
    print("  " + "-" * 72)
    for r in results:
        if r["npd_score"] is None:
            print(f"  {r['dataset']:<12} [skipped]")
            continue
        print(f"  {r['dataset']:<12} {r['npd_score']:>8.4f}")
    print("=" * 84)

    pd.DataFrame(results).to_csv("npd_results.csv", index=False)
    print(f"\n[INFO] Saved: npd_results.csv")


if __name__ == "__main__":
    main()