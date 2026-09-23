import re
import random
import itertools
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM

from encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cat.cat import compute_ss

_MAIN_DIR  = Path(__file__).resolve().parent
CROWS_CSV  = _MAIN_DIR / "crows_pairs_anonymized.csv"
MODEL_NAME = "bert-base-uncased"
SEED       = 42
N_MAX      = 100000
rng = np.random.default_rng(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def results_to_csv(results, filename):
    df = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    out = _MAIN_DIR / filename
    df.to_csv(out, index=False)
    return str(out)


print(f"Loading {MODEL_NAME} ...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print(f"  Device: {DEVICE}\n")


def _shuffle_meaningless(sentence: str) -> str:
    words = sentence.split()
    random.shuffle(words)
    return " ".join(words)



_STEREOSET_UNRELATED_POOL = None


def stereoset_unrelated_pool(n=500):
    global _STEREOSET_UNRELATED_POOL
    if _STEREOSET_UNRELATED_POOL is not None:
        return _STEREOSET_UNRELATED_POOL
    ds = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
    pool = []
    for row in ds:
        for sent, gl in zip(row["sentences"]["sentence"], row["sentences"]["gold_label"]):
            if gl == 2:
                pool.append(sent)
        if len(pool) >= n:
            break
    _STEREOSET_UNRELATED_POOL = pool
    print(f"  [INFO] StereoSet unrelated pool cached: {len(pool)} sentences")
    return pool



def load_crows(bias_type=None, n_max=N_MAX):
    df = pd.read_csv(CROWS_CSV)
    if bias_type:
        df = df[df["bias_type"] == bias_type]
    else:
        print(f"[INFO] CrowS-Pairs bias_type categories found: "
              f"{sorted(df['bias_type'].dropna().unique().tolist())}")
    stereo, anti, related = [], [], []
    for _, row in df.iterrows():
        if row["stereo_antistereo"] == "stereo":
            s, a = row["sent_more"], row["sent_less"]
        else:
            s, a = row["sent_less"], row["sent_more"]
        stereo.append(s); anti.append(a); related.append(_shuffle_meaningless(s))
        if len(stereo) >= n_max:
            break
    print(f"  CrowS-Pairs: {len(stereo)} triples, ALL categories pooled "
          f"(related = SYNTHETIC shuffle -> lms uninformative)")
    return stereo, anti, related



def load_stereoset(n_max=N_MAX):
    
    ds = load_dataset("McGill-NLP/stereoset", "intrasentence", split="validation")
    LABEL = {1: "stereo", 0: "anti", 2: "unrelated"}
    stereo, anti, related = [], [], []
    per_cat_counts = {}
    for row in ds:
        bucket = {"stereo": None, "anti": None, "unrelated": None}
        for sent, gl in zip(row["sentences"]["sentence"], row["sentences"]["gold_label"]):
            key = LABEL.get(gl)
            if key:
                bucket[key] = sent
        if all(bucket.values()):
            stereo.append(bucket["stereo"])
            anti.append(bucket["anti"])
            related.append(bucket["unrelated"])
            bt = row.get("bias_type", "unknown")
            per_cat_counts[bt] = per_cat_counts.get(bt, 0) + 1
        if len(stereo) >= n_max:
            break
    print(f"[INFO] StereoSet bias_type categories found: {sorted(per_cat_counts)}")
    for bt, n in sorted(per_cat_counts.items()):
        print(f"    {bt:<20} : {n} triples")
    print(f"  StereoSet: {len(stereo)} triples, ALL categories pooled "
          f"(native unrelated -> valid iCAT)")
    return stereo, anti, related



def load_xnli(n_max=N_MAX):
    try:
        ds = load_dataset("facebook/xnli", "en", split="train", streaming=True)
    except Exception as e:
        print(f"  [warn] XNLI load failed: {e}")
        return [], [], []
    ALIGN = [("christian", "muslim"), ("christians", "muslims"),
             ("christianity", "islam")]
    pool = stereoset_unrelated_pool() 
    seen, stereo, anti, related = set(), [], [], []
    for ex in ds:
        for field in ("premise", "hypothesis"):
            s = (ex.get(field) or "").strip()
            low = s.lower()
            for c, m in ALIGN:
                if re.search(rf"\b{c}\b", low): 
                    a_sent = re.sub(rf"\b{c}\b", m, s, flags=re.IGNORECASE)
                    if a_sent != s and s not in seen:
                        seen.add(s)
                        stereo.append(s)  
                        anti.append(a_sent)
                        related.append(rng.choice(pool))
                    break
                elif re.search(rf"\b{m}\b", low): 
                    s_sent = re.sub(rf"\b{m}\b", c, s, flags=re.IGNORECASE)
                    if s_sent != s and s_sent not in seen:
                        seen.add(s_sent)
                        stereo.append(s_sent) 
                        anti.append(s) 
                        related.append(rng.choice(pool))
                    break
            if len(stereo) >= n_max:
                break
        if len(stereo) >= n_max:
            break
    return stereo, anti, related


def run_dataset(name, stereo, anti, related, attr, valid_icat):
    n = min(len(stereo), len(anti), len(related))
    if n == 0:
        print(f"\n  [skip] {name} — no triples")
        return None
    print(f"\n{'-'*55}\n  iCAT | {name} ({n} triples) | {attr}\n{'-'*55}")
    ss, lms, icat, rows = compute_ss(model, tokenizer, stereo[:n], anti[:n], related[:n])
    print(f"  ss={ss:.2f}%  lms={lms:.2f}%  iCAT={icat:.2f}")
    return {"dataset": name, "cat_score": icat}


def main():
    print("\n" + "=" * 55)
    print(f"  Stereotype Score / iCAT  |  {MODEL_NAME}  |  one score per dataset")
    print("=" * 55)
    results = []
    s, a, r = load_stereoset()
    results.append(run_dataset("StereoSet", s, a, r, "race/gender/etc.", valid_icat=True))
    s, a, r = load_crows()
    results.append(run_dataset("CrowS-Pairs", s, a, r, "mixed", valid_icat=False))
    s, a, r = load_xnli()
    results.append(run_dataset("XNLI", s, a, r, "religion", valid_icat=False))
    results = [x for x in results if x is not None]

    summary = pd.DataFrame(results)
    print("\n" + "=" * 55)
    print(summary.to_string(index=False))
    print("=" * 55)
    print(f"\nSaved: {results_to_csv(summary, 'cat_results.csv')}")


if __name__ == "__main__":
    main()