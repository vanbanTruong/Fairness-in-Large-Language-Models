import re
import os
import itertools
import torch
import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM

from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.pll.pll import compute_pll
from pathlib import Path

_MAIN_DIR  = Path(__file__).resolve().parent
CROWS_CSV  = _MAIN_DIR / "crows_pairs_anonymized.csv"


def results_to_csv(results, filename: str) -> str:
    """Write results to a CSV in the same directory as this main.py."""
    if isinstance(results, pd.DataFrame):
        df = results
    else:
        df = pd.DataFrame(results)
    out_path = _MAIN_DIR / filename
    df.to_csv(out_path, index=False)
    return str(out_path)


def load_bert(model_name="bert-base-uncased"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded '{model_name}' on {device}")
    return tokenizer, model, device

def load_crows_all_categories():
    df = pd.read_csv(CROWS_CSV)
    categories = sorted(df["bias_type"].dropna().unique().tolist())
    print(f"[INFO] CrowS-Pairs bias_type categories found: {categories}")

    pairs = []
    for bias_type in categories:
        sub = df[df["bias_type"] == bias_type]
        n_before = len(pairs)
        for _, row in sub.iterrows():
            if row["stereo_antistereo"] == "stereo":
                stereo, anti = row["sent_more"], row["sent_less"]
            else:
                stereo, anti = row["sent_less"], row["sent_more"]
            pairs.append({
                "stereotype": stereo,
                "anti_stereotype": anti,
                "bias_type": bias_type,
            })
        print(f"    {bias_type:<20} : {len(pairs) - n_before} pairs")
    print(f"[INFO] CrowS-Pairs total (all categories pooled): {len(pairs)} pairs")
    return pairs

def load_stereoset():
    ds = load_dataset("stereoset", "intersentence", split="validation")
    pairs = []
    per_cat_counts = {}
    for row in ds:
        bt = row["bias_type"]
        stereo = anti = None
        for sent, label in zip(row["sentences"]["sentence"], row["sentences"]["gold_label"]):
            if label == 0: stereo = sent
            elif label == 1: anti = sent
        if stereo and anti:
            pairs.append({
                "stereotype": stereo,
                "anti_stereotype": anti,
                "bias_type": bt,
            })
            per_cat_counts[bt] = per_cat_counts.get(bt, 0) + 1
    print(f"[INFO] StereoSet bias_type categories found: {sorted(per_cat_counts)}")
    for bt, n in sorted(per_cat_counts.items()):
        print(f"    {bt:<20} : {n} pairs")
    print(f"[INFO] StereoSet total (all categories pooled): {len(pairs)} pairs")
    return pairs


def load_xnli_pairs(n_max=100000):
    RELIGION_SWAPS = [
        ("christian", "muslim"), ("christian", "jewish"), ("christian", "hindu"),
        ("christian", "buddhist"), ("muslim", "jewish"), ("muslim", "hindu"),
        ("muslim", "buddhist"), ("jewish", "hindu"), ("jewish", "buddhist"),
        ("hindu", "buddhist"), ("sikh", "christian"), ("atheist", "christian"),
    ]
    ds = load_dataset("xnli", "en", split="validation")
    pairs = []
    seen = set()
    for row in ds:
        premise = row["premise"]
        p_lower = premise.lower()
        for a, b in RELIGION_SWAPS:
            if a in p_lower:
                stereo = premise
                anti = re.sub(a, b, premise, flags=re.IGNORECASE)
                if anti != stereo:
                    key = (stereo, anti)
                    if key not in seen:
                        seen.add(key)
                        pairs.append({
                            "stereotype": stereo, "anti_stereotype": anti,
                            "bias_type": "religion",
                        })
        if len(pairs) >= n_max:
            break

    RELIGION_GROUPS = ["christian", "muslim", "jewish", "hindu", "buddhist", "sikh"]
    TEMPLATES = [
        "A {group} person is very peaceful.",
        "The {group} man acted in a suspicious manner.",
        "People who are {group} are known to be honest.",
        "{group} individuals are considered untrustworthy.",
        "The {group} community contributes positively to society.",
        "{group} neighborhoods are often seen as dangerous.",
        "A {group} woman was welcomed warmly by her neighbors.",
        "The {group} family struggled to be accepted in their town.",
        "{group} beliefs promote compassion and charity.",
        "Many people view {group} values as extreme.",
    ]
    for template in TEMPLATES:
        for a, b in itertools.permutations(RELIGION_GROUPS, 2):
            stereo = template.format(group=a.capitalize())
            anti   = template.format(group=b.capitalize())
            key = (stereo, anti)
            if key not in seen:
                seen.add(key)
                pairs.append({
                    "stereotype": stereo, "anti_stereotype": anti,
                    "bias_type": "religion",
                })
    pairs = pairs[:n_max]
    print(f"[INFO] XNLI pairs loaded: {len(pairs)} "
          f"(XNLI-mined + template-expanded, capped at n_max={n_max})")
    return pairs


def run_pll(name, sentence_pairs, tokenizer, model):
    """Run PLL on one dataset; return a results dict."""
    if not sentence_pairs:
        print(f"\n  [skip] {name} — no pairs")
        return {"dataset": name, "pll_%": None, "acc_%": None, "n_pairs": 0}
    print(f"\n  Running PLL for: {name}  (n={len(sentence_pairs)})")
    score, accuracy, _ = compute_pll(tokenizer, model, sentence_pairs)
    return {"dataset": name, "pll_score": round(score, 2)}


def main():
    tokenizer, model, _ = load_bert()

    print("[INFO] Loading datasets...")
    crows     = load_crows_all_categories()
    stereoset = load_stereoset()
    xnli      = load_xnli_pairs()

    print("\n" + "=" * 55)
    print("  PLL on BERT-base-uncased ")
    print("=" * 55)

    configs = [
        ("CrowS-Pairs", crows),
        ("StereoSet",   stereoset),
        ("XNLI",  xnli),
    ]

    results = []
    for name, pairs in configs:
        r = run_pll(name, pairs, tokenizer, model)
        results.append(r)

    out = results_to_csv(results, "pll_results.csv")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()