import random
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from encoder_decoder.intrinsic_bias.stereotypical_association.sd.sd import compute_sd, pronoun_accuracy, age_accuracy

MODEL_NAME     = "google/mt5-base"
MAX_NEW_TOKENS = 128
SEED           = 42
N_MAX          = 200
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
_MALE_DOMINATED_OCCS = {
    "developer", "mechanic", "driver", "janitor", "constructor",
    "laborer", "farmer", "guard", "chief", "lawyer", "physician",
    "carpenter", "manager", "analyst", "supervisor", "programmer",
    "surgeon", "architect", "engineer",
}


def load_winomt(n_max: int = N_MAX) -> Tuple[List, List, List, List]:
    stereo_sents, stereo_labels = [], []
    anti_sents,   anti_labels   = [], []
    try:
        with urllib.request.urlopen(_WINOMT_URL, timeout=15) as resp:
            for line in resp:
                parts = line.decode("utf-8").strip().split("\t")
                if len(parts) < 4:
                    continue
                gender    = parts[0].strip().lower()
                sent      = parts[2].strip()
                occ       = parts[3].strip().lower()
                is_male_occ = occ in _MALE_DOMINATED_OCCS
                is_stereo   = (gender == "male" and is_male_occ) or \
                              (gender == "female" and not is_male_occ)
                if is_stereo and len(stereo_sents) < n_max:
                    stereo_sents.append(sent)
                    stereo_labels.append(gender)
                elif not is_stereo and len(anti_sents) < n_max:
                    anti_sents.append(sent)
                    anti_labels.append(gender)
                if len(stereo_sents) >= n_max and len(anti_sents) >= n_max:
                    break
    except Exception as e:
        print(f"  [warn] WinoMT load failed: {e}")
    n = min(len(stereo_sents), len(anti_sents), n_max)
    print(f"  WinoMT: {n} stereo, {n} anti pairs loaded")
    return stereo_sents[:n], stereo_labels[:n], anti_sents[:n], anti_labels[:n]


def _gender_from_tokens(tokens):
    for tok in tokens:
        if tok.lower() in {"he", "him", "his", "himself"}:
            return "male"
        if tok.lower() in {"she", "her", "hers", "herself"}:
            return "female"
    return "unknown"


def load_winobias(n_max: int = N_MAX) -> Tuple[List, List, List, List]:
    try:
        ds_pro  = load_dataset("uclanlp/wino_bias", "type2_pro",  split="test")
        ds_anti = load_dataset("uclanlp/wino_bias", "type2_anti", split="test")
    except Exception as e:
        print(f"  [warn] WinoBias load failed: {e}")
        return [], [], [], []
    stereo_sents, stereo_labels = [], []
    anti_sents,   anti_labels   = [], []
    for row_pro, row_anti in zip(ds_pro, ds_anti):
        if len(stereo_sents) < n_max:
            stereo_sents.append(" ".join(row_pro["tokens"]))
            stereo_labels.append(_gender_from_tokens(row_pro["tokens"]))
        if len(anti_sents) < n_max:
            anti_sents.append(" ".join(row_anti["tokens"]))
            anti_labels.append(_gender_from_tokens(row_anti["tokens"]))
        if len(stereo_sents) >= n_max and len(anti_sents) >= n_max:
            break
    n = min(len(stereo_sents), len(anti_sents), n_max)
    print(f"  WinoBias: {n} stereo, {n} anti pairs loaded")
    return stereo_sents[:n], stereo_labels[:n], anti_sents[:n], anti_labels[:n]


def load_europarl_age(n_max: int = N_MAX) -> Tuple[List, List, List, List]:
    _OLD_MARKERS   = {"elderly", "senior", "pensioner", "pensioners",
                      "retiree", "retired", "veteran", "elder", "aged",
                      "older", "grandfather", "grandmother", "retirement",
                      "pension", "geriatric"}
    _YOUNG_MARKERS = {"young", "teenager", "youth", "adolescent", "student",
                      "junior", "millennial", "child", "apprentice",
                      "youngster", "younger", "generation"}
    stereo_sents, stereo_labels = [], []
    anti_sents,   anti_labels   = [], []
    try:
        ds = load_dataset("Helsinki-NLP/europarl", "en-fr",
                          split="train", streaming=True, trust_remote_code=True)
        for ex in ds:
            sent  = ex.get("translation", {}).get("en", "").strip()
            if not sent:
                continue
            words     = set(sent.lower().split())
            has_old   = bool(words & _OLD_MARKERS)
            has_young = bool(words & _YOUNG_MARKERS)
            if has_old and not has_young and len(stereo_sents) < n_max:
                stereo_sents.append(sent)
                stereo_labels.append("old")
            elif has_young and not has_old and len(anti_sents) < n_max:
                anti_sents.append(sent)
                anti_labels.append("young")
            if len(stereo_sents) >= n_max and len(anti_sents) >= n_max:
                break
    except Exception as e:
        print(f"  [warn] Europarl load failed: {e}")
    n = min(len(stereo_sents), len(anti_sents), n_max)
    print(f"  Europarl: {n} stereo (old), {n} anti (young) pairs loaded")
    return stereo_sents[:n], stereo_labels[:n], anti_sents[:n], anti_labels[:n]


def bootstrap_sd(rows: list, n_bootstrap: int = N_BOOTSTRAP,
                 seed: int = SEED) -> dict:
    rng        = np.random.default_rng(seed)
    stereo_arr = np.array([r for r in rows if r["split"] == "stereo"])
    anti_arr   = np.array([r for r in rows if r["split"] == "anti"])
    n_s        = len(stereo_arr)
    n_a        = len(anti_arr)

    ms_scores, ma_scores, ds_scores = [], [], []

    for _ in range(n_bootstrap):
        s_idx    = rng.choice(n_s, size=n_s, replace=True)
        a_idx    = rng.choice(n_a, size=n_a, replace=True)
        s_scores = [stereo_arr[i]["score"] for i in s_idx]
        a_scores = [anti_arr[i]["score"]   for i in a_idx]

        ms = float(np.mean(s_scores))
        ma = float(np.mean(a_scores))
        ds = ma - ms

        if all(np.isfinite([ms, ma, ds])):
            ms_scores.append(ms)
            ma_scores.append(ma)
            ds_scores.append(ds)

    print(f"    Bootstrap valid samples: {len(ms_scores)}/{n_bootstrap}")

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

    all_stereo = [r["score"] for r in rows if r["split"] == "stereo"]
    all_anti   = [r["score"] for r in rows if r["split"] == "anti"]
    ms_pt      = float(np.mean(all_stereo))
    ma_pt      = float(np.mean(all_anti))
    ds_pt      = ma_pt - ms_pt

    return {
        "m_stereo": agg(ms_scores, ms_pt),
        "m_anti":   agg(ma_scores, ma_pt),
        "delta_s":  agg(ds_scores, ds_pt),
    }


def run_dataset(name, stereo_sents, stereo_labels,
                anti_sents, anti_labels, sensitive_attr, metric_fn) -> dict:
    n = min(len(stereo_sents), len(anti_sents), N_MAX)
    if n == 0:
        print(f"\n  [skip] {name} — no pairs loaded")
        return {"dataset": name, "sensitive_attr": sensitive_attr,
                "n_pairs": 0,
                "m_stereo": None, "m_stereo_std": None,
                "m_stereo_ci_low": None, "m_stereo_ci_high": None,
                "m_stereo_sig": "",
                "m_anti": None, "m_anti_std": None,
                "m_anti_ci_low": None, "m_anti_ci_high": None,
                "m_anti_sig": "",
                "delta_s": None, "delta_s_std": None,
                "delta_s_ci_low": None, "delta_s_ci_high": None,
                "delta_s_sig": ""}

    print(f"\n{'─' * 60}")
    print(f"  SD   |  {name}  ({n} pairs per split)")
    print(f"  sensitive attr : {sensitive_attr}")
    print(f"  metric         : {metric_fn.__name__}")
    print(f"{'─' * 60}")

    m_stereo, m_anti, delta_s, rows = compute_sd(
        model, tokenizer,
        stereo_sents[:n], stereo_labels[:n],
        anti_sents[:n],   anti_labels[:n],
        MAX_NEW_TOKENS, metric_fn,
    )

    aggs = bootstrap_sd(rows)

    def fmt(agg):
        if agg["ci_low"] is None:
            return f"{agg['point']:.4f}   N/A              N/A"
        return (f"{agg['point']:.4f} ±{agg['std']:.4f} "
                f"[{agg['ci_low']:.4f}, {agg['ci_high']:.4f}]{agg['sig']}")

    print(f"  M_stereo : {fmt(aggs['m_stereo'])}")
    print(f"  M_anti   : {fmt(aggs['m_anti'])}")
    print(f"  ΔS       : {fmt(aggs['delta_s'])}  "
          f"({'anti-stereo better ✓' if delta_s > 0 else 'stereo better (biased)' if delta_s < 0 else 'fair'})")

    def flat(key, agg):
        return {
            f"{key}":         agg["point"]
        }

    row = {"dataset": name}
    row.update(flat("delta_s",  aggs["delta_s"]))
    return row


def main():
    print("\n" + "=" * 65)
    print(f"  Stereotype-based Disparity (SD)  |  {MODEL_NAME}")
    print("=" * 65)

    results = []

    s_s, s_l, a_s, a_l = load_winomt()
    results.append(run_dataset(
        "WinoMT", s_s, s_l, a_s, a_l, "gender", pronoun_accuracy))

    s_s, s_l, a_s, a_l = load_winobias()
    results.append(run_dataset(
        "WinoBias", s_s, s_l, a_s, a_l, "gender", pronoun_accuracy))

    s_s, s_l, a_s, a_l = load_europarl_age()
    results.append(run_dataset(
        "Europarl", s_s, s_l, a_s, a_l, "age", age_accuracy))


    pd.DataFrame(results).to_csv("sd_results.csv", index=False)
    print(f"\n[INFO] Saved: sd_results.csv")


if __name__ == "__main__":
    main()