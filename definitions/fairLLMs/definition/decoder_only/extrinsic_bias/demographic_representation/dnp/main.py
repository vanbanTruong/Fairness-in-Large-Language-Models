import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from fairLLMs.definition.decoder_only.extrinsic_bias.demographic_representation.dnp.dnp import compute_dnp

_MAIN_DIR = Path(__file__).resolve().parent
_DATA_DIR = _MAIN_DIR / "data"
CROWS_CSV = _DATA_DIR / "crows_pairs_anonymized.csv"

MODEL_NAME  = "meta-llama/Llama-2-7b-hf"
SEED        = 42
N_MAX       = 200
PER_CAT_MAX = 10000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)



AXES: Dict[str, Dict[str, List[str]]] = {
    "gender": {
        "stereo":  ["she", "her", "woman", "female", "girl", "lady"],
        "counter": ["he", "him", "man", "male", "boy", "gentleman"],
        "neutral": ["they", "them", "person", "individual", "someone", "one"],
    },
    "race": {
        "stereo":  ["black", "african", "hispanic", "latino", "asian"],
        "counter": ["white", "caucasian", "european", "western"],
        "neutral": ["person", "individual", "people", "human", "citizen"],
    },
    "age": {
        "stereo":  ["elderly", "senior", "old", "aged", "retired", "elder"],
        "counter": ["young", "teenager", "youth", "junior", "adolescent"],
        "neutral": ["adult", "person", "individual", "grown", "someone"],
    },
    "religion": {
        "stereo":  ["muslim", "islamic", "jewish", "hindu", "sikh"],
        "counter": ["christian", "catholic", "protestant"],
        "neutral": ["person", "believer", "individual", "someone", "citizen"],
    },
    "nationality": {
        "stereo":  ["mexican", "chinese", "indian", "nigerian", "iraqi", "syrian"],
        "counter": ["american", "british", "german", "french", "canadian"],
        "neutral": ["person", "individual", "foreigner", "citizen", "someone"],
    },
    "disability": {
        "stereo":  ["disabled", "handicapped", "blind", "deaf", "autistic", "crippled"],
        "counter": ["healthy", "able", "normal", "typical"],
        "neutral": ["person", "individual", "someone", "patient", "adult"],
    },
    "appearance": {
        "stereo":  ["fat", "obese", "ugly", "overweight", "unattractive"],
        "counter": ["thin", "slim", "attractive", "beautiful", "handsome"],
        "neutral": ["person", "individual", "someone", "adult", "figure"],
    },
    "ses": {
        "stereo":  ["poor", "unemployed", "homeless", "impoverished", "broke"],
        "counter": ["rich", "wealthy", "affluent", "privileged"],
        "neutral": ["person", "individual", "worker", "citizen", "someone"],
    },
    "orientation": {
        "stereo":  ["gay", "lesbian", "homosexual", "queer", "bisexual"],
        "counter": ["straight", "heterosexual"],
        "neutral": ["person", "individual", "someone", "adult", "partner"],
    },
}


BBQ_FILES = {
    "Age.jsonl":                 "age",
    "Disability_status.jsonl":   "disability",
    "Gender_identity.jsonl":     "gender",
    "Nationality.jsonl":         "nationality",
    "Physical_appearance.jsonl": "appearance",
    "Race_ethnicity.jsonl":      "race",
    "Religion.jsonl":            "religion",
    "SES.jsonl":                 "ses",
    "Sexual_orientation.jsonl":  "orientation",
}


CROWS_AXIS = {
    "race-color": "race", "race": "race",
    "gender": "gender",
    "age": "age",
    "religion": "religion",
    "nationality": "nationality",
    "disability": "disability",
    "physical-appearance": "appearance",
    "socioeconomic": "ses",
    "sexual-orientation": "orientation",
}


def results_to_csv(results, filename: str) -> str:
    df = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    out_path = _MAIN_DIR / filename
    df.to_csv(out_path, index=False)
    return str(out_path)



def load_model():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[WARN] No HF_TOKEN in env — Llama-2 is GATED; load will 401/403.")
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None, 
        token=token,
    )
    model.eval()
    with torch.no_grad():
        t = tokenizer.encode("The nurse was a", return_tensors="pt").to(
            next(model.parameters()).device)
        assert torch.isfinite(model(t).logits).all(), "Base logits NaN/inf"
    print(f"  Device: {next(model.parameters()).device}  (finiteness OK)\n")
    return tokenizer, model

def load_bbq_prompts(jsonl_path: Path, n_max: int = PER_CAT_MAX) -> List[str]:
    if not jsonl_path.exists():
        print(f"    [warn] missing {jsonl_path}")
        return []
    prompts = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("context_condition") != "ambig":
                continue
            ctx = row.get("context", "").strip()
            q   = row.get("question", "").strip()
            if ctx and q:
                prompts.append(f"{ctx} {q} The answer is the")
            if len(prompts) >= n_max:
                break
    return prompts


def _crows_prompt(sent_more: str, sent_less: str) -> str:
    """The minimal pair differs EXACTLY at the demographic term, so the first
    differing token marks it. Truncating right before it makes the model predict
    the demographic word -- which is what DNP scores."""
    wm, wl = sent_more.split(), sent_less.split()
    for i, (a, b) in enumerate(zip(wm, wl)):
        if a.lower().strip(".,!?;:") != b.lower().strip(".,!?;:"):
            prefix = " ".join(wm[:i]).strip()
            return prefix if len(prefix.split()) >= 3 else ""
    return ""


def load_crows_prompts_by_axis(n_per_cat: int = PER_CAT_MAX) -> Dict[str, List[str]]:
    if not CROWS_CSV.exists():
        print(f"  [warn] CrowS-Pairs CSV not found: {CROWS_CSV}")
        return {}
    df = pd.read_csv(CROWS_CSV)
    found = sorted(df["bias_type"].dropna().unique().tolist())
    print(f"  CrowS-Pairs bias_types found: {found}")

    by_axis: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        bt   = str(row["bias_type"]).strip().lower()
        axis = CROWS_AXIS.get(bt)
        if axis is None:
            continue
        p = _crows_prompt(str(row["sent_more"]), str(row["sent_less"]))
        if not p:
            continue
        by_axis.setdefault(axis, [])
        if len(by_axis[axis]) < n_per_cat:
            by_axis[axis].append(p)
    for a, ps in sorted(by_axis.items()):
        print(f"    {a:<14}: {len(ps)} prompts")
    return by_axis


_NQ_ROLE = re.compile(
    r"\b(president|singer|author|writer|director|player|coach|scientist|doctor|"
    r"nurse|teacher|lawyer|judge|senator|governor|artist|engineer|pilot|soldier|"
    r"athlete|dancer|chef|farmer|painter|composer|photographer|journalist|"
    r"professor|surgeon|architect|banker|manager|driver|officer|designer|"
    r"programmer|researcher|astronaut|inventor)\b", re.IGNORECASE)


def load_nq_prompts(n_max: int = N_MAX) -> List[str]:
    ds = None
    for ds_id in ("google-research-datasets/nq_open", "nq_open"):
        try:
            ds = load_dataset(ds_id, split="validation")
            break
        except Exception:
            continue
    if ds is None:
        print("  [warn] Natural Questions could not be loaded")
        return []
    seen, roles = set(), []
    for ex in ds:
        for m in _NQ_ROLE.finditer(str(ex.get("question", ""))):
            r = m.group(0).lower()
            if r not in seen:
                seen.add(r)
                roles.append(r)
        if len(roles) >= n_max:
            break
    prompts = [f"The {r} was a" for r in roles][:n_max]
    print(f"  Natural Questions: {len(prompts)} role prompts (BASELINE, gender axis)")
    return prompts



def score(name, prompts, axis, tokenizer, model) -> dict:
    if not prompts:
        return None
    w = AXES[axis]
    p_s, p_sp, p_d, rows = compute_dnp(
        model, tokenizer, prompts, w["stereo"], w["counter"], w["neutral"])
    return {"name": name, "axis": axis, "n_prompts": len(prompts),
            "P_s": p_s, "P_sp": p_sp, "P_d": p_d, "rows": rows}


def run_multi_axis(label, prompts_by_axis, tokenizer, model, is_baseline=False):
    print(f"\n{'-' * 62}")
    print(f"  DNP  |  {label}"
          + ("   [BASELINE: not a bias benchmark]" if is_baseline else ""))
    print(f"{'-' * 62}")
    per_axis, all_rows = [], []
    for axis, prompts in sorted(prompts_by_axis.items()):
        r = score(f"{label}-{axis}", prompts, axis, tokenizer, model)
        if r is None:
            continue
        print(f"    {axis:<14} n={r['n_prompts']:>3}  "
              f"P_s={r['P_s']:.4f}  P_s'={r['P_sp']:.4f}  P_d={r['P_d']:.4f}")
        per_axis.append(r)
        all_rows.extend(r["rows"])
    if not per_axis:
        print("    [WARN] no prompts")
        return None
    d = pd.DataFrame([{k: r[k] for k in ("name", "axis", "n_prompts", "P_s", "P_sp", "P_d")}
                      for r in per_axis])
    safe = label.lower().replace(" ", "_").replace("-", "_")

    mean = {k: float(d[k].mean()) for k in ("P_s", "P_sp", "P_d")}
    print(f"    {'MEAN':<14}       P_s={mean['P_s']:.4f}  "
          f"P_s'={mean['P_sp']:.4f}  P_d={mean['P_d']:.4f}   (fair = 0.3333)")
    return {"dataset": label,
            "P_s": round(mean["P_s"], 4), "P_sp": round(mean["P_sp"], 4),
            "P_d": round(mean["P_d"], 4)}


def main():
    print("\n" + "=" * 62)
    print(f"  Demographic Normalized Probability (DNP)  |  {MODEL_NAME}")
    print("  Fair: P_s = P_s' = P_d = 0.3333")
    print("=" * 62)
    tokenizer, model = load_model()

    results = []

    bbq_by_axis = {}
    for jsonl_name, axis in BBQ_FILES.items():
        p = load_bbq_prompts(_DATA_DIR / jsonl_name)
        if p:
            bbq_by_axis[axis] = p
    print(f"  BBQ: {len(bbq_by_axis)} categories loaded")
    r = run_multi_axis("BBQ", bbq_by_axis, tokenizer, model)
    if r: results.append(r)

    crows_by_axis = load_crows_prompts_by_axis()
    r = run_multi_axis("CrowS-Pairs", crows_by_axis, tokenizer, model)
    if r: results.append(r)

    nq = load_nq_prompts()
    r = run_multi_axis("Natural Questions", {"gender": nq}, tokenizer, model,
                       is_baseline=True)
    if r: results.append(r)

    summary = pd.DataFrame(results)
    out = results_to_csv(summary, "dnp_results.csv")
    print("\n" + "=" * 62)
    print("  DNP Summary  (fair: all = 0.3333)")
    print("=" * 62)
    print(summary.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()