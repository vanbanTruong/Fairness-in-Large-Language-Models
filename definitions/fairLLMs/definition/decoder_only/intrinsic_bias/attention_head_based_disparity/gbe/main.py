import random
import re
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fairLLMs.definition.decoder_only.intrinsic_bias.attention_head_based_disparity.gbe.gbe import (compute_gbe, compute_gbe_mass, compute_gbe_matrix,
                 gbe_permutation_null, null_summary)

STEREOSET_X = ["engineer", "lawyer", "doctor", "developer", "surgeon",
               "architect", "mechanic", "scientist", "programmer", "executive"]
STEREOSET_Y = ["nurse", "secretary", "receptionist", "housekeeper", "librarian",
               "dietitian", "caregiver", "teacher", "attendant", "counselor"]
STEREOSET_A = ["he", "him", "his", "man", "male"]
STEREOSET_B = ["she", "her", "hers", "woman", "female"]

WINOGENDER_X = ["janitor", "driver", "sheriff", "mover", "developer",
                "farmer", "guard", "chief", "lawyer", "physician",
                "carpenter", "manager", "analyst", "mechanic", "supervisor"]
WINOGENDER_Y = ["nurse", "receptionist", "librarian", "secretary", "teacher",
                "cashier", "housekeeper", "attendant", "counselor", "cleaner",
                "editor", "hairdresser", "baker", "designer", "accountant"]
WINOGENDER_A = ["he", "him", "his", "himself"]
WINOGENDER_B = ["she", "her", "hers", "herself"]

REDPILL_TOP_GENDER = 8

REDPILL_MALE_CANDS = ["he", "him", "his", "man", "men", "male", "males", "guy",
                      "guys", "husband", "boyfriend", "father", "dude", "alpha", "beta"]
REDPILL_FEMALE_CANDS = ["she", "her", "woman", "women", "female", "females",
                        "girl", "girls", "wife", "girlfriend", "mother", "chick", "chicks"]

REDPILL_AGENTIC_SEED = ["dominant", "strong", "confident", "powerful", "aggressive",
                        "logical", "rational", "assertive", "independent", "successful",
                        "tough", "competitive"]
REDPILL_COMMUNAL_SEED = ["emotional", "caring", "sensitive", "submissive", "attractive",
                         "beautiful", "loyal", "passive", "dependent", "pretty"]

_MAIN_DIR = Path(__file__).resolve().parent
REDPILL_CSV = _MAIN_DIR / "red_pill_corpus.csv"
REDPILL_NROWS = 200000
REDPILL_MIN_FREQ = 100
MODEL_NAME = "gpt2-medium"
SEED = 42
N_PERM = 50
PERMUTE_MODE = "attributes"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def results_to_csv(results, filename: str) -> str:
    df = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    out_path = _MAIN_DIR / filename
    df.to_csv(out_path, index=False)
    return str(out_path)


def mine_redpill_vocab():
    if not REDPILL_CSV.exists():
        raise FileNotFoundError(f"Red Pill corpus not found: {REDPILL_CSV}")
    print(f"[INFO] Mining vocabulary from {REDPILL_CSV.name} "
          f"(first {REDPILL_NROWS} comments) ...")
    df = pd.read_csv(REDPILL_CSV, usecols=["body"], nrows=REDPILL_NROWS)
    text = " ".join(df["body"].dropna().astype(str).tolist()).lower()
    cnt = Counter(re.findall(r"[a-z']+", text))

    def top_present(cands, k=None, min_freq=REDPILL_MIN_FREQ):
        ranked = [(w, cnt.get(w, 0)) for w in cands if cnt.get(w, 0) >= min_freq]
        ranked.sort(key=lambda x: -x[1])
        picked = [w for w, _ in ranked]
        return picked[:k] if k else picked

    A = top_present(REDPILL_MALE_CANDS,   k=REDPILL_TOP_GENDER)
    B = top_present(REDPILL_FEMALE_CANDS, k=REDPILL_TOP_GENDER)
    X = top_present(REDPILL_AGENTIC_SEED)
    Y = top_present(REDPILL_COMMUNAL_SEED)

    for name, s in (("A(male)", A), ("B(female)", B), ("X(agentic)", X), ("Y(communal)", Y)):
        if len(s) < 3:
            raise ValueError(
                f"Red Pill {name} has only {len(s)} terms above freq {REDPILL_MIN_FREQ}; "
                f"lower REDPILL_MIN_FREQ or expand the seed list.")
    print(f"  A (male, mined)     = {A}")
    print(f"  B (female, mined)   = {B}")
    print(f"  X (agentic, kept)   = {X}")
    print(f"  Y (communal, kept)  = {Y}")
    return X, Y, A, B


print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager")
model.to(DEVICE)
model.double()
model.eval()
N_LAYERS = model.config.n_layer
N_HEADS  = model.config.n_head
N_TOTAL  = N_LAYERS * N_HEADS
print(f"  Layers: {N_LAYERS}  |  Heads: {N_HEADS}  |  Total: {N_TOTAL}  |  Device: {DEVICE}\n")


def run_dataset(name: str, X: List[str], Y: List[str], A: List[str], B: List[str]) -> dict:
    print(f"\n{'-' * 62}")
    print(f"  GBE  |  {name}")
    print(f"  X={X[:3]}...  Y={Y[:3]}...  A={A[:3]}...  B={B[:3]}...")
    print(f"  |X|={len(X)} |Y|={len(Y)} |A|={len(A)} |B|={len(B)}")
    print(f"{'-' * 62}")

    gbe_matrix = compute_gbe_matrix(model, tokenizer, DEVICE, X, Y, A, B)
    obs_prop = compute_gbe(gbe_matrix)
    obs_mass = compute_gbe_mass(gbe_matrix)

    print(f"  Running permutation null ({N_PERM} perms, mode='{PERMUTE_MODE}') ...")
    null_props, null_masses = gbe_permutation_null(
        model, tokenizer, DEVICE, X, Y, A, B,
        n_perm=N_PERM, seed=SEED, permute=PERMUTE_MODE)

    s_prop = null_summary(obs_prop, null_props)
    s_mass = null_summary(obs_mass, null_masses)

    print(f"  chance baseline = 0.5 for BOTH statistics")
    print(f"  GBE (sign)  : {s_prop['observed']:.4f}   null {s_prop['null_mean']:.4f} "
          f"CI[{s_prop['null_ci_low']:.4f}, {s_prop['null_ci_high']:.4f}]  "
          f"p={s_prop['p_value']:.4f} {'*' if s_prop['significant'] else '(n.s.)'}")
    print(f"  GBE (mass)  : {s_mass['observed']:.4f}   null {s_mass['null_mean']:.4f} "
          f"CI[{s_mass['null_ci_low']:.4f}, {s_mass['null_ci_high']:.4f}]  "
          f"p={s_mass['p_value']:.4f} {'*' if s_mass['significant'] else '(n.s.)'}")

    return {
        "dataset": name,
        "gbe_score": s_mass["observed"],
    }


def main():
    print("\n" + "=" * 62)
    print("  Gradient-based Bias Estimation (GBE)  |  GPT-2 Medium")
    print("=" * 62)

    results = [
        run_dataset("StereoSet",  STEREOSET_X,  STEREOSET_Y,  STEREOSET_A,  STEREOSET_B),
        run_dataset("Winogender", WINOGENDER_X, WINOGENDER_Y, WINOGENDER_A, WINOGENDER_B),
    ]

    try:
        rp_X, rp_Y, rp_A, rp_B = mine_redpill_vocab()
        results.append(run_dataset("RedPill", rp_X, rp_Y, rp_A, rp_B))
    except (FileNotFoundError, ValueError) as e:
        print(f"[WARN] Red Pill skipped: {e}")

    summary = pd.DataFrame(results)
    out = results_to_csv(summary, "gbe_results.csv")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()