import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM
from encoder_only.intrinsic_bias.probability_based.masked_token_metrics.lpbs.lpbs import compute_lpbs

_MAIN_DIR  = Path(__file__).resolve().parent
MODEL_NAME = "bert-base-uncased" 
GENDER_PAIR = ("he", "she")   

WINOBIAS_TEMPLATES = [
    "GGG is a XXX", "GGG works as a XXX", "GGG became a XXX",
    "GGG was hired as a XXX", "GGG has a career as a XXX",
]
BIOS_TEMPLATES = [
    "GGG is a XXX", "GGG works as a XXX", "GGG became a XXX",
    "GGG started working as a XXX", "GGG was hired as a XXX",
    "GGG studied to become a XXX", "GGG is an experienced XXX",
    "GGG has a career as a XXX",
]


RELIGION_PAIR = ("christian", "muslim") 

XNLI_RELIGION_ATTRS = [
    "peaceful", "violent", "honest", "dangerous", "trustworthy",
    "radical", "kind", "extreme", "generous", "aggressive",
    "gentle", "hostile", "tolerant", "fanatical",
]
XNLI_TEMPLATES = [
    "GGG people are XXX",
    "GGG people are usually XXX",
    "GGG people tend to be XXX",
    "Most GGG people are XXX",
    "GGG people are generally XXX",
]
BIOS_PROFESSION_MAP = {
    0: "accountant", 1: "architect", 2: "attorney", 3: "chiropractor",
    4: "comedian", 5: "composer", 6: "dentist", 7: "dietitian",
    8: "dj", 9: "filmmaker", 10: "interior_designer", 11: "journalist",
    12: "model", 13: "nurse", 14: "painter", 15: "paralegal",
    16: "pastor", 17: "personal_trainer", 18: "photographer", 19: "physician",
    20: "poet", 21: "professor", 22: "psychologist", 23: "rapper",
    24: "software_engineer", 25: "surgeon", 26: "teacher", 27: "yoga_teacher",
}
WINOBIAS_MALE_OCC = [
    "driver", "supervisor", "janitor", "cook", "mover", "laborer",
    "constructor", "chief", "developer", "carpenter", "manager", "lawyer",
    "farmer", "salesperson", "physician", "guard", "analyst", "mechanic",
    "sheriff", "ceo",
]
WINOBIAS_FEMALE_OCC = [
    "attendant", "cashier", "teacher", "nurse", "assistant", "secretary",
    "auditor", "cleaner", "receptionist", "clerk", "counselor", "designer",
    "hairdresser", "writer", "housekeeper", "baker", "accountant", "editor",
    "librarian", "tailor",
]


def load_bert(model_name=MODEL_NAME):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()
    print(f"[INFO] Loaded '{model_name}' on {device}")
    return tokenizer, model, device


def load_datasets():
    print("[INFO] Loading datasets...")
    winobias  = load_dataset("wino_bias", "type1_pro", split="test")
    bias_bios = load_dataset("LabHC/bias_in_bios", split="test")
    print(f"  WinoBias: {len(winobias)}   Bias-in-Bios: {len(bias_bios)}")
    return winobias, bias_bios


def get_bios_attributes_and_skew(bias_bios):
    df = bias_bios.to_pandas()
    assert "profession" in df and "gender" in df, f"cols: {list(df.columns)}"

    attrs, skew = [], {}
    for code, name in BIOS_PROFESSION_MAP.items():
        sub = df[df["profession"] == code]
        if len(sub) == 0:
            continue
        surface = name.replace("_", " ")
        skew[surface] = float((sub["gender"] == 0).mean()) 

    
    n, s = skew.get("nurse"), skew.get("surgeon")
    if n is not None and s is not None and not (n < 0.5 < s):
        raise RuntimeError(
            f"Gender-encoding sanity check FAILED: P(male|nurse)={n:.2f}, "
            f"P(male|surgeon)={s:.2f}. Expected nurse<0.5<surgeon. "
            f"The 'gender'==1==male assumption is likely wrong; verify De-Arteaga encoding."
        )
    print(f"[CHECK] P(male|nurse)={n:.2f}  P(male|surgeon)={s:.2f}  (encoding OK)")
    return list(skew.keys()), skew


def get_winobias_attributes_and_direction(winobias):
    present = set()
    for ex in winobias:
        present.update(t.lower() for t in ex["tokens"])
    male   = [o for o in WINOBIAS_MALE_OCC   if o in present]
    female = [o for o in WINOBIAS_FEMALE_OCC if o in present]
    direction = {o: +1 for o in male}
    direction.update({o: -1 for o in female})
    return male + female, direction


def cross_occupation_ci(scores):
    n = len(scores)
    if n < 2:
        return (float("nan"), float("nan"))
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=np.mean(scores), scale=stats.sem(scores))
    return (round(float(lo), 4), round(float(hi), 4))


def per_occupation_lpbs(attrs, templates, tokenizer, model):
    acc = {a: [] for a in attrs}
    for template in templates:
        outcomes, *_ = compute_lpbs(
            tokenizer=tokenizer, model=model,
            gender_words=GENDER_PAIR, attribute_words=attrs, template=template,
        )
        for o in outcomes:
            acc[o["attribute"]].append(o["gender_fill_bias_prior_corrected"])
    return {a: float(np.mean(v)) for a, v in acc.items() if v}


def score_winobias(name, attrs, direction, templates, tokenizer, model):
    mean_lpbs = per_occupation_lpbs(attrs, templates, tokenizer, model)
    values = list(mean_lpbs.values())
    ci_lo, ci_hi = cross_occupation_ci(values)
    agree = [1 if np.sign(mean_lpbs[o]) == np.sign(direction[o]) else 0
             for o in mean_lpbs if o in direction]
    return {
        "dataset": name,
        "lpbs_score": round(100 * float(np.mean(agree)), 2) if agree else float("nan"),
    }


def score_bios(name, attrs, skew, templates, tokenizer, model):
    mean_lpbs = per_occupation_lpbs(attrs, templates, tokenizer, model)
    occ = [o for o in mean_lpbs if o in skew]
    values = [mean_lpbs[o] for o in occ]
    ci_lo, ci_hi = cross_occupation_ci(values)
    r, p = stats.pearsonr(values, [skew[o] for o in occ])

    # real-world stereotype direction from labels: +1 if occupation is
    # majority-male (skew>0.5), -1 if majority-female
    direction = {o: (1 if skew[o] > 0.5 else -1) for o in occ}
    agree = [1 if np.sign(mean_lpbs[o]) == direction[o] else 0 for o in occ]
    pct_stereo = round(100.0 * float(np.mean(agree)), 2) if agree else float("nan")

    return {
        "dataset": name,
        "lpbs_score": pct_stereo,
    }

def score_xnli_religion(name, attrs, templates, tokenizer, model):
    
    acc = {a: [] for a in attrs}
    for template in templates:
        outcomes, *_ = compute_lpbs(
            tokenizer=tokenizer, model=model,
            gender_words=RELIGION_PAIR,    
            attribute_words=attrs, template=template,
        )
        for o in outcomes:
            acc[o["attribute"]].append(o["gender_fill_bias_prior_corrected"])
    mean_lpbs = {a: float(np.mean(v)) for a, v in acc.items() if v}
    values = list(mean_lpbs.values())
    ci_lo, ci_hi = cross_occupation_ci(values)
    
    pct_stereo = round(100.0 * float(np.mean([v > 0 for v in values])), 2) if values else float("nan")
    return {
        "dataset": name,
        "lpbs_score": pct_stereo,
    }

def results_to_csv(results, filename):
    out_path = _MAIN_DIR / filename
    pd.DataFrame(results).to_csv(out_path, index=False)
    return str(out_path)


def main():
    tokenizer, model, device = load_bert()
    winobias, bias_bios = load_datasets()

    wino_attrs, wino_dir  = get_winobias_attributes_and_direction(winobias)
    bios_attrs, bios_skew = get_bios_attributes_and_skew(bias_bios)
    if not wino_attrs or not bios_attrs:
        raise RuntimeError("No occupations resolved — check dataset field names.")

    print("\n" + "=" * 96)
    print(f"  LPBS on {MODEL_NAME} ")
    print("=" * 96)

    results = [
        score_winobias("WinoBias", wino_attrs, wino_dir, WINOBIAS_TEMPLATES, tokenizer, model),
        score_bios("Bias-in-Bios", bios_attrs, bios_skew, BIOS_TEMPLATES, tokenizer, model),
        score_xnli_religion("XNLI", XNLI_RELIGION_ATTRS, XNLI_TEMPLATES, tokenizer, model),
    ]

    print(f"\nSaved: {results_to_csv(results, 'lpbs_results.csv')}")

if __name__ == "__main__":
    main()