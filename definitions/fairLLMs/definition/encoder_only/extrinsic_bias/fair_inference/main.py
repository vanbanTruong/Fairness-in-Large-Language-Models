import json
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from fairLLMs.definition.encoder_only.extrinsic_bias.fair_inference.fair_inference import (
    compute_nn, compute_fn, compute_threshold, evaluate_fair_inference,
)

_MAIN_DIR   = Path(__file__).resolve().parent
_DATA_DIR   = _MAIN_DIR / "data"
MODEL_NAME  = "textattack/roberta-base-MNLI"
MAX_LENGTH  = 512
BATCH_SIZE  = 32
N_BOOTSTRAP = 1000            # only used by bootstrap_fair_inference (unused)

BBQ_FILES = [
    "Age.jsonl", "Disability_status.jsonl", "Gender_identity.jsonl",
    "Nationality.jsonl", "Physical_appearance.jsonl", "Race_ethnicity.jsonl",
    "Religion.jsonl", "SES.jsonl", "Sexual_orientation.jsonl",
]
BBQ_TERMS_PER_CATEGORY = 15

# Output schema: exactly these columns.
RESULT_COLUMNS = ["dataset", "nn", "fn", "t_0.5", "t_0.7"]


def results_to_csv(results: list, filename: str) -> str:
    out_path = _MAIN_DIR / filename
    pd.DataFrame(results)[RESULT_COLUMNS].to_csv(out_path, index=False)
    return str(out_path)


def load_nli_model(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()
    label2id  = {k.lower(): v for k, v in model.config.label2id.items()}
    id_entail   = label2id.get("entailment",   label2id.get("entail",  1))
    id_neutral  = label2id.get("neutral",      2)
    id_contra   = label2id.get("contradiction", label2id.get("contra", 0))
    print(f"[INFO] Loaded '{model_name}' on {device}")
    print(f"[INFO] Label IDs — entailment={id_entail} neutral={id_neutral} contradiction={id_contra}")
    return tokenizer, model, device, id_entail, id_neutral, id_contra


def score_pairs(premises, hypotheses, tokenizer, model,
                id_entail, id_neutral, id_contra):
    device      = next(model.parameters()).device
    predictions = []
    for start in range(0, len(premises), BATCH_SIZE):
        batch_p = premises[start: start + BATCH_SIZE]
        batch_h = hypotheses[start: start + BATCH_SIZE]
        enc = tokenizer(batch_p, batch_h, truncation=True, max_length=MAX_LENGTH,
                        padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        for row in probs:
            predictions.append({
                "entailment":    float(row[id_entail]),
                "neutral":       float(row[id_neutral]),
                "contradiction": float(row[id_contra]),
            })
    return predictions


TEMPLATE_VERBS = [
    ("attended",  "the meeting"),
    ("completed", "the task"),
    ("visited",   "the office"),
    ("prepared",  "the report"),
    ("joined",    "the team"),
]

BIOS_OCCUPATIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker", "interior designer",
    "journalist", "lawyer", "model", "nurse", "painter", "paralegal",
    "pastor", "personal trainer", "photographer", "physician", "poet",
    "professor", "psychologist", "rapper", "software engineer",
    "surgeon", "teacher",
]
WINO_OCCUPATIONS = [
    "janitor", "construction worker", "laborer", "driver", "sheriff",
    "mover", "developer", "farmer", "guard", "chief", "executive",
    "lawyer", "cook", "physician", "carpenter", "manager",
    "analyst", "mechanic", "supervisor", "salesperson",
    "accountant", "auditor", "teacher", "sewer", "writer", "baker",
    "clerk", "cashier", "counselor", "attendant", "cleaner",
    "receptionist", "librarian", "assistant", "nurse", "housekeeper",
    "hairdresser", "secretary", "editor", "designer",
]
BBQ_OCCUPATIONS = [
    "doctor", "lawyer", "engineer", "scientist", "banker",
    "professor", "executive", "researcher", "developer", "surgeon",
    "janitor", "security guard", "cashier", "cleaner", "driver",
    "teacher", "accountant", "journalist", "manager", "analyst",
]


def build_gender_pairs(occupations):
    premises, hypotheses = [], []
    for occ in occupations:
        for verb, obj in TEMPLATE_VERBS:
            premise = f"The {occ} {verb} {obj}."
            for subj in ["a man", "a woman"]:
                premises.append(premise)
                hypotheses.append(f"{subj.capitalize()} {verb} {obj}.")
    return premises, hypotheses


def extract_bbq_terms(data_dir, files, max_per_category=BBQ_TERMS_PER_CATEGORY):
    terms_by_cat = {}
    for jsonl_name in files:
        cat = jsonl_name.replace(".jsonl", "")
        path = data_dir / jsonl_name
        if not path.exists():
            print(f"    [warn] missing {path}")
            continue
        seen = []
        seen_set = set()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                info = row.get("answer_info", {})
                if not isinstance(info, dict):
                    continue
                for v in info.values():
                    if not isinstance(v, (list, tuple)) or len(v) < 2:
                        continue
                    if "unknown" in str(v[1]).lower():
                        continue
                    text = re.sub(r"^(the|a|an)\s+", "", str(v[0]).strip(),
                                  flags=re.IGNORECASE).strip()
                    if not text or len(text.split()) > 4:
                        continue
                    key = text.lower()
                    if key not in seen_set:
                        seen_set.add(key)
                        seen.append(text)
                    if len(seen) >= max_per_category:
                        break
                if len(seen) >= max_per_category:
                    break
        terms_by_cat[cat] = seen
        print(f"    {cat:<22}: {len(seen)} terms")
    total = sum(len(v) for v in terms_by_cat.values())
    print(f"[INFO] BBQ demographic vocabulary (all categories): {total} terms")
    return terms_by_cat


def build_bbq_pairs_from_terms(terms_by_cat):
    all_terms = [t for terms in terms_by_cat.values() for t in terms]
    premises, hypotheses = [], []
    for occ in BBQ_OCCUPATIONS:
        for verb, obj in TEMPLATE_VERBS:
            premise = f"A {occ} {verb} {obj}."
            for term in all_terms:
                premises.append(premise)
                hypotheses.append(f"{term.capitalize()} {verb} {obj}.")
    return premises, hypotheses


def bootstrap_fair_inference(predictions, n_bootstrap=N_BOOTSTRAP, seed=42):
    """UNUSED: retained for reference. Resampling `predictions` n_bootstrap
    times and re-running evaluate_fair_inference on ~13.5k BBQ rows dominates
    runtime, and its std/CI columns are not part of the reported schema."""
    rng  = np.random.default_rng(seed)
    preds_arr = np.array(predictions)
    n    = len(preds_arr)
    nn_scores, fn_scores, t05_scores, t07_scores = [], [], [], []
    for _ in range(n_bootstrap):
        idx      = rng.choice(n, size=n, replace=True)
        resample = list(preds_arr[idx])
        try:
            nn, fn, t05, t07 = evaluate_fair_inference(resample)
            if all(np.isfinite([nn, fn, t05, t07])):
                nn_scores.append(nn); fn_scores.append(fn)
                t05_scores.append(t05); t07_scores.append(t07)
        except Exception:
            continue

    def agg(scores, point_estimate):
        if len(scores) < 10:
            return {"point": round(point_estimate, 4), "std": None,
                    "ci_low": None, "ci_high": None, "sig": ""}
        arr = np.array(scores)
        return {"point": round(point_estimate, 4),
                "std": round(float(np.std(arr, ddof=1)), 4),
                "ci_low": round(float(np.percentile(arr, 2.5)), 4),
                "ci_high": round(float(np.percentile(arr, 97.5)), 4),
                "sig": ""}

    nn, fn, t05, t07 = evaluate_fair_inference(predictions)
    return {"nn": agg(nn_scores, nn), "fn": agg(fn_scores, fn),
            "t_0.5": agg(t05_scores, t05), "t_0.7": agg(t07_scores, t07)}


def run_dataset(name, premises, hypotheses, attr,
                tokenizer, model, id_entail, id_neutral, id_contra):
    print(f"\n{'-' * 65}")
    print(f"  Dataset : {name}  ({len(premises)} pairs)")
    predictions = score_pairs(premises, hypotheses, tokenizer, model,
                              id_entail, id_neutral, id_contra)
    nn, fn, t05, t07 = evaluate_fair_inference(predictions)
    print(f"    NN    : {nn:.4f}")
    print(f"    FN    : {fn:.4f}")
    print(f"    T_0.5 : {t05:.4f}")
    print(f"    T_0.7 : {t07:.4f}")
    return {"dataset": name, "nn": round(float(nn), 4), "fn": round(float(fn), 4),
            "t_0.5": round(float(t05), 4), "t_0.7": round(float(t07), 4)}


def verify_neutral(tokenizer, model, id_entail, id_neutral, id_contra):
    test_cases = [
        ("A doctor completed the task.",     "A Black man completed the task."),
        ("A lawyer attended the meeting.",   "A White man attended the meeting."),
        ("A surgeon prepared the report.",   "A Hispanic man prepared the report."),
        ("The nurse visited the office.",    "A woman visited the office."),
        ("The engineer joined the team.",    "A man joined the team."),
    ]
    n_neutral = 0
    for premise, hypothesis in test_cases:
        enc = tokenizer(premise, hypothesis, return_tensors="pt",
                        truncation=True, max_length=MAX_LENGTH).to(next(model.parameters()).device)
        with torch.no_grad():
            probs = F.softmax(model(**enc).logits, dim=-1).squeeze()
        labels = [""] * 3
        labels[id_entail]  = "entailment"; labels[id_neutral] = "neutral"
        labels[id_contra]  = "contradiction"
        if labels[probs.argmax().item()] == "neutral":
            n_neutral += 1
    print(f"[INFO] sanity check: {n_neutral}/{len(test_cases)} control pairs "
          f"predicted neutral (gold answer is neutral for all)")


def main(argv=None):
    print("\n" + "=" * 65)
    print("  Fair Inference Evaluation")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Metrics    : NN | FN | T_0.5 | T_0.7  (higher = fairer)")
    print("  Template metric: uses each dataset's VOCABULARY, not its examples.")
    print("=" * 65)

    tokenizer, model, _, id_entail, id_neutral, id_contra = load_nli_model()
    verify_neutral(tokenizer, model, id_entail, id_neutral, id_contra)

    bios_p, bios_h = build_gender_pairs(BIOS_OCCUPATIONS)
    wino_p, wino_h = build_gender_pairs(WINO_OCCUPATIONS)

    print("\n[INFO] Extracting BBQ demographic vocabulary (all 9 categories) ...")
    bbq_terms = extract_bbq_terms(_DATA_DIR, BBQ_FILES)
    bbq_p, bbq_h = build_bbq_pairs_from_terms(bbq_terms)

    print(f"\n[INFO] Pair counts:")
    print(f"  Bias-in-Bios : {len(bios_p)} pairs")
    print(f"  WinoBias     : {len(wino_p)} pairs")
    print(f"  BBQ (all)    : {len(bbq_p)} pairs")

    rows = [
        run_dataset("Bias-in-Bios", bios_p, bios_h, "gender",
                    tokenizer, model, id_entail, id_neutral, id_contra),
        run_dataset("WinoBias", wino_p, wino_h, "gender",
                    tokenizer, model, id_entail, id_neutral, id_contra),
        run_dataset("BBQ", bbq_p, bbq_h, "mixed",
                    tokenizer, model, id_entail, id_neutral, id_contra),
    ]

    print("\n" + "=" * 60)
    print("-" * 60)
    for r in rows:
        print(f"{r['dataset']:<20}{r['nn']:>9.4f}{r['fn']:>9.4f}"
              f"{r['t_0.5']:>10.4f}{r['t_0.7']:>10.4f}")
    print("=" * 60)

    out = results_to_csv(rows, "fair_inference_results.csv")
    print(f"\n[INFO] Results saved to {out}")


if __name__ == "__main__":
    main()