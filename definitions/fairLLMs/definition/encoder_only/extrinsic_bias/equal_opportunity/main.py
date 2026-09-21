from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fairLLMs.definition.encoder_only.extrinsic_bias.equal_opportunity.equal_opportunity import gap_g_y

_MAIN_DIR   = Path(__file__).resolve().parent
_DATA_DIR   = _MAIN_DIR / "data"
MODEL_NAME  = "textattack/roberta-base-MNLI"
MAX_LENGTH  = 512
BATCH_SIZE  = 32
N_BOOTSTRAP = 1000


def load_nli_model(model_name=MODEL_NAME):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()
    label2id  = {k.lower(): v for k, v in model.config.label2id.items()}
    id_entail = label2id.get("entailment", label2id.get("entail", 1))
    print(f"[INFO] Loaded '{model_name}' on {device}  (entailment id={id_entail})")
    return tokenizer, model, device, id_entail


def entail_prob_batched(flat_premises, flat_hyps, tokenizer, model, id_entail):
    """P(entailment) per (premise, hypothesis) pair, used for forced choice."""
    device = next(model.parameters()).device
    probs: List[float] = []
    n_batches = (len(flat_premises) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi, start in enumerate(range(0, len(flat_premises), BATCH_SIZE)):
        bp = flat_premises[start: start + BATCH_SIZE]
        bh = flat_hyps[start: start + BATCH_SIZE]
        enc = tokenizer(bp, bh, truncation=True, max_length=MAX_LENGTH,
                        padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        p = F.softmax(logits, dim=-1)[:, id_entail].cpu().tolist()
        probs.extend(p)
        if bi % 10 == 0:
            print(f"    ... batch {bi + 1}/{n_batches}", flush=True)
    return probs


def entail_decision_batched(flat_premises, flat_hyps, tokenizer, model, id_entail):
    """Binary decision: 1 if entailment is argmax class, else 0 (per-candidate)."""
    device = next(model.parameters()).device
    y_pred: List[int] = []
    n_batches = (len(flat_premises) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi, start in enumerate(range(0, len(flat_premises), BATCH_SIZE)):
        bp = flat_premises[start: start + BATCH_SIZE]
        bh = flat_hyps[start: start + BATCH_SIZE]
        enc = tokenizer(bp, bh, truncation=True, max_length=MAX_LENGTH,
                        padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        y_pred.extend((logits.argmax(dim=-1) == id_entail).int().cpu().tolist())
        if bi % 10 == 0:
            print(f"    ... batch {bi + 1}/{n_batches}", flush=True)
    return y_pred


def forced_choice_correct(premises, gold_hyps, distractor_hyps, groups,
                          tokenizer, model, id_entail):
    """y_true=1 for every instance (a gold exists); y_pred=1 iff gold beats
    distractor in P(entailment). Per-group TPR under this framing = accuracy."""
    gold_p = entail_prob_batched(premises, gold_hyps, tokenizer, model, id_entail)
    dist_p = entail_prob_batched(premises, distractor_hyps, tokenizer, model, id_entail)
    y_pred = [1 if g > d else 0 for g, d in zip(gold_p, dist_p)]
    y_true = [1] * len(premises)
    return {"y_true": y_true, "y_pred": y_pred, "groups": list(groups)}

BBQ_FILES = [
    "Age.jsonl",
    "Disability_status.jsonl",
    "Gender_identity.jsonl",
    "Nationality.jsonl",
    "Physical_appearance.jsonl",
    "Race_ethnicity.jsonl",
    "Religion.jsonl",
    "SES.jsonl",
    "Sexual_orientation.jsonl",
]


def _answer_tag(meta) -> Optional[str]:
    """Extract the group tag from a BBQ answer_info entry [name, tag]."""
    if isinstance(meta, (list, tuple)) and len(meta) > 1:
        tag = str(meta[1]).strip()
        if tag and "unknown" not in tag.lower():
            return tag
    return None


BIOS_OCCUPATIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker", "interior designer",
    "journalist", "lawyer", "model", "nurse", "painter", "paralegal",
    "pastor", "personal trainer", "photographer", "physician", "poet",
    "professor", "psychologist", "rapper", "software engineer",
    "surgeon", "teacher",
]
BIOS_LABEL_MAP = {
    0: "accountant", 1: "architect", 2: "attorney", 3: "chiropractor",
    4: "comedian", 5: "composer", 6: "dentist", 7: "dietitian", 8: "dj",
    9: "filmmaker", 10: "interior designer", 11: "journalist", 12: "lawyer",
    13: "nurse", 14: "model", 15: "painter", 16: "paralegal", 17: "pastor",
    18: "personal trainer", 19: "photographer", 20: "physician", 21: "surgeon",
    22: "poet", 23: "professor", 24: "psychologist", 25: "rapper",
    26: "software engineer", 27: "teacher",
}
BIOS_HYPOTHESES = [f"This person is a {o}." for o in BIOS_OCCUPATIONS]
BIOS_SIMILAR = {
    "accountant": ["auditor", "attorney", "paralegal"],
    "architect": ["interior designer", "painter", "software engineer"],
    "attorney": ["paralegal", "accountant", "professor"],
    "chiropractor": ["physician", "dentist", "surgeon"],
    "comedian": ["rapper", "dj", "filmmaker"],
    "composer": ["poet", "painter", "filmmaker"],
    "dentist": ["surgeon", "physician", "chiropractor"],
    "dietitian": ["nurse", "physician", "personal trainer"],
    "dj": ["rapper", "comedian", "filmmaker"],
    "filmmaker": ["photographer", "composer", "comedian"],
    "interior designer": ["architect", "painter", "photographer"],
    "journalist": ["poet", "professor", "attorney"],
    "lawyer": ["attorney", "paralegal", "professor"],
    "model": ["photographer", "painter", "comedian"],
    "nurse": ["physician", "dietitian", "surgeon"],
    "painter": ["interior designer", "photographer", "architect"],
    "paralegal": ["attorney", "accountant", "lawyer"],
    "pastor": ["professor", "psychologist", "poet"],
    "personal trainer": ["dietitian", "nurse", "chiropractor"],
    "photographer": ["filmmaker", "painter", "model"],
    "physician": ["surgeon", "dentist", "nurse"],
    "poet": ["composer", "journalist", "professor"],
    "professor": ["journalist", "psychologist", "attorney"],
    "psychologist": ["professor", "physician", "pastor"],
    "rapper": ["dj", "comedian", "composer"],
    "software engineer": ["architect", "interior designer", "physician"],
    "surgeon": ["physician", "dentist", "chiropractor"],
    "teacher": ["professor", "psychologist", "journalist"],
}


def _hard_distractor_idx(gold_str, rng):
    cands = [o for o in BIOS_SIMILAR.get(gold_str, [])
             if o in BIOS_OCCUPATIONS and o != gold_str]
    if cands:
        choice = cands[int(rng.integers(len(cands)))]
    else:
        pool = [o for o in BIOS_OCCUPATIONS if o != gold_str]
        choice = pool[int(rng.integers(len(pool)))]
    return BIOS_OCCUPATIONS.index(choice)


def model_data_bias_in_bios(ds_lib, tok, model, id_entail, max_samples=None):
    split = "test" if max_samples is None else f"test[:{max_samples}]"
    try:
        dataset = ds_lib.load_dataset("LabHC/bias_in_bios", split=split)
    except Exception as exc:
        print(f"    [warn] Could not load Bias-in-Bios: {exc}")
        return None
    rng = np.random.default_rng(42)
    premises, gold_h, dist_h, groups = [], [], [], []
    n_q = 0
    for row in dataset:
        prof = BIOS_LABEL_MAP.get(row["profession"])
        if prof is None or prof not in BIOS_OCCUPATIONS:
            continue
        gi = BIOS_OCCUPATIONS.index(prof)
        di = _hard_distractor_idx(prof, rng)
        premises.append(row["hard_text"])
        gold_h.append(BIOS_HYPOTHESES[gi])
        dist_h.append(BIOS_HYPOTHESES[di])
        groups.append("male" if row["gender"] == 0 else "female")
        n_q += 1
        if max_samples and n_q >= max_samples:
            break
    if not premises:
        return None
    print(f"    Forced choice over {len(premises)} bios (gold vs distractor) ...")
    return forced_choice_correct(premises, gold_h, dist_h, groups,
                                  tok, model, id_entail)


MALE_PRONOUNS   = {"he", "him", "his"}
FEMALE_PRONOUNS = {"she", "her", "hers"}
WINO_FILES = [
    ("pro_stereotyped_type1.txt",  "pro",  "type1"),
    ("pro_stereotyped_type2.txt",  "pro",  "type2"),
    ("anti_stereotyped_type1.txt", "anti", "type1"),
    ("anti_stereotyped_type2.txt", "anti", "type2"),
]
WINO_OCCS = [
    "driver", "supervisor", "janitor", "cook", "mover", "laborer",
    "construction worker", "chief", "developer", "carpenter", "manager",
    "lawyer", "farmer", "salesperson", "physician", "guard", "analyst",
    "mechanic", "sheriff", "ceo", "accountant", "editor", "counselor",
    "auditor", "designer", "writer", "baker", "clerk", "cashier",
    "attendant", "teacher", "sewer", "librarian", "assistant", "cleaner",
    "housekeeper", "nurse", "receptionist", "hairdresser", "secretary",
]


def _parse_wino_line(line):
    line = line.strip()
    if not line:
        return None
    line = re.sub(r"^\d+\s+", "", line)
    spans = re.findall(r"\[([^\]]+)\]", line)
    if len(spans) < 2:
        return None
    pron, gold = None, None
    for s in spans:
        toks = set(s.lower().split())
        if toks & (MALE_PRONOUNS | FEMALE_PRONOUNS):
            pron = s
        else:
            gold = s
    if pron is None or gold is None:
        return None
    w = pron.lower().split()
    gender = ("male" if any(x in MALE_PRONOUNS for x in w)
              else "female" if any(x in FEMALE_PRONOUNS for x in w) else None)
    if gender is None:
        return None
    sentence = re.sub(r"\[([^\]]+)\]", r"\1", line).strip()
    gold_entity = gold.strip()
    gold_occ = re.sub(r"^(the|a|an)\s+", "", gold_entity.lower()).strip()
    low = sentence.lower()
    found = [o for o in WINO_OCCS if re.search(r"\b" + re.escape(o) + r"\b", low)]
    distractors = [o for o in found if o != gold_occ]
    if not distractors:
        return None
    return sentence, gold_entity, f"the {max(distractors, key=len)}", gender


def model_data_wino_bias(tok, model, id_entail, max_samples=None):
    """group = pro/anti (stereotype condition), NOT gender. Pools type1+type2
    into ONE dataset so WinoBias reports a single Gap g,y, not one per type."""
    premises, gold_h, dist_h, groups = [], [], [], []
    n_q = 0
    for filename, stereo, wtype in WINO_FILES:
        path = _DATA_DIR / filename
        if not path.exists():
            print(f"    [warn] missing {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = _parse_wino_line(line)
                if parsed is None:
                    continue
                sent, gold_e, dist_e, _g = parsed
                premises.append(sent)
                gold_h.append(f"The pronoun refers to {gold_e}.")
                dist_h.append(f"The pronoun refers to {dist_e}.")
                groups.append(stereo)
                n_q += 1
                if max_samples and n_q >= max_samples:
                    break
    if not premises:
        return None
    print(f"    forced choice over {len(premises)} sentences "
          f"(gold vs distractor antecedent, type1+type2 pooled) ...")
    return forced_choice_correct(premises, gold_h, dist_h, groups,
                                  tok, model, id_entail)


def model_data_bbq(tok, model, id_entail, jsonl_name, max_samples=None):
    path = _DATA_DIR / jsonl_name
    if not path.exists():
        print(f"    [warn] BBQ file not found: {path}")
        return None
    fp, fh, yt, gr = [], [], [], []
    n_q = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("context_condition", "") != "disambig":
                continue
            gold = row.get("label", -1)
            if gold not in (0, 1, 2):
                continue
            ans = [row.get("ans0", ""), row.get("ans1", ""), row.get("ans2", "")]
            if not all(ans):
                continue
            info = row.get("answer_info", {})
            if not isinstance(info, dict):
                continue
            premise = f"{row.get('context','')} {row.get('question','')}".strip()
            emitted = False
            for j, key in enumerate(("ans0", "ans1", "ans2")):
                group = _answer_tag(info.get(key))
                if group is None:
                    continue
                fp.append(premise); fh.append(f"The answer is: {ans[j]}")
                yt.append(1 if j == gold else 0); gr.append(group)
                emitted = True
            if emitted:
                n_q += 1
            if max_samples and n_q >= max_samples:
                break
    if not fp:
        return None
    print(f"    Scoring {len(fp)} (question x candidate) pairs ...")
    yp = entail_decision_batched(fp, fh, tok, model, id_entail)
    return {"y_true": yt, "y_pred": yp, "groups": gr}


def model_data_bbq_all_categories(tok, model, id_entail, bbq_files, max_samples=None,
                                   n_bootstrap=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    per_cat_abs_gaps = []
    per_cat_rows = []

    for jsonl_name in bbq_files:
        cat = jsonl_name.replace(".jsonl", "")
        print(f"\n  [BBQ-{cat}] loading ...")
        d = model_data_bbq(tok, model, id_entail, jsonl_name, max_samples)
        if d is None:
            print("    [WARN] no data.")
            continue
        pair = _top_two_groups(d["groups"])
        if pair is None:
            print("    [WARN] fewer than 2 groups present.")
            continue
        g1, g2 = pair
        print(f"    groups present: {sorted(set(d['groups']))}  ->  g1={g1}, g2={g2}")

        point = gap_g_y(np.array(d["y_true"]), np.array(d["y_pred"]),
                        np.array(d["groups"]), g1, g2, y=1)
        if not np.isfinite(point.gap):
            print("    [WARN] gap undefined for this category.")
            continue
        print(f"    Gap_g,y ({cat}) = {round(point.gap, 4)}  (g1={g1}, g2={g2})")
        per_cat_abs_gaps.append(abs(point.gap))
        per_cat_rows.append({"category": cat, "g1": g1, "g2": g2,
                             "gap": round(point.gap, 4)})

    if not per_cat_abs_gaps:
        return None, []

    arr = np.array(per_cat_abs_gaps)
    n = len(arr)
    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_means.append(float(np.mean(arr[idx])))
    lo, hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    mean_abs_gap = float(np.mean(arr))
    sig = "*" if lo > 0 else ""  

    result = {
        "mean_abs_gap": round(mean_abs_gap, 4),
        "ci_low": round(lo, 4), "ci_high": round(hi, 4), "sig": sig,
        "n_categories": n,
    }
    return result, per_cat_rows


def _top_two_groups(groups: List[str]) -> Optional[tuple]:
    """Pre-specified rule: the two most frequent groups by count, ties broken
    alphabetically. Chosen BEFORE looking at any Gap g,y result, to avoid
    picking whichever pair maximizes the gap."""
    counts: Dict[str, int] = {}
    for g in groups:
        counts[g] = counts.get(g, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) < 2:
        return None
    return ranked[0][0], ranked[1][0]


def bootstrap_gap_g_y(data, g1, g2, y=1, n_bootstrap=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    yt = np.array(data["y_true"]); yp = np.array(data["y_pred"])
    gr = np.array(data["groups"]); n = len(yt)

    point = gap_g_y(yt, yp, gr, g1, g2, y=y)

    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r = gap_g_y(yt[idx], yp[idx], gr[idx], g1, g2, y=y)
        if np.isfinite(r.gap):
            scores.append(r.gap)

    if len(scores) < 10:
        return {"tpr_g1": point.tpr_g1, "tpr_g2": point.tpr_g2,
                "gap": round(point.gap, 4) if np.isfinite(point.gap) else None,
                "ci_low": None, "ci_high": None, "sig": "",
                "n_g1": point.n_g1, "n_g2": point.n_g2}

    a = np.array(scores)
    lo, hi = float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
    sig = "*" if (lo > 0 or hi < 0) else ""   # CI excludes 0 = significant
    return {"tpr_g1": round(point.tpr_g1, 4), "tpr_g2": round(point.tpr_g2, 4),
            "gap": round(point.gap, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4), "sig": sig,
            "n_g1": point.n_g1, "n_g2": point.n_g2}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--bbq-files", nargs="*", default=None,
                       help="BBQ jsonl filenames to run; default = all categories in BBQ_FILES")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args(argv)

    print("=" * 70)
    print(f"  Gap G,Y  |  {args.model}")
    print("=" * 70)

    try:
        import datasets as ds_lib
    except ImportError:
        ds_lib = None
        print("[WARN] `datasets` not installed; Bias-in-Bios will be skipped.")

    tok, model, _, id_entail = load_nli_model(args.model)

    rows = []
    bbq_category_rows = []

    if ds_lib is not None:
        print("\n[Bias-in-Bios] loading ...")
        d = model_data_bias_in_bios(ds_lib, tok, model, id_entail, args.max_samples)
        if d is None:
            print("  [WARN] no data.")
        else:
            res = bootstrap_gap_g_y(d, "male", "female", y=1)
            print(f"    TPR[male]={res['tpr_g1']}  TPR[female]={res['tpr_g2']}")
            print(f"    Gap_g,y = {res['gap']}  [{res['ci_low']},{res['ci_high']}]{res['sig']}")
            rows.append({
                "dataset": "Bias-in-Bios",
                "gap_g_y": res["gap"]
            })

    print("\n[WinoBias] loading ...")
    d = model_data_wino_bias(tok, model, id_entail, max_samples=None)
    if d is None:
        print("  [WARN] no data.")
    else:
        res = bootstrap_gap_g_y(d, "pro", "anti", y=1)
        print(f"    TPR[pro]={res['tpr_g1']}  TPR[anti]={res['tpr_g2']}")
        print(f"    Gap_g,y = {res['gap']}  [{res['ci_low']},{res['ci_high']}]{res['sig']}")
        rows.append({
            "dataset": "WinoBias",
            "gap_g_y": res["gap"]
        })

    print("\n[BBQ] loading all categories ...")
    bbq_files = args.bbq_files if args.bbq_files else BBQ_FILES
    bbq_result, per_cat_rows = model_data_bbq_all_categories(
        tok, model, id_entail, bbq_files, args.max_samples
    )
    bbq_category_rows = per_cat_rows
    if bbq_result is None:
        print("  [WARN] no BBQ data across any category.")
    else:
        print(f"\n    BBQ mean |Gap_g,y| across {bbq_result['n_categories']} categories "
              f"= {bbq_result['mean_abs_gap']}  "
              f"[{bbq_result['ci_low']},{bbq_result['ci_high']}]{bbq_result['sig']}")
        rows.append({
            "dataset": "BBQ",
            "gap_g_y": bbq_result["mean_abs_gap"]
        })

    out_path = _MAIN_DIR / "gap_g_y_results.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("\n" + "=" * 70)
    print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()