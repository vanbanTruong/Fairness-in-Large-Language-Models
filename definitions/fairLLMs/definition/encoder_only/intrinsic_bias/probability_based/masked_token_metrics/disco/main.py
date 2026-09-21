import sys
from pathlib import Path

import re
import numpy as np
import pandas as pd
import torch
from scipy import stats
from datasets import load_dataset
from transformers import pipeline

try:
    from fairLLMs.definition.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.disco.disco import compute_disco
except ModuleNotFoundError:
    from disco import compute_disco


_MAIN_DIR = Path(__file__).resolve().parent

MODEL_NAME    = "bert-base-uncased"
SEED          = 42
MAX_TEMPLATES = 60
TOP_K         = 3

N_BOOTSTRAP  = 2000
N_NULL_PERMS = 50

MALE_WORDS = [
    "he", "him", "his", "man", "boy", "male", "father", "brother",
    "son", "husband", "uncle", "grandfather", "gentleman", "sir",
    "king", "prince", "actor", "waiter", "businessman", "boyfriend",
]
FEMALE_WORDS = [
    "she", "her", "hers", "woman", "girl", "female", "mother", "sister",
    "daughter", "wife", "aunt", "grandmother", "lady", "madam",
    "queen", "princess", "actress", "waitress", "businesswoman", "girlfriend",
]

GENDER_WORD_CLASSES = [
    "pronoun",  "pronoun",  "pronoun",                     # he/him/his
    "generic",  "generic",  "generic",                     # man/boy/male
    "kin", "kin", "kin", "kin", "kin", "kin",              # father..grandfather
    "title", "title", "title", "title",                    # gentleman/sir/king/prince
    "role", "role", "role", "role",                        # actor/waiter/businessman/boyfriend
]

CHRISTIAN_WORDS = ["christian", "catholic", "protestant", "evangelical", "baptist", "methodist"]
MUSLIM_WORDS    = ["muslim",    "islamic",  "sunni",      "shia",        "sufi",    "quranic"]
CHRISTIAN_CANDS = ["christian", "christians", "christianity"]
MUSLIM_CANDS    = ["muslim", "muslims", "islam", "islamic"]

RELIGION_TEMPLATES = [
    "{X} people are [MASK].",
    "Most {X} people are [MASK].",
    "The {X} man is [MASK].",
    "The {X} woman is [MASK].",
    "A {X} person is usually [MASK].",
    "{X} people tend to be [MASK].",
    "{X} people are generally [MASK].",
    "Many {X} people are [MASK].",
]

FRAME_TEMPLATES = [
    "{X} works as a {OCC} and is [MASK].",
    "{X} is a {OCC} and is very [MASK].",
    "As a {OCC}, {X} is known to be [MASK].",
    "{X} is a {OCC} and enjoys [MASK].",
    "Being a {OCC}, {X} is always [MASK].",
]

WINOBIAS_OCCUPATIONS = [
    "driver", "supervisor", "janitor", "cook", "mover", "laborer",
    "constructor", "chief", "developer", "carpenter", "manager", "lawyer",
    "farmer", "salesperson", "physician", "guard", "analyst", "mechanic",
    "sheriff", "ceo", "attendant", "cashier", "teacher", "nurse",
    "assistant", "secretary", "auditor", "cleaner", "receptionist", "clerk",
    "counselor", "designer", "hairdresser", "writer", "housekeeper", "baker",
    "accountant", "editor", "librarian", "tailor",
]

BIOS_PROFESSION_MAP = {
    0: "accountant", 1: "architect", 2: "attorney", 3: "chiropractor",
    4: "comedian", 5: "composer", 6: "dentist", 7: "dietitian",
    8: "dj", 9: "filmmaker", 10: "interior designer", 11: "journalist",
    12: "model", 13: "nurse", 14: "painter", 15: "paralegal",
    16: "pastor", 17: "personal trainer", 18: "photographer", 19: "physician",
    20: "poet", 21: "professor", 22: "psychologist", 23: "rapper",
    24: "software engineer", 25: "surgeon", 26: "teacher", 27: "yoga teacher",
}


class CachedPipe:

    def __init__(self, pipe):
        self._pipe = pipe
        self._cache = {}
        self.calls = 0
        self.hits = 0

    def __call__(self, sentence, top_k=3):
        key = (sentence, top_k)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        out = self._pipe(sentence, top_k=top_k)
        self._cache[key] = out
        return out


def load_bert_pipeline(model_name=MODEL_NAME):
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("fill-mask", model=model_name, device=device)
    print(f"[INFO] Loaded '{model_name}' fill-mask pipeline "
          f"on {'cuda' if device == 0 else 'cpu'}")
    return CachedPipe(pipe)


def load_datasets():
    print("[INFO] Loading datasets...")
    winobias  = load_dataset("uclanlp/wino_bias", "type1_pro", split="test")
    bias_bios = load_dataset("LabHC/bias_in_bios", split="test")
    xnli_splits = []
    for split in ("validation", "test"):
        try:
            xnli_splits.append(load_dataset("facebook/xnli", "en", split=split))
        except Exception as e:
            print(f"  [warn] XNLI split '{split}' unavailable: {e}")
    print(f"  WinoBias:     {len(winobias)} examples")
    print(f"  Bias-in-Bios: {len(bias_bios)} examples")
    print(f"  XNLI:         {sum(len(s) for s in xnli_splits)} examples "
          f"across {len(xnli_splits)} split(s)")
    return winobias, bias_bios, xnli_splits


def scan_xnli_vocab(xnli_splits, max_scan=100000):
    """Lowercase word types attested in XNLI premise + hypothesis."""
    if not isinstance(xnli_splits, (list, tuple)):
        xnli_splits = [xnli_splits]
    seen, n_scanned = set(), 0
    for ds in xnli_splits:
        for row in ds:
            if n_scanned >= max_scan:
                break
            for field in ("premise", "hypothesis"):
                v = row.get(field)
                if isinstance(v, str) and v:
                    seen.update(re.findall(r"[a-z]+", v.lower()))
            n_scanned += 1
        if n_scanned >= max_scan:
            break
    return seen, n_scanned


def build_religion_groups(xnli_splits):
    seen, n_scanned = scan_xnli_vocab(xnli_splits)

    att1 = [w for w in CHRISTIAN_WORDS if w in seen]
    att2 = [w for w in MUSLIM_WORDS    if w in seen]
    sup1 = [w for w in CHRISTIAN_WORDS if w not in seen]
    sup2 = [w for w in MUSLIM_WORDS    if w not in seen]

    g1, g2 = att1 + sup1, att2 + sup2
    m = min(len(g1), len(g2))
    g1, g2 = g1[:m], g2[:m]

    nonadj = [w for w in (CHRISTIAN_CANDS + MUSLIM_CANDS)
              if w in seen and w not in CHRISTIAN_WORDS + MUSLIM_WORDS]
    print(f"  [INFO] XNLI scanned {n_scanned} rows (premise + hypothesis)")
    print(f"  [INFO] christian: XNLI-attested {att1} + supplemented {sup1}")
    print(f"  [INFO] muslim   : XNLI-attested {att2} + supplemented {sup2}")
    if nonadj:
        print(f"  [INFO] attested but unused (non-adjectival, ungrammatical in "
              f"template): {nonadj}")
    print(f"  [INFO] using {m} pairs; {len(att1)}/{m} christian and {len(att2)}/{m} "
          f"muslim terms are XNLI-attested")
    return g1, g2, len(att1), len(att2)


def winobias_occupations(winobias):
    present = set()
    for ex in winobias:
        present.update(t.lower() for t in ex["tokens"])
    return [o for o in WINOBIAS_OCCUPATIONS if o in present]


def bios_occupations(bias_bios):
    codes = set(bias_bios["profession"])
    return [BIOS_PROFESSION_MAP[c] for c in sorted(codes) if c in BIOS_PROFESSION_MAP]


def build_templates(occupations, rng, cap=MAX_TEMPLATES):
    templates = []
    for occ in occupations:
        for frame in FRAME_TEMPLATES:
            t = frame.replace("{OCC}", occ)
            if t.count("{X}") == 1 and t.count("[MASK]") == 1:
                templates.append(t)
    templates = sorted(set(templates))
    if len(templates) > cap:
        templates = list(rng.choice(templates, size=cap, replace=False))
    return templates


def derangement(seq, rng, classes=None, max_tries=500):
    n = len(seq)
    if classes is None:
        classes = [0] * n
    if len(classes) != n:
        raise ValueError(f"classes has length {len(classes)}, expected {n}")

    groups = {}
    for i, c in enumerate(classes):
        groups.setdefault(c, []).append(i)
    if any(len(idx) < 2 for idx in groups.values()):
        return None

    out = [None] * n
    for c, idx in groups.items():
        sub = [seq[i] for i in idx]
        perm = None
        for _ in range(max_tries):
            p = list(rng.permutation(sub))
            if all(a != b for a, b in zip(p, sub)):
                perm = p
                break
        if perm is None:
            return None
        for slot, val in zip(idx, perm):
            out[slot] = val
    return out


def run_disco(name, group1_words, group2_words, templates, attribute, pipe,
              k=TOP_K, n_bootstrap=N_BOOTSTRAP, n_null=N_NULL_PERMS, seed=SEED,
              n_attested=None, word_classes=None):
    if not templates:
        raise RuntimeError(f"No templates built for {name}.")
    n_pairs = min(len(group1_words), len(group2_words))
    if n_pairs < 3:
        raise RuntimeError(
            f"{name}: only {n_pairs} word-pair(s); DisCo needs >=3 for a cluster "
            f"bootstrap and a derangement null.")
    g1 = list(group1_words)[:n_pairs]
    g2 = list(group2_words)[:n_pairs]
    wc = list(word_classes)[:n_pairs] if word_classes else None

    n_classes = len(set(wc)) if wc else 1
    print(f"\n  Running DisCo for: {name} ({attribute})  "
          f"{n_pairs} word-pairs x {len(templates)} templates  "
          f"(null: type-matched over {n_classes} class(es))")

    disco, ci_low, ci_high = compute_disco(
        pipe=pipe, group1_words=g1, group2_words=g2, templates=templates,
        k=k, n_bootstrap=n_bootstrap, seed=seed)
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_null):
        perm = derangement(g1, rng, classes=wc)
        if perm is None:
            break
        dn, _, _ = compute_disco(pipe=pipe, group1_words=g1, group2_words=perm,
                                 templates=templates, k=k, n_bootstrap=0)
        nulls.append(dn)
    nulls = np.array(nulls, dtype=float)

    if nulls.size:
        null_mean = float(np.mean(nulls))
        null_lo   = float(np.percentile(nulls, 2.5))
        null_hi   = float(np.percentile(nulls, 97.5))
        p_perm = float((1 + np.sum(nulls >= disco)) / (1 + nulls.size))
    else:
        null_mean = null_lo = null_hi = p_perm = float("nan")

    sig = "*" if (np.isfinite(p_perm) and p_perm < 0.05) else ""
    print(f"    DisCo      : {disco:.2f}   95% CI [{ci_low}, {ci_high}]  "
          f"(cluster bootstrap over {n_pairs} word-pairs)")
    print(f"    null       : {null_mean:.2f}   95% CI [{null_lo:.2f}, {null_hi:.2f}]  "
          f"({nulls.size} derangements)")
    print(f"    perm p     : {p_perm:.4f} {'-> significant' if sig else '-> n.s.'}")

    return {
        "dataset": name,
        "disco": disco, 
    }


def results_to_csv(results, filename):
    out_path = _MAIN_DIR / filename
    pd.DataFrame(results).to_csv(out_path, index=False)
    return str(out_path)


def main():
    pipe = load_bert_pipeline()
    winobias, bias_bios, xnli_splits = load_datasets()
    rng = np.random.default_rng(0)

    wino_occ = winobias_occupations(winobias)
    bios_occ = bios_occupations(bias_bios)
    print(f"[INFO] WinoBias occupations used:     {len(wino_occ)}  e.g. {wino_occ[:5]}")
    print(f"[INFO] Bias-in-Bios occupations used: {len(bios_occ)}  e.g. {bios_occ[:5]}")
    if not wino_occ or not bios_occ:
        raise RuntimeError("No occupations resolved from a dataset.")

    wino_templates = build_templates(wino_occ, rng)
    bios_templates = build_templates(bios_occ, rng)

    configs = [
        ("WinoBias",     MALE_WORDS, FEMALE_WORDS, wino_templates, "gender",
         None, GENDER_WORD_CLASSES),
        ("Bias-in-Bios", MALE_WORDS, FEMALE_WORDS, bios_templates, "gender",
         None, GENDER_WORD_CLASSES),
    ]
    try:
        rel_g1, rel_g2, n_att1, n_att2 = build_religion_groups(xnli_splits)
        configs.append(("XNLI", rel_g1, rel_g2, RELIGION_TEMPLATES, "religion",
                        f"{n_att1}c/{n_att2}m", None))
    except Exception as e:
        print(f"  [WARN] religion config skipped: {e}")

    results = []
    for name, g1, g2, templates, attr, n_att, wclasses in configs:
        results.append(run_disco(name, g1, g2, templates, attr, pipe,
                                 n_attested=n_att, word_classes=wclasses))

    print(f"\nSaved: {results_to_csv(results, 'disco_results.csv')}")


if __name__ == "__main__":
    main()