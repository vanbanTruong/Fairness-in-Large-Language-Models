import json
import os
import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from fairLLMs.definition.decoder_only.intrinsic_bias.stereotypical_association.sll.sll import compute_sll, sll_for_occupation

_MAIN_DIR   = Path(__file__).resolve().parent
_DATA_DIR   = _MAIN_DIR / "data"
BBQ_PER_CAT = 100000

_Q_PREFIX = re.compile(r"^\s*who\s+(?:was|is|were|are|had|has)\s+", re.IGNORECASE)


def _norm_tag(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def _question_trait(question: str) -> str:
    q = question.strip().rstrip("?").strip()
    q = _Q_PREFIX.sub("", q)
    return q.strip().lower()


MODEL_NAME = "meta-llama/Llama-2-7b-hf"
SEED       = 42
N_MAX      = 100000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BBQ_FILES = [
    "Age.jsonl", "Disability_status.jsonl", "Gender_identity.jsonl",
    "Nationality.jsonl", "Physical_appearance.jsonl", "Race_ethnicity.jsonl",
    "Religion.jsonl", "SES.jsonl", "Sexual_orientation.jsonl",
]

_SKIP_ANS = {
    "unknown", "can't be determined", "cannot be determined", "undetermined",
    "not enough information", "not answerable", "n/a",
}


def results_to_csv(results, filename: str) -> str:
    df = results if isinstance(results, pd.DataFrame) else pd.DataFrame(results)
    out_path = _MAIN_DIR / filename
    df.to_csv(out_path, index=False)
    return str(out_path)


def _clean(text: str) -> str:
    text = str(text).strip()
    for art in ("The ", "the ", "A ", "a ", "An ", "an "):
        if text.startswith(art):
            text = text[len(art):]
            break
    return text.strip().lower()


BIOS_PROFESSION_MAP = {
    0: "accountant", 1: "architect", 2: "attorney", 3: "chiropractor",
    4: "comedian", 5: "composer", 6: "dentist", 7: "dietitian", 8: "dj",
    9: "filmmaker", 10: "interior designer", 11: "journalist", 12: "model",
    13: "nurse", 14: "painter", 15: "paralegal", 16: "pastor",
    17: "personal trainer", 18: "photographer", 19: "physician", 20: "poet",
    21: "professor", 22: "psychologist", 23: "rapper", 24: "software engineer",
    25: "surgeon", 26: "teacher", 27: "yoga teacher",
}


def load_bios_pairs() -> Tuple[List[Tuple[str, str, str]], dict]:
    ds = load_dataset("LabHC/bias_in_bios", split="test")
    df = ds.to_pandas()
    skew = {} 
    for code, name in BIOS_PROFESSION_MAP.items():
        sub = df[df["profession"] == code]
        if len(sub) == 0:
            continue
        skew[name] = float((sub["gender"] == 0).mean())

    n, s = skew.get("nurse"), skew.get("surgeon")
    if n is not None and s is not None and not (n < 0.5 < s):
        raise RuntimeError(
            f"Gender-encoding sanity check FAILED: P(male|nurse)={n:.2f}, "
            f"P(male|surgeon)={s:.2f}; expected nurse<0.5<surgeon.")
    print(f"  [CHECK] P(male|nurse)={n:.2f}  P(male|surgeon)={s:.2f}  (encoding OK)")

    pairs = []
    for occ, p_male in skew.items():
        if p_male > 0.5:
            pairs.append((occ, "male", "female"))
        else:
            pairs.append((occ, "female", "male"))
    random.shuffle(pairs)
    print(f"  Bias-in-Bios: {len(pairs)} occupations (direction from labels)")
    return pairs[:N_MAX], skew


def load_bbq_pairs_all_categories() -> List[Tuple[str, str, str]]:
    pairs, seen, per_cat = [], set(), {}

    for jsonl_name in BBQ_FILES:
        cat = jsonl_name.replace(".jsonl", "")
        path = _DATA_DIR / jsonl_name
        if not path.exists():
            print(f"    [warn] missing {path}")
            per_cat[cat] = 0
            continue

        n_cat = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                if row.get("context_condition") != "ambig":
                    continue
                if row.get("question_polarity") != "neg":
                    continue

                concept = _question_trait(row.get("question", ""))
                if not concept or len(concept.split()) > 6:
                    continue

                stereo_groups = {_norm_tag(g) for g in
                                 row.get("additional_metadata", {})
                                    .get("stereotyped_groups", [])}
                if not stereo_groups:
                    continue

                info = row.get("answer_info", {})
                if not isinstance(info, dict):
                    continue

                opts = []
                for k in ("ans0", "ans1", "ans2"):
                    v = info.get(k)
                    if not (isinstance(v, (list, tuple)) and len(v) > 1):
                        continue
                    raw_tag = str(v[1]).strip()
                    tag = _norm_tag(raw_tag)
                    txt = _clean(v[0] if v[0] else row.get(k, ""))
                    if not txt or txt in _SKIP_ANS or "unknown" in tag:
                        continue
                    opts.append((txt, raw_tag, tag))
                if len(opts) < 2:
                    continue

                stereo_opt = next((o for o in opts if o[2] in stereo_groups), None)
                if stereo_opt is None:
                    continue
                counter_opt = next((o for o in opts if o[2] != stereo_opt[2]), None)
                if counter_opt is None:
                    continue

                key = (concept, stereo_opt[1], counter_opt[1])
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((concept, stereo_opt[1], counter_opt[1]))
                n_cat += 1
                if n_cat >= BBQ_PER_CAT:
                    break

        per_cat[cat] = n_cat
    return pairs


_NQ_ROLE = re.compile(
    r"\b(president|actor|actress|singer|author|writer|director|player|coach|"
    r"scientist|doctor|nurse|teacher|lawyer|judge|senator|governor|artist|"
    r"engineer|pilot|soldier|athlete|dancer|chef|farmer|painter|composer|"
    r"photographer|journalist|professor|surgeon|architect|banker|manager|"
    r"driver|officer|designer|programmer|researcher|astronaut|inventor)\b",
    re.IGNORECASE)


def load_nq_pairs() -> List[Tuple[str, str, str]]:
    try:
        ds = load_dataset("google-research-datasets/nq_open", split="validation")   # small: question/answer only
    except Exception as e:
        print(f"  [warn] Natural Questions load failed: {e}")
        return []
    concepts, seen = [], set()
    for ex in ds:
        q = str(ex.get("question", ""))
        for m in _NQ_ROLE.finditer(q):
            c = m.group(0).lower()
            if c not in seen:
                seen.add(c)
                concepts.append(c)
        if len(concepts) >= N_MAX:
            break
    pairs = [(c, "male", "female") for c in concepts][:N_MAX]
    print(f"  Natural Questions: {len(pairs)} concepts mined from real questions")
    print(f"    NOTE: fixed male/female axis -- CONTROL for gender skew, not a")
    print(f"          stereotype measure (NQ has no stereotype labels).")
    return pairs

def load_model():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[WARN] No HF_TOKEN in env. Llama-2 is GATED — set HF_TOKEN or the "
              "load will fail with a 401/403.")
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
        t = tokenizer.encode("The engineer said that", return_tensors="pt").to(
            next(model.parameters()).device)
        assert torch.isfinite(model(t).logits).all(), \
            "Base logits are NaN/inf — check dtype / attn_implementation"
    print(f"  Device: {DEVICE}  (finiteness check passed)\n")
    return tokenizer, model

def run_dataset(name: str, pairs: List[Tuple[str, str, str]],
                tokenizer, model, is_control: bool = False) -> dict:
    if not pairs:
        print(f"\n  [skip] {name} — no pairs")
        return None

    rows = []
    for occ, stereo, counter in pairs:
        per_occ = sll_for_occupation(model, tokenizer, DEVICE, occ, stereo, counter)
        rows.append({"occupation": occ, "stereo": stereo, "counter": counter, **per_occ})

    df = pd.DataFrame(rows)
    scores = {k: round(float(df[k].mean()),4) for k in ("NV", "CV", "IV") if k in df.columns}

    print(f"\n  Mean NV: {scores.get('NV', float('nan')):+.4f}  "
          f"CV: {scores.get('CV', float('nan')):+.4f}  "
          f"IV: {scores.get('IV', float('nan')):+.4f}")
    return {"dataset": name, **scores}


def main():
    print("\n" + "=" * 60)
    print(f"  Stereotypical Log-Likelihood (SLL)  |  {MODEL_NAME}")
    print("  NV=neutral  CV=competent  IV=incompetent")
    print("=" * 60)

    tokenizer, model = load_model()

    results = []
    bios_pairs, _skew = load_bios_pairs()
    r = run_dataset("Bias-in-Bios", bios_pairs, tokenizer, model)
    if r: results.append(r)

    r = run_dataset("BBQ", load_bbq_pairs_all_categories(),
                    tokenizer, model)
    if r: results.append(r)

    r = run_dataset("Natural Questions", load_nq_pairs(), tokenizer, model,
                    is_control=True)
    if r: results.append(r)

    summary = pd.DataFrame(results)
    out = results_to_csv(summary, "sll_results.csv")
    print("\n" + "=" * 60)
    print("  SLL Summary  (~0 = fair; + = stereo/male preference)")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()