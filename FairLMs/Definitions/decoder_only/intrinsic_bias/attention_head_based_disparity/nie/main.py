import os
import re
import random
from typing import List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from decoder_only.intrinsic_bias.attention_head_based_disparity.nie.nie import compute_nie_matrix, compute_nie, _capture_acts

_MAIN_DIR = os.path.dirname(__file__)
NIE_THRESHOLD = 0.003


def results_to_csv(results: list, filename: str) -> str:
    out_path = os.path.join(_MAIN_DIR, filename)
    pd.DataFrame(results).to_csv(out_path, index=False)
    return str(out_path)


MODEL_NAME = "gpt2-medium"
SEED       = 42
N_MAX      = 200
REDPILL_CSV = _MAIN_DIR / Path("red_pill_corpus.csv")
REDPILL_NROWS = 150000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
)
model.to(DEVICE)
model.double()
model.eval()
N_LAYERS = model.config.n_layer
N_HEADS  = model.config.n_head
HEAD_DIM = model.config.n_embd // N_HEADS
print(f"  Layers: {N_LAYERS}  |  Heads: {N_HEADS}  |  Device: {DEVICE}\n")


_GENDER_SUB = [
    (r"\bhe\b", "they"), (r"\bshe\b", "they"), (r"\bhim\b", "them"),
    (r"\bher\b", "them"), (r"\bhis\b", "their"), (r"\bhers\b", "theirs"),
    (r"\bhimself\b", "themselves"), (r"\bherself\b", "themselves"),
    (r"\bman\b", "person"), (r"\bwoman\b", "person"), (r"\bmen\b", "people"),
    (r"\bwomen\b", "people"), (r"\bmale\b", "person"), (r"\bfemale\b", "person"),
    (r"\bguy\b", "person"), (r"\bguys\b", "people"), (r"\bgirl\b", "person"),
    (r"\bgirls\b", "people"), (r"\bboy\b", "person"), (r"\bboys\b", "people"),
    (r"\bwife\b", "spouse"), (r"\bhusband\b", "spouse"),
    (r"\bmother\b", "parent"), (r"\bfather\b", "parent"),
    (r"\bchick\b", "person"), (r"\bchicks\b", "people"),
]

MALE_STEREO_OCCS = {
    "engineer", "surgeon", "physician", "developer", "carpenter", "lawyer",
    "manager", "analyst", "mechanic", "supervisor", "janitor", "driver",
    "sheriff", "farmer", "guard", "chief", "technician", "programmer",
    "scientist", "electrician", "plumber", "architect", "executive", "banker",
}
FEMALE_STEREO_OCCS = {
    "nurse", "receptionist", "secretary", "housekeeper", "librarian",
    "teacher", "cashier", "counselor", "attendant", "cleaner", "hairdresser",
    "dietitian", "paralegal", "designer", "editor", "baker", "clerk",
    "assistant", "therapist", "hygienist",
}


def stereo_spec(occ: str):
    o = occ.lower().strip()
    if o in MALE_STEREO_OCCS:
        return {"stereo": "he", "anti": "she", "cf_noun": "woman"}
    if o in FEMALE_STEREO_OCCS:
        return {"stereo": "she", "anti": "he", "cf_noun": "man"}
    return None


def neutralize(text: str, extra_occupations=None) -> str:
    out = text
    for pat, repl in _GENDER_SUB:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    if extra_occupations:
        for occ in extra_occupations:
            out = re.sub(r"\b" + re.escape(occ) + r"\b", "person", out, flags=re.IGNORECASE)
    return out


def _norm_label(l):
    if isinstance(l, str):
        return l
    return {0: "anti-stereotype", 1: "stereotype", 2: "unrelated"}.get(int(l), str(l))


def _fill_word(context, sentence):
    pre, _, post = context.partition("BLANK")
    pre, post = pre.strip(), post.strip()
    s = sentence.strip()
    if pre and s.lower().startswith(pre.lower()):
        s = s[len(pre):].strip()
    if post and s.lower().endswith(post.lower()):
        s = s[:len(s) - len(post)].strip()
    toks = s.split()
    return toks[0].strip(".,!?") if toks else None


def load_stereoset_prompts(n_max: int = 200, bias_type: str = "profession"):

    try:
        ds = load_dataset("McGill-NLP/stereoset", "intrasentence")
    except Exception as e:
        print(f"  [warn] StereoSet load failed: {e}")
        return []
    split = ds["validation"]
    prompts, skipped_target, skipped_fill = [], 0, 0
    for ex in split:
        if bias_type and ex.get("bias_type") != bias_type:
            continue
        context = ex["context"]
        target  = (ex.get("target") or "").strip()
        if "BLANK" not in context or not target:
            continue

        prefix = context.partition("BLANK")[0].strip()
        if target.lower() not in prefix.lower():
            skipped_target += 1
            continue

        sents = list(zip(ex["sentences"]["sentence"],
                         [_norm_label(l) for l in ex["sentences"]["gold_label"]]))
        stereo_sent = next((s for s, l in sents if l == "stereotype"), None)
        anti_sent   = next((s for s, l in sents if l == "anti-stereotype"), None)
        if stereo_sent is None or anti_sent is None:
            continue
        stereo_word = _fill_word(context, stereo_sent)
        anti_word   = _fill_word(context, anti_sent)
        if not stereo_word or not anti_word or stereo_word == anti_word:
            skipped_fill += 1
            continue

        cf_prefix = re.sub(re.escape(target), "person", prefix, flags=re.IGNORECASE)
        if cf_prefix.strip().lower() == prefix.strip().lower():
            continue

        prompts.append((prefix, cf_prefix, stereo_word, anti_word))
        if len(prompts) >= n_max:
            break
    print(f"  StereoSet: {len(prompts)} prompts "
          f"(target-neutralized cf; skipped {skipped_target} target-after-BLANK, "
          f"{skipped_fill} bad-fill)")
    return prompts


WINO_OCCS = [
    "nurse", "surgeon", "receptionist", "engineer", "librarian", "lawyer",
    "teacher", "developer", "housekeeper", "physician", "secretary", "manager",
    "cashier", "carpenter", "counselor", "supervisor", "hairdresser", "analyst",
    "attendant", "janitor", "doctor", "technician", "administrator", "paramedic",
    "accountant", "clerk", "electrician", "plumber", "mechanic", "scientist",
    "pathologist", "practitioner", "pharmacist", "therapist", "investigator",
    "chef", "dispatcher", "hygienist", "programmer", "inspector", "auditor",
    "worker", "employee", "advisor", "officer", "assistant", "specialist",
]


def load_winogender_prompts(n_max=N_MAX):
    ds = load_dataset("oskarvanderwal/winogender", "all", split="test")
    prompts = []
    for ex in ds:
        sent = ex.get("sentence") or ""
        m = re.search(r"\b(he|she|him|her|his)\b", sent, flags=re.IGNORECASE)
        if not m:
            continue
        prefix = sent[:m.start()].strip()
        if len(prefix.split()) < 3:
            continue

        found = None
        for occ in (MALE_STEREO_OCCS | FEMALE_STEREO_OCCS):
            if re.search(r"\b" + re.escape(occ) + r"\b", prefix, re.IGNORECASE):
                found = occ
                break
        if not found:
            continue
        spec = stereo_spec(found)
        if spec is None:
            continue
        cf_prefix = re.sub(r"\b" + re.escape(found) + r"\b", spec["cf_noun"],
                           prefix, flags=re.IGNORECASE)
        if cf_prefix.lower() == prefix.lower():
            continue
        prompts.append((prefix, cf_prefix, spec["stereo"], spec["anti"]))
        if len(prompts) >= n_max:
            break
    print(f"  Winogender: {len(prompts)} prompts (anti-stereotypical substitution cf)")
    return prompts
_PRON_OPP = {"he": "she", "she": "he", "him": "her", "her": "him", "his": "her"}
_CUE  = re.compile(r"\b(because|since|and|but|so|while|when|as)\b", re.IGNORECASE)
_PRON = re.compile(r"\b(he|she|him|her|his)\b", re.IGNORECASE)


_ALL_OCCS = MALE_STEREO_OCCS | FEMALE_STEREO_OCCS

def load_redpill_prompts(n_max=N_MAX, nrows=REDPILL_NROWS):
    df = pd.read_csv(REDPILL_CSV, usecols=["body"], nrows=nrows)
    prompts, seen = [], set()
    for body in df["body"].dropna().astype(str):
        pm = _PRON.search(body)
        if not pm:
            continue
        prefix = re.sub(r"\s+", " ", body[:pm.start()].strip())
        if not (4 <= len(prefix.split()) <= 40):
            continue
        found = next((o for o in _ALL_OCCS
                      if re.search(r"\b" + re.escape(o) + r"\b", prefix, re.IGNORECASE)), None)
        if not found:
            continue
        spec = stereo_spec(found)
        cf_prefix = re.sub(r"\b" + re.escape(found) + r"\b", spec["cf_noun"],
                           prefix, flags=re.IGNORECASE)
        if cf_prefix.lower() == prefix.lower() or prefix.lower() in seen:
            continue
        seen.add(prefix.lower())
        prompts.append((prefix, cf_prefix, spec["stereo"], spec["anti"]))
        if len(prompts) >= n_max:
            break
    print(f"  Red Pill: {len(prompts)} prompts (anti-stereotypical substitution cf)")
    return prompts

def _prompts_to_probes(prompts, tokenizer, DEVICE):
    probes = []
    for null_prompt, set_prompt, stereo_word, anti_word in prompts:
        stereo_ids = tokenizer.encode(" " + stereo_word, add_special_tokens=False)
        anti_ids   = tokenizer.encode(" " + anti_word,   add_special_tokens=False)
        if not stereo_ids or not anti_ids:
            continue
        probes.append({
            "prompt": null_prompt, "cf_text": set_prompt,
            "stereo_token_id": stereo_ids[0], "anti_token_id": anti_ids[0],
        })
    return probes


def run_dataset(name: str, prompts):
    print(f"\n{'-' * 55}")
    print(f"  NIE  |  {name}")
    print(f"{'-' * 55}")
    if not prompts:
        print("  No prompts loaded — skipping.")
        return {"dataset": name, "proportion_biased": float("nan"), "n_prompts": 0}
    probes     = _prompts_to_probes(prompts, tokenizer, DEVICE)
    nie_matrix = compute_nie_matrix(model, tokenizer, DEVICE, probes,
                                    N_LAYERS, N_HEADS, HEAD_DIM)
    prop       = compute_nie(nie_matrix, NIE_THRESHOLD)
    mean_abs = float(np.abs(nie_matrix).mean())
    max_abs  = float(np.abs(nie_matrix).max())
    print(f"  Top-decile head proportion: {prop:.4f}  |  mean|NIE|={mean_abs:.5f}  max|NIE|={max_abs:.5f}")
    rows = [{"layer": l, "head": h, "nie": nie_matrix[l, h]}
            for l in range(N_LAYERS) for h in range(N_HEADS)]
    return {"dataset": name, "nie_score": prop}


def main():
    print("\n" + "=" * 55)
    print("  Natural Indirect Effect (NIE)  |  GPT-2 Medium")
    print("=" * 55)
    results = [
        run_dataset("StereoSet",  load_stereoset_prompts()),
        run_dataset("Winogender", load_winogender_prompts()),
        run_dataset("RedPill",    load_redpill_prompts()),
    ]
    summary = pd.DataFrame(results)
    out = results_to_csv(summary.to_dict("records"), "nie_results.csv")
    print("\n" + "=" * 55)
    print("  NIE Summary")
    print("=" * 55)

if __name__ == "__main__":
    main()