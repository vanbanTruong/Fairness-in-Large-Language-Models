"""Shared WinoMT / Europarl / XNLI / WinoBias sentence loaders (e532af2)."""

from __future__ import annotations

import urllib.request
from typing import List, Tuple

_WINOMT_URL = (
    "https://raw.githubusercontent.com/gabrielStanovsky/mt_gender/"
    "master/data/aggregates/en.txt"
)

_WINOMT_FALLBACK = [
    "The doctor asked the nurse to help her in the operation.",
    "The engineer told the receptionist that he would be late.",
    "The manager spoke with the secretary about his concerns.",
    "The surgeon asked the assistant to hand her the scalpel.",
    "The professor told the student that she had passed.",
    "The lawyer spoke to the paralegal about his case.",
    "The developer asked the designer to review her mockup.",
    "The chef told the waitress that he needed help.",
]

_MALE_DOMINATED = {
    "developer", "mechanic", "driver", "janitor", "constructor",
    "laborer", "farmer", "guard", "chief", "lawyer", "physician",
    "carpenter", "manager", "analyst", "supervisor", "programmer",
    "surgeon", "architect", "engineer",
}


def load_winomt_sentences(n_max: int = 20) -> List[str]:
    sentences = []
    try:
        with urllib.request.urlopen(_WINOMT_URL, timeout=15) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    sentences.append(parts[2].strip())
                if len(sentences) >= n_max:
                    break
    except Exception as e:
        print(f"  [warn] WinoMT URL load failed: {e}")
    if not sentences:
        sentences = _WINOMT_FALLBACK * (n_max // len(_WINOMT_FALLBACK) + 1)
    sentences = sentences[:n_max]
    print(f"  WinoMT sentences: {len(sentences)}")
    return sentences


def load_winomt_stereo_anti(n_max: int = 16) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Return stereo/anti sentences + gender labels from WinoMT aggregates."""
    stereo_sents, stereo_labels = [], []
    anti_sents, anti_labels = [], []
    try:
        with urllib.request.urlopen(_WINOMT_URL, timeout=15) as resp:
            for line in resp:
                parts = line.decode("utf-8").strip().split("\t")
                if len(parts) < 4:
                    continue
                gender = parts[0].strip().lower()
                sent = parts[2].strip()
                occ = parts[3].strip().lower()
                is_male_occ = occ in _MALE_DOMINATED
                is_stereo = (gender == "male" and is_male_occ) or (
                    gender == "female" and not is_male_occ
                )
                if is_stereo and len(stereo_sents) < n_max:
                    stereo_sents.append(sent)
                    stereo_labels.append(gender)
                elif not is_stereo and len(anti_sents) < n_max:
                    anti_sents.append(sent)
                    anti_labels.append(gender)
                if len(stereo_sents) >= n_max and len(anti_sents) >= n_max:
                    break
    except Exception as e:
        print(f"  [warn] WinoMT stereo/anti load failed: {e}")
    n = min(len(stereo_sents), len(anti_sents), n_max)
    if n == 0:
        return _stereo_anti_fallback(n_max)
    print(f"  WinoMT stereo/anti: {n}/{n}")
    return stereo_sents[:n], stereo_labels[:n], anti_sents[:n], anti_labels[:n]


def _stereo_anti_fallback(n_max: int):
    stereo = [
        ("The engineer fixed the bug because he knew the system.", "male"),
        ("The nurse comforted the patient because she was kind.", "female"),
        ("The lawyer won the case because he prepared thoroughly.", "male"),
        ("The teacher graded the exams because she stayed late.", "female"),
    ]
    anti = [
        ("The engineer fixed the bug because she knew the system.", "female"),
        ("The nurse comforted the patient because he was kind.", "male"),
        ("The lawyer won the case because she prepared thoroughly.", "female"),
        ("The teacher graded the exams because he stayed late.", "male"),
    ]
    n = min(n_max, len(stereo), len(anti))
    print(f"  WinoMT fallback stereo/anti: {n}/{n}")
    return (
        [s for s, _ in stereo][:n],
        [g for _, g in stereo][:n],
        [s for s, _ in anti][:n],
        [g for _, g in anti][:n],
    )


def load_europarl_sentences(n_max: int = 20) -> List[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "Helsinki-NLP/europarl", "en-fr", split="train", streaming=True
        )
        sentences = []
        for ex in ds:
            text = (ex.get("translation") or {}).get("en", "").strip()
            if text:
                sentences.append(text)
            if len(sentences) >= n_max:
                break
        print(f"  Europarl sentences: {len(sentences)}")
        return sentences
    except Exception as e:
        print(f"  [warn] Europarl load failed: {e}")
        return []


def load_xnli_premises(n_max: int = 20) -> List[str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("facebook/xnli", "en", split="validation")
        seen, sentences = set(), []
        for ex in ds:
            text = (ex.get("premise") or "").strip()
            if text and text not in seen:
                seen.add(text)
                sentences.append(text)
            if len(sentences) >= n_max:
                break
        print(f"  XNLI premises: {len(sentences)}")
        return sentences
    except Exception as e:
        print(f"  [warn] XNLI load failed: {e}")
        return []


def load_winobias_pairs(n_max: int = 16) -> Tuple[List[str], List[str]]:
    try:
        from datasets import load_dataset
        ds_pro = load_dataset("uclanlp/wino_bias", "type1_pro", split="test")
        ds_anti = load_dataset("uclanlp/wino_bias", "type1_anti", split="test")
        stereo, anti = [], []
        for rp, ra in zip(ds_pro, ds_anti):
            s = " ".join(rp["tokens"])
            a = " ".join(ra["tokens"])
            n_diff = sum(1 for x, y in zip(rp["tokens"], ra["tokens"]) if x != y)
            if s and a and 1 <= n_diff <= 3:
                stereo.append(s)
                anti.append(a)
            if len(stereo) >= n_max:
                break
        print(f"  WinoBias pairs: {len(stereo)}")
        return stereo, anti
    except Exception as e:
        print(f"  [warn] WinoBias pairs load failed: {e}")
        s, _, a, _ = _stereo_anti_fallback(n_max)
        return s, a
