"""Original demographic AXES + prompt loaders from e532af2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from fairLMs.datasets import BBQ, CrowSPairs

AXES: Dict[str, Dict[str, List[str]]] = {
    "gender": {
        "stereo": ["she", "her", "woman", "female", "girl", "lady"],
        "counter": ["he", "him", "man", "male", "boy", "gentleman"],
        "neutral": ["they", "them", "person", "individual", "someone", "one"],
    },
    "race": {
        "stereo": ["black", "african", "hispanic", "latino", "asian"],
        "counter": ["white", "caucasian", "european", "western"],
        "neutral": ["person", "individual", "people", "human", "citizen"],
    },
    "age": {
        "stereo": ["elderly", "senior", "old", "aged", "retired", "elder"],
        "counter": ["young", "teenager", "youth", "junior", "adolescent"],
        "neutral": ["adult", "person", "individual", "grown", "someone"],
    },
    "religion": {
        "stereo": ["muslim", "islamic", "jewish", "hindu", "sikh"],
        "counter": ["christian", "catholic", "protestant"],
        "neutral": ["person", "believer", "individual", "someone", "citizen"],
    },
    "nationality": {
        "stereo": ["mexican", "chinese", "indian", "nigerian", "iraqi", "syrian"],
        "counter": ["american", "british", "german", "french", "canadian"],
        "neutral": ["person", "individual", "foreigner", "citizen", "someone"],
    },
    "disability": {
        "stereo": ["disabled", "handicapped", "blind", "deaf", "autistic", "crippled"],
        "counter": ["healthy", "able", "normal", "typical"],
        "neutral": ["person", "individual", "someone", "patient", "adult"],
    },
    "appearance": {
        "stereo": ["fat", "obese", "ugly", "overweight", "unattractive"],
        "counter": ["thin", "slim", "attractive", "beautiful", "handsome"],
        "neutral": ["person", "individual", "someone", "adult", "figure"],
    },
    "ses": {
        "stereo": ["poor", "unemployed", "homeless", "impoverished", "broke"],
        "counter": ["rich", "wealthy", "affluent", "privileged"],
        "neutral": ["person", "individual", "worker", "citizen", "someone"],
    },
    "orientation": {
        "stereo": ["gay", "lesbian", "homosexual", "queer", "bisexual"],
        "counter": ["straight", "heterosexual"],
        "neutral": ["person", "individual", "someone", "adult", "partner"],
    },
}

BBQ_AXIS = {
    "Age": "age",
    "Disability_status": "disability",
    "Gender_identity": "gender",
    "Nationality": "nationality",
    "Physical_appearance": "appearance",
    "Race_ethnicity": "race",
    "Religion": "religion",
    "SES": "ses",
    "Sexual_orientation": "orientation",
}

CROWS_AXIS = {
    "race-color": "race",
    "race": "race",
    "gender": "gender",
    "age": "age",
    "religion": "religion",
    "nationality": "nationality",
    "disability": "disability",
    "physical-appearance": "appearance",
    "socioeconomic": "ses",
    "sexual-orientation": "orientation",
}


def load_bbq_prompts(n_max: int = 30, per_cat: int = 5) -> Dict[str, List[str]]:
    by_axis: Dict[str, List[str]] = {a: [] for a in AXES}
    for row in BBQ(n_max=n_max * 3).load():
        axis = BBQ_AXIS.get(row.get("category", ""))
        if axis is None or len(by_axis[axis]) >= per_cat:
            continue
        ctx = str(row.get("context", "")).strip()
        q = str(row.get("question", "")).strip()
        if ctx and q:
            by_axis[axis].append(f"{ctx} {q} The answer is the")
    return {k: v for k, v in by_axis.items() if v}


def load_crows_prompts(n_max: int = 40) -> Dict[str, List[str]]:
    by_axis: Dict[str, List[str]] = {a: [] for a in AXES}
    for ex in CrowSPairs(n_max=n_max * 2).load():
        axis = CROWS_AXIS.get(str(ex.get("bias_type", "")).lower())
        if axis is None:
            continue
        sent = str(ex.get("stereotype") or "").strip()
        if not sent:
            continue
        # Truncate to a prompt-like prefix (drop last word)
        toks = sent.split()
        if len(toks) < 4:
            continue
        by_axis[axis].append(" ".join(toks[:-1]))
        if sum(len(v) for v in by_axis.values()) >= n_max:
            break
    return {k: v for k, v in by_axis.items() if v}
