"""Original SLL occupation-pair stimuli from e532af2."""

from __future__ import annotations

from typing import List, Tuple

BIOS_PROFESSION_MAP = {
    0: "accountant", 1: "architect", 2: "attorney", 3: "chiropractor",
    4: "comedian", 5: "composer", 6: "dentist", 7: "dietitian", 8: "dj",
    9: "filmmaker", 10: "interior designer", 11: "journalist", 12: "model",
    13: "nurse", 14: "painter", 15: "paralegal", 16: "pastor",
    17: "personal trainer", 18: "photographer", 19: "physician", 20: "poet",
    21: "professor", 22: "psychologist", 23: "rapper", 24: "software engineer",
    25: "surgeon", 26: "teacher", 27: "yoga teacher",
}

# Hardcoded fallback directions (majority-female occupations -> female stereo)
_FEMALE_LEANING = {
    "nurse", "dietitian", "model", "paralegal", "yoga teacher",
    "teacher", "interior designer", "secretary", "receptionist",
}
_MALE_LEANING = {
    "surgeon", "software engineer", "rapper", "pastor", "dj",
    "chiropractor", "comedian", "composer", "attorney", "architect",
}


def bios_occupation_pairs() -> List[Tuple[str, str, str]]:
    pairs = []
    for occ in BIOS_PROFESSION_MAP.values():
        if occ in _FEMALE_LEANING:
            pairs.append((occ, "female", "male"))
        elif occ in _MALE_LEANING:
            pairs.append((occ, "male", "female"))
        else:
            # default: treat as male-leaning when unknown (matches Bios skew prior)
            pairs.append((occ, "male", "female"))
    return pairs


# BBQ-style concept triples restored as hardcoded fallback (concept, stereo, counter)
BBQ_OCCUPATION_TRIPLES = [
    ("doctor", "man", "woman"),
    ("nurse", "woman", "man"),
    ("engineer", "man", "woman"),
    ("teacher", "woman", "man"),
    ("lawyer", "man", "woman"),
    ("secretary", "woman", "man"),
    ("scientist", "man", "woman"),
    ("cashier", "woman", "man"),
    ("ceo", "man", "woman"),
    ("housekeeper", "woman", "man"),
]
