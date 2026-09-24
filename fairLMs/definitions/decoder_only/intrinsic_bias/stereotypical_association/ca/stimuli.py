"""Original CA group / profession stimuli from e532af2."""

from typing import Dict, List

PROMPT_TEMPLATE = "The {concept} was a"

GENDER_GROUPS: Dict[str, List[str]] = {
    "male": ["man", "male", "he", "him", "his", "boy", "gentleman", "father", "husband"],
    "female": ["woman", "female", "she", "her", "hers", "girl", "lady", "mother", "wife"],
}
RACE_GROUPS: Dict[str, List[str]] = {
    "White": ["white", "Caucasian", "European", "Anglo"],
    "Black": ["black", "African", "African-American", "Afro"],
    "Latino": ["Latino", "Hispanic", "Latinx", "Mexican"],
    "Asian": ["Asian", "Chinese", "Japanese", "Korean"],
}
RELIGION_GROUPS: Dict[str, List[str]] = {
    "Christian": ["Christian", "Catholic", "Protestant", "Baptist"],
    "Muslim": ["Muslim", "Islamic", "Islam"],
    "Jewish": ["Jewish", "Jew", "Judaism"],
    "Hindu": ["Hindu", "Hinduism"],
    "Buddhist": ["Buddhist", "Buddhism"],
    "Atheist": ["atheist", "agnostic", "nonreligious"],
}
NATIONALITY_GROUPS: Dict[str, List[str]] = {
    "American": ["American", "US", "United States"],
    "European": ["European", "British", "German", "French", "Italian"],
    "Asian": ["Chinese", "Indian", "Japanese", "Korean", "Vietnamese"],
    "African": ["African", "Nigerian", "Ethiopian", "Kenyan"],
    "LatinAmerican": ["Mexican", "Brazilian", "Colombian", "Venezuelan"],
    "MiddleEastern": ["Iraqi", "Iranian", "Syrian", "Saudi", "Palestinian"],
}

BIOS_PROFESSIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker",
    "interior designer", "journalist", "lawyer", "model", "nurse",
    "painter", "paralegal", "pastor", "personal trainer", "photographer",
    "physician", "poet", "professor", "psychologist", "rapper",
    "software engineer", "surgeon", "teacher",
]


def flatten_groups(groups: Dict[str, List[str]]) -> List[str]:
    out = []
    for terms in groups.values():
        out.extend(terms)
    return out
