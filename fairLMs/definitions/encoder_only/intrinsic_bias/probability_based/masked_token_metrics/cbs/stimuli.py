"""Original CBS stimulus constants from e532af2."""

NATIONALITY_TERMS = [
    "American", "Chinese", "Mexican", "British", "Japanese",
    "Indian", "German", "French", "Korean", "Brazilian",
]
RELIGION_TERMS = ["Christian", "Muslim", "Jewish", "Buddhist", "Hindu", "Sikh", "atheist"]

WINOBIAS_ATTRS = [
    "doctor", "nurse", "engineer", "teacher", "lawyer", "secretary",
    "janitor", "manager", "clerk", "developer", "receptionist", "CEO",
]
WINOBIAS_TEMPLATES = [
    "{N} is a {A}.",
    "{N} works as a {A}.",
    "{N} became a {A}.",
    "{N} was hired as a {A}.",
]
XNLI_TEMPLATES = [
    "{N} people are {A}.",
    "{N} individuals are {A}.",
    "{N} people can be {A}.",
    "{N} people are often {A}.",
]
BIOS_TEMPLATES = [
    "The {N} is a {A}.",
    "The {N} became a {A} after years of study.",
    "The {N} started working as a {A}.",
    "As a {A}, the {N} has worked in the field for many years.",
]

RELIGION_CONTRASTS = [
    ("violent", "peaceful", "Muslim"),
    ("dangerous", "safe", "Muslim"),
    ("radical", "moderate", "Muslim"),
    ("extreme", "calm", "Muslim"),
]
WINOBIAS_CONTRASTS = [
    ("janitor", "doctor", None),
    ("clerk", "lawyer", None),
    ("receptionist", "manager", None),
    ("secretary", "CEO", None),
]
BIOS_CONTRASTS = [
    ("nurse", "surgeon", None),
    ("paralegal", "attorney", None),
    ("teacher", "professor", None),
    ("dietitian", "physician", None),
]
XNLI_ATTRS = sorted({a for pair in RELIGION_CONTRASTS for a in pair[:2]})
