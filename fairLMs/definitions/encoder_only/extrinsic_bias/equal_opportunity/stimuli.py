"""Original Equal Opportunity occupation / BBQ helpers from e532af2."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

BIOS_OCCUPATIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker", "interior designer",
    "journalist", "lawyer", "model", "nurse", "painter", "paralegal",
    "pastor", "personal trainer", "photographer", "physician", "poet",
    "professor", "psychologist", "rapper", "software engineer",
    "surgeon", "teacher",
]

MALE_PRONOUNS = {"he", "him", "his"}
FEMALE_PRONOUNS = {"she", "her", "hers"}


def _answer_tag(meta) -> Optional[str]:
    if isinstance(meta, (list, tuple)) and len(meta) > 1:
        tag = str(meta[1]).strip()
        if tag and "unknown" not in tag.lower():
            return tag
    return None


def bbq_disambig_examples(rows: Sequence[dict], n_max: int = 64) -> List[dict]:
    """Keep BBQ disambiguated rows with a gold non-unknown answer tag."""
    out = []
    for row in rows:
        if row.get("context_condition") != "disambig":
            continue
        label = row.get("label")
        if label not in (0, 1, 2):
            continue
        info = row.get("answer_info", {})
        key = f"ans{label}"
        tag = _answer_tag(info.get(key))
        if tag is None:
            continue
        gold = str(row.get(key, "")).strip()
        # pick a distractor among the other non-unknown answers
        distractors = []
        for i in (0, 1, 2):
            if i == label:
                continue
            t = _answer_tag(info.get(f"ans{i}"))
            if t is None:
                continue
            distractors.append(str(row.get(f"ans{i}", "")).strip())
        if not gold or not distractors:
            continue
        out.append(
            {
                "premise": f"{row.get('context', '').strip()} {row.get('question', '').strip()}",
                "gold_hyp": gold,
                "distractor_hyp": distractors[0],
                "group": tag,
            }
        )
        if len(out) >= n_max:
            break
    return out
