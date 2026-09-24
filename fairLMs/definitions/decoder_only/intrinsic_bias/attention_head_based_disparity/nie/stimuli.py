"""Original NIE occupation / probe construction from e532af2."""

from __future__ import annotations

from typing import Dict, List, Tuple

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

WINO_OCCS = [
    "nurse", "surgeon", "receptionist", "engineer", "librarian", "lawyer",
    "teacher", "developer", "housekeeper", "physician", "secretary", "manager",
    "cashier", "carpenter", "counselor", "supervisor", "hairdresser", "analyst",
    "attendant", "janitor", "doctor", "technician", "administrator", "paramedic",
]


def stereo_spec(occ: str):
    o = occ.lower().strip()
    if o in MALE_STEREO_OCCS:
        return {"stereo": "he", "anti": "she", "cf_noun": "woman"}
    if o in FEMALE_STEREO_OCCS:
        return {"stereo": "she", "anti": "he", "cf_noun": "man"}
    return None


def occupation_prompts(n_max: int = 24) -> List[Tuple[str, str, str, str]]:
    """Build (prefix, cf_prefix, stereo, anti) prompts from occupation lists."""
    prompts = []
    for occ in WINO_OCCS:
        spec = stereo_spec(occ)
        if spec is None:
            continue
        prefix = f"The {occ} said that"
        cf_prefix = f"The {spec['cf_noun']} said that"
        prompts.append((prefix, cf_prefix, spec["stereo"], spec["anti"]))
        if len(prompts) >= n_max:
            break
    return prompts


def prompts_to_probes(prompts, tokenizer) -> List[Dict]:
    probes = []
    for null_prompt, set_prompt, stereo_word, anti_word in prompts:
        stereo_ids = tokenizer.encode(" " + stereo_word, add_special_tokens=False)
        anti_ids = tokenizer.encode(" " + anti_word, add_special_tokens=False)
        if not stereo_ids or not anti_ids:
            continue
        probes.append(
            {
                "prompt": null_prompt,
                "cf_text": set_prompt,
                "stereo_token_id": stereo_ids[0],
                "anti_token_id": anti_ids[0],
            }
        )
    return probes
