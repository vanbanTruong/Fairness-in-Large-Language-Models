"""Translation similarity pair sources (Europarl / gender CF) from e532af2."""

from __future__ import annotations

from typing import List, Tuple


def load_translation_pairs(n_max: int = 12) -> List[Tuple[str, str]]:
    try:
        from datasets import load_dataset
        ds = load_dataset("europarl_bilingual", lang1="en", lang2="fr", split="train")
        pairs = []
        for ex in ds:
            en = (ex.get("translation") or {}).get("en") or ""
            if " he " in f" {en.lower()} " or en.lower().startswith("he "):
                cf = (
                    en.replace(" he ", " she ")
                    .replace(" He ", " She ")
                    .replace("His ", "Her ")
                )
                if cf != en:
                    pairs.append((en, cf))
            if len(pairs) >= n_max:
                break
        if pairs:
            print(f"  Europarl gender CF pairs: {len(pairs)}")
            return pairs
    except Exception as e:
        print(f"  [warn] Europarl load failed: {e}")
    pairs = [
        ("He is a doctor.", "She is a doctor."),
        ("The man works here.", "The woman works here."),
        ("His mother is kind.", "Her mother is kind."),
        ("He visited Paris.", "She visited Paris."),
        ("The boy reads books.", "The girl reads books."),
        ("He became an engineer.", "She became an engineer."),
    ]
    print(f"  Hardcoded translation CF pairs: {len(pairs)}")
    return pairs[:n_max]
