"""WinoBias dataset loader and occupation lists."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from fairLMs.datasets.base import FairnessDataset, optional_limit

WINOBIAS_MALE_OCC = [
    "driver",
    "supervisor",
    "janitor",
    "cook",
    "mover",
    "laborer",
    "constructor",
    "chief",
    "developer",
    "carpenter",
    "manager",
    "lawyer",
    "farmer",
    "salesperson",
    "physician",
    "guard",
    "analyst",
    "mechanic",
    "sheriff",
    "ceo",
]

WINOBIAS_FEMALE_OCC = [
    "attendant",
    "cashier",
    "teacher",
    "nurse",
    "assistant",
    "secretary",
    "auditor",
    "cleaner",
    "receptionist",
    "clerk",
    "counselor",
    "designer",
    "hairdresser",
    "writer",
    "housekeeper",
    "baker",
    "accountant",
    "editor",
    "librarian",
    "tailor",
]


class WinoBias(FairnessDataset):
    """Load WinoBias from Hugging Face.

    Tries ``wino_bias`` then ``uclanlp/wino_bias``.
    """

    name = "wino_bias"
    data_origin = "Hugging Face Hub, downloaded at first use"

    def __init__(
        self,
        config: str = "type1_pro",
        split: str = "test",
        n_max: Optional[int] = None,
        hf_path: Optional[str] = None,
        revision: Optional[str] = None,
    ):
        self.config = config
        self.split = split
        self.n_max = n_max
        self.hf_path = hf_path
        self.revision = revision
        self._cache: Optional[List[dict]] = None

    def _load_hf(self):
        from datasets import load_dataset

        errors = []
        candidates = (
            [self.hf_path] if self.hf_path else ["wino_bias", "uclanlp/wino_bias"]
        )
        for path in candidates:
            if not path:
                continue
            try:
                loaded = load_dataset(
                    path, self.config, split=self.split, revision=self.revision
                )
                self._resolved_hf_path = path
                self._resolved_fingerprint = getattr(loaded, "_fingerprint", None)
                return loaded
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: {exc}")
        raise RuntimeError(
            "Failed to load WinoBias from Hugging Face. Tried: " + "; ".join(errors)
        )

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        ds = self._load_hf()
        examples = [dict(row) for row in ds]
        self._cache = examples
        return optional_limit(examples, self.n_max)

    def attributes_and_direction(self) -> Tuple[List[str], Dict[str, int]]:
        """Occupations present in the split with stereotypical direction.

        Direction: +1 male-stereotyped, -1 female-stereotyped.
        """
        examples = self.load() if self._cache is None else self._cache
        present = set()
        for ex in examples:
            tokens = ex.get("tokens") or []
            present.update(t.lower() for t in tokens)
        male = [o for o in WINOBIAS_MALE_OCC if o in present]
        female = [o for o in WINOBIAS_FEMALE_OCC if o in present]
        direction = {o: +1 for o in male}
        direction.update({o: -1 for o in female})
        return male + female, direction
