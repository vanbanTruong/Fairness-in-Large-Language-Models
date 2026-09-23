"""Bias-in-Bios dataset loader and profession helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from fairLMs.datasets.base import FairnessDataset, optional_limit

# De-Arteaga et al. / LabHC profession id → surface form
BIOS_PROFESSION_MAP = {
    0: "accountant",
    1: "architect",
    2: "attorney",
    3: "chiropractor",
    4: "comedian",
    5: "composer",
    6: "dentist",
    7: "dietitian",
    8: "dj",
    9: "filmmaker",
    10: "interior_designer",
    11: "journalist",
    12: "model",
    13: "nurse",
    14: "painter",
    15: "paralegal",
    16: "pastor",
    17: "personal_trainer",
    18: "photographer",
    19: "physician",
    20: "poet",
    21: "professor",
    22: "psychologist",
    23: "rapper",
    24: "software_engineer",
    25: "surgeon",
    26: "teacher",
    27: "yoga_teacher",
}


class BiasInBios(FairnessDataset):
    """Load the Hugging Face ``LabHC/bias_in_bios`` dataset.

    Each example mirrors the HF row, with an added ``profession_name`` field
    when the profession id is known.
    """

    name = "bias_in_bios"
    data_origin = "Hugging Face Hub, downloaded at first use"

    def __init__(
        self,
        split: str = "test",
        n_max: Optional[int] = None,
        hf_path: str = "LabHC/bias_in_bios",
        revision: Optional[str] = None,
    ):
        self.split = split
        self.n_max = n_max
        self.hf_path = hf_path
        self.revision = revision
        self._cache: Optional[List[dict]] = None
        self._raw = None

    def _load_hf(self):
        from datasets import load_dataset

        loaded = load_dataset(self.hf_path, split=self.split, revision=self.revision)
        self._resolved_hf_path = self.hf_path
        self._resolved_fingerprint = getattr(loaded, "_fingerprint", None)
        return loaded

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        ds = self._load_hf()
        self._raw = ds
        examples: List[dict] = []
        for row in ds:
            item = dict(row)
            code = item.get("profession")
            if code in BIOS_PROFESSION_MAP:
                item["profession_name"] = BIOS_PROFESSION_MAP[code].replace("_", " ")
            examples.append(item)
            if self.n_max is not None and len(examples) >= self.n_max:
                break

        self._cache = examples
        return optional_limit(examples, self.n_max)

    def profession_gender_skew(self, male_gender_code: int = 0) -> Dict[str, float]:
        """Return P(male | profession) using ``gender == male_gender_code``.

        De-Arteaga encoding used in this repo: gender 0 = male, 1 = female.
        """
        if self._raw is None:
            self.load()
        df = self._raw.to_pandas()
        skew: Dict[str, float] = {}
        for code, name in BIOS_PROFESSION_MAP.items():
            sub = df[df["profession"] == code]
            if len(sub) == 0:
                continue
            surface = name.replace("_", " ")
            skew[surface] = float((sub["gender"] == male_gender_code).mean())
        return skew

    def attributes_and_skew(
        self, validate_encoding: bool = True
    ) -> Tuple[List[str], Dict[str, float]]:
        skew = self.profession_gender_skew()
        if validate_encoding:
            n, s = skew.get("nurse"), skew.get("surgeon")
            if n is not None and s is not None and not (n < 0.5 < s):
                raise RuntimeError(
                    f"Gender-encoding sanity check FAILED: P(male|nurse)={n:.2f}, "
                    f"P(male|surgeon)={s:.2f}. Expected nurse<0.5<surgeon."
                )
        return list(skew.keys()), skew
