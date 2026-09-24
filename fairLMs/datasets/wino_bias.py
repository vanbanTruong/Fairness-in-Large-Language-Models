"""WinoBias dataset loader and occupation lists."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from fairLMs.datasets._paths import dataset_resource_dir
from fairLMs.datasets._sources import require_choice
from fairLMs.datasets.base import FairnessDataset, optional_limit

PathLike = Union[str, Path]

WINOBIAS_CONFIGS = ("type1_anti", "type1_pro", "type2_anti", "type2_pro")
WINOBIAS_SPLITS = ("validation", "test")
WINOBIAS_HF_PATH = "uclanlp/wino_bias"
WINOBIAS_HF_REVISION = "3f31267586e4408e3b3f77ec22198fd24ea8dc1d"

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
    """Load the bundled WinoBias validation or test split.

    The eight small Parquet splits are distributed with FairLMs for offline
    use. Pass ``root=`` for another copy, or set ``hf_path=`` / ``revision=``
    to request a Hugging Face version explicitly.
    """

    name = "wino_bias"
    data_origin = "bundled with the package; optional Hugging Face/local override"

    def __init__(
        self,
        config: str = "type1_pro",
        split: str = "test",
        root: Optional[PathLike] = None,
        n_max: Optional[int] = None,
        hf_path: Optional[str] = None,
        revision: Optional[str] = None,
    ):
        self.config = require_choice(config, WINOBIAS_CONFIGS, "WinoBias config")
        self.split = require_choice(split, WINOBIAS_SPLITS, "WinoBias split")
        self.root = root
        self.n_max = n_max
        self.hf_path = hf_path
        self.revision = revision
        self._cache: Optional[List[dict]] = None

    @property
    def _filename(self) -> Path:
        return Path(self.config) / f"{self.split}-00000-of-00001.parquet"

    def _local_path(self) -> Path:
        if self.root is None:
            return dataset_resource_dir("wino_bias", [self._filename]) / self._filename

        base = Path(self.root).expanduser()
        for candidate in (base, base / "wino_bias", base / "WinoBias"):
            path = candidate / self._filename
            if path.is_file():
                return path.resolve()
        raise FileNotFoundError(
            f"WinoBias not found under {base}: expected {self._filename}."
        )

    def _load_local(self) -> List[dict]:
        frame = pd.read_parquet(self._local_path())
        examples: List[dict] = []
        for raw in frame.to_dict("records"):
            # Pandas returns Parquet list columns as ndarrays. Match the plain
            # lists returned by datasets.load_dataset so both sources expose
            # exactly the same public record shape.
            examples.append(
                {
                    key: value.tolist() if hasattr(value, "tolist") else value
                    for key, value in raw.items()
                }
            )
        return examples

    def _load_hf(self):
        from datasets import load_dataset

        errors = []
        candidates = [self.hf_path] if self.hf_path else [WINOBIAS_HF_PATH]
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

        if self.root is not None or (self.hf_path is None and self.revision is None):
            examples = self._load_local()
        else:
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
