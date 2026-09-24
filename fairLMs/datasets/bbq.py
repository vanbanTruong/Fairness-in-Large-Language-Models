"""BBQ dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

from fairLMs.datasets.base import FairnessDataset, optional_limit
from fairLMs.datasets._sources import hub_file

PathLike = Union[str, Path]

DEFAULT_BBQ_CATEGORIES = (
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Religion",
    "SES",
    "Sexual_orientation",
)


class BBQ(FairnessDataset):
    """Load BBQ examples from the Hub cache or a local JSONL directory.

    Each example is the original BBQ row dict plus::

        {"category": "<CategoryName>"}
    """

    name = "bbq"
    data_origin = "Hugging Face Hub, downloaded at first use"

    def __init__(
        self,
        data_dir: Optional[PathLike] = None,
        categories: Optional[Sequence[str]] = None,
        context_condition: Optional[str] = None,
        n_max: Optional[int] = None,
        hf_path: str = "heegyu/bbq",
        revision: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.categories = [
            name.removesuffix(".jsonl")
            for name in (
                list(categories)
                if categories is not None
                else list(DEFAULT_BBQ_CATEGORIES)
            )
        ]
        self.context_condition = context_condition
        self.n_max = n_max
        self.hf_path = hf_path
        self.revision = revision
        self._cache: Optional[List[dict]] = None
        self._resolved_files: dict[str, Path] = {}

    def _category_file(self, category: str) -> Path:
        if category in self._resolved_files:
            return self._resolved_files[category]
        if self.data_dir is None:
            resolved = hub_file(
                self.hf_path,
                f"data/{category}.jsonl",
                self.revision,
            )
        else:
            root = Path(self.data_dir).expanduser()
            if not root.is_dir():
                raise FileNotFoundError(f"BBQ data directory not found: {root}")
            resolved = root / f"{category}.jsonl"
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"BBQ category {category!r} not found under {root}: {resolved}"
                )
        self._resolved_files[category] = resolved
        return resolved

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        examples: List[dict] = []
        for category_name in self.categories:
            path = self._category_file(category_name)
            category = path.stem
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if (
                        self.context_condition is not None
                        and row.get("context_condition") != self.context_condition
                    ):
                        continue
                    row = dict(row)
                    row["category"] = category
                    examples.append(row)
                    if self.n_max is not None and len(examples) >= self.n_max:
                        self._cache = examples
                        return examples

        self._cache = examples
        return optional_limit(examples, self.n_max)

    @property
    def directory(self) -> Path:
        """Directory containing the requested local or cached category files."""
        files = [self._category_file(category) for category in self.categories]
        if not files:
            raise ValueError("BBQ requires at least one category.")
        return files[0].parent
