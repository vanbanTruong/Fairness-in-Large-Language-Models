"""BBQ dataset loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

from fairLMs.datasets.base import FairnessDataset, optional_limit
from fairLMs.utils.paths import bbq_category_files, resolve_bbq_dir

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
    """Load BBQ examples from bundled (or legacy) jsonl category files.

    Each example is the original BBQ row dict plus::

        {"category": "<CategoryName>"}
    """

    name = "bbq"
    data_origin = "bundled with the package"

    def __init__(
        self,
        data_dir: Optional[PathLike] = None,
        categories: Optional[Sequence[str]] = None,
        context_condition: Optional[str] = None,
        n_max: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.categories = list(categories) if categories is not None else list(
            DEFAULT_BBQ_CATEGORIES
        )
        self.context_condition = context_condition
        self.n_max = n_max
        self._cache: Optional[List[dict]] = None

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        files = bbq_category_files(self.data_dir, self.categories)
        examples: List[dict] = []
        for path in files:
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
        return resolve_bbq_dir(self.data_dir)
