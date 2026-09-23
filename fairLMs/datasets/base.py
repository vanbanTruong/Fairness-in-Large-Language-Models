"""Base protocol for reusable fairness datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Sequence


class FairnessDataset(ABC):
    """Minimal dataset interface used by shared loaders.

    Concrete datasets return a list (or other sequence) of example dicts from
    :meth:`load`. Example schemas are documented on each subclass.
    """

    name: str = "dataset"

    #: Where :meth:`load` gets its bytes, as the generated Loaders table prints
    #: it. Declared rather than inferred: the loaders reach the same data three
    #: different ways (bundled file, Hub download, caller-supplied directory)
    #: and no single import or call name distinguishes them.
    data_origin: str = "bundled with the package"

    @abstractmethod
    def load(self) -> Sequence[Any]:
        """Load and return examples for this dataset."""

    def __iter__(self) -> Iterable[Any]:
        return iter(self.load())

    def __len__(self) -> int:
        return len(self.load())


class StereotypePairExample(dict):
    """Dict with keys: stereotype, anti_stereotype, bias_type (optional)."""


class StereotypeTripleExample(dict):
    """Dict with keys: stereotype, anti_stereotype, unrelated, bias_type."""


class BBQExample(dict):
    """Dict mirroring one BBQ jsonl row plus ``category``."""


def optional_limit(examples: Sequence[Any], n_max: Optional[int]) -> Sequence[Any]:
    if n_max is None:
        return examples
    return examples[:n_max]
