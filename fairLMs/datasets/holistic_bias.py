"""HolisticBias loader."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

from fairLMs.datasets._sources import hub_file, none_if_na, require_choice
from fairLMs.datasets.base import FairnessDataset, optional_limit

PathLike = Union[str, Path]

HOLISTIC_BIAS_HF_PATH = "fairnlp/holistic-bias"

#: ``sentences`` is the ~473k-row sentence release; ``nouns`` is the 18k-row
#: descriptor/noun-phrase inventory those sentences are built from.
HOLISTIC_BIAS_CONFIGS = {"sentences": "sentences.csv", "nouns": "nouns.csv"}

_CHUNK_ROWS = 50_000


class HolisticBias(FairnessDataset):
    """Load HolisticBias sentences or their descriptor/noun inventory.

    Each example mirrors one row of the release, with ``bias_type`` set to the
    row's demographic ``axis``. For ``config="sentences"``::

        {
            "text": str,            # "I'm a wheelchair user."
            "axis": str,            # ability, gender_and_sex, race_ethnicity, ...
            "bucket": str,
            "descriptor": str,
            "noun": str,
            "noun_phrase": str,
            "template": str,
            "bias_type": str,       # == axis
            ...
        }

    ``config="nouns"`` returns the same columns minus ``text``/``template``.

    The sentence release is ~100 MB. ``n_max`` is applied while reading, so a
    small sample does not parse the whole file.

    Parameters
    ----------
    config:
        ``"sentences"`` (default) or ``"nouns"``.
    axes:
        Keep only these demographic axes.
    root:
        A local HolisticBias checkout, read as ``data/{config}.csv``. Without
        one the file comes from the Hugging Face repo.
    """

    name = "holistic_bias"
    data_origin = "Hugging Face Hub, downloaded at first use"

    def __init__(
        self,
        config: str = "sentences",
        axes: Optional[Sequence[str]] = None,
        root: Optional[PathLike] = None,
        n_max: Optional[int] = None,
        hf_path: str = HOLISTIC_BIAS_HF_PATH,
        revision: Optional[str] = None,
    ):
        self.config = require_choice(
            config, tuple(HOLISTIC_BIAS_CONFIGS), "HolisticBias config"
        )
        self.axes = list(axes) if axes is not None else None
        self.root = root
        self.n_max = n_max
        self.hf_path = hf_path
        self.revision = revision
        self._cache: Optional[List[dict]] = None

    def _source_path(self) -> Path:
        filename = HOLISTIC_BIAS_CONFIGS[self.config]
        if self.root is None:
            return hub_file(self.hf_path, filename, self.revision)
        base = Path(self.root).expanduser()
        for candidate in (base, base / "HolisticBias"):
            for relative in (Path("data") / filename, Path(filename)):
                if (candidate / relative).exists():
                    return candidate / relative
        raise FileNotFoundError(
            f"HolisticBias not found under {base}: expected data/{filename}."
        )

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        examples: List[dict] = []
        reader = pd.read_csv(self._source_path(), chunksize=_CHUNK_ROWS)
        for chunk in reader:
            if self.axes is not None:
                chunk = chunk[chunk["axis"].isin(self.axes)]
            for row in chunk.to_dict("records"):
                # Mirrored columns stay verbatim; the derived one is normalised
                # so a missing axis reads as None rather than nan.
                row["bias_type"] = none_if_na(row.get("axis"))
                examples.append(row)
                if self.n_max is not None and len(examples) >= self.n_max:
                    self._cache = examples
                    return examples

        self._cache = examples
        return optional_limit(examples, self.n_max)

    def axis_counts(self) -> dict:
        """Example count per demographic axis, for representativeness checks."""
        counts: dict = {}
        for ex in self._cache if self._cache is not None else self.load():
            axis = ex.get("axis")
            counts[axis] = counts.get(axis, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: str(kv[0])))
