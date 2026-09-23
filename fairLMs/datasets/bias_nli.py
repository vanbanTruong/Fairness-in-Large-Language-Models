"""Bias-NLI loader."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from fairLMs.datasets._sources import require_choice, resolve_root
from fairLMs.datasets.base import FairnessDataset, optional_limit

PathLike = Union[str, Path]

BIAS_NLI_HOMEPAGE = "https://github.com/sunipa/On-Measuring-and-Mitigating-Biased-Inferences-of-Word-Embeddings"

BIAS_NLI_SPLITS = ("train", "validation", "test")

#: SNLI's label order, which the Bias-NLI release inherits.
BIAS_NLI_LABELS = ("entailment", "neutral", "contradiction")


class BiasNLI(FairnessDataset):
    """Load the retained Bias-NLI release as premise/hypothesis pairs.

    Each example is a dict::

        {
            "premise": str,
            "hypothesis": str,
            "label": int,        # 0 entailment, 1 neutral, 2 contradiction
            "label_name": str,
        }

    What this returns is the three-column release archived under the Bias-NLI
    name, not the template-expanded evaluation set of Dev et al. (2020). The
    two are distributed separately and there is no deterministic join between
    them, so the loader keeps the release verbatim and claims nothing about the
    construction of individual rows. The generator archive (``generate_templates.py``,
    the stereotype lists and the word lists) is a separate object this loader
    does not read; expand it upstream if you need the neutral-by-construction
    inference pairs.

    Bias-NLI has no canonical Hugging Face release -- the mirrors differ in row
    count because they come from different generation or export runs -- so this
    loader reads a local copy. Pass ``root=`` pointing at a directory holding
    either ``{split}-00000-of-00001.parquet`` or
    ``csv/{split}-00000-of-00001.csv``.
    """

    name = "bias_nli"
    data_origin = "local copy required (`root=`)"

    def __init__(
        self,
        root: Optional[PathLike] = None,
        split: str = "test",
        n_max: Optional[int] = None,
    ):
        self.root = root
        self.split = require_choice(split, BIAS_NLI_SPLITS, "Bias-NLI split")
        self.n_max = n_max
        self._cache: Optional[List[dict]] = None

    def _read(self) -> pd.DataFrame:
        stem = f"{self.split}-00000-of-00001"
        parquet = Path(f"{stem}.parquet")
        csv = Path("csv") / f"{stem}.csv"
        base = resolve_root(
            self.root,
            dataset="Bias-NLI",
            directory="Bias-NLI",
            sentinels=(parquet, csv),
            homepage=BIAS_NLI_HOMEPAGE,
        )
        if (base / parquet).exists():
            return pd.read_parquet(base / parquet)
        return pd.read_csv(base / csv)

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        df = self._read()
        examples: List[dict] = []
        for row in df.to_dict("records"):
            label = row.get("label")
            label = int(label) if pd.notna(label) else None
            examples.append(
                {
                    "premise": row.get("premise"),
                    "hypothesis": row.get("hypothesis"),
                    "label": label,
                    "label_name": (
                        BIAS_NLI_LABELS[label]
                        if label is not None and 0 <= label < len(BIAS_NLI_LABELS)
                        else None
                    ),
                }
            )
            if self.n_max is not None and len(examples) >= self.n_max:
                self._cache = examples
                return examples

        self._cache = examples
        return optional_limit(examples, self.n_max)

    def label_counts(self) -> Dict[str, int]:
        """Count per released label."""
        counts: Dict[str, int] = {}
        for ex in self._cache if self._cache is not None else self.load():
            key = ex.get("label_name")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: str(kv[0])))
