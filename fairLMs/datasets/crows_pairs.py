"""CrowS-Pairs dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

from fairLMs.datasets.base import FairnessDataset, optional_limit
from fairLMs.datasets._paths import resolve_crows_pairs_csv

PathLike = Union[str, Path]


class CrowSPairs(FairnessDataset):
    """Load CrowS-Pairs as stereotype / anti-stereotype sentence pairs.

    ``stereotype`` is always the dataset's ``sent_more`` (the more stereotyping
    sentence) and ``anti_stereotype`` its ``sent_less``, for stereo and
    anti-stereo rows alike, matching the published metric. Each example is a
    dict::

        {
            "stereotype": str,
            "anti_stereotype": str,
            "bias_type": str,
            "stereo_antistereo": str,  # original CrowS label
        }
    """

    name = "crows_pairs"
    data_origin = "bundled with the package"

    def __init__(
        self,
        path: Optional[PathLike] = None,
        bias_type: Optional[str] = None,
        n_max: Optional[int] = None,
    ):
        self.path = path
        self.bias_type = bias_type
        self.n_max = n_max
        self._cache: Optional[List[dict]] = None

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        csv_path = resolve_crows_pairs_csv(self.path)
        df = pd.read_csv(csv_path)
        if self.bias_type is not None:
            df = df[df["bias_type"] == self.bias_type]

        # ``sent_more`` is the more stereotyping sentence in *every* row; the
        # ``stereo_antistereo`` label says whether the pair demonstrates or
        # violates a stereotype about a disadvantaged group, not which side is
        # more stereotypical. The published metric (Nangia et al., 2020,
        # metric.py) therefore counts a preference for ``sent_more`` for both
        # labels, and so does this loader. Swapping the sides for the 218
        # anti-stereotype rows, as an earlier version did, inverted their
        # contribution and lowered the bert-base-uncased score from the
        # published 60.5 to 58.5.
        pairs: List[dict] = []
        for _, row in df.iterrows():
            pairs.append(
                {
                    "stereotype": row["sent_more"],
                    "anti_stereotype": row["sent_less"],
                    "bias_type": row.get("bias_type"),
                    "stereo_antistereo": row.get("stereo_antistereo"),
                }
            )

        self._cache = pairs
        return optional_limit(pairs, self.n_max)

    def bias_types(self) -> List[str]:
        """Return sorted bias_type labels present in the full CSV (ignores n_max)."""
        full = CrowSPairs(path=self.path, bias_type=None, n_max=None)
        examples = full.load()
        return sorted(
            {
                ex["bias_type"]
                for ex in examples
                if ex.get("bias_type") is not None and str(ex["bias_type"]) != "nan"
            }
        )
