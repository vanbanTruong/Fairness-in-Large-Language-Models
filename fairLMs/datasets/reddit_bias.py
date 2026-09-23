"""RedditBias loader."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from fairLMs.datasets._sources import require_choice, resolve_root
from fairLMs.datasets.base import FairnessDataset, optional_limit

PathLike = Union[str, Path]

REDDIT_BIAS_HOMEPAGE = "https://github.com/umanlp/RedditBias"

#: Per demographic axis: the processed comment corpus, the reduced
#: counterfactual pair tables, and the phrase-level annotation file. The file
#: names are the ones the upstream release ships under ``data/{axis}/``.
REDDIT_BIAS_AXES: Dict[str, Dict[str, object]] = {
    "gender": {
        "bias_type": "gender",
        "minority": "female",
        "majority": "male",
        "comments": "reddit_comments_gender_female_processed.csv",
        "pairs": (
            "reddit_comments_gender_male_biased_test_reduced.csv",
            "reddit_comments_gender_male_biased_valid_reduced.csv",
        ),
        "phrases": "reddit_comments_gender_female_processed_phrase_annotated.csv",
    },
    "race": {
        "bias_type": "race",
        "minority": "black",
        "majority": "white",
        "comments": "reddit_comments_race_black_processed.csv",
        "pairs": (
            "reddit_comments_race_white_biased_test_reduced.csv",
            "reddit_comments_race_white_biased_valid_reduced.csv",
        ),
        "phrases": "reddit_comments_race_black_processed_phrase_annotated.csv",
    },
    "orientation": {
        "bias_type": "sexual_orientation",
        "minority": "lgbtq",
        "majority": "straight",
        "comments": "reddit_comments_orientation_lgbtq_processed.csv",
        "pairs": (
            "reddit_comments_orientation_straight_biased_test_reduced.csv",
            "reddit_comments_orientation_straight_biased_valid_reduced.csv",
        ),
        "phrases": "reddit_comments_orientation_lgbtq_processed_phrase_annotated.csv",
    },
    "religion1": {
        "bias_type": "religion",
        "minority": "jews",
        "majority": "christians",
        "comments": "reddit_comments_religion1_christians_processed.csv",
        "pairs": (
            "reddit_comments_religion1_christians_biased_test_reduced.csv",
            "reddit_comments_religion1_christians_biased_valid_reduced.csv",
        ),
        "phrases": "reddit_comments_religion1_jews_processed_phrase_annotated.csv",
    },
    "religion2": {
        "bias_type": "religion",
        "minority": "muslims",
        "majority": "christians",
        "comments": "reddit_comments_religion2_christians_processed.csv",
        "pairs": (
            "reddit_comments_religion2_christians_biased_test_reduced.csv",
            "reddit_comments_religion2_christians_biased_valid_reduced.csv",
        ),
        "phrases": "reddit_comments_religion2_muslims_processed_phrase_annotated.csv",
    },
}

REDDIT_BIAS_SUBSETS = ("comments", "pairs", "phrases")


def _as_terms(value) -> List[str]:
    """``"['girl']"`` -> ``["girl"]``: the pair tables store the demographic
    slot as the repr of a Python list."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str):
        return []
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [text]
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    return [text]


def _read_csv(path: Path) -> pd.DataFrame:
    """Read one RedditBias CSV.

    The religion phrase-annotation files are Latin-1, not UTF-8, so a plain
    ``read_csv`` raises ``UnicodeDecodeError`` on them.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Could not decode {path}")


class RedditBias(FairnessDataset):
    """Load RedditBias comments, counterfactual pairs, or phrase annotations.

    Three different measurement objects live under one release, selected with
    ``subset``; they are not interchangeable.

    ``subset="comments"`` (default) returns the processed corpus::

        {"text": str, "axis": str, "group": str, "bias_type": str}

    ``subset="pairs"`` returns the reduced counterfactual tables, where the
    stored sentence is the *substituted* one and ``initial_demo`` /
    ``replaced_demo`` name the swap that produced it::

        {
            "text": str,              # comments_processed, after substitution
            "original": str,          # comments, before substitution
            "group_initial": list[str],   # the demographic terms swapped out
            "group_replaced": list[str],  # and the terms swapped in
            "perplexity": float,
            "axis": str, "bias_type": str, "source": str,
        }

    ``subset="phrases"`` returns the human phrase-level annotations::

        {"comment": str, "phrase": str, "bias_sent": int, "bias_phrase": int, ...}

    RedditBias is distributed from its own repository, not the Hugging Face
    Hub, so pass ``root=`` pointing at a checkout (or at the parent directory
    holding ``RedditBias/``).

    Parameters
    ----------
    axis:
        One of :data:`REDDIT_BIAS_AXES`. ``religion1`` is the Jewish/Christian
        axis and ``religion2`` the Muslim/Christian one.
    """

    name = "reddit_bias"
    data_origin = "local copy required (`root=`)"

    def __init__(
        self,
        root: Optional[PathLike] = None,
        axis: str = "gender",
        subset: str = "comments",
        n_max: Optional[int] = None,
    ):
        self.root = root
        self.axis = require_choice(axis, tuple(REDDIT_BIAS_AXES), "RedditBias axis")
        self.subset = require_choice(subset, REDDIT_BIAS_SUBSETS, "RedditBias subset")
        self.n_max = n_max
        self._cache: Optional[List[dict]] = None

    @property
    def spec(self) -> Dict[str, object]:
        return REDDIT_BIAS_AXES[self.axis]

    def _data_dir(self) -> Path:
        sentinel = Path("data") / self.axis / str(self.spec["comments"])
        base = resolve_root(
            self.root,
            dataset="RedditBias",
            directory="RedditBias",
            sentinels=(sentinel,),
            homepage=REDDIT_BIAS_HOMEPAGE,
        )
        return base / "data" / self.axis

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        data_dir = self._data_dir()
        spec = self.spec
        examples: List[dict] = []

        if self.subset == "comments":
            df = _read_csv(data_dir / str(spec["comments"]))
            for row in df.to_dict("records"):
                examples.append(
                    {
                        "text": row.get("comments_processed"),
                        "axis": self.axis,
                        "group": spec["minority"],
                        "bias_type": spec["bias_type"],
                    }
                )
                if self.n_max is not None and len(examples) >= self.n_max:
                    break

        elif self.subset == "pairs":
            for filename in spec["pairs"]:  # type: ignore[union-attr]
                path = data_dir / str(filename)
                if not path.exists():
                    raise FileNotFoundError(f"RedditBias pair table not found: {path}.")
                df = _read_csv(path)
                for row in df.to_dict("records"):
                    examples.append(
                        {
                            "text": row.get("comments_processed"),
                            "original": row.get("comments"),
                            "group_initial": _as_terms(row.get("initial_demo")),
                            "group_replaced": _as_terms(row.get("replaced_demo")),
                            "perplexity": row.get("perplexity"),
                            "axis": self.axis,
                            "bias_type": spec["bias_type"],
                            "source": str(filename),
                        }
                    )
                    if self.n_max is not None and len(examples) >= self.n_max:
                        break
                if self.n_max is not None and len(examples) >= self.n_max:
                    break

        else:  # phrases
            df = _read_csv(data_dir / str(spec["phrases"]))
            df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
            for row in df.to_dict("records"):
                row["axis"] = self.axis
                row["bias_type"] = spec["bias_type"]
                examples.append(row)
                if self.n_max is not None and len(examples) >= self.n_max:
                    break

        self._cache = examples
        return optional_limit(examples, self.n_max)
