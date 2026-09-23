"""Grep-BiasIR loader."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from fairLMs.datasets._sources import require_choice, resolve_root
from fairLMs.datasets.base import FairnessDataset, optional_limit

PathLike = Union[str, Path]

GREP_BIASIR_HOMEPAGE = "https://github.com/KlaraKrieg/GrepBiasIR"

#: The seven societal domains the queries are drawn from.
GREP_BIASIR_DOMAINS = (
    "Appearance",
    "Career",
    "Child Care",
    "Cognitive Capabilities",
    "Domestic Work",
    "Physical Capabilities",
    "Sex & Relationship",
)

GREP_BIASIR_CONFIGS = ("documents", "queries")

#: The document-side gender marking; ``N`` is the gender-neutral writing. The
#: released files also carry five rows labelled ``both`` and ``botrh``, so
#: ``content_gender`` is not validated against this tuple -- filtering on a
#: value that is not here is how you reach those rows.
GREP_BIASIR_DOCUMENT_GENDERS = ("F", "M", "N")


class GrepBiasIR(FairnessDataset):
    """Load Grep-BiasIR gender-annotated query/document pairs.

    ``config="documents"`` (default) returns one example per relevance
    judgement::

        {
            "q_id": int, "d_id": int,
            "query": str,
            "doc_title": str, "document": str,
            "relevant": int,             # 1 relevant, 0 non-relevant
            "content_gender": str,       # F / M / N (see the constant)
            "exp_stereotype": str,       # the annotators' expected direction
            "domain": str,
            "bias_type": "gender",
        }

    Each query comes with the same document written three ways -- female,
    male and neutral -- so a retrieval system's ranking difference across them
    is attributable to the gendered wording rather than to relevance.

    ``config="queries"`` returns the 117 released queries alone::

        {"q_id": int, "category": str, "query": str, "bias_type": "gender"}

    Grep-BiasIR is distributed from its own repository, not the Hugging Face
    Hub, so pass ``root=`` pointing at a checkout (or at the parent directory
    holding ``Grep-BiasIR/``).
    """

    name = "grep_biasir"
    data_origin = "local copy required (`root=`)"

    def __init__(
        self,
        root: Optional[PathLike] = None,
        config: str = "documents",
        domains: Optional[Sequence[str]] = None,
        content_gender: Optional[str] = None,
        n_max: Optional[int] = None,
    ):
        self.root = root
        self.config = require_choice(config, GREP_BIASIR_CONFIGS, "Grep-BiasIR config")
        self.domains = (
            [
                require_choice(d, GREP_BIASIR_DOMAINS, "Grep-BiasIR domain")
                for d in domains
            ]
            if domains is not None
            else list(GREP_BIASIR_DOMAINS)
        )
        self.content_gender = content_gender
        self.n_max = n_max
        self._cache: Optional[List[dict]] = None

    def _data_dir(self) -> Path:
        base = resolve_root(
            self.root,
            dataset="Grep-BiasIR",
            directory="Grep-BiasIR",
            sentinels=(Path("data") / "queries.csv",),
            homepage=GREP_BIASIR_HOMEPAGE,
        )
        return base / "data"

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        data_dir = self._data_dir()
        examples: List[dict] = []

        if self.config == "queries":
            df = pd.read_csv(data_dir / "queries.csv")
            for row in df.to_dict("records"):
                if row.get("category") not in self.domains:
                    continue
                row["bias_type"] = "gender"
                examples.append(row)
                if self.n_max is not None and len(examples) >= self.n_max:
                    break
        else:
            for domain in self.domains:
                path = data_dir / f"queries-documents_{domain}.csv"
                if not path.exists():
                    raise FileNotFoundError(
                        f"Grep-BiasIR domain file not found: {path}."
                    )
                df = pd.read_csv(path)
                for row in df.to_dict("records"):
                    if (
                        self.content_gender is not None
                        and row.get("content_gender") != self.content_gender
                    ):
                        continue
                    row["domain"] = domain
                    row["bias_type"] = "gender"
                    examples.append(row)
                    if self.n_max is not None and len(examples) >= self.n_max:
                        break
                if self.n_max is not None and len(examples) >= self.n_max:
                    break

        self._cache = examples
        return optional_limit(examples, self.n_max)

    def content_gender_counts(self) -> Dict[str, int]:
        """Row counts per document gender writing, which should be balanced."""
        counts: Dict[str, int] = {}
        for ex in self._cache if self._cache is not None else self.load():
            key = ex.get("content_gender")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: str(kv[0])))
