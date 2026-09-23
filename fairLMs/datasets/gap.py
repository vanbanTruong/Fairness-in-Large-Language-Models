"""GAP gendered-ambiguous-pronoun coreference loader."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from fairLMs.datasets._sources import hub_file, require_choice
from fairLMs.datasets.base import FairnessDataset, optional_limit

PathLike = Union[str, Path]

GAP_HF_PATH = "google-research-datasets/gap"
# The repo still ships a loading script, which datasets>=3 will not run. Its
# auto-converted parquet branch holds the same rows and needs no script.
GAP_HF_REVISION = "refs/convert/parquet"

#: Split name -> (upstream TSV, path on the Hub parquet branch). Upstream calls
#: the 2000-row training split "development"; the Hub calls it "train".
GAP_SPLITS: Dict[str, tuple] = {
    "train": ("gap-development.tsv", "default/train/0000.parquet"),
    "validation": ("gap-validation.tsv", "default/validation/0000.parquet"),
    "test": ("gap-test.tsv", "default/test/0000.parquet"),
}

MASCULINE_PRONOUNS = {"he", "him", "his"}
FEMININE_PRONOUNS = {"she", "her", "hers"}


class GAP(FairnessDataset):
    """Load GAP ambiguous-pronoun coreference examples.

    Each example mirrors one release row, with a derived ``pronoun_gender``::

        {
            "ID": str,
            "Text": str,
            "Pronoun": str,
            "Pronoun-offset": int,
            "A": str, "A-offset": int, "A-coref": bool,
            "B": str, "B-offset": int, "B-coref": bool,
            "URL": str,
            "pronoun_gender": str,   # masculine / feminine
            "bias_type": "gender",
        }

    GAP is gender-balanced by construction, so the usual measurement is the gap
    between masculine and feminine resolution accuracy on the same task, not a
    score on the corpus as a whole.

    Parameters
    ----------
    split:
        One of :data:`GAP_SPLITS`. ``"development"`` is accepted as the
        upstream name for ``"train"``.
    pronoun_gender:
        Keep only ``"masculine"`` or ``"feminine"`` rows.
    root:
        A local GAP checkout, read as ``data/gap-{split}.tsv``. Without one the
        split comes from the Hugging Face parquet branch.
    """

    name = "gap"
    data_origin = "Hugging Face Hub, downloaded at first use"

    def __init__(
        self,
        split: str = "test",
        pronoun_gender: Optional[str] = None,
        root: Optional[PathLike] = None,
        n_max: Optional[int] = None,
        hf_path: str = GAP_HF_PATH,
        revision: str = GAP_HF_REVISION,
    ):
        if split == "development":
            split = "train"
        self.split = require_choice(split, tuple(GAP_SPLITS), "GAP split")
        if pronoun_gender not in (None, "masculine", "feminine"):
            raise ValueError(
                "GAP annotates masculine and feminine pronouns only; got "
                f"pronoun_gender={pronoun_gender!r}."
            )
        self.pronoun_gender = pronoun_gender
        self.root = root
        self.n_max = n_max
        self.hf_path = hf_path
        self.revision = revision
        self._cache: Optional[List[dict]] = None

    def _read(self) -> pd.DataFrame:
        tsv_name, parquet_name = GAP_SPLITS[self.split]
        if self.root is None:
            return pd.read_parquet(hub_file(self.hf_path, parquet_name, self.revision))
        base = Path(self.root).expanduser()
        for candidate in (base, base / "GAP"):
            for relative in (Path("data") / tsv_name, Path(tsv_name)):
                if (candidate / relative).exists():
                    return pd.read_csv(candidate / relative, sep="\t")
        raise FileNotFoundError(
            f"GAP not found under {base}: expected data/{tsv_name}."
        )

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        df = self._read()
        examples: List[dict] = []
        for row in df.to_dict("records"):
            pronoun = str(row.get("Pronoun", "")).strip().lower()
            if pronoun in MASCULINE_PRONOUNS:
                gender = "masculine"
            elif pronoun in FEMININE_PRONOUNS:
                gender = "feminine"
            else:
                gender = None
            if self.pronoun_gender is not None and gender != self.pronoun_gender:
                continue
            # The TSV spells the coreference labels TRUE/FALSE; the parquet
            # branch already has bools. Normalise so both paths agree.
            for key in ("A-coref", "B-coref"):
                value = row.get(key)
                if isinstance(value, str):
                    row[key] = value.strip().upper() == "TRUE"
                else:
                    row[key] = bool(value)
            row["pronoun_gender"] = gender
            row["bias_type"] = "gender"
            examples.append(row)
            if self.n_max is not None and len(examples) >= self.n_max:
                self._cache = examples
                return examples

        self._cache = examples
        return optional_limit(examples, self.n_max)

    def pronoun_gender_counts(self) -> Dict[str, int]:
        """Masculine/feminine row counts, the balance GAP is built around."""
        counts: Dict[str, int] = {}
        for ex in self._cache if self._cache is not None else self.load():
            key = ex.get("pronoun_gender")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: str(kv[0])))
