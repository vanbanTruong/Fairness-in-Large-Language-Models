"""Equity Evaluation Corpus (EEC) loader."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

from fairLMs.datasets._sources import hub_file, none_if_na
from fairLMs.datasets.base import FairnessDataset, optional_limit

PathLike = Union[str, Path]

EEC_HF_PATH = "peixian/equity_evaluation_corpus"
EEC_FILENAME = "Equity-Evaluation-Corpus.csv"

# The release header carries a space ("Emotion word") and title case, neither of
# which survives attribute access; the loader renames once, here.
_COLUMNS = {
    "ID": "id",
    "Sentence": "sentence",
    "Template": "template",
    "Person": "person",
    "Gender": "gender",
    "Race": "race",
    "Emotion": "emotion",
    "Emotion word": "emotion_word",
}


class EquityEvaluationCorpus(FairnessDataset):
    """Load the Equity Evaluation Corpus as gender/race-varied sentences.

    Each example is a dict::

        {
            "id": str,
            "sentence": str,       # "Alonzo feels angry."
            "template": str,       # "<person subject> feels <emotion word>."
            "person": str,         # the substituted noun phrase or name
            "gender": str | None,  # male / female, None for gender-neutral rows
            "race": str | None,    # African-American / European, None otherwise
            "emotion": str | None, # anger, joy, fear, sadness
            "emotion_word": str | None,
            "bias_type": str,      # "race" for the name rows, else "gender"
        }

    EEC is a matched-template corpus: sentences differing only in the person
    term are meant to be scored and compared, so ``template`` plus
    ``emotion_word`` identifies a comparison group.

    Every row carries a gender label. Two thirds also carry a race label,
    because those rows substitute a racially associated first name, which is
    gendered too. ``bias_type`` records whether a row has that race label; the
    race rows support a gender contrast as well.

    Parameters
    ----------
    bias_type:
        ``"race"`` keeps the rows carrying a race label, ``"gender"`` the rows
        that vary gender alone (pronouns and noun phrases such as
        "this woman").
    root:
        A local EEC checkout, read as ``data/Equity-Evaluation-Corpus.csv``.
        Without one the CSV comes from the Hugging Face repo.
    """

    name = "eec"
    data_origin = "Hugging Face Hub, downloaded at first use"

    def __init__(
        self,
        bias_type: Optional[str] = None,
        root: Optional[PathLike] = None,
        n_max: Optional[int] = None,
        hf_path: str = EEC_HF_PATH,
        revision: Optional[str] = None,
    ):
        if bias_type not in (None, "gender", "race"):
            raise ValueError(
                f"EEC varies gender and race only; got bias_type={bias_type!r}."
            )
        self.bias_type = bias_type
        self.root = root
        self.n_max = n_max
        self.hf_path = hf_path
        self.revision = revision
        self._cache: Optional[List[dict]] = None

    def _source_path(self) -> Path:
        if self.root is None:
            return hub_file(self.hf_path, EEC_FILENAME, self.revision)
        base = Path(self.root).expanduser()
        for candidate in (base, base / "EEC"):
            for relative in (Path("data") / EEC_FILENAME, Path(EEC_FILENAME)):
                if (candidate / relative).exists():
                    return candidate / relative
        raise FileNotFoundError(
            f"EEC not found under {base}: expected data/{EEC_FILENAME}."
        )

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        df = pd.read_csv(self._source_path())
        df = df.rename(columns=_COLUMNS)

        examples: List[dict] = []
        for raw in df.to_dict("records"):
            # Per value rather than frame-wide: under pandas 3 a `str` column's
            # gaps survive `df.where(pd.notna(df), None)` as `nan`, and `nan is
            # not None` would put every row on the race axis.
            row = {key: none_if_na(value) for key, value in raw.items()}
            # The race half of the corpus is the first-name rows; the rest
            # substitute a pronoun or noun phrase and vary gender only.
            axis = "race" if row.get("race") is not None else "gender"
            if self.bias_type is not None and axis != self.bias_type:
                continue
            row["bias_type"] = axis
            examples.append(row)
            if self.n_max is not None and len(examples) >= self.n_max:
                self._cache = examples
                return examples

        self._cache = examples
        return optional_limit(examples, self.n_max)

    def emotions(self) -> List[str]:
        """Sorted emotion labels present (the neutral rows have none)."""
        examples = self._cache if self._cache is not None else self.load()
        return sorted({str(ex["emotion"]) for ex in examples if ex.get("emotion")})


#: Short alias, matching how the benchmark is cited.
EEC = EquityEvaluationCorpus
