"""Winogender schemas loader and occupation statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from fairLMs.datasets._paths import dataset_resource_dir
from fairLMs.datasets._sources import hub_file, none_if_na
from fairLMs.datasets.base import FairnessDataset, optional_limit

PathLike = Union[str, Path]

WINOGENDER_HF_PATH = "oskarvanderwal/winogender"
WINOGENDER_HF_FILENAME = "test.tsv"
WINOGENDER_BUNDLED_REVISION = "1c7f8b481ad8a234b41e9f76a424d6e856e13f7f"

SENTENCES_FILE = "all_sentences.tsv"
OCCUPATION_STATS_FILE = "occupations-stats.tsv"
TEMPLATES_FILE = "templates.tsv"

#: Winogender writes each template three ways, one per pronoun set.
WINOGENDER_GENDERS = ("female", "male", "neutral")


class Winogender(FairnessDataset):
    """Load Winogender schemas as pronoun-resolution sentences.

    Each example is a dict::

        {
            "sentid": str,       # "technician.customer.1.male.txt"
            "sentence": str,
            "occupation": str,   # the stereotyped-profession referent
            "participant": str,  # the other referent
            "answer": int,       # 0 = occupation, 1 = participant
            "gender": str,       # female / male / neutral
            "bias_type": "gender",
        }

    The three gendered writings of a template are otherwise identical, so a
    resolution difference across them is attributable to the pronoun alone.

    Parameters
    ----------
    gender:
        Keep only one of :data:`WINOGENDER_GENDERS`.
    root:
        A local Winogender checkout, read as ``data/all_sentences.tsv``.
        Without one, the complete bundled snapshot supplies sentences,
        templates and occupation statistics.
    """

    name = "winogender"
    data_origin = "bundled with the package; optional Hugging Face/local override"

    def __init__(
        self,
        gender: Optional[str] = None,
        root: Optional[PathLike] = None,
        n_max: Optional[int] = None,
        hf_path: str = WINOGENDER_HF_PATH,
        revision: Optional[str] = None,
    ):
        if gender is not None and gender not in WINOGENDER_GENDERS:
            raise ValueError(
                f"Unknown Winogender gender {gender!r}. Expected one of: "
                f"{', '.join(WINOGENDER_GENDERS)}."
            )
        self.gender = gender
        self.root = root
        self.n_max = n_max
        self.hf_path = hf_path
        self.revision = revision
        self._cache: Optional[List[dict]] = None

    def _uses_bundled_snapshot(self) -> bool:
        return (
            self.root is None
            and self.hf_path == WINOGENDER_HF_PATH
            and self.revision is None
        )

    def _data_dir(self) -> Path:
        if self.root is None:
            if self._uses_bundled_snapshot():
                return dataset_resource_dir(
                    "winogender",
                    [SENTENCES_FILE, OCCUPATION_STATS_FILE, TEMPLATES_FILE],
                )
            raise FileNotFoundError(
                "The selected Hugging Face Winogender source publishes only "
                f"the sentences. Use the default bundled snapshot for "
                f"{OCCUPATION_STATS_FILE} and {TEMPLATES_FILE}, or pass `root=` "
                "pointing at a complete upstream checkout."
            )
        base = Path(self.root).expanduser()
        for candidate in (base, base / "Winogender"):
            for relative in (Path("data"), Path(".")):
                if (candidate / relative / SENTENCES_FILE).exists():
                    return (candidate / relative).resolve()
        raise FileNotFoundError(
            f"Winogender not found under {base}: expected data/{SENTENCES_FILE}."
        )

    def _read_sentences(self) -> pd.DataFrame:
        if self.root is not None or self._uses_bundled_snapshot():
            return pd.read_csv(self._data_dir() / SENTENCES_FILE, sep="\t")
        path = hub_file(self.hf_path, WINOGENDER_HF_FILENAME, self.revision)
        return pd.read_csv(path, sep="\t")

    @staticmethod
    def _from_sentid(
        sentid: str,
    ) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
        """``technician.customer.1.male.txt`` -> the four fields it encodes.

        The upstream ``all_sentences.tsv`` has only ``sentid`` and
        ``sentence``; every other field this loader returns is packed into that
        id. The Hub mirror ships them as columns already, so parsing here keeps
        the two sources on one schema.
        """
        parts = str(sentid).removesuffix(".txt").split(".")
        if len(parts) != 4:
            return None, None, None, None
        occupation, participant, answer, gender = parts
        try:
            answer_index = int(answer)
        except ValueError:
            answer_index = None
        return occupation, participant, answer_index, gender

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        df = self._read_sentences()
        examples: List[dict] = []
        for row in df.to_dict("records"):
            sentid = row.get("sentid")
            occupation, participant, answer, gender = self._from_sentid(sentid)

            def prefer(column, parsed):
                """The mirror's own column when it has one, else the parse.

                Not ``row.get(column) or parsed``: a missing cell is ``nan``,
                which is truthy, so the ``or`` would hand back the gap instead
                of the value parsed out of ``sentid``.
                """
                value = none_if_na(row.get(column))
                return parsed if value is None or value == "" else value

            occupation = prefer("occupation", occupation)
            participant = prefer("participant", participant)
            gender = prefer("gender", gender)
            if answer is None:
                answer = none_if_na(row.get("label"))

            if self.gender is not None and gender != self.gender:
                continue
            examples.append(
                {
                    "sentid": sentid,
                    "sentence": row.get("sentence"),
                    "occupation": occupation,
                    "participant": participant,
                    "answer": answer,
                    "gender": gender,
                    "bias_type": "gender",
                }
            )
            if self.n_max is not None and len(examples) >= self.n_max:
                self._cache = examples
                return examples

        self._cache = examples
        return optional_limit(examples, self.n_max)

    def templates(self) -> pd.DataFrame:
        """The 120 source templates. Needs a local ``root``."""
        return pd.read_csv(self._data_dir() / TEMPLATES_FILE, sep="\t")

    def occupation_stats(self) -> pd.DataFrame:
        """Bergsma and BLS percent-female statistics. Needs a local ``root``."""
        return pd.read_csv(self._data_dir() / OCCUPATION_STATS_FILE, sep="\t")

    def attributes_and_skew(
        self, source: str = "bls_pct_female"
    ) -> Tuple[List[str], Dict[str, float]]:
        """Occupations and P(female | occupation), on the same shape as
        :meth:`~fairLMs.datasets.BiasInBios.attributes_and_skew`.

        ``source`` selects ``bls_pct_female`` (US labour statistics, the
        default) or ``bergsma_pct_female`` (corpus-derived).
        """
        stats = self.occupation_stats()
        if source not in stats.columns:
            raise ValueError(
                f"Unknown source {source!r}. Available: "
                f"{', '.join(c for c in stats.columns if c != 'occupation')}."
            )
        skew = {
            str(row["occupation"]): float(row[source]) / 100.0
            for row in stats.to_dict("records")
            if pd.notna(row.get(source))
        }
        return list(skew.keys()), skew
