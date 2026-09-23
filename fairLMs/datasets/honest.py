"""HONEST hurtful-completion template loader."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

from fairLMs.datasets._sources import hub_file, none_if_na, require_choice
from fairLMs.datasets.base import FairnessDataset, optional_limit

PathLike = Union[str, Path]

HONEST_HF_PATH = "MilaNLProc/honest"

#: ``binary`` covers six languages; ``queer_nonqueer`` is English-only.
HONEST_CONFIGS = ("binary", "queer_nonqueer")
HONEST_BINARY_LANGUAGES = ("en", "es", "fr", "it", "pt", "ro")


class HONEST(FairnessDataset):
    """Load HONEST masked templates for hurtful-completion scoring.

    Each example is a dict::

        {
            "template_masked": str,  # e.g. "the woman should work as a [M]."
            "raw": str,              # the template with an [I] identity slot
            "identity": str,         # the filled identity term
            "number": str,           # singular / plural
            "category": str,         # identity grouping, e.g. female
            "type": str,             # template family, e.g. occupation
            "language": str,
            "config": str,
        }

    The ``[M]`` slot is what a model fills; HONEST then scores the completions
    against its hurtful-lexicon categories, which this loader does not ship.

    Parameters
    ----------
    config:
        ``"binary"`` (default) or ``"queer_nonqueer"``.
    language:
        Two-letter code. Only ``"en"`` exists for ``queer_nonqueer``.
    root:
        A local HONEST checkout, read as
        ``data/{config}/{language}_template.tsv``. Without one the templates
        come from the Hugging Face repo.
    """

    name = "honest"
    data_origin = "Hugging Face Hub, downloaded at first use"

    def __init__(
        self,
        config: str = "binary",
        language: str = "en",
        root: Optional[PathLike] = None,
        n_max: Optional[int] = None,
        hf_path: str = HONEST_HF_PATH,
        revision: Optional[str] = None,
    ):
        self.config = require_choice(config, HONEST_CONFIGS, "HONEST config")
        self.language = language
        if config == "queer_nonqueer" and language != "en":
            raise ValueError(
                "HONEST publishes queer_nonqueer templates in English only; "
                f"got language={language!r}."
            )
        if config == "binary":
            require_choice(language, HONEST_BINARY_LANGUAGES, "HONEST language")
        self.root = root
        self.n_max = n_max
        self.hf_path = hf_path
        self.revision = revision
        self._cache: Optional[List[dict]] = None

    def _local_path(self) -> Path:
        base = Path(self.root).expanduser()
        relative = Path("data") / self.config / f"{self.language}_template.tsv"
        for candidate in (base, base / "HONEST"):
            if (candidate / relative).exists():
                return candidate / relative
        raise FileNotFoundError(
            f"HONEST templates not found under {base}: expected {relative}."
        )

    def _source_path(self) -> Path:
        if self.root is not None:
            return self._local_path()
        # The Hub repo names the file by config; the upstream checkout by folder.
        filename = f"data/{self.language}/{self.language}_{self.config}_template.tsv"
        return hub_file(self.hf_path, filename, self.revision)

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        df = pd.read_csv(self._source_path(), sep="\t")
        examples: List[dict] = []
        for _, row in df.iterrows():
            examples.append(
                {
                    "template_masked": row["template_masked"],
                    "raw": none_if_na(row.get("raw")),
                    "identity": none_if_na(row.get("identity")),
                    "number": none_if_na(row.get("number")),
                    "category": none_if_na(row.get("category")),
                    "type": none_if_na(row.get("type")),
                    "language": self.language,
                    "config": self.config,
                }
            )

        self._cache = examples
        return optional_limit(examples, self.n_max)

    def identities(self) -> List[str]:
        """Sorted identity terms present in this config/language."""
        examples = self._cache if self._cache is not None else self.load()
        return sorted({str(ex["identity"]) for ex in examples if ex.get("identity")})
