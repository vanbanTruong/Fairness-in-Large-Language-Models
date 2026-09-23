"""UnQover underspecified-question loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

from fairLMs.datasets._sources import require_choice, resolve_root
from fairLMs.datasets.base import FairnessDataset, optional_limit

PathLike = Union[str, Path]

UNQOVER_HOMEPAGE = "https://github.com/allenai/unqover"

#: Subject class -> (activity list, slot template) as ``generate_questions.sh``
#: pairs them. These are the stems of the released ``*.source.json`` files.
UNQOVER_SUBJECTS = {
    "mixed_gender": ("occupationrev1", "gendernoact", "gender"),
    "ethnicity": ("biasedethnicity", "ethnicitynoact", "ethnicity"),
    "country": ("biasedcountry", "countrynoact", "nationality"),
    "religion": ("biasedreligion", "religionnoact", "religion"),
}

#: ``None`` is the extractive-QA writing; ``bert`` and ``roberta`` are the
#: masked-LM writings, which differ only in the mask token.
UNQOVER_MODELS = (None, "bert", "roberta")

_CHUNK_CHARS = 1 << 20


def _stream_json_object(path: Path) -> Iterator[Tuple[str, dict]]:
    """Yield ``(key, value)`` from a top-level JSON object, incrementally.

    The released slotmaps run from 70 MB to 2.2 GB; ``json.load`` on the
    largest of them needs tens of gigabytes of memory, which defeats passing a
    small ``n_max``. This walks the file with ``raw_decode`` over a sliding
    buffer instead, so a caller that wants a hundred examples reads about a
    megabyte.
    """
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as handle:
        buffer = handle.read(_CHUNK_CHARS)
        start = buffer.find("{")
        if start < 0:
            raise ValueError(f"{path} does not start a JSON object.")
        buffer = buffer[start + 1 :]

        def fill() -> bool:
            nonlocal buffer
            more = handle.read(_CHUNK_CHARS)
            if not more:
                return False
            buffer += more
            return True

        def skip(characters: str) -> bool:
            """Drop leading ``characters``; False at end of input."""
            nonlocal buffer
            while True:
                index = 0
                while index < len(buffer) and buffer[index] in characters:
                    index += 1
                if index < len(buffer):
                    buffer = buffer[index:]
                    return True
                buffer = ""
                if not fill():
                    return False

        def decode():
            """raw_decode the head of the buffer, growing it until it parses."""
            nonlocal buffer
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                except ValueError:
                    if not fill():
                        raise
                    continue
                buffer = buffer[end:]
                return value

        while True:
            if not skip(" \t\r\n,"):
                return
            if buffer[0] == "}":
                return
            key = decode()
            if not skip(" \t\r\n:"):
                return
            yield str(key), decode()


class UnQover(FairnessDataset):
    """Load UnQover underspecified question/context examples.

    Each example is one slotmap entry -- a context naming two subjects, and the
    two questions that ask which of them the attribute applies to::

        {
            "key": str,            # the release's slot key, kept verbatim
            "context": str,        # "A Christian man got off the flight to ..."
            "question_0": str,     # "Who had a reputation of being barbarian?"
            "question_1": str,     # the negated counterpart
            "answer_0": str,       # subject 0, an answer choice for both
            "answer_1": str,       # subject 1
            "subject_0": str, "subject_1": str,
            "attribute": str,      # the activity cluster, e.g. "rude"
            "template_id": str,
            "bias_type": str,      # gender / ethnicity / nationality / religion
        }

    The context is underspecified on purpose: neither subject is the answer, so
    any consistent preference between them across the two questions is the
    measured signal.

    UnQover ships a generator rather than a fixed evaluation file, and the
    generated slotmaps run to gigabytes, so this loader reads a local
    ``data/slotmap_*.source.json`` produced by the upstream
    ``scripts/generate_questions.sh``. Pass ``root=`` pointing at the directory
    holding them. Files are streamed, so ``n_max`` stops reading early.

    Parameters
    ----------
    subject:
        One of :data:`UNQOVER_SUBJECTS`.
    model:
        ``None`` for the extractive-QA slotmap, or ``"bert"`` / ``"roberta"``
        for the masked-LM writings.
    """

    name = "unqover"
    data_origin = "local copy required (`root=`)"

    def __init__(
        self,
        root: Optional[PathLike] = None,
        subject: str = "religion",
        model: Optional[str] = None,
        n_max: Optional[int] = None,
    ):
        self.root = root
        self.subject = require_choice(
            subject, tuple(UNQOVER_SUBJECTS), "UnQover subject"
        )
        if model not in UNQOVER_MODELS:
            raise ValueError(
                f"Unknown UnQover model {model!r}. Expected None, 'bert' or "
                "'roberta'."
            )
        self.model = model
        self.n_max = n_max
        self._cache: Optional[List[dict]] = None

    @property
    def filename(self) -> str:
        """The released slotmap file this configuration reads."""
        activity, slot, _ = UNQOVER_SUBJECTS[self.subject]
        subject = self.subject.replace("_", "")
        if self.model is None:
            return f"slotmap_{subject}_{activity}_{slot}.source.json"
        return f"slotmap_{subject}{self.model}_{activity}_{slot}lm.source.json"

    def _source_path(self) -> Path:
        relative = Path("data") / self.filename
        base = resolve_root(
            self.root,
            dataset="UnQover",
            directory="UnQover",
            sentinels=(relative, Path("data") / "data" / self.filename),
            homepage=UNQOVER_HOMEPAGE,
        )
        for candidate in (base / relative, base / "data" / "data" / self.filename):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"UnQover slotmap not found: {base / relative}.")

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        bias_type = UNQOVER_SUBJECTS[self.subject][2]
        examples: List[dict] = []
        for key, value in _stream_json_object(self._source_path()):
            # generate_underspecified_templates.py writes the key as
            # cluster0|cluster1|subj0|subj1|template_id|attribute|act0|act1
            fields = key.split("|")
            subject_0 = fields[2] if len(fields) > 3 else None
            subject_1 = fields[3] if len(fields) > 3 else None
            template_id = fields[4] if len(fields) > 4 else None
            attribute = fields[5] if len(fields) > 5 else None
            q0 = value.get("q0") or {}
            q1 = value.get("q1") or {}
            examples.append(
                {
                    "key": key,
                    "context": value.get("context"),
                    "question_0": q0.get("question"),
                    "question_1": q1.get("question"),
                    "answer_0": (q0.get("ans0") or {}).get("text"),
                    "answer_1": (q0.get("ans1") or {}).get("text"),
                    "subject_0": subject_0,
                    "subject_1": subject_1,
                    "attribute": attribute,
                    "template_id": template_id,
                    "bias_type": bias_type,
                }
            )
            if self.n_max is not None and len(examples) >= self.n_max:
                break

        self._cache = examples
        return optional_limit(examples, self.n_max)
