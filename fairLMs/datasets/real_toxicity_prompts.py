"""RealToxicityPrompts loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

from fairLMs.datasets._sources import hub_file
from fairLMs.datasets.base import FairnessDataset, optional_limit

PathLike = Union[str, Path]

RTP_HF_PATH = "allenai/real-toxicity-prompts"
RTP_FILENAME = "prompts.jsonl"

#: The eight Perspective API attributes scored on every prompt and continuation.
RTP_ATTRIBUTES = (
    "toxicity",
    "severe_toxicity",
    "identity_attack",
    "insult",
    "threat",
    "profanity",
    "sexually_explicit",
    "flirtation",
)


class RealToxicityPrompts(FairnessDataset):
    """Load RealToxicityPrompts naturally-occurring generation prompts.

    Each example is a dict::

        {
            "text": str,              # the prompt to continue
            "challenging": bool,      # in the 1.2k challenging subset
            "continuation": str,      # the document's own continuation
            "filename": str,          # source document id
            "toxicity": float | None, # and the other RTP_ATTRIBUTES,
            ...                       #   as scored on the *prompt*
        }

    The scores are the released Perspective API values for the prompt itself,
    so they describe the stimulus, not a model. Toxicity of a generation is
    something the caller measures on its own continuations.

    Parameters
    ----------
    challenging_only:
        Keep only the challenging subset.
    min_toxicity:
        Keep prompts whose released ``toxicity`` is at least this value. Rows
        with no toxicity score are dropped when this is set.
    root:
        A directory holding ``prompts.jsonl`` (the ~65 MB release). Without one
        the file is downloaded from the Hugging Face repo.
    """

    name = "real_toxicity_prompts"
    data_origin = "Hugging Face Hub, downloaded at first use"

    def __init__(
        self,
        challenging_only: bool = False,
        min_toxicity: Optional[float] = None,
        root: Optional[PathLike] = None,
        n_max: Optional[int] = None,
        hf_path: str = RTP_HF_PATH,
        revision: Optional[str] = None,
    ):
        self.challenging_only = challenging_only
        self.min_toxicity = min_toxicity
        self.root = root
        self.n_max = n_max
        self.hf_path = hf_path
        self.revision = revision
        self._cache: Optional[List[dict]] = None

    def _source_path(self) -> Path:
        if self.root is None:
            return hub_file(self.hf_path, RTP_FILENAME, self.revision)
        base = Path(self.root).expanduser()
        for candidate in (base, base / "RealToxicityPrompts"):
            if (candidate / RTP_FILENAME).exists():
                return candidate / RTP_FILENAME
            if candidate.name == RTP_FILENAME:
                return candidate
        raise FileNotFoundError(
            f"RealToxicityPrompts not found under {base}: expected " f"{RTP_FILENAME}."
        )

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        examples: List[dict] = []
        # 99k lines, read one at a time so `n_max` stops early instead of
        # parsing the whole 65 MB release first.
        with open(self._source_path(), "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = row.get("prompt") or {}
                continuation = row.get("continuation") or {}

                if self.challenging_only and not row.get("challenging"):
                    continue
                toxicity = prompt.get("toxicity")
                if self.min_toxicity is not None and (
                    toxicity is None or toxicity < self.min_toxicity
                ):
                    continue

                example = {
                    "text": prompt.get("text"),
                    "challenging": bool(row.get("challenging")),
                    "continuation": continuation.get("text"),
                    "filename": row.get("filename"),
                }
                example.update({a: prompt.get(a) for a in RTP_ATTRIBUTES})
                examples.append(example)
                if self.n_max is not None and len(examples) >= self.n_max:
                    self._cache = examples
                    return examples

        self._cache = examples
        return optional_limit(examples, self.n_max)
