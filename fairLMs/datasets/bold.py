"""BOLD (Bias in Open-ended Language Generation Dataset) loader."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

from fairLMs.datasets._sources import hub_file, require_choice
from fairLMs.datasets.base import FairnessDataset, optional_limit

PathLike = Union[str, Path]

BOLD_HF_PATH = "AmazonScience/bold"

#: BOLD's five prompt domains, and the bias axis each one varies.
BOLD_DOMAINS = (
    "gender",
    "political_ideology",
    "profession",
    "race",
    "religious_ideology",
)

BOLD_DOMAIN_BIAS_TYPES = {
    "gender": "gender",
    "political_ideology": "political_ideology",
    "profession": "profession",
    "race": "race",
    "religious_ideology": "religion",
}


def _as_list(value) -> List[str]:
    """BOLD's Hub mirror stores ``prompts`` as the *repr* of a Python list."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return [value]
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        return [value]
    return [str(value)]


class BOLD(FairnessDataset):
    """Load BOLD open-ended generation prompts, one example per prompt.

    Each example is a dict::

        {
            "prompt": str,           # the open-ended prefix to continue
            "domain": str,           # gender, race, profession, ...
            "category": str,         # BOLD's within-domain group
            "name": str,             # the Wikipedia entity the prefix came from
            "bias_type": str,
            "wikipedia": str | None, # source sentence, when include_wikipedia
        }

    BOLD supplies prompts, not model outputs: a generation metric continues each
    ``prompt`` itself and scores the continuation.

    Parameters
    ----------
    domains:
        Which of :data:`BOLD_DOMAINS` to load. Defaults to all five.
    root:
        A local BOLD checkout. Reads ``data/prompts/{domain}_prompt.json`` (and
        ``data/wikipedia/{domain}_wiki.json``) from it. Without one the prompts
        come from the Hugging Face mirror.
    include_wikipedia:
        Attach the Wikipedia sentence each prefix was truncated from.
    """

    name = "bold"
    data_origin = "Hugging Face Hub, downloaded at first use"

    def __init__(
        self,
        domains: Optional[Sequence[str]] = None,
        root: Optional[PathLike] = None,
        include_wikipedia: bool = False,
        n_max: Optional[int] = None,
        hf_path: str = BOLD_HF_PATH,
        revision: Optional[str] = None,
    ):
        self.domains = [
            require_choice(d, BOLD_DOMAINS, "BOLD domain")
            for d in (domains if domains is not None else BOLD_DOMAINS)
        ]
        self.root = root
        self.include_wikipedia = include_wikipedia
        self.n_max = n_max
        self.hf_path = hf_path
        self.revision = revision
        self._cache: Optional[List[dict]] = None

    def _local_domain(self, domain: str) -> List[dict]:
        """Read the upstream layout: nested ``{category: {name: [prompt]}}``."""
        base = Path(self.root).expanduser()
        for candidate in (base, base / "BOLD"):
            prompts_path = candidate / "data" / "prompts" / f"{domain}_prompt.json"
            if prompts_path.exists():
                break
        else:
            raise FileNotFoundError(
                f"BOLD prompts for {domain!r} not found under {base}: expected "
                f"data/prompts/{domain}_prompt.json."
            )

        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        wiki = {}
        wiki_path = candidate / "data" / "wikipedia" / f"{domain}_wiki.json"
        if self.include_wikipedia and wiki_path.exists():
            wiki = json.loads(wiki_path.read_text(encoding="utf-8"))

        rows = []
        for category, entities in prompts.items():
            for entity, entity_prompts in entities.items():
                rows.append(
                    {
                        "domain": domain,
                        "category": category,
                        "name": entity,
                        "prompts": entity_prompts,
                        "wikipedia": wiki.get(category, {}).get(entity),
                    }
                )
        return rows

    def _hub_domain(self, domain: str) -> List[dict]:
        """Read the Hub mirror: one JSON object per line, already flattened."""
        path = hub_file(self.hf_path, f"{domain}_prompt_wiki.json", self.revision)
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        examples: List[dict] = []
        for domain in self.domains:
            rows = (
                self._local_domain(domain)
                if self.root is not None
                else self._hub_domain(domain)
            )
            for row in rows:
                wikipedia = _as_list(row.get("wikipedia") or [])
                for index, prompt in enumerate(_as_list(row.get("prompts") or [])):
                    example = {
                        "prompt": prompt,
                        "domain": row.get("domain", domain),
                        "category": row.get("category"),
                        "name": row.get("name"),
                        "bias_type": BOLD_DOMAIN_BIAS_TYPES[domain],
                    }
                    if self.include_wikipedia:
                        example["wikipedia"] = (
                            wikipedia[index] if index < len(wikipedia) else None
                        )
                    examples.append(example)
                    if self.n_max is not None and len(examples) >= self.n_max:
                        self._cache = examples
                        return examples

        self._cache = examples
        return optional_limit(examples, self.n_max)

    def categories(self) -> List[str]:
        """Sorted within-domain category labels present in the loaded domains."""
        return sorted(
            {
                ex["category"]
                for ex in (self._cache if self._cache is not None else self.load())
                if ex.get("category")
            }
        )
