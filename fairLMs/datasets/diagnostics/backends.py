"""Reference backends for the three backend-dependent construction slots.

``b_equiv``, ``b_gram`` and ``b_diff_dep`` read a quantity that no lightweight
kernel can produce: a sentence embedding, a grammatical-error count and a
dependency-tree depth. The diagnostics themselves only know the protocols in
:mod:`fairLMs.datasets.diagnostics.construction` (``EmbeddingBackend``,
``GrammarCheckerBackend``, ``DependencyParserBackend``); this module supplies
one implementation of each so that the slots can run out of the box.

Every backend records a ``revision`` string that ends up in the component's
provenance, because a semantic-similarity or depth value is only comparable
against values produced by the same model and version.

Heavy dependencies are imported lazily, on first use, and only the embedding
backend relies on packages that the core distribution already installs
(``torch`` and ``transformers``). The grammar and parser backends need the
``grammar`` and ``parse`` extras respectively::

    pip install "fairLMs[grammar]"   # language-tool-python (needs a Java runtime)
    pip install "fairLMs[parse]"     # spaCy; then: python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

__all__ = [
    "HuggingFaceEmbeddingBackend",
    "LanguageToolGrammarBackend",
    "SpacyDependencyBackend",
]


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _require_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a sequence of strings, not a single string.")
    out: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(
                f"texts[{index}] must be a str, got {type(text).__name__}."
            )
        out.append(text)
    return out


# ---------------------------------------------------------------------------
# b_equiv: sentence embeddings from any Hugging Face encoder
# ---------------------------------------------------------------------------


@dataclass
class HuggingFaceEmbeddingBackend:
    """Mean- or CLS-pooled sentence embeddings from a Hugging Face encoder.

    The default checkpoint is the SBERT model the paper names for
    :math:`B_{\\mathrm{equiv}}`. Any encoder loadable through ``AutoModel``
    works; the checkpoint, its resolved commit and the pooling rule are all
    recorded in :attr:`revision`.

    Parameters
    ----------
    model_name:
        Hub id or local path.
    model_revision:
        Optional Hub revision (branch, tag or commit) to pin.
    pooling:
        ``"mean"`` (attention-masked mean of the last hidden state) or
        ``"cls"`` (first token).
    max_length, batch_size, device:
        Tokenizer truncation length, encoding batch size and torch device.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_revision: Optional[str] = None
    pooling: str = "mean"
    max_length: int = 256
    batch_size: int = 32
    device: Optional[str] = None
    _tokenizer: Any = field(default=None, init=False, repr=False, compare=False)
    _model: Any = field(default=None, init=False, repr=False, compare=False)
    _resolved_revision: Optional[str] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if self.pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {self.pooling!r}.")
        if not isinstance(self.max_length, int) or self.max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

    @classmethod
    def from_components(
        cls,
        tokenizer: Any,
        model: Any,
        *,
        revision: str,
        pooling: str = "mean",
        max_length: int = 256,
        batch_size: int = 32,
    ) -> "HuggingFaceEmbeddingBackend":
        """Wrap an already loaded tokenizer and encoder.

        ``revision`` labels the pair in provenance; use the checkpoint name and
        commit when they are known, or a fixture label in tests.
        """
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be a non-empty string.")
        backend = cls(
            model_name=revision,
            pooling=pooling,
            max_length=max_length,
            batch_size=batch_size,
        )
        backend._tokenizer = tokenizer
        backend._model = model
        return backend

    @property
    def revision(self) -> str:
        resolved = self._resolved_revision or self.model_revision or "default"
        return (
            f"{self.model_name}@{resolved};pooling={self.pooling};"
            f"transformers={_package_version('transformers')}"
        )

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            from transformers import AutoModel, AutoTokenizer

            from fairLMs.datasets._sources import from_pretrained

            kwargs: dict[str, Any] = {}
            if self.model_revision is not None:
                kwargs["revision"] = self.model_revision
            self._tokenizer = from_pretrained(
                AutoTokenizer.from_pretrained, self.model_name, **kwargs
            )
            self._model = from_pretrained(
                AutoModel.from_pretrained, self.model_name, **kwargs
            )
            commit = getattr(getattr(self._model, "config", None), "_commit_hash", None)
            if isinstance(commit, str) and commit:
                self._resolved_revision = commit
        self._model.eval()
        if self.device is not None:
            self._model.to(self.device)
        return self._tokenizer, self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one pooled vector per text, in order, as Python floats."""
        import torch

        items = _require_texts(texts)
        if not items:
            return []
        tokenizer, model = self._load()
        device = next(model.parameters()).device
        vectors: list[list[float]] = []
        with torch.no_grad():
            for start in range(0, len(items), self.batch_size):
                batch = items[start : start + self.batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                hidden = model(**encoded).last_hidden_state
                if self.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                vectors.extend(pooled.float().cpu().tolist())
        return vectors


# ---------------------------------------------------------------------------
# b_gram: grammatical-error counts from LanguageTool
# ---------------------------------------------------------------------------


@dataclass
class LanguageToolGrammarBackend:
    """Count LanguageTool matches per text.

    Requires the ``grammar`` extra (``language-tool-python``), which downloads
    the LanguageTool server on first use and needs a Java runtime. The error
    count is the number of rule matches LanguageTool reports for the text in
    the configured language; the tool version and language go into
    :attr:`revision`.
    """

    language: str = "en-US"
    _tool: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("language must be a non-empty string such as 'en-US'.")

    @property
    def revision(self) -> str:
        return (
            f"language_tool_python={_package_version('language-tool-python')};"
            f"language={self.language}"
        )

    def _load(self) -> Any:
        if self._tool is None:
            try:
                import language_tool_python
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "b_gram's reference backend needs language_tool_python; "
                    "install it with `pip install \"fairLMs[grammar]\"` (a Java "
                    "runtime is required) or supply your own GrammarCheckerBackend."
                ) from exc
            self._tool = language_tool_python.LanguageTool(self.language)
        return self._tool

    def count_errors(self, texts: Sequence[str]) -> list[int]:
        """Return the number of LanguageTool matches per text, in order."""
        items = _require_texts(texts)
        if not items:
            return []
        tool = self._load()
        return [len(tool.check(text)) for text in items]

    def close(self) -> None:
        """Shut down the LanguageTool server if one was started."""
        if self._tool is not None:
            close = getattr(self._tool, "close", None)
            if callable(close):
                close()
            self._tool = None


# ---------------------------------------------------------------------------
# b_diff_dep: dependency-tree depth from spaCy
# ---------------------------------------------------------------------------


@dataclass
class SpacyDependencyBackend:
    """Maximum dependency-tree depth per text from a spaCy pipeline.

    Requires the ``parse`` extra (``spacy``) and a downloaded pipeline such as
    ``en_core_web_sm``. Depth is the number of tokens on the longest path from
    a sentence root to a leaf, so a one-token sentence has depth 1; a text
    with several sentences reports the deepest one, and an empty text reports
    0. The pipeline name and version go into :attr:`revision`.
    """

    model: str = "en_core_web_sm"
    _nlp: Any = field(default=None, init=False, repr=False, compare=False)

    #: Recorded in component provenance so a reader knows which depth this is.
    depth_definition: str = field(
        default=(
            "tokens on the longest root-to-leaf path of the dependency tree, "
            "maximum over sentences; empty text is 0"
        ),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty spaCy pipeline name.")

    @property
    def revision(self) -> str:
        pipeline_version = "unloaded"
        if self._nlp is not None:
            pipeline_version = str(getattr(self._nlp, "meta", {}).get("version", "unknown"))
        return (
            f"spacy={_package_version('spacy')};pipeline={self.model}@{pipeline_version}"
        )

    def _load(self) -> Any:
        if self._nlp is None:
            try:
                import spacy
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "b_diff_dep's reference backend needs spaCy; install it with "
                    "`pip install \"fairLMs[parse]\"` and download a pipeline with "
                    "`python -m spacy download en_core_web_sm`, or supply your own "
                    "DependencyParserBackend."
                ) from exc
            try:
                self._nlp = spacy.load(self.model, exclude=["ner", "lemmatizer"])
            except OSError as exc:  # pragma: no cover - depends on the environment
                raise OSError(
                    f"spaCy pipeline {self.model!r} is not installed; run "
                    f"`python -m spacy download {self.model}`."
                ) from exc
        return self._nlp

    @staticmethod
    def _tree_depth(root: Any) -> int:
        """Iteratively compute tokens on the longest root-to-leaf path."""
        depth = 0
        stack = [(root, 1)]
        while stack:
            token, level = stack.pop()
            depth = max(depth, level)
            for child in token.children:
                stack.append((child, level + 1))
        return depth

    def depths(self, texts: Sequence[str]) -> list[int]:
        """Return the maximum dependency depth per text, in order."""
        items = _require_texts(texts)
        if not items:
            return []
        nlp = self._load()
        out: list[int] = []
        for doc in nlp.pipe(items):
            depth = 0
            for sentence in doc.sents:
                depth = max(depth, self._tree_depth(sentence.root))
            out.append(depth)
        return out
