"""Evidence containers introduced by the mitigation layer.

**Reuse before inventing.** Most mitigators consume a container that already
exists: :class:`~fairLMs.metrics.PromptPairs`,
:class:`~fairLMs.metrics.GroupWordPairs`,
:class:`~fairLMs.metrics.DemographicPrompts` and the rest of
:mod:`fairLMs.metrics.data`, or
:class:`~fairLMs.diagnostics.LabeledScoredGroups` for the group-fairness
post-processing methods. Only the shapes with no existing home are defined here.

``LabeledScoredGroups`` is deliberately **not** here: it composes
:class:`~fairLMs.diagnostics.ScoredGroups`, and the diagnostics layer uses it
too, so it lives next to the type it composes and is re-exported from
:mod:`fairLMs.mitigation` for convenience.

Every container validates at construction and **refuses** malformed input rather
than coercing, dropping or silently repairing it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from fairLMs.diagnostics._utils import (
    freeze_json_mapping,
    normalize_string_sequence,
    require_nonempty_string,
)

__all__ = [
    "AttributeLabeledVectors",
    "CandidateSets",
    "CorpusWithBenignPool",
    "CorpusWithLexicon",
    "GENERATOR_RANK",
    "GroupLabeledRecords",
    "InfluenceScoredCorpus",
    "PromptSpec",
    "SwapLexicon",
    "TextRecords",
]


#: Reserved ``CandidateSets.quality_name``, recorded by
#: :class:`~fairLMs.mitigation.OutputReranking` when the caller declares
#: no ``q`` and the generator's own ordering stands in for it. A caller cannot
#: claim the name, or a fitted rule could not tell the two apart.
GENERATOR_RANK = "generator_rank"


def _check_texts(value: Any, field_name: str) -> Tuple[str, ...]:
    return normalize_string_sequence(value, field_name=field_name, allow_empty=False)


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class TextRecords:
    """Free-text rows with an explicit field mapping, for corpus transforms.

    Follows the convention of :mod:`fairLMs.diagnostics.evidence`: the caller
    declares which field carries the text, rather than the container guessing a
    column name.
    """

    texts: Sequence[str]
    source: str
    ids: Optional[Sequence[str]] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "texts", _check_texts(self.texts, "texts"))
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )
        if self.ids is not None:
            ids = normalize_string_sequence(self.ids, field_name="ids")
            if len(ids) != len(self.texts):
                raise ValueError(
                    "ids and texts must contain the same number of rows; got "
                    f"{len(ids)} ids and {len(self.texts)} texts."
                )
            if len(set(ids)) != len(ids):
                raise ValueError("ids must not contain duplicates.")
            object.__setattr__(self, "ids", ids)
        object.__setattr__(
            self, "provenance", freeze_json_mapping(self.provenance, path="provenance")
        )

    @property
    def n_rows(self) -> int:
        return len(self.texts)

    def to_dict(self) -> dict:
        return {
            "texts": list(self.texts),
            "source": self.source,
            "ids": None if self.ids is None else list(self.ids),
        }


@dataclass(frozen=True, kw_only=True)
class SwapLexicon:
    """An explicit, bidirectional surface-form swap lexicon.

    Counterfactual data augmentation is only as defensible as the lexicon behind
    it, so the lexicon is evidence the caller supplies and that is recorded in
    provenance - never an inferred or bundled default.
    """

    axis: str
    pairs: Sequence[Tuple[str, str]]
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )
        cleaned = []
        seen = set()
        for index, pair in enumerate(self.pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(
                    f"pairs[{index}] must be a 2-tuple (term, counterpart); "
                    f"got {pair!r}."
                )
            left = require_nonempty_string(pair[0], f"pairs[{index}][0]")
            right = require_nonempty_string(pair[1], f"pairs[{index}][1]")
            if left.lower() == right.lower():
                raise ValueError(
                    f"pairs[{index}] maps {left!r} to itself; nothing to swap."
                )
            for term in (left.lower(), right.lower()):
                if term in seen:
                    raise ValueError(
                        f"pairs[{index}] reuses the term {term!r}; a term may "
                        f"appear in exactly one pair or the swap is ambiguous."
                    )
                seen.add(term)
            cleaned.append((left, right))
        if not cleaned:
            raise ValueError("pairs must not be empty.")
        object.__setattr__(self, "pairs", tuple(cleaned))

    def mapping(self) -> dict:
        """Return the case-insensitive term -> counterpart map, both directions."""
        table = {}
        for left, right in self.pairs:
            table[left.lower()] = right
            table[right.lower()] = left
        return table

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "pairs": [list(pair) for pair in self.pairs],
            "source": self.source,
        }


@dataclass(frozen=True, kw_only=True)
class GroupLabeledRecords:
    """Rows carrying both an outcome label and a protected-group membership.

    What reweighting needs: the joint ``(y, a)`` cell of every row, declared
    rather than inferred.
    """

    axis: str
    groups: Sequence[str]
    labels: Sequence[str]
    label_name: str
    source: str
    texts: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self, "label_name", require_nonempty_string(self.label_name, "label_name")
        )
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )
        groups = _check_texts(self.groups, "groups")
        labels = _check_texts(self.labels, "labels")
        if len(groups) != len(labels):
            raise ValueError(
                "groups and labels must contain the same number of rows; got "
                f"{len(groups)} groups and {len(labels)} labels."
            )
        if len(set(groups)) < 2:
            raise ValueError("groups must contain at least two observed categories.")
        if len(set(labels)) < 2:
            raise ValueError("labels must contain at least two observed outcomes.")
        if self.texts is not None:
            texts = _check_texts(self.texts, "texts")
            if len(texts) != len(groups):
                raise ValueError(
                    "texts and groups must contain the same number of rows; got "
                    f"{len(texts)} texts and {len(groups)} groups."
                )
            object.__setattr__(self, "texts", texts)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "labels", labels)

    @property
    def n_rows(self) -> int:
        return len(self.groups)

    @property
    def cells(self) -> Tuple[Tuple[str, str], ...]:
        """Sorted observed ``(label, group)`` cells."""
        return tuple(sorted(set(zip(self.labels, self.groups))))

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "groups": list(self.groups),
            "labels": list(self.labels),
            "label_name": self.label_name,
            "source": self.source,
            "texts": None if self.texts is None else list(self.texts),
        }


@dataclass(frozen=True, kw_only=True)
class CorpusWithLexicon:
    """A corpus paired with the swap lexicon to rewrite it by.

    A named container rather than a bare 2-tuple: the pairing is validated once,
    it appears in ``accepts`` as something a reader can look up, and a
    transposed ``(lexicon, records)`` call is refused at construction.
    """

    records: TextRecords
    lexicon: SwapLexicon

    def __post_init__(self) -> None:
        if not isinstance(self.records, TextRecords):
            raise TypeError(
                f"records must be a TextRecords, got {type(self.records).__name__}."
            )
        if not isinstance(self.lexicon, SwapLexicon):
            raise TypeError(
                f"lexicon must be a SwapLexicon, got {type(self.lexicon).__name__}."
            )

    def to_dict(self) -> dict:
        return {"records": self.records.to_dict(), "lexicon": self.lexicon.to_dict()}


@dataclass(frozen=True, kw_only=True)
class CorpusWithBenignPool:
    """A labelled corpus paired with the benign identity-term examples to add."""

    corpus: GroupLabeledRecords
    pool: TextRecords

    def __post_init__(self) -> None:
        if not isinstance(self.corpus, GroupLabeledRecords):
            raise TypeError(
                "corpus must be a GroupLabeledRecords, got "
                f"{type(self.corpus).__name__}."
            )
        if not isinstance(self.pool, TextRecords):
            raise TypeError(
                f"pool must be a TextRecords, got {type(self.pool).__name__}."
            )
        if self.corpus.texts is None:
            raise ValueError(
                "corpus must carry texts to be augmented with text examples. "
                "Build GroupLabeledRecords(..., texts=[...])."
            )

    def to_dict(self) -> dict:
        return {"corpus": self.corpus.to_dict(), "pool": self.pool.to_dict()}


@dataclass(frozen=True, kw_only=True)
class PromptSpec:
    """Instruction templates prepended to a query, plus the queries themselves.

    Used by ``debiasing_prompt`` (pre) and ``self_debiasing`` (intra). Each
    template must carry a ``{query}`` placeholder, so the composition is
    explicit rather than positional string concatenation.
    """

    templates: Sequence[str]
    queries: Sequence[str] = ()
    attribute: Optional[str] = None

    def __post_init__(self) -> None:
        templates = _check_texts(self.templates, "templates")
        for index, template in enumerate(templates):
            if "{query}" not in template:
                raise ValueError(
                    f"templates[{index}] must contain a '{{query}}' placeholder "
                    f"so the query position is explicit; got {template!r}."
                )
        object.__setattr__(self, "templates", templates)
        if self.queries:
            object.__setattr__(self, "queries", _check_texts(self.queries, "queries"))
        else:
            object.__setattr__(self, "queries", ())
        if self.attribute is not None:
            object.__setattr__(
                self, "attribute", require_nonempty_string(self.attribute, "attribute")
            )

    def render(self, query: str) -> Tuple[str, ...]:
        """Return *query* under every template."""
        require_nonempty_string(query, "query")
        return tuple(t.replace("{query}", query) for t in self.templates)

    def to_dict(self) -> dict:
        return {
            "templates": list(self.templates),
            "queries": list(self.queries),
            "attribute": self.attribute,
        }


# ---------------------------------------------------------------------------
# In-processing
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class InfluenceScoredCorpus:
    """A training corpus, a flagged harmful subset, and **precomputed** scores.

    ``fairLMs`` ships the IF-Guide *objective*, not the influence machinery.
    Estimating influence (Hessian inverse, EK-FAC, ...) is an optional backend
    and never a core dependency, so the scores arrive here already computed and
    the component is testable on scores the caller supplies.
    """

    n_examples: int
    flagged: Sequence[int]
    influence: Sequence[float]
    source: str

    def __post_init__(self) -> None:
        if isinstance(self.n_examples, bool) or not isinstance(self.n_examples, int):
            raise TypeError(
                f"n_examples must be an int, got {type(self.n_examples).__name__}."
            )
        if self.n_examples < 1:
            raise ValueError("n_examples must be at least 1.")
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )

        flagged = tuple(self.flagged)
        for index, value in enumerate(flagged):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"flagged[{index}] must be an int index, "
                    f"got {type(value).__name__}."
                )
            if not 0 <= value < self.n_examples:
                raise ValueError(
                    f"flagged[{index}] is {value}, outside the corpus range "
                    f"[0, {self.n_examples})."
                )
        if not flagged:
            raise ValueError("flagged must not be empty; there is nothing to suppress.")
        if len(set(flagged)) != len(flagged):
            raise ValueError("flagged must not contain duplicate indices.")

        influence = tuple(self.influence)
        if len(influence) != len(flagged):
            raise ValueError(
                "influence must carry one score per flagged example; got "
                f"{len(influence)} scores for {len(flagged)} flagged indices."
            )
        cleaned = []
        for index, value in enumerate(influence):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"influence[{index}] must be a real number, "
                    f"got {type(value).__name__}."
                )
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"influence[{index}] must be finite.")
            cleaned.append(number)

        object.__setattr__(self, "flagged", flagged)
        object.__setattr__(self, "influence", tuple(cleaned))

    def to_dict(self) -> dict:
        return {
            "n_examples": self.n_examples,
            "flagged": list(self.flagged),
            "influence": list(self.influence),
            "source": self.source,
        }


@dataclass(frozen=True, kw_only=True)
class AttributeLabeledVectors:
    """Representations paired with the protected attribute to be removed.

    What a linear attribute probe is fitted on, and therefore what INLP needs.
    Held as a container rather than a bare array so the axis and the source of
    the labels are recorded alongside the numbers.
    """

    axis: str
    vectors: Any
    labels: Sequence[str]
    source: str
    representation_layer: Optional[str] = None
    pooling: Optional[str] = None
    pair_ids: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        import numpy as np

        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )
        vectors = np.array(self.vectors, dtype=float, copy=True)
        if vectors.ndim != 2:
            raise ValueError(
                f"vectors must be a 2-D (n_rows, n_features) array; got shape "
                f"{vectors.shape}."
            )
        if not np.isfinite(vectors).all():
            raise ValueError("vectors must be finite; got NaN or infinity.")
        labels = _check_texts(self.labels, "labels")
        if len(labels) != vectors.shape[0]:
            raise ValueError(
                "labels and vectors must contain the same number of rows; got "
                f"{len(labels)} labels and {vectors.shape[0]} vectors."
            )
        if len(set(labels)) < 2:
            raise ValueError(
                "labels must contain at least two observed attribute values; a "
                "probe cannot be fitted against a constant."
            )
        if vectors.shape[1] == 0:
            raise ValueError("vectors must have at least one feature.")
        for field in ("representation_layer", "pooling"):
            if getattr(self, field) is not None:
                require_nonempty_string(getattr(self, field), field)
        if self.pair_ids is not None:
            ids = _check_texts(self.pair_ids, "pair_ids")
            if len(ids) != len(labels):
                raise ValueError("pair_ids must align with vectors.")
            object.__setattr__(self, "pair_ids", ids)
        vectors.setflags(write=False)
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "labels", labels)

    @property
    def n_rows(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.vectors.shape[1])

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "labels": list(self.labels),
            "source": self.source,
            "shape": [self.n_rows, self.n_features],
        }


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class CandidateSets:
    """Per-query candidate generations plus the injected scorers a reranker needs.

    Both scorers are **declared, never chosen** by the library: reranking
    reorders by whatever notions of bias and quality the caller is willing to
    defend, and burying defaults here would make those choices invisible.

    ``scorer`` is the bias term ``f`` of
    :class:`~fairLMs.mitigation.OutputReranking`, which maximizes
    ``lambda * q + (1 - lambda) * f`` and so prefers *higher* ``f``: a caller
    who wants bias penalized declares ``f`` oriented that way, for instance as a
    negated bias score. ``quality`` is the ``q`` term, in whatever units the
    caller declares; when it is omitted, the reranker reads ``q`` off the
    generator's own ordering and records that as
    ``quality_name='generator_rank'``.
    """

    queries: Sequence[str]
    candidates: Sequence[Sequence[str]]
    scorer: Callable[[str, str], float]
    scorer_name: str
    quality: Optional[Callable[[str, str], float]] = None
    quality_name: Optional[str] = None

    def __post_init__(self) -> None:
        queries = _check_texts(self.queries, "queries")
        rows = tuple(self.candidates)
        if len(rows) != len(queries):
            raise ValueError(
                "candidates must carry one candidate list per query; got "
                f"{len(rows)} lists for {len(queries)} queries."
            )
        cleaned = []
        for index, row in enumerate(rows):
            items = normalize_string_sequence(
                row, field_name=f"candidates[{index}]", allow_empty=False
            )
            if len(set(items)) != len(items):
                raise ValueError(
                    f"candidates[{index}] contains duplicates; a ranking over "
                    f"duplicate candidates is not well defined."
                )
            cleaned.append(items)
        if not callable(self.scorer):
            raise TypeError(
                f"scorer must be callable (query, candidate) -> float; got "
                f"{type(self.scorer).__name__}."
            )
        object.__setattr__(
            self,
            "scorer_name",
            require_nonempty_string(self.scorer_name, "scorer_name"),
        )
        if self.quality is not None:
            if not callable(self.quality):
                raise TypeError(
                    f"quality must be callable (query, candidate) -> float; got "
                    f"{type(self.quality).__name__}."
                )
            # A declared quality scorer has to be nameable, so a fitted rule
            # records which `q` produced it.
            quality_name = require_nonempty_string(self.quality_name, "quality_name")
            if quality_name == GENERATOR_RANK:
                raise ValueError(
                    f"quality_name {GENERATOR_RANK!r} is reserved for the "
                    "generator's own ordering; name the declared quality scorer "
                    "something else."
                )
            object.__setattr__(self, "quality_name", quality_name)
        elif self.quality_name is not None:
            raise ValueError(
                "quality_name was given without quality; pass the callable "
                "itself, or drop the name to rank by the generator's own "
                "ordering."
            )
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "candidates", tuple(cleaned))

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    def to_dict(self) -> dict:
        return {
            "queries": list(self.queries),
            "candidates": [list(row) for row in self.candidates],
            "scorer_name": self.scorer_name,
            "quality_name": self.quality_name,
        }
