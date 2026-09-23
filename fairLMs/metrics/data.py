"""Typed, validated input containers for metrics that need structured data.

Metrics such as WEAT / SEAT / CEAT do not consume a flat corpus; they need
*four labelled sets* (two target groups, two attribute poles). Historically each
metric accepted these through loose ``**kwargs`` (``T1_terms``, ``T1_vecs``,
``T1_contexts``, ``A_terms`` aliased to ``A1_terms``, …), which meant the
accepted shape was undocumented, unvalidated, and different per metric.

These containers make the shape explicit and validate it at construction, so a
malformed input fails immediately with a useful message instead of surfacing as
a ``matmul`` shape error deep inside numpy.
"""

from __future__ import annotations

import collections.abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from fairLMs.datasets.base import FairnessDataset

__all__ = [
    "RECORD_CORPUS",
    "WordSets",
    "VectorSets",
    "ContextSets",
    "SentenceTriples",
    "GroupWordPairs",
    "ContrastSpec",
    "LabeledSentences",
    "StereotypeLabelled",
    "GroupPredictions",
    "ScorePair",
    "PromptPairs",
    "DemographicPrompts",
    "ProbeSet",
    "ConceptSpec",
    "OccupationTriples",
    "QuerySpec",
    "GroupProperties",
]

#: What a corpus metric declares in ``accepts``. Not every metric consumes a
#: purpose-built container: several score a plain corpus of records, which
#: reaches them either as a :class:`~fairLMs.datasets.FairnessDataset` or as an
#: already-materialised sequence of examples. Declaring both is what those
#: metrics actually take; narrowing it to one would be a false declaration.
RECORD_CORPUS: Tuple[type, ...] = (FairnessDataset, collections.abc.Sequence)

_ROLES = ("target_1", "target_2", "attribute_1", "attribute_2")


def _check_str_seq(value: Any, role: str) -> tuple:
    if isinstance(value, str) or not isinstance(value, (list, tuple, Sequence)):
        raise TypeError(
            f"{role} must be a sequence of strings, got {type(value).__name__}. "
            f"(A bare string is not accepted. Wrap it in a list.)"
        )
    items = tuple(value)
    if not items:
        raise ValueError(
            f"{role} must contain at least one term, got an empty sequence."
        )
    bad = [x for x in items if not isinstance(x, str)]
    if bad:
        raise TypeError(
            f"{role} must contain only strings; found {type(bad[0]).__name__} "
            f"({bad[0]!r})."
        )
    return items


def _check_seq(value: Any, role: str) -> tuple:
    """Non-empty sequence of anything (labels may be ints, floats, or strings)."""
    if isinstance(value, str) or not isinstance(value, (list, tuple, Sequence)):
        raise TypeError(f"{role} must be a sequence, got {type(value).__name__}.")
    items = tuple(value)
    if not items:
        raise ValueError(f"{role} must not be empty.")
    return items


def _check_same_length(**named: Sequence[Any]) -> None:
    lengths = {name: len(seq) for name, seq in named.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"These must all have the same length, got {lengths}.")


def _check_dict_seq(value: Any, role: str) -> Dict[str, tuple]:
    """Mapping of name -> non-empty sequence of strings."""
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{role} must be a mapping of group name -> list of terms, got "
            f"{type(value).__name__}."
        )
    if not value:
        raise ValueError(f"{role} must contain at least one group.")
    return {k: _check_str_seq(v, f"{role}[{k!r}]") for k, v in value.items()}


@dataclass(frozen=True)
class WordSets:
    """Four word lists for an association test.

    ``target_1`` / ``target_2`` are the two groups being compared (e.g.
    European-American vs African-American names); ``attribute_1`` /
    ``attribute_2`` are the two attribute poles (e.g. pleasant vs unpleasant).

    Example
    -------
    >>> sets = WordSets(
    ...     target_1=["Adam", "Chip"],
    ...     target_2=["Alonzo", "Jamel"],
    ...     attribute_1=["freedom", "health"],
    ...     attribute_2=["abuse", "murder"],
    ... )
    >>> sets.target_1              # lists are normalized to tuples
    ('Adam', 'Chip')
    """

    target_1: Sequence[str]
    target_2: Sequence[str]
    attribute_1: Sequence[str]
    attribute_2: Sequence[str]

    def __post_init__(self) -> None:
        for role in _ROLES:
            object.__setattr__(self, role, _check_str_seq(getattr(self, role), role))

    def require_balanced_targets(self, metric_name: str) -> None:
        """Raise unless the two target lists are the same length.

        SEAT compares per-target association scores pairwise, so it needs equal
        sizes; WEAT does not. Metric-specific constraints live here rather than
        in ``__post_init__`` so the container stays reusable.
        """
        if len(self.target_1) != len(self.target_2):
            raise ValueError(
                f"{metric_name} requires target_1 and target_2 to have the same "
                f"length, got {len(self.target_1)} and {len(self.target_2)}."
            )


@dataclass(frozen=True)
class VectorSets:
    """Precomputed embeddings for an association test, one array per role.

    Each role is ``(n_terms, n_dims)``. Use this to run WEAT without a model,
    e.g. on static embeddings you already have.
    """

    target_1: Any
    target_2: Any
    attribute_1: Any
    attribute_2: Any

    def __post_init__(self) -> None:
        dims = {}
        for role in _ROLES:
            arr = np.asarray(getattr(self, role), dtype=float)
            if arr.ndim != 2:
                raise ValueError(
                    f"{role} must be 2-D (n_terms, n_dims), got shape {arr.shape}. "
                    f"Pass a list of vectors, not a single vector."
                )
            if arr.shape[0] == 0:
                raise ValueError(f"{role} must contain at least one vector.")
            object.__setattr__(self, role, arr)
            dims[role] = arr.shape[1]
        if len(set(dims.values())) > 1:
            raise ValueError(
                f"All roles must share an embedding dimension, got {dims}."
            )

    @property
    def n_dims(self) -> int:
        return int(np.asarray(self.target_1).shape[1])


@dataclass(frozen=True)
class ContextSets:
    """Per-term context sentences for a contextualized association test (CEAT).

    Each role maps a term to the sentences it should be embedded in::

        ContextSets(
            target_1={"Adam": ["Adam went home.", "Adam is here."]},
            target_2={"Alonzo": ["Alonzo went home.", "Alonzo is here."]},
            attribute_1={"freedom": ["Freedom matters.", "We value freedom."]},
            attribute_2={"abuse": ["Abuse is harmful.", "They reported abuse."]},
        )
    """

    target_1: Mapping[str, Sequence[str]]
    target_2: Mapping[str, Sequence[str]]
    attribute_1: Mapping[str, Sequence[str]]
    attribute_2: Mapping[str, Sequence[str]]

    def __post_init__(self) -> None:
        for role in _ROLES:
            mapping = getattr(self, role)
            if not isinstance(mapping, Mapping):
                raise TypeError(
                    f"{role} must be a mapping of term -> list of context "
                    f"sentences, got {type(mapping).__name__}. "
                    f"(A flat list of sentences is not accepted; CEAT samples "
                    f"contexts per term, so terms must be keyed.)"
                )
            if not mapping:
                raise ValueError(f"{role} must contain at least one term.")
            cleaned = {}
            for term, contexts in mapping.items():
                cleaned[term] = _check_str_seq(contexts, f"{role}[{term!r}]")
            object.__setattr__(self, role, cleaned)

    def min_contexts(self) -> int:
        """Fewest contexts available for any single term across all roles."""
        return min(
            len(ctxs) for role in _ROLES for ctxs in getattr(self, role).values()
        )

    def require_contexts(self, sample_size: int, metric_name: str) -> None:
        """Raise if any term has fewer contexts than ``sample_size``."""
        for role in _ROLES:
            for term, contexts in getattr(self, role).items():
                if len(contexts) < sample_size:
                    raise ValueError(
                        f"{metric_name}: term {term!r} in {role} has only "
                        f"{len(contexts)} context(s) but sample_size="
                        f"{sample_size}. Provide more contexts or lower "
                        f"sample_size."
                    )


# ---------------------------------------------------------------------------
# Sentence-level structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SentenceTriples:
    """Stereotype / anti-stereotype / unrelated sentence triples (StereoSet CAT)."""

    stereotype: Sequence[str]
    anti_stereotype: Sequence[str]
    unrelated: Sequence[str]
    contexts: Optional[Sequence[Optional[str]]] = None

    def __post_init__(self) -> None:
        for role in ("stereotype", "anti_stereotype", "unrelated"):
            object.__setattr__(self, role, _check_str_seq(getattr(self, role), role))
        _check_same_length(
            stereotype=self.stereotype,
            anti_stereotype=self.anti_stereotype,
            unrelated=self.unrelated,
        )

        if self.contexts is not None:
            if len(self.contexts) != len(self.stereotype) or any(
                c is not None and (not isinstance(c, str) or not c.strip())
                for c in self.contexts
            ):
                raise ValueError(
                    "contexts must align with triples and contain non-empty strings or None."
                )
            object.__setattr__(self, "contexts", tuple(self.contexts))

    def __len__(self) -> int:
        return len(self.stereotype)

    @classmethod
    def from_examples(cls, examples: Sequence[Any]) -> "SentenceTriples":
        """Build from dicts (``stereotype``/``anti_stereotype``/``unrelated``) or 3-tuples."""
        stereo, anti, unrel, contexts = [], [], [], []
        for ex in examples:
            contexts.append(
                ex.get("scoring_context") if isinstance(ex, Mapping) else None
            )
            if isinstance(ex, Mapping):
                try:
                    stereo.append(ex["stereotype"])
                    anti.append(ex["anti_stereotype"])
                    unrel.append(ex["unrelated"])
                except KeyError as exc:
                    raise ValueError(
                        f"Example dict is missing key {exc}. Triples need "
                        f"'stereotype', 'anti_stereotype' and 'unrelated'."
                    ) from exc
            elif isinstance(ex, (list, tuple)) and len(ex) >= 3:
                stereo.append(ex[0])
                anti.append(ex[1])
                unrel.append(ex[2])
            else:
                raise TypeError(
                    f"Cannot read a triple from {type(ex).__name__}; expected a "
                    f"mapping or a 3-element sequence."
                )
        return cls(
            stereo,
            anti,
            unrel,
            contexts=contexts if any(c is not None for c in contexts) else None,
        )


@dataclass(frozen=True)
class LabeledSentences:
    """Sentences paired with labels (counterfactual AUC, stereotype divergence)."""

    sentences: Sequence[str]
    labels: Sequence[Any]
    pair_ids: Optional[Sequence[Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sentences", _check_str_seq(self.sentences, "sentences")
        )
        object.__setattr__(self, "labels", _check_seq(self.labels, "labels"))
        _check_same_length(sentences=self.sentences, labels=self.labels)
        if self.pair_ids is not None:
            object.__setattr__(self, "pair_ids", _check_seq(self.pair_ids, "pair_ids"))
            _check_same_length(sentences=self.sentences, pair_ids=self.pair_ids)

    def __len__(self) -> int:
        return len(self.sentences)

    @classmethod
    def from_examples(cls, examples: Sequence[Any]) -> "LabeledSentences":
        sents, labels = [], []
        for ex in examples:
            if isinstance(ex, Mapping):
                sents.append(ex["sentence"])
                labels.append(ex["label"])
            elif isinstance(ex, (list, tuple)) and len(ex) >= 2:
                sents.append(ex[0])
                labels.append(ex[1])
            else:
                raise TypeError(
                    f"Cannot read (sentence, label) from {type(ex).__name__}."
                )
        return cls(sents, labels)


@dataclass(frozen=True)
class StereotypeLabelled:
    """Two labelled sentence sets: stereotypical and anti-stereotypical (SD)."""

    stereotype: LabeledSentences
    anti_stereotype: LabeledSentences

    def __post_init__(self) -> None:
        for role in ("stereotype", "anti_stereotype"):
            value = getattr(self, role)
            if not isinstance(value, LabeledSentences):
                raise TypeError(
                    f"{role} must be a LabeledSentences, got {type(value).__name__}."
                )


# ---------------------------------------------------------------------------
# Group / prediction structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GroupPredictions:
    """Ground-truth labels, predictions, and a protected group per instance."""

    y_true: Sequence[Any]
    y_pred: Sequence[Any]
    groups: Sequence[Any]

    def __post_init__(self) -> None:
        for role in ("y_true", "y_pred", "groups"):
            object.__setattr__(self, role, _check_seq(getattr(self, role), role))
        _check_same_length(y_true=self.y_true, y_pred=self.y_pred, groups=self.groups)

    def __len__(self) -> int:
        return len(self.y_true)

    def unique_groups(self) -> Tuple[Any, ...]:
        seen = []
        for g in self.groups:
            if g not in seen:
                seen.append(g)
        return tuple(seen)

    @classmethod
    def from_examples(cls, examples: Sequence[Any]) -> "GroupPredictions":
        y_true, y_pred, groups = [], [], []
        for ex in examples:
            if not isinstance(ex, Mapping):
                raise TypeError(
                    f"Expected mappings with 'y_true'/'y_pred'/'group', got "
                    f"{type(ex).__name__}."
                )
            y_true.append(ex["y_true"])
            y_pred.append(ex["y_pred"])
            groups.append(ex.get("group", ex.get("groups")))
        return cls(y_true, y_pred, groups)


@dataclass(frozen=True)
class ScorePair:
    """Per-item scores for a stereotyped set and its counter-stereotyped twin."""

    stereotype: Sequence[float]
    counter_stereotype: Sequence[float]

    def __post_init__(self) -> None:
        for role in ("stereotype", "counter_stereotype"):
            object.__setattr__(self, role, _check_seq(getattr(self, role), role))

    @classmethod
    def from_examples(cls, examples: Sequence[Any]) -> "ScorePair":
        s, sp = [], []
        for ex in examples:
            if isinstance(ex, Mapping):
                s.append(ex["scores_s"])
                sp.append(ex["scores_sp"])
            elif isinstance(ex, (list, tuple)) and len(ex) >= 2:
                s.append(ex[0])
                sp.append(ex[1])
            else:
                raise TypeError(f"Cannot read a score pair from {type(ex).__name__}.")
        return cls(s, sp)


# ---------------------------------------------------------------------------
# Prompt-based structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PromptPairs:
    """Factual prompts and their counterfactual (attribute-swapped) twins."""

    factual: Sequence[str]
    counterfactual: Sequence[str]

    def __post_init__(self) -> None:
        for role in ("factual", "counterfactual"):
            object.__setattr__(self, role, _check_str_seq(getattr(self, role), role))
        _check_same_length(factual=self.factual, counterfactual=self.counterfactual)

    def __len__(self) -> int:
        return len(self.factual)

    @classmethod
    def from_examples(cls, examples: Sequence[Any]) -> "PromptPairs":
        fact, cf = [], []
        for ex in examples:
            if isinstance(ex, Mapping):
                fact.append(ex.get("factual") or ex.get("stereotype"))
                cf.append(ex.get("counterfactual") or ex.get("anti_stereotype"))
            elif isinstance(ex, (list, tuple)) and len(ex) >= 2:
                fact.append(ex[0])
                cf.append(ex[1])
            else:
                raise TypeError(f"Cannot read a prompt pair from {type(ex).__name__}.")
        return cls(fact, cf)


@dataclass(frozen=True)
class DemographicPrompts:
    """Prompts plus the demographic word lists used to score continuations.

    ``neutral`` is only required by metrics that normalise against a neutral
    baseline (DNP); leave it ``None`` for metrics that do not (DRD).
    """

    prompts: Sequence[str]
    stereotype_words: Sequence[str]
    counter_words: Sequence[str]
    neutral_words: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompts", _check_str_seq(self.prompts, "prompts"))
        for role in ("stereotype_words", "counter_words"):
            object.__setattr__(self, role, _check_str_seq(getattr(self, role), role))
        if self.neutral_words is not None:
            object.__setattr__(
                self,
                "neutral_words",
                _check_str_seq(self.neutral_words, "neutral_words"),
            )

    def require_neutral(self, metric_name: str) -> None:
        if self.neutral_words is None:
            raise ValueError(
                f"{metric_name} normalises against a neutral baseline, so "
                f"DemographicPrompts(neutral_words=...) is required."
            )


@dataclass(frozen=True)
class ConceptSpec:
    """Concepts to probe, a prompt template, and group term lists (CA)."""

    concepts: Sequence[str]
    prompt_template: str
    group_terms: Mapping[str, Sequence[str]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "concepts", _check_str_seq(self.concepts, "concepts"))
        if not isinstance(self.prompt_template, str):
            raise TypeError(
                f"prompt_template must be a string, got "
                f"{type(self.prompt_template).__name__}."
            )
        if "{concept}" not in self.prompt_template:
            raise ValueError(
                f"prompt_template must contain a '{{concept}}' placeholder; got "
                f"{self.prompt_template!r}. (Group terms are counted in the "
                f"generated text, not substituted into the prompt.)"
            )
        object.__setattr__(
            self, "group_terms", _check_dict_seq(self.group_terms, "group_terms")
        )


@dataclass(frozen=True)
class QuerySpec:
    """Queries plus the prompt builders and group values used to probe them (SNS)."""

    queries: Sequence[str]
    neutral_prompt_fn: Callable[[str], str]
    group_prompt_fn: Callable[[str, str], str]
    group_values: Sequence[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "queries", _check_str_seq(self.queries, "queries"))
        object.__setattr__(
            self, "group_values", _check_str_seq(self.group_values, "group_values")
        )
        for role in ("neutral_prompt_fn", "group_prompt_fn"):
            if not callable(getattr(self, role)):
                raise TypeError(f"{role} must be callable.")


@dataclass(frozen=True)
class GroupProperties:
    """Groups, properties, and the two prompt templates BiasAsker compares."""

    groups: Sequence[str]
    properties: Sequence[str]
    ab_template: str
    rb_template: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", _check_str_seq(self.groups, "groups"))
        object.__setattr__(
            self, "properties", _check_str_seq(self.properties, "properties")
        )
        if len(self.groups) < 2:
            raise ValueError(
                f"groups must contain at least two groups to compare, got "
                f"{len(self.groups)}."
            )
        for role, required in (
            ("ab_template", ("{gi}", "{gj}", "{prop}")),
            ("rb_template", ("{g}", "{prop}")),
        ):
            tmpl = getattr(self, role)
            if not isinstance(tmpl, str):
                raise TypeError(f"{role} must be a string, got {type(tmpl).__name__}.")
            missing = [p for p in required if p not in tmpl]
            if missing:
                raise ValueError(
                    f"{role} is missing placeholder(s) {', '.join(missing)}; got "
                    f"{tmpl!r}."
                )


# ---------------------------------------------------------------------------
# Masked-token / attention structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GroupWordPairs:
    """Aligned demographic word pairs, e.g. ("he","she"), ("man","woman") (DisCo)."""

    group_1: Sequence[str]
    group_2: Sequence[str]

    def __post_init__(self) -> None:
        for role in ("group_1", "group_2"):
            object.__setattr__(self, role, _check_str_seq(getattr(self, role), role))
        if len(self.group_1) != len(self.group_2):
            raise ValueError(
                f"group_1 and group_2 must be the same length (they are compared "
                f"pairwise), got {len(self.group_1)} and {len(self.group_2)}."
            )
        if list(self.group_1) == list(self.group_2):
            raise ValueError("group_1 and group_2 are identical; nothing to contrast.")


@dataclass(frozen=True)
class ContrastSpec:
    """Group terms, (negative, positive, stereo_group) triples, and templates (CBS)."""

    group_terms: Sequence[str]
    contrast_pairs: Sequence[Tuple[str, str, Optional[str]]]
    templates: Sequence[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "group_terms", _check_str_seq(self.group_terms, "group_terms")
        )
        object.__setattr__(
            self, "templates", _check_str_seq(self.templates, "templates")
        )
        triples = _check_seq(self.contrast_pairs, "contrast_pairs")
        cleaned = []
        for i, item in enumerate(triples):
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError(
                    f"contrast_pairs[{i}] must be a 3-tuple "
                    f"(negative, positive, stereo_group), got {item!r}. "
                    f"Use None for stereo_group when there is no declared target."
                )
            neg, pos, grp = item
            if grp is not None and grp not in self.group_terms:
                raise ValueError(
                    f"contrast_pairs[{i}] declares stereo_group {grp!r}, which is "
                    f"not in group_terms {list(self.group_terms)}."
                )
            cleaned.append((neg, pos, grp))
        object.__setattr__(self, "contrast_pairs", tuple(cleaned))


@dataclass(frozen=True)
class ProbeSet:
    """Causal-mediation probes: prompt, counterfactual text, and token ids (NIE)."""

    probes: Sequence[Mapping[str, Any]]

    _REQUIRED = ("prompt", "cf_text", "stereo_token_id", "anti_token_id")

    def __post_init__(self) -> None:
        items = _check_seq(self.probes, "probes")
        for i, probe in enumerate(items):
            if not isinstance(probe, Mapping):
                raise TypeError(
                    f"probes[{i}] must be a mapping with keys "
                    f"{', '.join(self._REQUIRED)}, got {type(probe).__name__}."
                )
            missing = [k for k in self._REQUIRED if k not in probe]
            if missing:
                raise ValueError(
                    f"probes[{i}] is missing {', '.join(missing)}. Each probe needs "
                    f"{', '.join(self._REQUIRED)}."
                )
        object.__setattr__(self, "probes", tuple(items))

    def __len__(self) -> int:
        return len(self.probes)


@dataclass(frozen=True)
class OccupationTriples:
    """(occupation, stereotype_word, counter_word) triples (SLL)."""

    triples: Sequence[Tuple[str, str, str]]

    def __post_init__(self) -> None:
        items = _check_seq(self.triples, "triples")
        for i, item in enumerate(items):
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError(
                    f"triples[{i}] must be a 3-tuple "
                    f"(occupation, stereotype_word, counter_word), got {item!r}."
                )
        object.__setattr__(self, "triples", tuple(tuple(t) for t in items))

    def __len__(self) -> int:
        return len(self.triples)
