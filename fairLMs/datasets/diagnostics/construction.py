"""Construction-bias diagnostics: the eight-slot ``b_constr`` vector.

Construction bias is reported as a **vector** over eight stable, ordered slots.
Every slot is always present in the report, every slot carries its own
applicability status, and there is deliberately **no aggregate construction
score** anywhere in this module: a mixed ready / not-applicable / blocked
vector is the normal, expected result.

Five slots are implemented on the lightweight standard-library path
(``b_min``, ``b_diff_len``, ``b_frame``, ``b_opt``, ``b_temp``).  The other
three (``b_equiv``, ``b_gram``, ``b_diff_dep``) require an optional backend
that this release does not ship.  Each is synthesized rather than registered
as a stub, and the backend is checked *last*: the slot is BLOCKED naming the
specific backend requirement it needs only once it has been requested and the
evidence view it needs is present for the audited axis, and is
NOT_APPLICABLE for the ordinary reason before that.  A missing backend blocks
only its own slot: it never raises, never changes another slot's status, and
never prevents the report from being produced.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from fairLMs.datasets.diagnostics._kernels import max_pairwise_gap, token_levenshtein
from fairLMs.datasets.diagnostics._messages import (
    DESCRIPTIVE_INTERPRETATION,
    STRESS_TEST_WARNING,
)
from fairLMs.datasets.diagnostics._utils import (
    freeze_json_mapping,
    json_digest,
    normalize_enum,
    normalize_string_sequence,
    require_nonempty_string,
    thaw_json,
)

from .base import (
    ComponentPlan,
    ComponentResult,
    DatasetDiagnostic,
    DiagnosticReport,
    DiagnosticStatus,
)
from .evidence import (
    DatasetEvidence,
    GroupedTexts,
    OptionItems,
    PairedTexts,
    TemplateGroups,
)
from .spec import DatasetAuditSpec, DesignStance, TargetKind


# --------------------------------------------------------------------------
# Frozen slot vector
# --------------------------------------------------------------------------

CONSTRUCTION_SLOTS: Final[tuple[str, ...]] = (
    "b_min",
    "b_equiv",
    "b_gram",
    "b_diff_len",
    "b_diff_dep",
    "b_frame",
    "b_opt",
    "b_temp",
)
LIGHTWEIGHT_CONSTRUCTION_SLOTS: Final[tuple[str, ...]] = (
    "b_min",
    "b_diff_len",
    "b_frame",
    "b_opt",
    "b_temp",
)
BACKEND_CONSTRUCTION_SLOTS: Final[tuple[str, ...]] = (
    "b_equiv",
    "b_gram",
    "b_diff_dep",
)

# ---------------------------------------------------------------------------
# Optional backend protocols
# ---------------------------------------------------------------------------
# The three backend-dependent slots are blocked in this release. These protocols
# exist so the names the refusal metadata cites are real types a caller can
# implement against, and so a future backend has a contract to satisfy rather
# than one invented at integration time. Nothing here imports a model, a parser
# or a checker: they are structural types, checked at type-check time only.


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Sentence embeddings for the ``b_equiv`` semantic-equivalence slot."""

    #: Identifies the model and revision that produced the vectors. Recorded in
    #: provenance, because a similarity is only comparable against itself.
    revision: str

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one fixed-width vector per text, in the order given."""


@runtime_checkable
class GrammarCheckerBackend(Protocol):
    """Grammatical-error counts for the ``b_gram`` slot."""

    #: Checker name, version and language, recorded in provenance.
    revision: str

    def count_errors(self, texts: Sequence[str]) -> Sequence[int]:
        """Return the number of flagged errors per text, in the order given."""


@runtime_checkable
class DependencyParserBackend(Protocol):
    """Dependency depth for the ``b_diff_dep`` slot."""

    #: Parser name, model and version, recorded in provenance.
    revision: str

    def depths(self, texts: Sequence[str]) -> Sequence[int]:
        """Return the dependency-tree depth per text, in the order given."""


CONSTRUCTION_BACKEND_REQUIREMENTS: Final[Mapping[str, Mapping[str, str]]] = (
    MappingProxyType(
        {
            "b_equiv": MappingProxyType(
                {
                    "reason_code": "embedding_backend_unavailable",
                    "required_backend": "sentence_embedding",
                    "required_protocol": "EmbeddingBackend",
                    "availability": "optional_backend",
                    "required_view": "paired_texts",
                    "milestone": "0.5.0",
                }
            ),
            "b_gram": MappingProxyType(
                {
                    "reason_code": "grammar_backend_unavailable",
                    "required_backend": "grammar_checker",
                    "required_protocol": "GrammarCheckerBackend",
                    "availability": "optional_backend",
                    "required_view": "paired_texts",
                    "milestone": "0.5.0",
                }
            ),
            "b_diff_dep": MappingProxyType(
                {
                    "reason_code": "dependency_parser_backend_unavailable",
                    "required_backend": "dependency_parser",
                    "required_protocol": "DependencyParserBackend",
                    "availability": "optional_backend",
                    "required_view": "grouped_texts",
                    "milestone": "0.5.0",
                }
            ),
        }
    )
)

_BACKEND_BLOCKED_REASONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "b_equiv": (
            "b_equiv requires a sentence-embedding backend; no EmbeddingBackend "
            "was supplied. Pass backend=..., for example "
            "fairLMs.datasets.diagnostics.backends.HuggingFaceEmbeddingBackend()."
        ),
        "b_gram": (
            "b_gram requires a grammar-checking backend; no GrammarCheckerBackend "
            "was supplied. Pass backend=..., for example "
            "fairLMs.datasets.diagnostics.backends.LanguageToolGrammarBackend()."
        ),
        "b_diff_dep": (
            "b_diff_dep requires a dependency-parser backend; no "
            "DependencyParserBackend was supplied. Pass backend=..., for example "
            "fairLMs.datasets.diagnostics.backends.SpacyDependencyBackend()."
        ),
    }
)

_SLOT_VIEWS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "b_min": "paired_texts",
        "b_equiv": "paired_texts",
        "b_gram": "paired_texts",
        "b_diff_len": "grouped_texts",
        "b_diff_dep": "grouped_texts",
        "b_frame": "grouped_texts",
        "b_opt": "option_items",
        "b_temp": "template_groups",
    }
)

_MAX_LISTED_IDS: Final[int] = 20


# --------------------------------------------------------------------------
# Report-level warnings
# --------------------------------------------------------------------------

CONSTRUCTION_VECTOR_WARNING = (
    "Construction bias is reported as a vector of independent components in "
    "the declared slot order. The package deliberately publishes no aggregate "
    "construction score, and blocked or not-applicable slots must not be read "
    "as zero."
)
BACKEND_BLOCKED_WARNING = (
    "Component(s) {slots} are blocked because their optional backend is "
    "unavailable; every other construction component is unaffected."
)
TOKENIZATION_DIVERGENCE_WARNING = (
    "b_diff_len measured length in declared surface tokens, not the paper's "
    "parser tokens; the value is not numerically comparable to a published "
    "spaCy-tokenized B_diff_len."
)
INJECTED_PREDICATE_WARNING = (
    "The frame predicate is an injected callable; its declared definition is "
    "recorded, but the result is not replayable from provenance alone."
)


def _enum_value(value: Any) -> Any:
    """Return a stable scalar for enum-like public values."""
    return getattr(value, "value", value)


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean, got {type(value).__name__}.")
    return value


def _compile_pattern(pattern: str, *, field_name: str) -> "re.Pattern[str]":
    require_nonempty_string(pattern, field_name)
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(
            f"{field_name} must be a valid regular expression: {exc}."
        ) from exc
    return compiled


# --------------------------------------------------------------------------
# Shared configuration objects
# --------------------------------------------------------------------------


class TokenizationMode(str, Enum):
    """How a declared surface tokenization splits a text into tokens."""

    WHITESPACE = "whitespace"
    REGEX = "regex"


@dataclass(frozen=True, kw_only=True)
class TokenizationRule:
    """Explicit, portable surface tokenization used for length counting.

    This is the declared lightweight replacement for the paper's parser token
    counts, so the base installation needs no parser. There is deliberately no
    ``lowercase`` knob: token *counts*, the only quantity these diagnostics
    use, are invariant to case under both modes, and an inert knob invites
    confusion.
    """

    mode: TokenizationMode = TokenizationMode.WHITESPACE
    pattern: Optional[str] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    rule_digest: str = field(init=False)
    _compiled: Optional["re.Pattern[str]"] = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self) -> None:
        mode = normalize_enum(self.mode, TokenizationMode, "mode")
        object.__setattr__(self, "mode", mode)
        if mode is TokenizationMode.REGEX:
            if self.pattern is None:
                raise ValueError("pattern is required when mode is 'regex'.")
            compiled = _compile_pattern(self.pattern, field_name="pattern")
            if compiled.search("") is not None:
                raise ValueError("pattern must not match the empty string.")
            object.__setattr__(self, "_compiled", compiled)
        elif self.pattern is not None:
            raise ValueError("pattern must be None when mode is 'whitespace'.")
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )
        object.__setattr__(
            self,
            "rule_digest",
            json_digest(
                {"mode": mode.value, "pattern": self.pattern},
                path="rule_digest",
            ),
        )

    def tokenize(self, text: str) -> tuple[str, ...]:
        """Split *text* into declared surface tokens."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}.")
        if self.mode is TokenizationMode.WHITESPACE:
            return tuple(text.split())
        assert self._compiled is not None  # guaranteed by __post_init__
        return tuple(self._compiled.findall(text))

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "mode": self.mode.value,
            "pattern": self.pattern,
            "rule_digest": self.rule_digest,
            "provenance": thaw_json(self.provenance),
        }


@dataclass(frozen=True, kw_only=True)
class IdentityMaskConfig:
    """Declared identity terms replaced by a placeholder before comparison.

    There is deliberately no "mask disabled" mode: ``b_min`` without an
    identity mask measures raw edit distance, which is a different estimand,
    so the absence of a mask blocks rather than silently re-labelling.
    """

    identity_terms: Sequence[str]
    placeholder: str = "[ID]"
    token_normalization_pattern: str = r"[^\w]"
    match_phrases: bool = True
    provenance: Mapping[str, Any] = field(default_factory=dict)
    canonical_single_terms: tuple[str, ...] = field(init=False)
    canonical_phrase_forms: tuple[tuple[str, ...], ...] = field(init=False)
    mask_digest: str = field(init=False)
    _normalizer: Optional["re.Pattern[str]"] = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )
    _max_phrase_length: int = field(
        init=False,
        repr=False,
        compare=False,
        default=0,
    )

    def __post_init__(self) -> None:
        terms = normalize_string_sequence(
            self.identity_terms,
            field_name="identity_terms",
            allow_empty=False,
        )
        object.__setattr__(
            self,
            "placeholder",
            require_nonempty_string(self.placeholder, "placeholder"),
        )
        normalizer = _compile_pattern(
            self.token_normalization_pattern,
            field_name="token_normalization_pattern",
        )
        object.__setattr__(self, "_normalizer", normalizer)
        object.__setattr__(
            self,
            "match_phrases",
            _require_bool(self.match_phrases, "match_phrases"),
        )

        canonical_terms = sorted({term.strip().lower() for term in terms})
        if not canonical_terms:
            raise ValueError("identity_terms must contain at least one term.")

        single_terms: set[str] = set()
        phrase_forms: set[tuple[str, ...]] = set()
        for term in canonical_terms:
            pieces = tuple(
                piece
                for piece in (
                    self._normalize_token(part)
                    for part in re.split(r"[\s_-]+", term)
                )
                if piece
            )
            if len(pieces) >= 2:
                phrase_forms.add(pieces)
                single_terms.add("".join(pieces))
            elif len(pieces) == 1:
                single_terms.add(pieces[0])
            stripped = self._normalize_token(term)
            if stripped:
                single_terms.add(stripped)
        if not single_terms and not phrase_forms:
            raise ValueError(
                "identity_terms must contain at least one term that survives "
                "token normalization."
            )

        object.__setattr__(self, "identity_terms", tuple(canonical_terms))
        object.__setattr__(self, "canonical_single_terms", tuple(sorted(single_terms)))
        object.__setattr__(
            self,
            "canonical_phrase_forms",
            tuple(sorted(phrase_forms)),
        )
        object.__setattr__(
            self,
            "_max_phrase_length",
            max((len(form) for form in phrase_forms), default=0),
        )
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )
        object.__setattr__(
            self,
            "mask_digest",
            json_digest(
                {
                    "identity_terms": list(self.identity_terms),
                    "placeholder": self.placeholder,
                    "token_normalization_pattern": self.token_normalization_pattern,
                    "match_phrases": self.match_phrases,
                    "single_terms": list(self.canonical_single_terms),
                    "phrase_forms": [
                        list(form) for form in self.canonical_phrase_forms
                    ],
                },
                path="mask_digest",
            ),
        )

    def _normalize_token(self, token: str) -> str:
        assert self._normalizer is not None  # guaranteed by __post_init__
        return self._normalizer.sub("", token.lower())

    def mask(self, text: str) -> tuple[str, ...]:
        """Return *text* as tokens with every declared identity span masked.

        Words are split on whitespace and normalized for matching only.
        Declared phrases are matched longest-first, then single terms; an
        unmatched word contributes its normalized token, and a word that
        normalizes to nothing contributes no token at all (the canonical
        implementation joins on spaces and re-splits, which drops it).
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}.")
        words = text.split()
        normalized = [self._normalize_token(word) for word in words]
        phrase_set = set(self.canonical_phrase_forms)
        single_set = set(self.canonical_single_terms)

        masked: list[str] = []
        index = 0
        while index < len(words):
            if self.match_phrases and phrase_set:
                span = min(self._max_phrase_length, len(words) - index)
                matched_length = 0
                while span >= 2:
                    if tuple(normalized[index : index + span]) in phrase_set:
                        matched_length = span
                        break
                    span -= 1
                if matched_length:
                    masked.append(self.placeholder)
                    index += matched_length
                    continue
            token = normalized[index]
            if token in single_set:
                masked.append(self.placeholder)
            elif token:
                masked.append(token)
            index += 1
        return tuple(masked)

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "identity_terms": list(self.identity_terms),
            "placeholder": self.placeholder,
            "token_normalization_pattern": self.token_normalization_pattern,
            "match_phrases": self.match_phrases,
            "single_terms": list(self.canonical_single_terms),
            "phrase_forms": [list(form) for form in self.canonical_phrase_forms],
            "mask_digest": self.mask_digest,
            "provenance": thaw_json(self.provenance),
        }


@dataclass(frozen=True, kw_only=True)
class OptionRoleContrast:
    """Which two declared option roles form the stereotype contrast.

    The contrast is always stated by role name. Option position is never a
    role and is never consulted.
    """

    stereotype_role: str
    anti_stereotype_role: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stereotype_role",
            require_nonempty_string(self.stereotype_role, "stereotype_role"),
        )
        object.__setattr__(
            self,
            "anti_stereotype_role",
            require_nonempty_string(
                self.anti_stereotype_role, "anti_stereotype_role"
            ),
        )
        if self.stereotype_role == self.anti_stereotype_role:
            raise ValueError(
                "stereotype_role and anti_stereotype_role must name distinct roles."
            )
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "stereotype_role": self.stereotype_role,
            "anti_stereotype_role": self.anti_stereotype_role,
            "orientation": "stereotype_minus_anti_stereotype",
            "provenance": thaw_json(self.provenance),
        }


class FrameMatchMode(str, Enum):
    """How a declared framing predicate matches its patterns."""

    SUBSTRING = "substring"
    REGEX = "regex"


#: The paper's canonical self-identification phrase set, in the paper's order.
_PAPER_FRAME_PHRASES: Final = ("i am", "i'm", "as a", "as an")

#: The same phrase set anchored at word boundaries, which is what
#: :data:`SELF_IDENTIFICATION_FRAME` ships. Read as unanchored substrings,
#: ``"as a"`` and ``"as an"`` fire inside ordinary words -- ``"he w-as a-
#: doctor"``, ``"she h-as an- idea"``, ``"overse-as an-d abroad"`` -- and
#: ``"i am"`` fires inside ``"Hawai-i am-azing"``. Those are not framing
#: choices, so matching them inflates every group's frame rate. Because
#: b_frame reports the *gap* between group rates the error does not cancel:
#: it tracks whichever group happens to use more past-tense ``"was a"``
#: phrasing, manufacturing exactly the construction artifact the component
#: exists to detect. One pattern per phrase, rather than a merged
#: ``\bas an?\b``, keeps the recorded patterns one-to-one with the paper's
#: four phrases for the audit trail.
_PAPER_FRAME_PATTERNS: Final = (r"\bi am\b", r"\bi'm\b", r"\bas a\b", r"\bas an\b")


@dataclass(frozen=True, kw_only=True)
class FramePredicate:
    """Declared, fully serializable, replayable framing predicate."""

    frame_name: str
    definition: str
    patterns: Sequence[str]
    match_mode: FrameMatchMode = FrameMatchMode.SUBSTRING
    case_fold: bool = True
    provenance: Mapping[str, Any] = field(default_factory=dict)
    predicate_digest: str = field(init=False)
    predicate_kind: ClassVar[str] = "declared_patterns"
    replayable: ClassVar[bool] = True
    _compiled: tuple["re.Pattern[str]", ...] = field(
        init=False,
        repr=False,
        compare=False,
        default=(),
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame_name",
            require_nonempty_string(self.frame_name, "frame_name"),
        )
        object.__setattr__(
            self,
            "definition",
            require_nonempty_string(self.definition, "definition"),
        )
        patterns = normalize_string_sequence(
            self.patterns,
            field_name="patterns",
            allow_empty=False,
        )
        if len(set(patterns)) != len(patterns):
            raise ValueError("patterns must not contain duplicate patterns.")
        object.__setattr__(self, "patterns", patterns)
        object.__setattr__(
            self,
            "match_mode",
            normalize_enum(self.match_mode, FrameMatchMode, "match_mode"),
        )
        object.__setattr__(
            self,
            "case_fold",
            _require_bool(self.case_fold, "case_fold"),
        )
        # ``matches`` folds the *haystack* only, so under substring matching a
        # pattern carrying an uppercase character could never fire: every
        # group's frame rate would silently be 0/n and the slot would publish
        # a ready 0.0 from evidence that was never actually examined.  The
        # predicate is the estimand, so a mis-cased declared pattern is a
        # declaration error and is refused rather than silently coerced.
        # ``FrameMatchMode.REGEX`` is unaffected: it compiles with
        # ``re.IGNORECASE``, which folds both sides.
        if self.match_mode is FrameMatchMode.SUBSTRING and self.case_fold:
            mis_cased = tuple(
                pattern for pattern in patterns if pattern != pattern.lower()
            )
            if mis_cased:
                raise ValueError(
                    "case_fold=True substring matching lower-cases the text "
                    "only, so every declared pattern must already be "
                    f"lower-case; got {list(mis_cased)!r}. Supply "
                    f"{[pattern.lower() for pattern in mis_cased]!r}, or set "
                    "case_fold=False for a case-sensitive predicate, or use "
                    "match_mode=FrameMatchMode.REGEX, which folds both sides."
                )
        if self.match_mode is FrameMatchMode.REGEX:
            flags = re.IGNORECASE if self.case_fold else 0
            compiled = []
            for pattern in patterns:
                try:
                    compiled.append(re.compile(pattern, flags))
                except re.error as exc:
                    raise ValueError(
                        f"pattern {pattern!r} must be a valid regular "
                        f"expression: {exc}."
                    ) from exc
            object.__setattr__(self, "_compiled", tuple(compiled))
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )
        object.__setattr__(
            self,
            "predicate_digest",
            json_digest(
                {
                    "frame_name": self.frame_name,
                    "definition": self.definition,
                    "predicate_kind": self.predicate_kind,
                    "replayable": self.replayable,
                    "patterns": list(self.patterns),
                    "match_mode": self.match_mode.value,
                    "case_fold": self.case_fold,
                },
                path="predicate_digest",
            ),
        )

    @property
    def paper_alignment(self) -> str:
        """Classify the predicate against the paper's canonical frame set.

        Three outcomes rather than two, because "the paper's phrase set" and
        "the paper's phrase set read as raw substrings" are different
        estimands and only one of them is worth shipping:

        ``paper_phrase_set_word_anchored``
            The canonical phrases matched at word boundaries. What
            :data:`SELF_IDENTIFICATION_FRAME` ships.
        ``paper_exact``
            The canonical phrases read as unanchored substrings, which is
            literally what a published implementation does and therefore the
            only setting whose numbers are bit-comparable to a published
            value. It also fires inside ``"was a"`` and ``"has an"``; a caller
            who needs that comparability can still declare it, and this label
            is how the report says so.
        ``generalized_frame_predicate``
            Anything else.
        """
        if not self.case_fold:
            return "generalized_frame_predicate"
        patterns = tuple(self.patterns)
        if self.match_mode is FrameMatchMode.REGEX:
            if patterns == _PAPER_FRAME_PATTERNS:
                return "paper_phrase_set_word_anchored"
            return "generalized_frame_predicate"
        if patterns == _PAPER_FRAME_PHRASES:
            return "paper_exact"
        return "generalized_frame_predicate"

    def matches(self, text: str) -> bool:
        """Return whether *text* exhibits the declared frame.

        Under ``FrameMatchMode.SUBSTRING`` only the haystack is folded; this
        is safe because ``__post_init__`` refuses a pattern that is not
        already lower-case whenever ``case_fold`` is set.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}.")
        if self.match_mode is FrameMatchMode.SUBSTRING:
            haystack = text.lower() if self.case_fold else text
            return any(pattern in haystack for pattern in self.patterns)
        return any(compiled.search(text) is not None for compiled in self._compiled)

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "frame_name": self.frame_name,
            "definition": self.definition,
            "predicate_kind": self.predicate_kind,
            "replayable": self.replayable,
            "patterns": list(self.patterns),
            "match_mode": self.match_mode.value,
            "case_fold": self.case_fold,
            "predicate_digest": self.predicate_digest,
            "provenance": thaw_json(self.provenance),
        }


@dataclass(frozen=True, kw_only=True)
class InjectedFramePredicate:
    """Caller-injected callable predicate; provenance is honest about replay.

    The callable itself is never serialized, so a result produced with an
    injected predicate is documented but not replayable from provenance alone.
    """

    frame_name: str
    definition: str
    predicate_id: str
    predicate_version: str
    predicate: Callable[[str], bool] = field(repr=False, compare=False)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    predicate_digest: str = field(init=False)
    predicate_kind: ClassVar[str] = "injected_callable"
    replayable: ClassVar[bool] = False

    def __post_init__(self) -> None:
        for name in ("frame_name", "definition", "predicate_id", "predicate_version"):
            object.__setattr__(
                self, name, require_nonempty_string(getattr(self, name), name)
            )
        if not callable(self.predicate):
            raise TypeError(
                f"predicate must be callable, got {type(self.predicate).__name__}."
            )
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )
        object.__setattr__(
            self,
            "predicate_digest",
            json_digest(
                {
                    "frame_name": self.frame_name,
                    "definition": self.definition,
                    "predicate_kind": self.predicate_kind,
                    "replayable": self.replayable,
                    "predicate_id": self.predicate_id,
                    "predicate_version": self.predicate_version,
                },
                path="predicate_digest",
            ),
        )

    def matches(self, text: str) -> bool:
        """Return the injected predicate's verdict, refusing a non-bool."""
        outcome = self.predicate(text)
        if not isinstance(outcome, bool):
            raise TypeError(
                "the injected frame predicate must return a bool, got "
                f"{type(outcome).__name__}."
            )
        return outcome

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "frame_name": self.frame_name,
            "definition": self.definition,
            "predicate_kind": self.predicate_kind,
            "replayable": self.replayable,
            "predicate_id": self.predicate_id,
            "predicate_version": self.predicate_version,
            "predicate_digest": self.predicate_digest,
            "provenance": thaw_json(self.provenance),
        }


FramePredicateLike = FramePredicate | InjectedFramePredicate

SELF_IDENTIFICATION_FRAME: Final = FramePredicate(
    frame_name="self_identification",
    definition=(
        "A text frames the subject in the first person or as a member of a "
        "category, detected by the paper's canonical self-identification "
        "phrase set matched at word boundaries. The phrases are the paper's; "
        "the word anchoring is a declared divergence from reading them as raw "
        "substrings, which would also fire inside 'was a', 'has an' and "
        "'Hawaii amazing' and so report framing where none occurs."
    ),
    patterns=_PAPER_FRAME_PATTERNS,
    match_mode=FrameMatchMode.REGEX,
    case_fold=True,
)


# --------------------------------------------------------------------------
# Shared applicability helpers
# --------------------------------------------------------------------------


def _component_override(spec: DatasetAuditSpec, component: str) -> Optional[Any]:
    """Return the caller-declared override for *component*, when there is one.

    ``component_overrides`` is an additive ``DatasetAuditSpec`` field; the
    lookup is defensive so this module keeps working against a spec revision
    that predates it.
    """
    overrides = getattr(spec, "component_overrides", None)
    if not isinstance(overrides, Mapping):
        return None
    return overrides.get(component)


def _protected_axes(spec: DatasetAuditSpec) -> tuple[str, ...]:
    """Return the declared protected axes, tolerating an older spec."""
    axes = getattr(spec, "protected_axes", ())
    if isinstance(axes, (str, bytes)) or not isinstance(axes, Sequence):
        return ()
    return tuple(axes)


def _shared_plan_prefix(
    *,
    component: str,
    spec: DatasetAuditSpec,
    axis: str,
    check_axis: bool = True,
) -> Optional[ComponentPlan]:
    """Apply the fixed precedence shared by every construction component.

    Steps 1-3 (not requested, caller override, unsupported target kind) and
    step 5 (undeclared axis) are identical everywhere. Step 4, evidence-view
    geometry, is owned by :func:`audit_construction` because a diagnostic
    raises ``TypeError`` on wrong evidence and must never be handed an absent
    view.
    """
    requested = spec.requested_components
    if requested is not None and component not in requested:
        return ComponentPlan(
            component=component,
            status=DiagnosticStatus.NOT_APPLICABLE,
            reason_code="component_not_requested",
            reason=(
                f"The {component} component was not requested by the audit "
                "specification."
            ),
            details={"axis": axis},
        )

    override = _component_override(spec, component)
    if override is not None:
        return ComponentPlan(
            component=component,
            status=override.status,
            reason_code="applicability_override",
            reason=override.reason,
            details={
                "applicability_override": True,
                "declared_reason_code": override.declared_reason_code,
                "override_source": "spec",
            },
        )

    if spec.target_kind is not TargetKind.BENCHMARK_DATASET:
        return ComponentPlan(
            component=component,
            status=DiagnosticStatus.NOT_APPLICABLE,
            reason_code="target_kind_not_supported",
            reason=(
                f"{component} applies to benchmark-dataset construction, not "
                f"target kind {_enum_value(spec.target_kind)!r}."
            ),
            details={"axis": axis},
        )

    if check_axis:
        declared_axes = _protected_axes(spec)
        if declared_axes and axis not in declared_axes:
            return ComponentPlan(
                component=component,
                status=DiagnosticStatus.BLOCKED,
                reason_code="axis_not_declared",
                reason=(
                    f"Axis {axis!r} is not declared in protected_axes "
                    f"{list(declared_axes)!r}."
                ),
                details={
                    "axis": axis,
                    "protected_axes": list(declared_axes),
                },
            )
    return None


def _empty_declared_groups(counts: Mapping[str, int]) -> list[str]:
    return [label for label in sorted(counts) if counts[label] == 0]


def _empty_group_plan(
    *,
    component: str,
    axis: str,
    empty_groups: Sequence[str],
    support: Sequence[str],
    noun: str = "texts",
) -> ComponentPlan:
    return ComponentPlan(
        component=component,
        status=DiagnosticStatus.BLOCKED,
        reason_code="empty_declared_group",
        reason=(
            f"Declared group(s) {list(empty_groups)!r} contain no {noun}; an "
            "empty declared group is blocked rather than reported as zero."
        ),
        details={
            "axis": axis,
            "support": list(support),
            "empty_groups": list(empty_groups),
        },
    )


def _non_ready_result(
    plan: ComponentPlan,
    *,
    spec: DatasetAuditSpec,
    provenance: Mapping[str, Any],
) -> ComponentResult:
    """Convert a non-ready plan into a value-free component result."""
    payload = dict(provenance)
    if plan.reason_code == "applicability_override":
        override = _component_override(spec, plan.component)
        if override is not None:
            payload["override"] = override.to_dict()
    return ComponentResult(
        component=plan.component,
        status=plan.status,
        details=dict(plan.details),
        provenance=payload,
        reason_code=plan.reason_code,
        reason=plan.reason,
    )


def _numeric_failure(
    *,
    component: str,
    details: Mapping[str, Any],
    provenance: Mapping[str, Any],
    reason: str,
) -> ComponentResult:
    return ComponentResult(
        component=component,
        status=DiagnosticStatus.FAILED,
        details=dict(details),
        provenance=dict(provenance),
        reason_code="numeric_computation_failed",
        reason=reason,
    )


def _grouped_evidence_metadata(evidence: GroupedTexts) -> dict[str, Any]:
    return {
        "axis": evidence.axis,
        "source": evidence.source,
        "support": list(evidence.support),
        "total": evidence.total,
        "provenance": dict(evidence.provenance),
    }


# --------------------------------------------------------------------------
# b_min
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class MinimalPairResidual(DatasetDiagnostic):
    """Mean normalized token edit distance between identity-masked pair sides."""

    name: ClassVar[str] = "b_min"
    identity_mask: Optional[IdentityMaskConfig] = None

    def __post_init__(self) -> None:
        if self.identity_mask is not None and not isinstance(
            self.identity_mask, IdentityMaskConfig
        ):
            raise TypeError(
                "identity_mask must be an IdentityMaskConfig or None, got "
                f"{type(self.identity_mask).__name__}."
            )

    @property
    def paper_alignment(self) -> str:
        """The lightweight path reproduces the paper's b_min definition."""
        return "paper_exact"

    def _provenance(self, evidence: PairedTexts) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "evidence": {
                "axis": evidence.axis,
                "source": evidence.source,
                "pairing_basis": evidence.pairing_basis,
                "condition_roles": list(evidence.condition_roles),
                "pair_count": evidence.pair_count,
                "provenance": dict(evidence.provenance),
            }
        }
        if self.identity_mask is not None:
            payload["identity_mask"] = self.identity_mask.to_dict()
        return payload

    def plan(self, evidence: PairedTexts, spec: DatasetAuditSpec) -> ComponentPlan:
        """Decide b_min applicability without measuring any distance."""
        if not isinstance(evidence, PairedTexts):
            raise TypeError(
                f"evidence must be PairedTexts, got {type(evidence).__name__}."
            )
        if not isinstance(spec, DatasetAuditSpec):
            raise TypeError(
                f"spec must be DatasetAuditSpec, got {type(spec).__name__}."
            )

        prefix = _shared_plan_prefix(
            component=self.name, spec=spec, axis=evidence.axis
        )
        if prefix is not None:
            return prefix

        if self.identity_mask is None:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.BLOCKED,
                reason_code="missing_identity_mask",
                reason=(
                    "b_min requires an explicit identity-term mask; identity "
                    "terms are never inferred from the paired texts."
                ),
                details={"axis": evidence.axis, "pair_count": evidence.pair_count},
            )

        return ComponentPlan(
            component=self.name,
            status=DiagnosticStatus.READY,
            details={
                "axis": evidence.axis,
                "pair_count": evidence.pair_count,
                "condition_roles": list(evidence.condition_roles),
                "pairing_basis": evidence.pairing_basis,
            },
        )

    def compute(self, evidence: PairedTexts, spec: DatasetAuditSpec) -> ComponentResult:
        """Compute the mean residual edit distance after identity masking."""
        plan = self.plan(evidence, spec)
        provenance = self._provenance(evidence)
        if plan.status is not DiagnosticStatus.READY:
            return _non_ready_result(plan, spec=spec, provenance=provenance)

        mask = self.identity_mask
        assert mask is not None  # guaranteed by the ready plan

        distances: list[float] = []
        masked_token_counts: list[int] = []
        degenerate_ids: list[Any] = []
        for start in range(0, evidence.total, 2):
            pair_id = evidence.pair_ids[start]
            left = mask.mask(evidence.texts[start])
            right = mask.mask(evidence.texts[start + 1])
            masked_token_counts.append(len(left))
            masked_token_counts.append(len(right))
            denominator = max(len(left), len(right))
            if denominator == 0:
                degenerate_ids.append(pair_id)
                continue
            distances.append(token_levenshtein(left, right) / denominator)

        if degenerate_ids:
            return ComponentResult(
                component=self.name,
                status=DiagnosticStatus.BLOCKED,
                details={
                    "axis": evidence.axis,
                    "pair_count": evidence.pair_count,
                    "degenerate_pair_ids": list(degenerate_ids[:_MAX_LISTED_IDS]),
                    "degenerate_pair_count": len(degenerate_ids),
                },
                provenance=provenance,
                reason_code="degenerate_masked_pair",
                reason=(
                    "At least one pair reduces to two empty token sequences "
                    "after identity masking, so its normalization denominator "
                    "is zero; a zero denominator is blocked rather than "
                    "clamped to one."
                ),
            )

        value = math.fsum(distances) / len(distances)
        mean_masked_token_count = math.fsum(masked_token_counts) / len(
            masked_token_counts
        )
        if not math.isfinite(value):
            return _numeric_failure(
                component=self.name,
                details={"axis": evidence.axis, "pair_count": evidence.pair_count},
                provenance=provenance,
                reason=(
                    "The mean normalized edit distance did not produce a finite "
                    "value."
                ),
            )

        non_zero_ratio = sum(1 for distance in distances if distance > 0.0) / len(
            distances
        )
        return ComponentResult(
            component=self.name,
            status=DiagnosticStatus.READY,
            value=value,
            details={
                "axis": evidence.axis,
                "pair_count": evidence.pair_count,
                "condition_roles": list(evidence.condition_roles),
                "pairing_basis": evidence.pairing_basis,
                "mean_normalized_edit_distance": value,
                "non_zero_ratio": non_zero_ratio,
                "minimum_normalized_edit_distance": min(distances),
                "maximum_normalized_edit_distance": max(distances),
                "mean_masked_token_count": mean_masked_token_count,
                "identity_term_count": len(mask.identity_terms),
                "estimator": "uniform_mass_per_pair",
                "unit": "normalized_token_edit_distance",
                "tokenization": "whitespace_after_identity_masking",
                "paper_alignment": self.paper_alignment,
            },
            assumptions=(
                DESCRIPTIVE_INTERPRETATION,
                "The residual distance is measured after masking only the "
                "declared identity terms; any edit the mask does not cover is "
                "counted as construction residue.",
                f"Pairing basis: {evidence.pairing_basis}",
                "The package validates pair geometry but cannot verify that a "
                "pair differs solely by the declared intervention.",
                f"Design stance: {_enum_value(spec.design_stance)}.",
            ),
            provenance=provenance,
        )


# --------------------------------------------------------------------------
# b_diff_len
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class LengthDisparity(DatasetDiagnostic):
    """Maximum pairwise group mean-length gap over the sample-weighted mean."""

    name: ClassVar[str] = "b_diff_len"
    tokenization: TokenizationRule = field(
        default_factory=lambda: TokenizationRule(
            mode=TokenizationMode.REGEX, pattern=r"\b[\w_]+\b"
        )
    )

    def __post_init__(self) -> None:
        if not isinstance(self.tokenization, TokenizationRule):
            raise TypeError(
                "tokenization must be a TokenizationRule, got "
                f"{type(self.tokenization).__name__}."
            )

    @property
    def paper_alignment(self) -> str:
        """Surface tokens are a declared divergence from parser tokens."""
        return "generalized_surface_tokenization"

    def _provenance(self, evidence: GroupedTexts) -> dict[str, Any]:
        return {
            "evidence": _grouped_evidence_metadata(evidence),
            "tokenization": self.tokenization.to_dict(),
        }

    def plan(self, evidence: GroupedTexts, spec: DatasetAuditSpec) -> ComponentPlan:
        """Decide b_diff_len applicability without counting any token."""
        if not isinstance(evidence, GroupedTexts):
            raise TypeError(
                f"evidence must be GroupedTexts, got {type(evidence).__name__}."
            )
        if not isinstance(spec, DatasetAuditSpec):
            raise TypeError(
                f"spec must be DatasetAuditSpec, got {type(spec).__name__}."
            )

        prefix = _shared_plan_prefix(
            component=self.name, spec=spec, axis=evidence.axis
        )
        if prefix is not None:
            return prefix

        empty_groups = _empty_declared_groups(evidence.group_sample_counts)
        if empty_groups:
            return _empty_group_plan(
                component=self.name,
                axis=evidence.axis,
                empty_groups=empty_groups,
                support=evidence.support,
            )

        return ComponentPlan(
            component=self.name,
            status=DiagnosticStatus.READY,
            details={
                "axis": evidence.axis,
                "support": list(evidence.support),
                "sample_count": evidence.total,
            },
        )

    def compute(
        self, evidence: GroupedTexts, spec: DatasetAuditSpec
    ) -> ComponentResult:
        """Compute the sample-weighted normalized group length disparity."""
        plan = self.plan(evidence, spec)
        provenance = self._provenance(evidence)
        if plan.status is not DiagnosticStatus.READY:
            return _non_ready_result(plan, spec=spec, provenance=provenance)

        token_counts = [
            len(self.tokenization.tokenize(text)) for text in evidence.texts
        ]
        group_token_totals = {label: 0 for label in evidence.support}
        for label, count in zip(evidence.groups, token_counts):
            group_token_totals[label] += count

        group_sample_counts = dict(evidence.group_sample_counts)
        group_mean_lengths = {
            label: group_token_totals[label] / group_sample_counts[label]
            for label in evidence.support
        }
        max_gap, widest_pair = max_pairwise_gap(group_mean_lengths)

        total_token_count = sum(token_counts)
        # Rule 4: one pooled token total over one pooled unit total. This is
        # deliberately NOT the unweighted mean of the per-group means.
        pooled_mean_length = total_token_count / evidence.total

        base_details = {
            "axis": evidence.axis,
            "support": list(evidence.support),
            "group_mean_lengths": group_mean_lengths,
            "group_sample_counts": group_sample_counts,
            "group_token_totals": group_token_totals,
            "sample_count": evidence.total,
            "total_token_count": total_token_count,
            "max_absolute_gap": max_gap,
            "widest_pair": list(widest_pair) if widest_pair is not None else None,
            "pooled_mean_length": pooled_mean_length,
            "denominator_rule": "sample_weighted_pooled_mean",
            "tokenization": self.tokenization.to_dict(),
            "unit": "ratio_to_pooled_mean_token_count",
            "estimand": "normalized_maximum_pairwise_group_mean_length_gap",
            "paper_alignment": self.paper_alignment,
        }

        if pooled_mean_length == 0.0:
            return ComponentResult(
                component=self.name,
                status=DiagnosticStatus.FAILED,
                details=base_details,
                provenance=provenance,
                reason_code="zero_length_denominator",
                reason=(
                    "The sample-weighted pooled mean length is zero, so the "
                    "normalized disparity is undefined."
                ),
            )

        value = max_gap / pooled_mean_length
        if not math.isfinite(value):
            return _numeric_failure(
                component=self.name,
                details=base_details,
                provenance=provenance,
                reason="The normalized length disparity is not a finite value.",
            )

        return ComponentResult(
            component=self.name,
            status=DiagnosticStatus.READY,
            value=value,
            details=base_details,
            assumptions=(
                DESCRIPTIVE_INTERPRETATION,
                "Length is measured in declared surface tokens, not the "
                "paper's parser tokens, so the value is not numerically "
                "comparable to a published spaCy-tokenized B_diff_len.",
                "The disparity denominator is the sample-weighted pooled mean "
                "length over every text, not an unweighted mean of group means.",
                f"Design stance: {_enum_value(spec.design_stance)}.",
            ),
            provenance=provenance,
        )


# --------------------------------------------------------------------------
# b_opt
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class OptionLengthBias(DatasetDiagnostic):
    """Mean signed option-length difference between two declared roles.

    Rule 5 is enforced twice: the evidence declares a role for every option
    row, and this diagnostic requires an explicit statement of which declared
    role is the stereotype. Option position is structurally unreachable from
    the kernel.
    """

    name: ClassVar[str] = "b_opt"
    role_contrast: Optional[OptionRoleContrast] = None
    tokenization: TokenizationRule = field(default_factory=TokenizationRule)

    def __post_init__(self) -> None:
        if self.role_contrast is not None and not isinstance(
            self.role_contrast, OptionRoleContrast
        ):
            raise TypeError(
                "role_contrast must be an OptionRoleContrast or None, got "
                f"{type(self.role_contrast).__name__}."
            )
        if not isinstance(self.tokenization, TokenizationRule):
            raise TypeError(
                "tokenization must be a TokenizationRule, got "
                f"{type(self.tokenization).__name__}."
            )

    @property
    def paper_alignment(self) -> str:
        """Whitespace tokens reproduce the paper's option length counting."""
        if self.tokenization.mode is TokenizationMode.WHITESPACE:
            return "paper_exact"
        return "generalized_tokenization"

    def _provenance(self, evidence: OptionItems) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "evidence": {
                "axis": evidence.axis,
                "source": evidence.source,
                "question_family": evidence.question_family,
                "declared_roles": list(evidence.declared_roles),
                "item_count": evidence.item_count,
                "provenance": dict(evidence.provenance),
            },
            "tokenization": self.tokenization.to_dict(),
        }
        if self.role_contrast is not None:
            payload["role_contrast"] = self.role_contrast.to_dict()
        return payload

    @staticmethod
    def _options_by_item(
        evidence: OptionItems,
    ) -> tuple[list[Any], dict[str, dict[str, str]]]:
        """Return canonical item order and the role -> option map per item."""
        order: list[Any] = []
        seen: set[str] = set()
        options: dict[str, dict[str, str]] = {}
        for item_id, role, option in zip(
            evidence.item_ids, evidence.roles, evidence.options
        ):
            key = str(item_id)
            if key not in seen:
                seen.add(key)
                order.append(item_id)
                options[key] = {}
            options[key][role] = option
        return order, options

    def plan(self, evidence: OptionItems, spec: DatasetAuditSpec) -> ComponentPlan:
        """Decide b_opt applicability without measuring any option length."""
        if not isinstance(evidence, OptionItems):
            raise TypeError(
                f"evidence must be OptionItems, got {type(evidence).__name__}."
            )
        if not isinstance(spec, DatasetAuditSpec):
            raise TypeError(
                f"spec must be DatasetAuditSpec, got {type(spec).__name__}."
            )

        prefix = _shared_plan_prefix(
            component=self.name, spec=spec, axis=evidence.axis
        )
        if prefix is not None:
            return prefix

        if self.role_contrast is None:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.BLOCKED,
                reason_code="missing_option_role_contrast",
                reason=(
                    "b_opt requires an explicit stereotype/anti-stereotype role "
                    "contrast; option roles are never inferred from position."
                ),
                details={
                    "axis": evidence.axis,
                    "declared_roles": list(evidence.declared_roles),
                    "item_count": evidence.item_count,
                },
            )

        contrast_roles = (
            self.role_contrast.stereotype_role,
            self.role_contrast.anti_stereotype_role,
        )
        missing_roles = [
            role for role in contrast_roles if role not in evidence.declared_roles
        ]
        if missing_roles:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.BLOCKED,
                reason_code="option_role_not_declared",
                reason=(
                    f"The role contrast names role(s) {missing_roles!r}, which "
                    "the option evidence does not declare."
                ),
                details={
                    "axis": evidence.axis,
                    "declared_roles": list(evidence.declared_roles),
                    "missing_roles": missing_roles,
                    "stereotype_role": self.role_contrast.stereotype_role,
                    "anti_stereotype_role": self.role_contrast.anti_stereotype_role,
                },
            )

        order, options = self._options_by_item(evidence)
        incomplete = [
            item_id
            for item_id in order
            if any(role not in options[str(item_id)] for role in contrast_roles)
        ]
        if incomplete:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.BLOCKED,
                reason_code="incomplete_option_contrast",
                reason=(
                    "At least one item does not carry exactly one option for "
                    "each contrasted role; incomplete items are refused, never "
                    "dropped."
                ),
                details={
                    "axis": evidence.axis,
                    "declared_roles": list(evidence.declared_roles),
                    "stereotype_role": self.role_contrast.stereotype_role,
                    "anti_stereotype_role": self.role_contrast.anti_stereotype_role,
                    "incomplete_item_ids": list(incomplete[:_MAX_LISTED_IDS]),
                    "incomplete_item_count": len(incomplete),
                },
            )

        return ComponentPlan(
            component=self.name,
            status=DiagnosticStatus.READY,
            details={
                "axis": evidence.axis,
                "item_count": evidence.item_count,
                "declared_roles": list(evidence.declared_roles),
                "stereotype_role": self.role_contrast.stereotype_role,
                "anti_stereotype_role": self.role_contrast.anti_stereotype_role,
            },
        )

    def compute(self, evidence: OptionItems, spec: DatasetAuditSpec) -> ComponentResult:
        """Compute the signed mean stereotype-minus-anti-stereotype length."""
        plan = self.plan(evidence, spec)
        provenance = self._provenance(evidence)
        if plan.status is not DiagnosticStatus.READY:
            return _non_ready_result(plan, spec=spec, provenance=provenance)

        contrast = self.role_contrast
        assert contrast is not None  # guaranteed by the ready plan
        order, options = self._options_by_item(evidence)

        differences: list[int] = []
        stereotype_lengths: list[int] = []
        anti_stereotype_lengths: list[int] = []
        for item_id in order:
            item = options[str(item_id)]
            stereotype_length = len(
                self.tokenization.tokenize(item[contrast.stereotype_role])
            )
            anti_length = len(
                self.tokenization.tokenize(item[contrast.anti_stereotype_role])
            )
            stereotype_lengths.append(stereotype_length)
            anti_stereotype_lengths.append(anti_length)
            differences.append(stereotype_length - anti_length)

        item_count = len(differences)
        value = math.fsum(differences) / item_count
        base_details = {
            "axis": evidence.axis,
            "item_count": item_count,
            "declared_roles": list(evidence.declared_roles),
            "stereotype_role": contrast.stereotype_role,
            "anti_stereotype_role": contrast.anti_stereotype_role,
            "question_family": evidence.question_family,
            "mean_signed_length_difference": value,
            "mean_stereotype_length": math.fsum(stereotype_lengths) / item_count,
            "mean_anti_stereotype_length": math.fsum(anti_stereotype_lengths)
            / item_count,
            "stereotype_longer_ratio": sum(1 for d in differences if d > 0)
            / item_count,
            "anti_stereotype_longer_ratio": sum(1 for d in differences if d < 0)
            / item_count,
            "equal_length_ratio": sum(1 for d in differences if d == 0) / item_count,
            "role_orientation": "stereotype_minus_anti_stereotype",
            "directionality": "signed",
            "unit": "token_count_difference",
            "tokenization": self.tokenization.to_dict(),
            "paper_alignment": self.paper_alignment,
        }
        if not math.isfinite(value):
            return _numeric_failure(
                component=self.name,
                details=base_details,
                provenance=provenance,
                reason="The mean signed option length difference is not finite.",
            )

        return ComponentResult(
            component=self.name,
            status=DiagnosticStatus.READY,
            value=value,
            details=base_details,
            assumptions=(
                DESCRIPTIVE_INTERPRETATION,
                "Option roles are taken only from the declared role labels; the "
                "package never infers a stereotype or anti-stereotype role from "
                "option position or ordering.",
                f"Question family: {evidence.question_family}",
                "The value is signed: a negative mean means the anti-stereotype "
                "option is longer on average.",
                f"Design stance: {_enum_value(spec.design_stance)}.",
            ),
            provenance=provenance,
        )


# --------------------------------------------------------------------------
# b_frame
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class FramingDisparity(DatasetDiagnostic):
    """Maximum pairwise gap in the per-group rate of a declared frame."""

    name: ClassVar[str] = "b_frame"
    predicate: Optional[FramePredicateLike] = None

    def __post_init__(self) -> None:
        if self.predicate is not None and not isinstance(
            self.predicate, (FramePredicate, InjectedFramePredicate)
        ):
            raise TypeError(
                "predicate must be a FramePredicate, an InjectedFramePredicate, "
                f"or None, got {type(self.predicate).__name__}."
            )

    def _provenance(self, evidence: GroupedTexts) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "evidence": _grouped_evidence_metadata(evidence),
        }
        if self.predicate is not None:
            payload["frame_predicate"] = self.predicate.to_dict()
        return payload

    def plan(self, evidence: GroupedTexts, spec: DatasetAuditSpec) -> ComponentPlan:
        """Decide b_frame applicability without evaluating the predicate."""
        if not isinstance(evidence, GroupedTexts):
            raise TypeError(
                f"evidence must be GroupedTexts, got {type(evidence).__name__}."
            )
        if not isinstance(spec, DatasetAuditSpec):
            raise TypeError(
                f"spec must be DatasetAuditSpec, got {type(spec).__name__}."
            )

        prefix = _shared_plan_prefix(
            component=self.name, spec=spec, axis=evidence.axis
        )
        if prefix is not None:
            return prefix

        if self.predicate is None:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.BLOCKED,
                reason_code="missing_frame_predicate",
                reason=(
                    "b_frame requires an explicitly declared or injected frame "
                    "predicate; the predicate definition is the estimand."
                ),
                details={
                    "axis": evidence.axis,
                    "support": list(evidence.support),
                },
            )

        empty_groups = _empty_declared_groups(evidence.group_sample_counts)
        if empty_groups:
            return _empty_group_plan(
                component=self.name,
                axis=evidence.axis,
                empty_groups=empty_groups,
                support=evidence.support,
            )

        return ComponentPlan(
            component=self.name,
            status=DiagnosticStatus.READY,
            details={
                "axis": evidence.axis,
                "support": list(evidence.support),
                "frame_name": self.predicate.frame_name,
                "predicate_kind": self.predicate.predicate_kind,
            },
        )

    def compute(
        self, evidence: GroupedTexts, spec: DatasetAuditSpec
    ) -> ComponentResult:
        """Compute the widest pairwise gap in per-group frame rates."""
        plan = self.plan(evidence, spec)
        provenance = self._provenance(evidence)
        if plan.status is not DiagnosticStatus.READY:
            return _non_ready_result(plan, spec=spec, provenance=provenance)

        predicate = self.predicate
        assert predicate is not None  # guaranteed by the ready plan

        group_frame_counts = {label: 0 for label in evidence.support}
        for index, (label, text) in enumerate(zip(evidence.groups, evidence.texts)):
            try:
                hit = predicate.matches(text)
            except Exception as exc:
                return ComponentResult(
                    component=self.name,
                    status=DiagnosticStatus.FAILED,
                    details={
                        "exception_type": type(exc).__name__,
                        "failed_text_index": index,
                    },
                    provenance=provenance,
                    reason_code="frame_predicate_failed",
                    reason=(
                        "The frame predicate failed on a text unit with "
                        f"{type(exc).__name__}."
                    ),
                )
            if hit:
                group_frame_counts[label] += 1

        group_sample_counts = dict(evidence.group_sample_counts)
        group_frame_rates = {
            label: group_frame_counts[label] / group_sample_counts[label]
            for label in evidence.support
        }
        value, widest_pair = max_pairwise_gap(group_frame_rates)
        pooled_hits = sum(group_frame_counts.values())

        details: dict[str, Any] = {
            "axis": evidence.axis,
            "support": list(evidence.support),
            "frame_name": predicate.frame_name,
            "predicate_kind": predicate.predicate_kind,
            "predicate_digest": predicate.predicate_digest,
            "replayable": predicate.replayable,
            "group_frame_rates": group_frame_rates,
            "group_frame_counts": group_frame_counts,
            "group_sample_counts": group_sample_counts,
            "sample_count": evidence.total,
            "overall_frame_rate": pooled_hits / evidence.total,
            "widest_pair": list(widest_pair) if widest_pair is not None else None,
            "unit": "proportion",
            "estimand": "maximum_absolute_pairwise_frame_rate_gap",
        }
        if isinstance(predicate, FramePredicate):
            details["paper_alignment"] = predicate.paper_alignment
        if not math.isfinite(value):
            return _numeric_failure(
                component=self.name,
                details=details,
                provenance=provenance,
                reason="The frame rate gap is not a finite value.",
            )

        assumptions = [
            DESCRIPTIVE_INTERPRETATION,
            f"Frame definition: {predicate.definition}",
            "The frame predicate is the estimand: a different predicate "
            "measures a different quantity, and results are comparable only "
            "under an identical predicate.",
        ]
        if isinstance(predicate, InjectedFramePredicate):
            assumptions.append(
                "The predicate is an injected callable; its definition is "
                "recorded but the result is not replayable from provenance "
                "alone."
            )
        assumptions.append(f"Design stance: {_enum_value(spec.design_stance)}.")

        return ComponentResult(
            component=self.name,
            status=DiagnosticStatus.READY,
            value=value,
            details=details,
            assumptions=tuple(assumptions),
            provenance=provenance,
        )


# --------------------------------------------------------------------------
# b_temp
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class TemplateImbalance(DatasetDiagnostic):
    """Unique-template count imbalance across declared groups, with coverage.

    A ready result always reports the imbalance **and** a defined coverage
    ratio: both an empty declared group and an unavailable coverage
    denominator are explicit blocked rules, never a successful zero.
    """

    name: ClassVar[str] = "b_temp"
    ratio_min_count: int = 1

    def __post_init__(self) -> None:
        value = self.ratio_min_count
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                "ratio_min_count must be an integer, got "
                f"{type(value).__name__}."
            )
        converted = int(value)
        if converted < 1:
            raise ValueError("ratio_min_count must be at least 1.")
        object.__setattr__(self, "ratio_min_count", converted)

    def _provenance(self, evidence: TemplateGroups) -> dict[str, Any]:
        return {
            "evidence": {
                "axis": evidence.axis,
                "source": evidence.source,
                "support": list(evidence.support),
                "total": evidence.total,
                "template_identity_rule": evidence.template_identity_rule,
                "provenance": dict(evidence.provenance),
            },
            "parameters": {"ratio_min_count": self.ratio_min_count},
        }

    def plan(self, evidence: TemplateGroups, spec: DatasetAuditSpec) -> ComponentPlan:
        """Decide b_temp applicability without computing any ratio."""
        if not isinstance(evidence, TemplateGroups):
            raise TypeError(
                f"evidence must be TemplateGroups, got {type(evidence).__name__}."
            )
        if not isinstance(spec, DatasetAuditSpec):
            raise TypeError(
                f"spec must be DatasetAuditSpec, got {type(spec).__name__}."
            )

        prefix = _shared_plan_prefix(
            component=self.name, spec=spec, axis=evidence.axis
        )
        if prefix is not None:
            return prefix

        empty_groups = _empty_declared_groups(evidence.group_instance_counts)
        if empty_groups:
            return _empty_group_plan(
                component=self.name,
                axis=evidence.axis,
                empty_groups=empty_groups,
                support=evidence.support,
                noun="template instantiations",
            )

        unique_counts = dict(evidence.group_unique_template_counts)
        if not any(count >= self.ratio_min_count for count in unique_counts.values()):
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.BLOCKED,
                reason_code="template_coverage_denominator_unavailable",
                reason=(
                    f"No declared group reached ratio_min_count="
                    f"{self.ratio_min_count}, so the template coverage ratio is "
                    "undefined."
                ),
                details={
                    "axis": evidence.axis,
                    "support": list(evidence.support),
                    "group_unique_template_counts": unique_counts,
                    "ratio_min_count": self.ratio_min_count,
                },
            )

        return ComponentPlan(
            component=self.name,
            status=DiagnosticStatus.READY,
            details={
                "axis": evidence.axis,
                "support": list(evidence.support),
                "ratio_min_count": self.ratio_min_count,
                "template_identity_rule": evidence.template_identity_rule,
            },
        )

    def compute(
        self, evidence: TemplateGroups, spec: DatasetAuditSpec
    ) -> ComponentResult:
        """Compute the unique-template imbalance and its coverage ratio."""
        plan = self.plan(evidence, spec)
        provenance = self._provenance(evidence)
        if plan.status is not DiagnosticStatus.READY:
            return _non_ready_result(plan, spec=spec, provenance=provenance)

        unique_counts = dict(evidence.group_unique_template_counts)
        instance_counts = dict(evidence.group_instance_counts)
        max_unique = max(unique_counts.values())
        min_unique = min(unique_counts.values())
        value = float(max_unique - min_unique)

        qualifying = {
            label: count
            for label, count in unique_counts.items()
            if count >= self.ratio_min_count
        }
        min_positive = min(qualifying.values())
        coverage_denominator_group = min(
            label for label, count in qualifying.items() if count == min_positive
        )
        coverage_ratio = max_unique / min_positive
        duplication_rates = {
            label: 1.0 - unique_counts[label] / instance_counts[label]
            for label in evidence.support
        }

        details = {
            "axis": evidence.axis,
            "support": list(evidence.support),
            "group_unique_template_counts": unique_counts,
            "group_instance_counts": instance_counts,
            "group_duplication_rates": duplication_rates,
            "max_unique_count": max_unique,
            "min_unique_count": min_unique,
            "coverage_ratio": coverage_ratio,
            "coverage_denominator_group": coverage_denominator_group,
            "ratio_min_count": self.ratio_min_count,
            "template_identity_rule": evidence.template_identity_rule,
            "total_instances": evidence.total,
            "unit": "unique_template_count_difference",
            "coverage_ratio_definition": (
                "max_unique / min_unique_at_or_above_ratio_min_count"
            ),
            "scale_note": (
                "unnormalized count difference; not comparable across datasets "
                "of different size"
            ),
        }
        if not math.isfinite(value) or not math.isfinite(coverage_ratio):
            return _numeric_failure(
                component=self.name,
                details=details,
                provenance=provenance,
                reason=(
                    "The template imbalance or its coverage ratio is not a "
                    "finite value."
                ),
            )

        return ComponentResult(
            component=self.name,
            status=DiagnosticStatus.READY,
            value=value,
            details=details,
            assumptions=(
                DESCRIPTIVE_INTERPRETATION,
                f"Template identity rule: {evidence.template_identity_rule}",
                "The imbalance is an unnormalized count difference in template "
                "units and is not comparable across datasets of different size.",
                f"Design stance: {_enum_value(spec.design_stance)}.",
            ),
            provenance=provenance,
        )


# --------------------------------------------------------------------------
# Backend-dependent components: b_equiv, b_gram, b_diff_dep
# --------------------------------------------------------------------------


def _backend_requirement_details(slot: str) -> dict[str, Any]:
    requirement = CONSTRUCTION_BACKEND_REQUIREMENTS[slot]
    return {
        "slot": slot,
        "required_backend": requirement["required_backend"],
        "required_protocol": requirement["required_protocol"],
        "availability": requirement["availability"],
        "milestone": requirement["milestone"],
    }


def _missing_backend_plan(
    slot: str, *, axis: str, extra: Optional[Mapping[str, Any]] = None
) -> ComponentPlan:
    """Block a backend-dependent slot whose backend was not supplied."""
    requirement = CONSTRUCTION_BACKEND_REQUIREMENTS[slot]
    details = _backend_requirement_details(slot)
    details["axis"] = axis
    if extra:
        details.update(extra)
    return ComponentPlan(
        component=slot,
        status=DiagnosticStatus.BLOCKED,
        reason_code=requirement["reason_code"],
        reason=_BACKEND_BLOCKED_REASONS[slot],
        details=details,
    )


def _check_backend(backend: Any, *, slot: str, method: str) -> None:
    """Validate a supplied backend against the slot's protocol, structurally."""
    if backend is None:
        return
    protocol = CONSTRUCTION_BACKEND_REQUIREMENTS[slot]["required_protocol"]
    revision = getattr(backend, "revision", None)
    if not callable(getattr(backend, method, None)) or not (
        isinstance(revision, str) and revision.strip()
    ):
        raise TypeError(
            f"backend for {slot} must implement {protocol}: a non-empty str "
            f"`revision` and a callable `{method}`; got {type(backend).__name__}."
        )


def _backend_provenance(backend: Any, *, slot: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol": CONSTRUCTION_BACKEND_REQUIREMENTS[slot]["required_protocol"],
        "implementation": f"{type(backend).__module__}.{type(backend).__qualname__}",
        "revision": str(backend.revision),
    }
    definition = getattr(backend, "depth_definition", None)
    if isinstance(definition, str) and definition:
        payload["depth_definition"] = definition
    return payload


def _backend_failure(
    *,
    component: str,
    details: Mapping[str, Any],
    provenance: Mapping[str, Any],
    reason_code: str,
    reason: str,
) -> ComponentResult:
    return ComponentResult(
        component=component,
        status=DiagnosticStatus.FAILED,
        details=dict(details),
        provenance=dict(provenance),
        reason_code=reason_code,
        reason=reason,
    )


def _call_backend(callable_: Any, texts: Sequence[str]) -> tuple[Any, Optional[str]]:
    """Run one backend call, returning ``(output, error)``."""
    try:
        return callable_(list(texts)), None
    except Exception as exc:  # a backend is foreign code; its failure is a result
        return None, f"{type(exc).__name__}: {exc}"


def _record_backend_revision(
    backend: Any, *, details: dict[str, Any], provenance: Mapping[str, Any]
) -> None:
    """Re-read ``revision`` after the backend has actually run.

    Both reference backends resolve their real identity lazily: the spaCy
    pipeline version and the Hub commit hash are only known once ``_load()``
    has run, and ``_load()`` runs inside the first ``depths()`` / ``encode()``
    call. Reading ``revision`` beforehand therefore records
    ``pipeline=en_core_web_sm@unloaded`` or ``@default`` -- and since a
    component is built once per audit, that is the ordinary case rather than
    an edge case. The recorded revision is the one field that says which model
    produced the numbers, so it is refreshed here, on every path out of the
    call, including the failure paths where a backend may have loaded and then
    raised.

    A non-ready ``plan`` is deliberately left alone: planning does no work by
    contract, so its pre-load revision is honest about what had run.
    """
    revision = str(backend.revision)
    details["backend_revision"] = revision
    recorded = provenance.get("backend")
    if isinstance(recorded, dict):
        recorded["revision"] = revision


def _validate_count_output(
    output: Any, *, expected: int, what: str
) -> tuple[Optional[list[int]], Optional[str]]:
    if isinstance(output, (str, bytes)) or not isinstance(output, Sequence):
        return None, f"the backend returned {type(output).__name__} instead of a sequence of {what}."
    if len(output) != expected:
        return None, f"the backend returned {len(output)} values for {expected} texts."
    values: list[int] = []
    for index, value in enumerate(output):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            return None, f"{what}[{index}] must be a non-negative integer, got {value!r}."
        values.append(int(value))
    return values, None


def _validate_vector_output(
    output: Any, *, expected: int
) -> tuple[Optional[list[list[float]]], Optional[str]]:
    if isinstance(output, (str, bytes)) or not isinstance(output, Sequence):
        return None, f"the backend returned {type(output).__name__} instead of a sequence of vectors."
    if len(output) != expected:
        return None, f"the backend returned {len(output)} vectors for {expected} texts."
    vectors: list[list[float]] = []
    width: Optional[int] = None
    for index, vector in enumerate(output):
        if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
            return None, f"vector[{index}] is not a sequence of numbers."
        values: list[float] = []
        for component in vector:
            # `numbers.Real`, matching the `numbers.Integral` policy in
            # `_validate_count_output`. The concrete `(int, float)` pair this
            # replaced admitted `numpy.float64` -- which subclasses `float` --
            # while rejecting `numpy.float32`, the default dtype of almost
            # every embedding model, so the two sibling validators disagreed
            # about what a number is and the failure looked arbitrary.
            if isinstance(component, bool) or not isinstance(component, Real):
                return None, f"vector[{index}] contains a non-numeric component."
            value = float(component)
            if not math.isfinite(value):
                return None, f"vector[{index}] contains a non-finite component."
            values.append(value)
        if not values:
            return None, f"vector[{index}] is empty."
        if width is None:
            width = len(values)
        elif len(values) != width:
            return None, "the backend returned vectors of different widths."
        vectors.append(values)
    return vectors, None


def _cosine(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    """Cosine similarity, or ``None`` when either vector has zero norm."""
    dot = math.fsum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(math.fsum(a * a for a in left))
    right_norm = math.sqrt(math.fsum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    value = dot / (left_norm * right_norm)
    return max(-1.0, min(1.0, value))


def _paired_evidence_metadata(evidence: PairedTexts) -> dict[str, Any]:
    return {
        "axis": evidence.axis,
        "source": evidence.source,
        "pairing_basis": evidence.pairing_basis,
        "condition_roles": list(evidence.condition_roles),
        "pair_count": evidence.pair_count,
        "provenance": dict(evidence.provenance),
    }


@dataclass(frozen=True, kw_only=True)
class SemanticEquivalence(DatasetDiagnostic):
    """One minus the mean cosine similarity of identity-masked pair sides.

    Implements the paper's :math:`B_{\\mathrm{equiv}} = 1 - \\mathbb{E}[\\cos(E(M(x)),
    E(M(x')))]`. Both the identity mask and the embedding backend are estimand
    declarations: neither is inferred, and the component is ``blocked`` by
    name until both are supplied.
    """

    name: ClassVar[str] = "b_equiv"
    identity_mask: Optional[IdentityMaskConfig] = None
    backend: Optional[EmbeddingBackend] = None

    def __post_init__(self) -> None:
        if self.identity_mask is not None and not isinstance(
            self.identity_mask, IdentityMaskConfig
        ):
            raise TypeError(
                "identity_mask must be an IdentityMaskConfig or None, got "
                f"{type(self.identity_mask).__name__}."
            )
        _check_backend(self.backend, slot=self.name, method="encode")

    @property
    def paper_alignment(self) -> str:
        """The formula is the paper's; the value depends on the declared backend."""
        return "paper_exact_given_backend"

    def _provenance(self, evidence: PairedTexts) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "evidence": _paired_evidence_metadata(evidence),
            "backend_requirement": dict(CONSTRUCTION_BACKEND_REQUIREMENTS[self.name]),
        }
        if self.identity_mask is not None:
            payload["identity_mask"] = self.identity_mask.to_dict()
        if self.backend is not None:
            payload["backend"] = _backend_provenance(self.backend, slot=self.name)
        return payload

    def plan(self, evidence: PairedTexts, spec: DatasetAuditSpec) -> ComponentPlan:
        """Decide b_equiv applicability without encoding any text."""
        if not isinstance(evidence, PairedTexts):
            raise TypeError(
                f"evidence must be PairedTexts, got {type(evidence).__name__}."
            )
        if not isinstance(spec, DatasetAuditSpec):
            raise TypeError(
                f"spec must be DatasetAuditSpec, got {type(spec).__name__}."
            )
        prefix = _shared_plan_prefix(
            component=self.name, spec=spec, axis=evidence.axis
        )
        if prefix is not None:
            return prefix
        if self.backend is None:
            return _missing_backend_plan(
                self.name,
                axis=evidence.axis,
                extra={"pair_count": evidence.pair_count},
            )
        if self.identity_mask is None:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.BLOCKED,
                reason_code="missing_identity_mask",
                reason=(
                    "b_equiv compares the two sides after masking the declared "
                    "identity terms; identity terms are never inferred from the "
                    "paired texts."
                ),
                details={"axis": evidence.axis, "pair_count": evidence.pair_count},
            )
        return ComponentPlan(
            component=self.name,
            status=DiagnosticStatus.READY,
            details={
                "axis": evidence.axis,
                "pair_count": evidence.pair_count,
                "condition_roles": list(evidence.condition_roles),
                "pairing_basis": evidence.pairing_basis,
                "backend_revision": str(self.backend.revision),
            },
        )

    def compute(self, evidence: PairedTexts, spec: DatasetAuditSpec) -> ComponentResult:
        """Embed the masked sides and report one minus the mean cosine."""
        plan = self.plan(evidence, spec)
        provenance = self._provenance(evidence)
        if plan.status is not DiagnosticStatus.READY:
            return _non_ready_result(plan, spec=spec, provenance=provenance)

        mask = self.identity_mask
        backend = self.backend
        assert mask is not None and backend is not None  # guaranteed by the plan

        base_details = {
            "axis": evidence.axis,
            "pair_count": evidence.pair_count,
            "condition_roles": list(evidence.condition_roles),
            "pairing_basis": evidence.pairing_basis,
            "backend_revision": str(backend.revision),
        }

        masked: list[str] = []
        degenerate_ids: list[Any] = []
        for start in range(0, evidence.total, 2):
            left = " ".join(mask.mask(evidence.texts[start]))
            right = " ".join(mask.mask(evidence.texts[start + 1]))
            if not left or not right:
                degenerate_ids.append(evidence.pair_ids[start])
            masked.extend((left, right))
        if degenerate_ids:
            return ComponentResult(
                component=self.name,
                status=DiagnosticStatus.BLOCKED,
                details={
                    **base_details,
                    "degenerate_pair_ids": list(degenerate_ids[:_MAX_LISTED_IDS]),
                    "degenerate_pair_count": len(degenerate_ids),
                },
                provenance=provenance,
                reason_code="degenerate_masked_pair",
                reason=(
                    "At least one pair side is empty after identity masking, so "
                    "it has no content to embed; an empty side is blocked rather "
                    "than embedded as an empty string."
                ),
            )

        output, error = _call_backend(backend.encode, masked)
        _record_backend_revision(backend, details=base_details, provenance=provenance)
        if error is not None:
            return _backend_failure(
                component=self.name,
                details=base_details,
                provenance=provenance,
                reason_code="backend_call_failed",
                reason=f"The embedding backend raised {error}",
            )
        vectors, problem = _validate_vector_output(output, expected=len(masked))
        if vectors is None:
            return _backend_failure(
                component=self.name,
                details=base_details,
                provenance=provenance,
                reason_code="backend_output_invalid",
                reason=f"The embedding backend output is unusable: {problem}",
            )

        similarities: list[float] = []
        zero_norm_ids: list[Any] = []
        for index in range(0, len(vectors), 2):
            similarity = _cosine(vectors[index], vectors[index + 1])
            if similarity is None:
                zero_norm_ids.append(evidence.pair_ids[index])
                continue
            similarities.append(similarity)
        if zero_norm_ids:
            return _backend_failure(
                component=self.name,
                details={
                    **base_details,
                    "zero_norm_pair_ids": list(zero_norm_ids[:_MAX_LISTED_IDS]),
                    "zero_norm_pair_count": len(zero_norm_ids),
                },
                provenance=provenance,
                reason_code="zero_norm_embedding",
                reason=(
                    "The backend returned an all-zero vector for at least one "
                    "masked side, so its cosine similarity is undefined."
                ),
            )

        mean_similarity = math.fsum(similarities) / len(similarities)
        value = 1.0 - mean_similarity
        if not math.isfinite(value):
            return _numeric_failure(
                component=self.name,
                details=base_details,
                provenance=provenance,
                reason="The mean cosine similarity did not produce a finite value.",
            )
        distances = [1.0 - similarity for similarity in similarities]
        return ComponentResult(
            component=self.name,
            status=DiagnosticStatus.READY,
            value=value,
            details={
                **base_details,
                "mean_cosine_similarity": mean_similarity,
                "minimum_cosine_similarity": min(similarities),
                "maximum_cosine_similarity": max(similarities),
                "pairs_with_distance_above_0_10": sum(1 for d in distances if d > 0.10),
                "pairs_with_distance_above_0_20": sum(1 for d in distances if d > 0.20),
                "embedding_width": len(vectors[0]),
                "identity_term_count": len(mask.identity_terms),
                "estimator": "uniform_mass_per_pair",
                "unit": "one_minus_mean_cosine_similarity",
                "paper_alignment": self.paper_alignment,
            },
            assumptions=(
                DESCRIPTIVE_INTERPRETATION,
                "Similarity is measured between the identity-masked sides, so "
                "the value reflects residual meaning differences the mask does "
                "not cover, under the declared embedding model.",
                "The counts above 0.10 and 0.20 use the paper's STS-based "
                "reference points; they are descriptive anchors, not validated "
                "fairness-audit thresholds.",
                f"Pairing basis: {evidence.pairing_basis}",
                f"Design stance: {_enum_value(spec.design_stance)}.",
            ),
            provenance=provenance,
        )


@dataclass(frozen=True, kw_only=True)
class GrammarConsistency(DatasetDiagnostic):
    """Mean absolute difference in grammatical-error counts within a pair.

    Implements the paper's :math:`B_{\\mathrm{gram}} = \\mathbb{E}[|\\mathrm{err}(x)
    - \\mathrm{err}(x')|]` over the unmasked pair sides. The grammar checker is
    an estimand declaration and is never inferred.
    """

    name: ClassVar[str] = "b_gram"
    backend: Optional[GrammarCheckerBackend] = None

    def __post_init__(self) -> None:
        _check_backend(self.backend, slot=self.name, method="count_errors")

    @property
    def paper_alignment(self) -> str:
        """The formula is the paper's; the counts depend on the declared checker."""
        return "paper_exact_given_backend"

    def _provenance(self, evidence: PairedTexts) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "evidence": _paired_evidence_metadata(evidence),
            "backend_requirement": dict(CONSTRUCTION_BACKEND_REQUIREMENTS[self.name]),
        }
        if self.backend is not None:
            payload["backend"] = _backend_provenance(self.backend, slot=self.name)
        return payload

    def plan(self, evidence: PairedTexts, spec: DatasetAuditSpec) -> ComponentPlan:
        """Decide b_gram applicability without checking any text."""
        if not isinstance(evidence, PairedTexts):
            raise TypeError(
                f"evidence must be PairedTexts, got {type(evidence).__name__}."
            )
        if not isinstance(spec, DatasetAuditSpec):
            raise TypeError(
                f"spec must be DatasetAuditSpec, got {type(spec).__name__}."
            )
        prefix = _shared_plan_prefix(
            component=self.name, spec=spec, axis=evidence.axis
        )
        if prefix is not None:
            return prefix
        if self.backend is None:
            return _missing_backend_plan(
                self.name,
                axis=evidence.axis,
                extra={"pair_count": evidence.pair_count},
            )
        return ComponentPlan(
            component=self.name,
            status=DiagnosticStatus.READY,
            details={
                "axis": evidence.axis,
                "pair_count": evidence.pair_count,
                "condition_roles": list(evidence.condition_roles),
                "pairing_basis": evidence.pairing_basis,
                "backend_revision": str(self.backend.revision),
            },
        )

    def compute(self, evidence: PairedTexts, spec: DatasetAuditSpec) -> ComponentResult:
        """Count errors on both sides and report the mean absolute difference."""
        plan = self.plan(evidence, spec)
        provenance = self._provenance(evidence)
        if plan.status is not DiagnosticStatus.READY:
            return _non_ready_result(plan, spec=spec, provenance=provenance)

        backend = self.backend
        assert backend is not None  # guaranteed by the plan
        base_details = {
            "axis": evidence.axis,
            "pair_count": evidence.pair_count,
            "condition_roles": list(evidence.condition_roles),
            "pairing_basis": evidence.pairing_basis,
            "backend_revision": str(backend.revision),
        }

        output, error = _call_backend(backend.count_errors, evidence.texts)
        _record_backend_revision(backend, details=base_details, provenance=provenance)
        if error is not None:
            return _backend_failure(
                component=self.name,
                details=base_details,
                provenance=provenance,
                reason_code="backend_call_failed",
                reason=f"The grammar backend raised {error}",
            )
        counts, problem = _validate_count_output(
            output, expected=evidence.total, what="error counts"
        )
        if counts is None:
            return _backend_failure(
                component=self.name,
                details=base_details,
                provenance=provenance,
                reason_code="backend_output_invalid",
                reason=f"The grammar backend output is unusable: {problem}",
            )

        differences = [
            abs(counts[index] - counts[index + 1])
            for index in range(0, evidence.total, 2)
        ]
        value = math.fsum(differences) / len(differences)
        if not math.isfinite(value):
            return _numeric_failure(
                component=self.name,
                details=base_details,
                provenance=provenance,
                reason="The mean error-count difference is not a finite value.",
            )

        conditions = getattr(evidence, "conditions", None)
        per_condition_totals: dict[str, list[int]] = {}
        if isinstance(conditions, Sequence) and len(conditions) == evidence.total:
            for label, count in zip(conditions, counts):
                per_condition_totals.setdefault(str(label), []).append(count)
        mean_errors_by_condition = {
            label: math.fsum(values) / len(values)
            for label, values in sorted(per_condition_totals.items())
        }
        return ComponentResult(
            component=self.name,
            status=DiagnosticStatus.READY,
            value=value,
            details={
                **base_details,
                "mean_absolute_error_difference": value,
                "pairs_with_difference": sum(1 for d in differences if d > 0),
                "maximum_absolute_error_difference": max(differences),
                "mean_error_count": math.fsum(counts) / len(counts),
                "mean_error_count_by_condition": mean_errors_by_condition,
                "estimator": "uniform_mass_per_pair",
                "unit": "grammar_error_count_difference",
                "paper_alignment": self.paper_alignment,
            },
            assumptions=(
                DESCRIPTIVE_INTERPRETATION,
                "Error counts are those reported by the declared grammar checker "
                "on the unmasked pair sides; the value is comparable only across "
                "audits that declare the same checker revision.",
                f"Pairing basis: {evidence.pairing_basis}",
                f"Design stance: {_enum_value(spec.design_stance)}.",
            ),
            provenance=provenance,
        )


@dataclass(frozen=True, kw_only=True)
class DependencyDepthDisparity(DatasetDiagnostic):
    """Maximum pairwise group mean dependency-depth gap over the pooled mean.

    Implements the paper's :math:`B_{\\mathrm{diff\\text{-}dep}}` with the same
    denominator rule as :class:`LengthDisparity`: the sample-weighted pooled
    mean depth over every text. The parser is an estimand declaration.
    """

    name: ClassVar[str] = "b_diff_dep"
    backend: Optional[DependencyParserBackend] = None

    def __post_init__(self) -> None:
        _check_backend(self.backend, slot=self.name, method="depths")

    @property
    def paper_alignment(self) -> str:
        """The formula is the paper's; the depths depend on the declared parser."""
        return "paper_exact_given_backend"

    def _provenance(self, evidence: GroupedTexts) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "evidence": _grouped_evidence_metadata(evidence),
            "backend_requirement": dict(CONSTRUCTION_BACKEND_REQUIREMENTS[self.name]),
        }
        if self.backend is not None:
            payload["backend"] = _backend_provenance(self.backend, slot=self.name)
        return payload

    def plan(self, evidence: GroupedTexts, spec: DatasetAuditSpec) -> ComponentPlan:
        """Decide b_diff_dep applicability without parsing any text."""
        if not isinstance(evidence, GroupedTexts):
            raise TypeError(
                f"evidence must be GroupedTexts, got {type(evidence).__name__}."
            )
        if not isinstance(spec, DatasetAuditSpec):
            raise TypeError(
                f"spec must be DatasetAuditSpec, got {type(spec).__name__}."
            )
        prefix = _shared_plan_prefix(
            component=self.name, spec=spec, axis=evidence.axis
        )
        if prefix is not None:
            return prefix
        if self.backend is None:
            return _missing_backend_plan(
                self.name,
                axis=evidence.axis,
                extra={"support": list(evidence.support), "sample_count": evidence.total},
            )
        empty_groups = _empty_declared_groups(evidence.group_sample_counts)
        if empty_groups:
            return _empty_group_plan(
                component=self.name,
                axis=evidence.axis,
                empty_groups=empty_groups,
                support=evidence.support,
            )
        return ComponentPlan(
            component=self.name,
            status=DiagnosticStatus.READY,
            details={
                "axis": evidence.axis,
                "support": list(evidence.support),
                "sample_count": evidence.total,
                "backend_revision": str(self.backend.revision),
            },
        )

    def compute(
        self, evidence: GroupedTexts, spec: DatasetAuditSpec
    ) -> ComponentResult:
        """Parse every text and report the normalized group depth disparity."""
        plan = self.plan(evidence, spec)
        provenance = self._provenance(evidence)
        if plan.status is not DiagnosticStatus.READY:
            return _non_ready_result(plan, spec=spec, provenance=provenance)

        backend = self.backend
        assert backend is not None  # guaranteed by the plan
        base_details: dict[str, Any] = {
            "axis": evidence.axis,
            "support": list(evidence.support),
            "sample_count": evidence.total,
            "backend_revision": str(backend.revision),
        }

        output, error = _call_backend(backend.depths, evidence.texts)
        _record_backend_revision(backend, details=base_details, provenance=provenance)
        if error is not None:
            return _backend_failure(
                component=self.name,
                details=base_details,
                provenance=provenance,
                reason_code="backend_call_failed",
                reason=f"The dependency-parser backend raised {error}",
            )
        depths, problem = _validate_count_output(
            output, expected=evidence.total, what="depths"
        )
        if depths is None:
            return _backend_failure(
                component=self.name,
                details=base_details,
                provenance=provenance,
                reason_code="backend_output_invalid",
                reason=f"The dependency-parser backend output is unusable: {problem}",
            )

        group_depth_totals = {label: 0 for label in evidence.support}
        for label, depth in zip(evidence.groups, depths):
            group_depth_totals[label] += depth
        group_sample_counts = dict(evidence.group_sample_counts)
        group_mean_depths = {
            label: group_depth_totals[label] / group_sample_counts[label]
            for label in evidence.support
        }
        max_gap, widest_pair = max_pairwise_gap(group_mean_depths)
        total_depth = sum(depths)
        # Same rule as b_diff_len: one pooled depth total over one pooled unit
        # total, deliberately not the unweighted mean of per-group means.
        pooled_mean_depth = total_depth / evidence.total

        base_details.update(
            {
                "group_mean_depths": group_mean_depths,
                "group_sample_counts": group_sample_counts,
                "group_depth_totals": group_depth_totals,
                "total_depth": total_depth,
                "max_absolute_gap": max_gap,
                "widest_pair": list(widest_pair) if widest_pair is not None else None,
                "pooled_mean_depth": pooled_mean_depth,
                "denominator_rule": "sample_weighted_pooled_mean",
                "unit": "ratio_to_pooled_mean_depth",
                "estimand": "normalized_maximum_pairwise_group_mean_depth_gap",
                "paper_alignment": self.paper_alignment,
            }
        )
        if pooled_mean_depth == 0.0:
            return ComponentResult(
                component=self.name,
                status=DiagnosticStatus.FAILED,
                details=base_details,
                provenance=provenance,
                reason_code="zero_depth_denominator",
                reason=(
                    "The sample-weighted pooled mean depth is zero, so the "
                    "normalized disparity is undefined."
                ),
            )
        value = max_gap / pooled_mean_depth
        if not math.isfinite(value):
            return _numeric_failure(
                component=self.name,
                details=base_details,
                provenance=provenance,
                reason="The normalized depth disparity is not a finite value.",
            )
        return ComponentResult(
            component=self.name,
            status=DiagnosticStatus.READY,
            value=value,
            details=base_details,
            assumptions=(
                DESCRIPTIVE_INTERPRETATION,
                "Depth is whatever the declared parser backend reports per text; "
                "the value is comparable only across audits that declare the "
                "same parser revision.",
                "The disparity denominator is the sample-weighted pooled mean "
                "depth over every text, not an unweighted mean of group means.",
                f"Design stance: {_enum_value(spec.design_stance)}.",
            ),
            provenance=provenance,
        )


# --------------------------------------------------------------------------
# Backend-dependent slots
# --------------------------------------------------------------------------


def _view_not_supplied_result(
    component: str,
    *,
    axis: str,
    view: str,
    provenance: Optional[Mapping[str, Any]] = None,
) -> ComponentResult:
    """Report absent evidence geometry as not-applicable, never as zero."""
    return ComponentResult(
        component=component,
        status=DiagnosticStatus.NOT_APPLICABLE,
        details={"axis": axis, "required_view": view},
        provenance=dict(provenance) if provenance is not None else {},
        reason_code="evidence_view_not_supplied",
        reason=(
            f"{component} requires a {view} evidence view for axis {axis!r}; "
            "none was supplied."
        ),
    )


# --------------------------------------------------------------------------
# The eight-slot vector
# --------------------------------------------------------------------------


def construction_vector(report: DiagnosticReport) -> tuple[ComponentResult, ...]:
    """Return the eight construction results in canonical slot order.

    ``DiagnosticReport.components`` is alphabetized, so this function,
    :data:`CONSTRUCTION_SLOTS` and ``report.provenance["slot_order"]`` are the
    only authorities for the declared order.
    """
    if not isinstance(report, DiagnosticReport):
        raise TypeError(
            f"report must be a DiagnosticReport, got {type(report).__name__}."
        )
    missing = [slot for slot in CONSTRUCTION_SLOTS if slot not in report.components]
    if missing:
        raise KeyError(f"report is missing construction slot(s): {missing!r}.")
    return tuple(report.components[slot] for slot in CONSTRUCTION_SLOTS)


_DEFAULT_SLOT_CLASSES: Final[Mapping[str, type]] = MappingProxyType(
    {
        "b_min": MinimalPairResidual,
        "b_equiv": SemanticEquivalence,
        "b_gram": GrammarConsistency,
        "b_diff_len": LengthDisparity,
        "b_diff_dep": DependencyDepthDisparity,
        "b_frame": FramingDisparity,
        "b_opt": OptionLengthBias,
        "b_temp": TemplateImbalance,
    }
)


def _normalize_construction_diagnostics(
    diagnostics: Sequence[DatasetDiagnostic],
) -> dict[str, DatasetDiagnostic]:
    """Validate an explicit diagnostic selection against the slot vocabulary."""
    if isinstance(diagnostics, (str, bytes)) or not isinstance(diagnostics, Sequence):
        raise TypeError("diagnostics must be an ordered sequence of diagnostics.")
    selected: dict[str, DatasetDiagnostic] = {}
    names: list[str] = []
    for index, diagnostic in enumerate(diagnostics):
        if not isinstance(diagnostic, DatasetDiagnostic):
            raise TypeError(
                f"diagnostics[{index}] must be a DatasetDiagnostic, "
                f"got {type(diagnostic).__name__}."
            )
        name = require_nonempty_string(diagnostic.name, f"diagnostics[{index}].name")
        if name not in CONSTRUCTION_SLOTS:
            raise ValueError(
                f"diagnostics[{index}].name {name!r} is not a construction "
                f"slot; expected one of {list(CONSTRUCTION_SLOTS)}."
            )
        names.append(name)
        selected[name] = diagnostic
    if len(set(names)) != len(names):
        raise ValueError("diagnostics must not contain duplicate component names.")
    return selected


def audit_construction(
    evidence: DatasetEvidence,
    spec: DatasetAuditSpec,
    *,
    axis: str,
    diagnostics: Sequence[DatasetDiagnostic] = (),
) -> DiagnosticReport:
    """Run the construction vector for one axis and return all eight slots.

    Every slot is always present, every slot carries its own applicability
    status, and no aggregate construction score is produced.
    """
    if not isinstance(evidence, DatasetEvidence):
        raise TypeError(
            f"evidence must be a DatasetEvidence, got {type(evidence).__name__}."
        )
    if not isinstance(spec, DatasetAuditSpec):
        raise TypeError(f"spec must be a DatasetAuditSpec, got {type(spec).__name__}.")
    axis = require_nonempty_string(axis, "axis")

    supplied = _normalize_construction_diagnostics(diagnostics)
    instances = {
        slot: (
            supplied[slot] if slot in supplied else _DEFAULT_SLOT_CLASSES[slot]()
        )
        for slot in CONSTRUCTION_SLOTS
    }

    views: dict[str, Any] = {}
    views_used: dict[str, str] = {}
    for slot in CONSTRUCTION_SLOTS:
        view_name = _SLOT_VIEWS[slot]
        view = getattr(evidence, view_name).get(axis)
        views[slot] = view
        if view is not None:
            views_used[slot] = view_name

    results: dict[str, ComponentResult] = {}
    for slot in CONSTRUCTION_SLOTS:
        diagnostic = instances[slot]
        prefix = _shared_plan_prefix(
            component=slot, spec=spec, axis=axis, check_axis=False
        )
        if prefix is not None:
            results[slot] = _non_ready_result(prefix, spec=spec, provenance={})
            continue
        view = views[slot]
        if view is None:
            # A diagnostic raises TypeError on wrong evidence and must never
            # be handed an absent view, so the absence is decided here.
            results[slot] = _view_not_supplied_result(
                slot, axis=axis, view=_SLOT_VIEWS[slot]
            )
            continue
        plan = diagnostic.plan(view, spec)
        try:
            results[slot] = diagnostic.compute(view, spec)
        except Exception as exc:
            if plan.status is not DiagnosticStatus.READY:
                raise
            results[slot] = ComponentResult(
                component=slot,
                status=DiagnosticStatus.FAILED,
                details={"exception_type": type(exc).__name__},
                provenance={
                    "evidence": {"axis": view.axis, "source": view.source}
                },
                reason_code="computation_failed",
                reason=(
                    "Applicability was established, but the construction "
                    f"component failed with {type(exc).__name__}."
                ),
            )

    blocked_backends = [
        slot
        for slot in BACKEND_CONSTRUCTION_SLOTS
        if results[slot].status is DiagnosticStatus.BLOCKED
        and (results[slot].reason_code or "").endswith("_backend_unavailable")
    ]
    warnings: list[str] = [CONSTRUCTION_VECTOR_WARNING]
    if blocked_backends:
        ordered = [slot for slot in CONSTRUCTION_SLOTS if slot in blocked_backends]
        warnings.append(BACKEND_BLOCKED_WARNING.format(slots=", ".join(ordered)))
    if results["b_diff_len"].status is DiagnosticStatus.READY:
        warnings.append(TOKENIZATION_DIVERGENCE_WARNING)
    if results["b_frame"].status is DiagnosticStatus.READY and isinstance(
        getattr(instances["b_frame"], "predicate", None), InjectedFramePredicate
    ):
        warnings.append(INJECTED_PREDICATE_WARNING)
    if spec.design_stance is DesignStance.STRESS_TEST and any(
        result.status is DiagnosticStatus.READY for result in results.values()
    ):
        warnings.append(STRESS_TEST_WARNING)

    deduplicated: list[str] = []
    for warning in warnings:
        if warning not in deduplicated:
            deduplicated.append(warning)

    return DiagnosticReport(
        spec=spec,
        components=results,
        warnings=tuple(deduplicated),
        provenance={
            "entry_point": "audit_construction",
            "axis": axis,
            "slot_order": list(CONSTRUCTION_SLOTS),
            "implemented_slots": list(CONSTRUCTION_SLOTS),
            "backend_slots": list(BACKEND_CONSTRUCTION_SLOTS),
            "target_name": evidence.target_name,
            "evidence_views": list(evidence.available_views),
            "views_used": dict(sorted(views_used.items())),
            "diagnostics": {
                slot: type(instances[slot]).__qualname__
                for slot in CONSTRUCTION_SLOTS
            },
        },
    )


__all__ = [
    "BACKEND_CONSTRUCTION_SLOTS",
    "CONSTRUCTION_BACKEND_REQUIREMENTS",
    "CONSTRUCTION_SLOTS",
    "LIGHTWEIGHT_CONSTRUCTION_SLOTS",
    "SELF_IDENTIFICATION_FRAME",
    "DependencyDepthDisparity",
    "DependencyParserBackend",
    "EmbeddingBackend",
    "FrameMatchMode",
    "FramePredicate",
    "FramingDisparity",
    "GrammarCheckerBackend",
    "GrammarConsistency",
    "IdentityMaskConfig",
    "InjectedFramePredicate",
    "LengthDisparity",
    "MinimalPairResidual",
    "OptionLengthBias",
    "OptionRoleContrast",
    "SemanticEquivalence",
    "TemplateImbalance",
    "TokenizationMode",
    "TokenizationRule",
    "audit_construction",
    "construction_vector",
]
