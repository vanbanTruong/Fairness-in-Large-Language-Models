"""Stereotype leakage (``b_leak``) and its separate text-to-count extraction stage.

The component reports the *smoothed normalized mutual information* between a
declared set of group terms and a declared set of trait terms, computed over a
**complete** group-by-trait count matrix.  Two stages are deliberately kept
apart:

``LeakageExtractionConfig`` / ``SurfaceCooccurrenceExtractor``
    An immutable, digest-identified configuration that turns raw text into a
    count matrix.  Extraction is a distinct callable object; ``compute()``
    never performs it implicitly.

``StereotypeLeakage``
    A pure numeric kernel over the validated count matrix.  Both the raw-text
    path and the supplied-matrix path enter the *same* kernel with the same
    ``AssociationCounts`` object, so their agreement is structural rather than
    asserted.

Degenerate geometries are handled deliberately.  A single group term or a
single trait term is refused by ``AssociationCounts`` and
``LeakageExtractionConfig`` at construction (both require at least two distinct
terms), so no unreachable ``degenerate_pair_space`` code is invented here.  A
zero marginal is impossible *in the reals* because the additive smoothing
constant is required to be strictly positive, which places mass on every
declared cell; it is still reachable in IEEE-754 doubles at the extremes of the
validated ``smoothing_alpha`` domain (a marginal product that underflows to
zero, or a smoothed total that overflows to ``inf``), so the numeric kernel is
total and such input returns ``failed`` with
``degenerate_normalization_denominator`` or ``numeric_computation_failed`` --
carrying the marginals, entropies, smoothing constant and pair-space size --
rather than raising out of ``compute()``.  An all-zero matrix is a first-class case: it is a *valid*
canonical zero when an extraction record proves an extraction ran (or when the
caller explicitly opts in), and is BLOCKED otherwise, so malformed or
incomplete evidence can never become a successful zero.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
)

from ._kernels import stable_log_ratio
from ._messages import STRESS_TEST_WARNING
from ._utils import (
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
from .evidence import AssociationCounts, LeakageExtractionRecord, TextEvidence
from .spec import DatasetAuditSpec, DesignStance, TargetKind


EXTRACTOR_VERSION: Final[str] = "surface_cooccurrence/1"

LEAKAGE_INTENT_ASSUMPTION: Final[str] = (
    "An observed group-trait association may be the benchmark's intended test "
    "signal or an unintended construction artifact; this statistic cannot "
    "distinguish them."
)
LEAKAGE_INTENT_WARNING: Final[str] = (
    "An observed group-trait association may be the benchmark's intended test "
    "signal or an unintended construction artifact; the leakage statistic "
    "cannot distinguish them."
)
ZERO_HIT_WARNING: Final[str] = (
    "The extraction completed and found zero group-trait co-occurrences. The "
    "reported leakage of zero reflects an absence of lexicon matches in this "
    "corpus, not evidence that the dataset is free of stereotype association."
)
COUNTING_BASIS_ASSUMPTION: Final[str] = (
    "Co-occurrence is counted group-anchored over a symmetric token window, so "
    "one trait token near several group tokens contributes several events."
)
SMOOTHING_ASSUMPTION: Final[str] = (
    "Additive smoothing places mass on every declared cell, so the reported "
    "NMI shrinks toward zero as the declared pair space grows; values are "
    "comparable only across identical lexicons."
)
LOG_BASE_ASSUMPTION: Final[str] = (
    "Normalized mutual information is invariant to the log base, but mutual "
    "information and PMI are not."
)

_WINDOW_RULE: Final[str] = "symmetric_token_window_excluding_center"
_COUNTING_UNIT: Final[str] = "group_anchored_ordered_cooccurrence"
_SMOOTHING_SCHEME: Final[str] = "additive_over_complete_pair_space"
_PAIR_SPACE: Final[str] = "complete"
_ESTIMAND: Final[str] = "normalized_mutual_information"
_NMI_DEFINITION: Final[str] = "2 * MI / (H_group + H_trait)"
_TOP_PMI_PAIRS_SCOPE: Final[str] = "observed_cells_only"
_UNIT: Final[str] = "dimensionless"

_PAPER_TOKEN_PATTERN: Final[str] = r"\b[\w_]+\b"
_PAPER_WINDOW: Final[int] = 5

_MI_NEGATIVE_TOLERANCE: Final[float] = 1e-12
_NMI_UPPER_TOLERANCE: Final[float] = 1e-9


class TokenMatchAttribute(str, Enum):
    """Token attribute the extraction lexicons are matched against."""

    SURFACE = "surface"


class LogBase(str, Enum):
    """Declared logarithm base for the information-theoretic kernel."""

    BASE_2 = "base_2"
    NATURAL = "natural"


def _enum_value(value: Any) -> Any:
    """Return a stable scalar for enum-like public values."""

    return getattr(value, "value", value)


def _log_function(base: LogBase):
    """Return the logarithm implementing the declared base."""

    if base is LogBase.BASE_2:
        return math.log2
    return math.log


def _information_unit(base: LogBase) -> str:
    """Return the unit of a mutual information measured in *base*."""

    return "bits" if base is LogBase.BASE_2 else "nats"


def _normalize_string_mapping(value: Any, *, field_name: str) -> Mapping[str, str]:
    """Validate a ``str -> str`` mapping and return a sorted immutable copy."""

    if not isinstance(value, Mapping):
        raise TypeError(
            f"{field_name} must be a mapping of string -> string, "
            f"got {type(value).__name__}."
        )
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(
                f"{field_name} keys must be strings, got {type(key).__name__}."
            )
        if not key.strip():
            raise ValueError(f"{field_name} keys must be non-empty strings.")
        if not isinstance(item, str):
            raise TypeError(
                f"{field_name}[{key!r}] must be a string, "
                f"got {type(item).__name__}."
            )
        if not item.strip():
            raise ValueError(f"{field_name}[{key!r}] must be a non-empty string.")
        normalized[key] = item
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, kw_only=True)
class LeakageExtractionConfig:
    """Immutable, digest-identified rule for turning raw text into counts.

    The configuration is the estimand of the extraction stage: lexicons, token
    normalization, window size and match attribute are all declared here, never
    inferred from the corpus, and all of them reach report provenance.
    """

    group_lexicon: Sequence[str]
    trait_lexicon: Sequence[str]
    window: int = _PAPER_WINDOW
    match_attribute: TokenMatchAttribute = TokenMatchAttribute.SURFACE
    token_pattern: str = _PAPER_TOKEN_PATTERN
    lowercase: bool = True
    replace_substrings: Mapping[str, str] = field(default_factory=dict)
    surface_map: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    canonical_group_terms: tuple[str, ...] = field(init=False)
    canonical_trait_terms: tuple[str, ...] = field(init=False)
    collapsed_terms: Mapping[str, tuple[str, ...]] = field(init=False)
    pair_space_size: int = field(init=False)
    config_digest: str = field(init=False)
    _token_matcher: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        group_lexicon = normalize_string_sequence(
            self.group_lexicon,
            field_name="group_lexicon",
            allow_empty=False,
        )
        if len(set(group_lexicon)) != len(group_lexicon):
            raise ValueError("group_lexicon must not contain duplicate raw terms.")
        trait_lexicon = normalize_string_sequence(
            self.trait_lexicon,
            field_name="trait_lexicon",
            allow_empty=False,
        )
        if len(set(trait_lexicon)) != len(trait_lexicon):
            raise ValueError("trait_lexicon must not contain duplicate raw terms.")
        object.__setattr__(self, "group_lexicon", group_lexicon)
        object.__setattr__(self, "trait_lexicon", trait_lexicon)

        if isinstance(self.window, bool) or not isinstance(self.window, Integral):
            raise TypeError(
                f"window must be an integer, got {type(self.window).__name__}."
            )
        window = int(self.window)
        if window < 1:
            raise ValueError("window must be at least 1.")
        object.__setattr__(self, "window", window)

        object.__setattr__(
            self,
            "match_attribute",
            normalize_enum(
                self.match_attribute,
                TokenMatchAttribute,
                "match_attribute",
            ),
        )

        token_pattern = require_nonempty_string(self.token_pattern, "token_pattern")
        try:
            matcher = re.compile(token_pattern)
        except re.error as exc:
            raise ValueError(
                f"token_pattern must be a valid regular expression: {exc}."
            ) from exc
        if matcher.search("") is not None:
            raise ValueError("token_pattern must not match the empty string.")
        if matcher.groups > 1:
            raise ValueError(
                "token_pattern must not contain more than one capturing group; "
                "re.findall would return group tuples instead of tokens."
            )
        object.__setattr__(self, "token_pattern", token_pattern)
        object.__setattr__(self, "_token_matcher", matcher)

        if not isinstance(self.lowercase, bool):
            raise TypeError(
                f"lowercase must be a boolean, got {type(self.lowercase).__name__}."
            )

        replace_substrings = _normalize_string_mapping(
            self.replace_substrings,
            field_name="replace_substrings",
        )
        surface_map = _normalize_string_mapping(
            self.surface_map,
            field_name="surface_map",
        )
        object.__setattr__(self, "replace_substrings", replace_substrings)
        object.__setattr__(self, "surface_map", surface_map)

        group_forms = self._canonicalize_lexicon(group_lexicon)
        trait_forms = self._canonicalize_lexicon(trait_lexicon)

        canonical_group_terms = tuple(sorted(group_forms))
        if len(canonical_group_terms) < 2:
            raise ValueError(
                "group_lexicon must canonicalize to at least two distinct terms."
            )
        canonical_trait_terms = tuple(sorted(trait_forms))
        if len(canonical_trait_terms) < 2:
            raise ValueError(
                "trait_lexicon must canonicalize to at least two distinct terms."
            )
        overlap = set(canonical_group_terms).intersection(canonical_trait_terms)
        if overlap:
            raise ValueError(
                "group_lexicon and trait_lexicon must not share canonical "
                f"tokens: {sorted(overlap)!r}."
            )
        object.__setattr__(self, "canonical_group_terms", canonical_group_terms)
        object.__setattr__(self, "canonical_trait_terms", canonical_trait_terms)

        collapsed = {
            token: tuple(sorted(raw_terms))
            for token, raw_terms in {**group_forms, **trait_forms}.items()
            if len(raw_terms) > 1
        }
        object.__setattr__(
            self,
            "collapsed_terms",
            MappingProxyType(dict(sorted(collapsed.items()))),
        )
        object.__setattr__(
            self,
            "pair_space_size",
            len(canonical_group_terms) * len(canonical_trait_terms),
        )
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )
        object.__setattr__(self, "config_digest", json_digest(self._digest_payload()))

    def _canonical_token(self, term: str) -> str:
        """Reduce one raw lexicon term to exactly one canonical match token."""

        text = term.lower() if self.lowercase else term
        for key in sorted(self.replace_substrings):
            text = text.replace(key, self.replace_substrings[key])
        tokens = [
            self.surface_map.get(token.lower(), token)
            for token in self._token_matcher.findall(text)
        ]
        if len(tokens) != 1:
            raise ValueError(
                f"lexicon term {term!r} must normalize to exactly one token; "
                f"got {tokens!r}. Add a replace_substrings rule so multiword "
                "terms collapse before token matching."
            )
        return tokens[0]

    def _canonicalize_lexicon(
        self,
        lexicon: Sequence[str],
    ) -> dict[str, list[str]]:
        """Map every canonical token to the raw spellings that produced it."""

        forms: dict[str, list[str]] = {}
        for term in lexicon:
            forms.setdefault(self._canonical_token(term), []).append(term)
        return forms

    @property
    def paper_alignment(self) -> str:
        """Classify the extraction against the paper's canonical definition."""

        if (
            self.match_attribute is TokenMatchAttribute.SURFACE
            and self.window == _PAPER_WINDOW
            and self.lowercase
            and self.token_pattern == _PAPER_TOKEN_PATTERN
        ):
            return "paper_exact"
        return "generalized_extraction"

    @property
    def alignment_assumption(self) -> str:
        """Return a report-ready statement of the extraction's scope."""

        if self.paper_alignment == "paper_exact":
            return (
                "The text-to-count extraction exactly matches the paper's "
                "surface regex token window definition."
            )
        return (
            "The extraction settings are a generalized package extension, not "
            "the paper's canonical surface regex token window definition."
        )

    def _digest_payload(self) -> dict[str, Any]:
        """Return the digest-covered payload: everything but digest and prose."""

        return {
            "group_lexicon": list(self.group_lexicon),
            "trait_lexicon": list(self.trait_lexicon),
            "canonical_group_terms": list(self.canonical_group_terms),
            "canonical_trait_terms": list(self.canonical_trait_terms),
            "collapsed_terms": {
                token: list(raw_terms)
                for token, raw_terms in self.collapsed_terms.items()
            },
            "window": self.window,
            "window_rule": _WINDOW_RULE,
            "match_attribute": self.match_attribute.value,
            "token_pattern": self.token_pattern,
            "lowercase": self.lowercase,
            "replace_substrings": dict(self.replace_substrings),
            "surface_map": dict(self.surface_map),
            "counting_unit": _COUNTING_UNIT,
            "pair_space_size": self.pair_space_size,
            "paper_alignment": self.paper_alignment,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation of the complete rule."""

        payload = self._digest_payload()
        payload["config_digest"] = self.config_digest
        payload["provenance"] = thaw_json(self.provenance)
        return payload


class LeakageExtractor(Protocol):
    """The text-to-count stage contract; deliberately not runtime-checkable."""

    extractor_id: str
    extractor_version: str
    config: LeakageExtractionConfig
    config_digest: str

    def extract(self, evidence: TextEvidence) -> AssociationCounts:
        """Turn declared text evidence into a complete count matrix."""

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe description of this extractor."""


_EXTRACTOR_PROTOCOL_MEMBERS: Final[tuple[str, ...]] = (
    "config",
    "config_digest",
    "extract",
    "extractor_id",
    "extractor_version",
    "to_dict",
)


def _require_extractor(obj: Any) -> Any:
    """Return *obj* after checking it satisfies the extractor protocol."""

    missing = [
        name for name in _EXTRACTOR_PROTOCOL_MEMBERS if not hasattr(obj, name)
    ]
    if missing:
        raise TypeError(
            "extractor must implement the LeakageExtractor protocol "
            f"(missing: {', '.join(missing)}), got {type(obj).__name__}."
        )
    return obj


@dataclass(frozen=True, kw_only=True)
class SurfaceCooccurrenceExtractor:
    """Canonical surface regex token-window co-occurrence extractor."""

    config: LeakageExtractionConfig
    extractor_id: ClassVar[str] = "surface_cooccurrence"
    extractor_version: ClassVar[str] = EXTRACTOR_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.config, LeakageExtractionConfig):
            raise TypeError(
                "config must be a LeakageExtractionConfig, "
                f"got {type(self.config).__name__}."
            )
        if self.config.match_attribute is not TokenMatchAttribute.SURFACE:
            raise ValueError(
                "SurfaceCooccurrenceExtractor requires match_attribute 'surface'."
            )

    @property
    def config_digest(self) -> str:
        """Return the digest of the extraction configuration in force."""

        return self.config.config_digest

    def tokenize(self, text: str) -> tuple[str, ...]:
        """Apply the declared normalization and return surface match tokens."""

        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}.")
        config = self.config
        value = text.lower() if config.lowercase else text
        for key in sorted(config.replace_substrings):
            value = value.replace(key, config.replace_substrings[key])
        surface_map = config.surface_map
        return tuple(
            surface_map.get(token.lower(), token)
            for token in config._token_matcher.findall(value)
        )

    def extract(self, evidence: TextEvidence) -> AssociationCounts:
        """Count group-anchored trait co-occurrences over the declared window."""

        if not isinstance(evidence, TextEvidence):
            raise TypeError(
                f"evidence must be TextEvidence, got {type(evidence).__name__}."
            )
        config = self.config
        window = config.window
        group_terms = frozenset(config.canonical_group_terms)
        trait_terms = frozenset(config.canonical_trait_terms)

        pair_counts: dict[tuple[str, str], int] = {}
        token_count = 0
        matched_group_positions = 0
        matched_trait_positions = 0
        event_count = 0

        for text in evidence.texts:
            tokens = self.tokenize(text)
            length = len(tokens)
            token_count += length
            for index, token in enumerate(tokens):
                if token in trait_terms:
                    matched_trait_positions += 1
                if token not in group_terms:
                    continue
                matched_group_positions += 1
                start = max(0, index - window)
                stop = min(length, index + window + 1)
                for neighbour_index in range(start, stop):
                    if neighbour_index == index:
                        continue
                    neighbour = tokens[neighbour_index]
                    if neighbour not in trait_terms:
                        continue
                    key = (token, neighbour)
                    pair_counts[key] = pair_counts.get(key, 0) + 1
                    event_count += 1

        record = LeakageExtractionRecord(
            extractor_id=self.extractor_id,
            extractor_version=self.extractor_version,
            config_digest=config.config_digest,
            text_digest=evidence.text_digest,
            text_unit_count=evidence.total,
            token_count=token_count,
            matched_group_positions=matched_group_positions,
            matched_trait_positions=matched_trait_positions,
            event_count=event_count,
            window=window,
        )
        return AssociationCounts.from_pair_counts(
            pair_counts,
            axis=evidence.axis,
            group_terms=config.canonical_group_terms,
            trait_terms=config.canonical_trait_terms,
            source=evidence.source,
            counting_basis=(
                "surface lexical co-occurrence within a symmetric "
                f"+/-{window}-token window, group-anchored, self position "
                "excluded"
            ),
            extraction=record,
            provenance={
                "extraction": self.to_dict(),
                "text_evidence": {
                    "axis": evidence.axis,
                    "source": evidence.source,
                    "total": evidence.total,
                    "text_digest": evidence.text_digest,
                    "provenance": thaw_json(evidence.provenance),
                },
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe description of this extractor."""

        return {
            "extractor_id": self.extractor_id,
            "extractor_version": self.extractor_version,
            "config": self.config.to_dict(),
            "config_digest": self.config_digest,
        }


def _resolve_extractor(config: LeakageExtractionConfig) -> Any:
    """Map a declared extraction configuration to its extractor implementation."""

    if not isinstance(config, LeakageExtractionConfig):
        raise TypeError(
            "leakage_extraction must be a LeakageExtractionConfig, "
            f"got {type(config).__name__}."
        )
    if config.match_attribute is TokenMatchAttribute.SURFACE:
        return SurfaceCooccurrenceExtractor(config=config)
    raise ValueError(  # pragma: no cover - SURFACE is the only member today.
        "no extractor is implemented for match attribute "
        f"{config.match_attribute.value!r}."
    )


def _association_metadata(counts: AssociationCounts) -> dict[str, Any]:
    """Copy count-matrix source metadata into result provenance."""

    return {
        "axis": counts.axis,
        "source": counts.source,
        "counting_basis": counts.counting_basis,
        "total_events": counts.total_events,
        "pair_space_size": counts.pair_space_size,
        "observed_cell_count": counts.observed_cell_count,
        "lexicon_digest": counts.lexicon_digest,
        "matrix_digest": counts.matrix_digest,
        "provenance": thaw_json(counts.provenance),
    }


def _text_metadata(evidence: TextEvidence) -> dict[str, Any]:
    """Copy text-evidence source metadata into result provenance."""

    return {
        "axis": evidence.axis,
        "source": evidence.source,
        "total": evidence.total,
        "text_digest": evidence.text_digest,
        "provenance": thaw_json(evidence.provenance),
    }


def _evidence_metadata(evidence: Any) -> dict[str, Any]:
    """Dispatch provenance metadata on the supplied evidence view."""

    if isinstance(evidence, AssociationCounts):
        return _association_metadata(evidence)
    return _text_metadata(evidence)


def _marginal_entropy(
    probabilities: Iterable[float],
    *,
    log: Callable[[float], float],
) -> float:
    """Shannon entropy of a marginal, total over degenerate float input.

    A strictly positive smoothing constant places positive mass on every
    declared cell, so in the reals every marginal is strictly positive. In
    IEEE-754 doubles it need not be: a smoothed total that overflows to
    ``inf`` drives every cell -- and therefore every marginal -- to exactly
    ``0.0``. ``math.log(0.0)`` raises ``ValueError``, so the degenerate case
    returns ``nan`` instead and the caller's
    ``degenerate_normalization_denominator`` guard reports it as a numeric
    failure with the estimator settings attached.
    """

    terms = []
    for probability in probabilities:
        if not (probability > 0.0) or not math.isfinite(probability):
            return math.nan
        terms.append(probability * log(probability))
    return -math.fsum(terms)


def _leakage_statistics(
    counts: AssociationCounts,
    *,
    smoothing_alpha: float,
    log_base: LogBase,
) -> dict[str, Any]:
    """Pure kernel over a complete count matrix; no text, no dataset identity.

    The smoothed joint spans the **complete** declared pair space, so every
    cell carries positive mass and both marginals are strictly positive.  The
    canonical runner's ``if total <= 0: return 0.0`` early exit is deliberately
    absent: that branch is exactly the "malformed evidence becomes a successful
    zero" this component refuses, and a strictly positive smoothing constant
    makes it unreachable in any case.

    That argument holds in the reals, not in floating point.  At the extremes
    of the validated ``smoothing_alpha`` domain the marginal product can
    underflow to zero and the smoothed total can overflow to ``inf``, so this
    kernel never raises on degenerate float input: it propagates ``inf`` or
    ``nan``, which ``compute()`` converts into an explicit ``failed`` result
    with a numeric reason code instead of an escaping ``ZeroDivisionError``.
    """

    groups = counts.group_terms
    traits = counts.trait_terms
    log = _log_function(log_base)

    total = counts.total_events + smoothing_alpha * counts.pair_space_size
    joint = {
        group: {
            trait: (counts.counts[group][trait] + smoothing_alpha) / total
            for trait in traits
        }
        for group in groups
    }
    group_probabilities = {
        group: math.fsum(joint[group][trait] for trait in traits) for group in groups
    }
    trait_probabilities = {
        trait: math.fsum(joint[group][trait] for group in groups) for trait in traits
    }
    pointwise = {
        group: {
            trait: stable_log_ratio(
                joint[group][trait],
                group_probabilities[group] * trait_probabilities[trait],
                base=log_base,
            )
            for trait in traits
        }
        for group in groups
    }
    mutual_information = math.fsum(
        joint[group][trait] * pointwise[group][trait]
        for group in groups
        for trait in traits
    )
    entropy_group = _marginal_entropy(group_probabilities.values(), log=log)
    entropy_trait = _marginal_entropy(trait_probabilities.values(), log=log)
    return {
        "smoothed_total": total,
        "joint": joint,
        "group_probabilities": group_probabilities,
        "trait_probabilities": trait_probabilities,
        "pointwise": pointwise,
        "mutual_information": mutual_information,
        "entropy_group": entropy_group,
        "entropy_trait": entropy_trait,
    }


@dataclass(frozen=True, kw_only=True)
class StereotypeLeakage(DatasetDiagnostic):
    """Smoothed normalized mutual information between group and trait terms."""

    name: ClassVar[str] = "b_leak"
    extractor: Optional[LeakageExtractor] = None
    smoothing_alpha: float = 1.0
    log_base: LogBase = LogBase.BASE_2
    top_pmi_pairs: int = 15
    include_count_matrix: bool = False
    allow_unverified_zero: bool = False

    def __post_init__(self) -> None:
        if self.extractor is not None:
            _require_extractor(self.extractor)

        value = self.smoothing_alpha
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                "smoothing_alpha must be a real number (booleans are not accepted)"
            )
        try:
            converted = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                "smoothing_alpha must be representable as a finite float"
            ) from exc
        if not math.isfinite(converted) or converted < sys.float_info.min:
            raise ValueError(
                "smoothing_alpha must be a finite float at least as large as "
                f"sys.float_info.min ({sys.float_info.min!r})"
            )
        object.__setattr__(self, "smoothing_alpha", converted)

        object.__setattr__(
            self,
            "log_base",
            normalize_enum(self.log_base, LogBase, "log_base"),
        )

        if isinstance(self.top_pmi_pairs, bool) or not isinstance(
            self.top_pmi_pairs, Integral
        ):
            raise TypeError(
                "top_pmi_pairs must be an integer, "
                f"got {type(self.top_pmi_pairs).__name__}."
            )
        top_pmi_pairs = int(self.top_pmi_pairs)
        if top_pmi_pairs < 0:
            raise ValueError("top_pmi_pairs must be non-negative.")
        object.__setattr__(self, "top_pmi_pairs", top_pmi_pairs)

        for name in ("include_count_matrix", "allow_unverified_zero"):
            flag = getattr(self, name)
            if not isinstance(flag, bool):
                raise TypeError(
                    f"{name} must be a boolean, got {type(flag).__name__}."
                )

    @property
    def paper_alignment(self) -> str:
        """Classify the estimator against the paper's parity preset."""

        if self.smoothing_alpha == 1.0 and self.log_base is LogBase.BASE_2:
            return "paper_exact"
        return "generalized_estimator"

    @property
    def alignment_assumption(self) -> str:
        """Return a report-ready statement of the estimator's scope."""

        if self.paper_alignment == "paper_exact":
            return (
                "The estimator exactly matches the paper's add-one smoothed, "
                "base-2 normalized mutual information definition."
            )
        return (
            "The smoothing constant or logarithm base is a generalized package "
            "extension, not the paper's add-one smoothed base-2 definition."
        )

    @property
    def information_unit(self) -> str:
        """Return the unit of the mutual information reported in details."""

        return _information_unit(self.log_base)

    def _resolved_extractor(self, spec: DatasetAuditSpec) -> Any:
        """Return the extractor in force, preferring the diagnostic's own."""

        if self.extractor is not None:
            return self.extractor
        config = getattr(spec, "leakage_extraction", None)
        if config is None:
            return None
        return _resolve_extractor(config)

    def _estimator_provenance(self) -> dict[str, Any]:
        return {
            "smoothing_alpha": self.smoothing_alpha,
            "smoothing_scheme": _SMOOTHING_SCHEME,
            "log_base": self.log_base.value,
            "information_unit": self.information_unit,
            "pair_space": _PAIR_SPACE,
            "top_pmi_pairs": self.top_pmi_pairs,
            "paper_alignment": self.paper_alignment,
        }

    def plan(
        self,
        evidence: AssociationCounts | TextEvidence,
        spec: DatasetAuditSpec,
    ) -> ComponentPlan:
        """Decide applicability without extracting or computing anything."""

        if not isinstance(evidence, (AssociationCounts, TextEvidence)):
            raise TypeError(
                "evidence must be AssociationCounts or TextEvidence, "
                f"got {type(evidence).__name__}."
            )
        if not isinstance(spec, DatasetAuditSpec):
            raise TypeError(
                f"spec must be DatasetAuditSpec, got {type(spec).__name__}."
            )

        axis = evidence.axis

        requested = spec.requested_components
        if requested is not None and self.name not in requested:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.NOT_APPLICABLE,
                reason_code="component_not_requested",
                reason=(
                    "The b_leak component was not requested by the audit "
                    "specification."
                ),
                details={"axis": axis},
            )

        override = getattr(spec, "component_overrides", None)
        override = None if override is None else override.get(self.name)
        if override is not None:
            return ComponentPlan(
                component=self.name,
                status=override.status,
                reason_code="applicability_override",
                reason=override.reason,
                details={
                    "axis": axis,
                    "applicability_override": True,
                    "declared_reason_code": override.declared_reason_code,
                    "override_source": "spec",
                },
            )

        if spec.target_kind is not TargetKind.BENCHMARK_DATASET:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.NOT_APPLICABLE,
                reason_code="target_kind_not_supported",
                reason=(
                    "b_leak applies to benchmark-dataset composition, not "
                    f"target kind {_enum_value(spec.target_kind)!r}. "
                    "Association measured over generated output is output "
                    "association, not dataset leakage."
                ),
                details={"axis": axis},
            )

        protected_axes = getattr(spec, "protected_axes", ()) or ()
        if protected_axes and axis not in protected_axes:
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.BLOCKED,
                reason_code="axis_not_declared",
                reason=(
                    f"Axis {axis!r} is not declared in spec.protected_axes "
                    f"{list(protected_axes)!r}."
                ),
                details={
                    "axis": axis,
                    "protected_axes": list(protected_axes),
                },
            )

        extractor = self._resolved_extractor(spec)

        if isinstance(evidence, TextEvidence):
            if extractor is None:
                return ComponentPlan(
                    component=self.name,
                    status=DiagnosticStatus.BLOCKED,
                    reason_code="missing_extraction_config",
                    reason=(
                        "b_leak requires an explicit text-to-count extraction "
                        "configuration when raw text evidence is supplied."
                    ),
                    details={"axis": axis, "evidence_path": "raw_text"},
                )
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.READY,
                details={
                    "axis": axis,
                    "evidence_path": "raw_text",
                    "group_term_count": len(extractor.config.canonical_group_terms),
                    "trait_term_count": len(extractor.config.canonical_trait_terms),
                    "pair_space_size": extractor.config.pair_space_size,
                    "smoothing_alpha": self.smoothing_alpha,
                    "log_base": self.log_base.value,
                    "paper_alignment": self.paper_alignment,
                },
            )

        if extractor is not None:
            declared_groups = extractor.config.canonical_group_terms
            declared_traits = extractor.config.canonical_trait_terms
            if (
                tuple(evidence.group_terms) != tuple(declared_groups)
                or tuple(evidence.trait_terms) != tuple(declared_traits)
            ):
                return ComponentPlan(
                    component=self.name,
                    status=DiagnosticStatus.BLOCKED,
                    reason_code="lexicon_support_mismatch",
                    reason=(
                        "The supplied association count matrix and the "
                        "configured extraction declare different canonical "
                        "group or trait terms."
                    ),
                    details={
                        "axis": axis,
                        "evidence_group_terms": list(evidence.group_terms),
                        "extraction_group_terms": list(declared_groups),
                        "group_term_symmetric_difference": sorted(
                            set(evidence.group_terms).symmetric_difference(
                                declared_groups
                            )
                        ),
                        "evidence_trait_terms": list(evidence.trait_terms),
                        "extraction_trait_terms": list(declared_traits),
                        "trait_term_symmetric_difference": sorted(
                            set(evidence.trait_terms).symmetric_difference(
                                declared_traits
                            )
                        ),
                    },
                )
            if (
                evidence.extraction is not None
                and evidence.extraction.config_digest != extractor.config_digest
            ):
                return ComponentPlan(
                    component=self.name,
                    status=DiagnosticStatus.BLOCKED,
                    reason_code="extraction_config_mismatch",
                    reason=(
                        "The supplied association count matrix was extracted "
                        "under a different extraction configuration than the "
                        "one configured for this audit."
                    ),
                    details={
                        "axis": axis,
                        "evidence_config_digest": (
                            evidence.extraction.config_digest
                        ),
                        "extraction_config_digest": extractor.config_digest,
                    },
                )

        if (
            evidence.total_events == 0
            and evidence.extraction is None
            and not self.allow_unverified_zero
        ):
            return ComponentPlan(
                component=self.name,
                status=DiagnosticStatus.BLOCKED,
                reason_code="zero_counts_without_extraction_record",
                reason=(
                    "An all-zero association matrix with no extraction record "
                    "cannot be distinguished from missing evidence; supply the "
                    "extraction record or set allow_unverified_zero explicitly."
                ),
                details={
                    "axis": axis,
                    "total_events": evidence.total_events,
                    "pair_space_size": evidence.pair_space_size,
                    "allow_unverified_zero": self.allow_unverified_zero,
                },
            )

        return ComponentPlan(
            component=self.name,
            status=DiagnosticStatus.READY,
            details={
                "axis": axis,
                "evidence_path": "association_counts",
                "group_term_count": len(evidence.group_terms),
                "trait_term_count": len(evidence.trait_terms),
                "pair_space_size": evidence.pair_space_size,
                "smoothing_alpha": self.smoothing_alpha,
                "log_base": self.log_base.value,
                "paper_alignment": self.paper_alignment,
            },
        )

    def compute(
        self,
        evidence: AssociationCounts | TextEvidence,
        spec: DatasetAuditSpec,
    ) -> ComponentResult:
        """Compute smoothed normalized mutual information over the pair space."""

        plan = self.plan(evidence, spec)
        extractor = self._resolved_extractor(spec)

        if plan.status is not DiagnosticStatus.READY:
            provenance: dict[str, Any] = {"evidence": _evidence_metadata(evidence)}
            if extractor is not None:
                provenance["extraction"] = extractor.config.to_dict()
            overrides = getattr(spec, "component_overrides", None)
            override = None if overrides is None else overrides.get(self.name)
            if override is not None:
                provenance["override"] = override.to_dict()
            return ComponentResult(
                component=plan.component,
                status=plan.status,
                details=dict(plan.details),
                provenance=provenance,
                reason_code=plan.reason_code,
                reason=plan.reason,
            )

        evidence_path = plan.details["evidence_path"]
        if evidence_path == "raw_text":
            try:
                counts = extractor.extract(evidence)
            except Exception as exc:
                provenance = {
                    "evidence": _evidence_metadata(evidence),
                    "extraction": extractor.config.to_dict(),
                    "extractor": {
                        "extractor_id": extractor.extractor_id,
                        "extractor_version": extractor.extractor_version,
                    },
                    "estimator": self._estimator_provenance(),
                }
                return ComponentResult(
                    component=self.name,
                    status=DiagnosticStatus.FAILED,
                    details={"exception_type": type(exc).__name__},
                    provenance=provenance,
                    reason_code="extraction_failed",
                    reason=(
                        "The declared text-to-count extraction failed with "
                        f"{type(exc).__name__}."
                    ),
                )
        else:
            counts = evidence

        provenance = {"evidence": _association_metadata(counts)}
        if extractor is not None:
            provenance["extraction"] = extractor.config.to_dict()
        if counts.extraction is not None:
            provenance["extraction_record"] = counts.extraction.to_dict()
        if evidence_path == "raw_text":
            provenance["extractor"] = {
                "extractor_id": extractor.extractor_id,
                "extractor_version": extractor.extractor_version,
            }
        provenance["estimator"] = self._estimator_provenance()

        statistics = _leakage_statistics(
            counts,
            smoothing_alpha=self.smoothing_alpha,
            log_base=self.log_base,
        )
        mutual_information = statistics["mutual_information"]
        entropy_group = statistics["entropy_group"]
        entropy_trait = statistics["entropy_trait"]
        denominator = entropy_group + entropy_trait

        # Additive smoothing keeps every marginal strictly positive in the
        # reals, but a smoothed total that overflows to ``inf`` (or a marginal
        # product that underflows to ``0.0``) breaks that in IEEE-754. Count
        # the degenerate marginals so a numeric failure still reports why.
        smoothed_total = statistics["smoothed_total"]
        nonpositive_marginals = sum(
            1
            for probability in (
                *statistics["group_probabilities"].values(),
                *statistics["trait_probabilities"].values(),
            )
            if not (probability > 0.0) or not math.isfinite(probability)
        )
        base_details = {
            "axis": counts.axis,
            "evidence_path": evidence_path,
            "total_events": counts.total_events,
            "pair_space_size": counts.pair_space_size,
            "smoothing_alpha": self.smoothing_alpha,
            "log_base": self.log_base.value,
            "smoothed_total": (
                smoothed_total if math.isfinite(smoothed_total) else None
            ),
            "nonpositive_marginal_count": nonpositive_marginals,
            "entropy_group": entropy_group if math.isfinite(entropy_group) else None,
            "entropy_trait": entropy_trait if math.isfinite(entropy_trait) else None,
        }

        if not math.isfinite(denominator) or denominator <= 0.0:
            return ComponentResult(
                component=self.name,
                status=DiagnosticStatus.FAILED,
                details=dict(base_details),
                provenance=provenance,
                reason_code="degenerate_normalization_denominator",
                reason=(
                    "The sum of the marginal entropies is not a positive "
                    "finite number, so normalized mutual information is "
                    "undefined."
                ),
            )

        numeric_zero_clamped = False
        numeric_upper_clamped = False

        if (
            not math.isfinite(mutual_information)
            or mutual_information < -_MI_NEGATIVE_TOLERANCE
        ):
            return ComponentResult(
                component=self.name,
                status=DiagnosticStatus.FAILED,
                details={
                    **base_details,
                    "mutual_information": (
                        mutual_information
                        if math.isfinite(mutual_information)
                        else None
                    ),
                },
                provenance=provenance,
                reason_code="numeric_computation_failed",
                reason=(
                    "The mutual information calculation did not produce a "
                    "finite non-negative value."
                ),
            )
        if mutual_information < 0.0:
            mutual_information = 0.0
            numeric_zero_clamped = True

        normalized = 2.0 * statistics["mutual_information"] / denominator
        if (
            not math.isfinite(normalized)
            or normalized < -_MI_NEGATIVE_TOLERANCE
            or normalized > 1.0 + _NMI_UPPER_TOLERANCE
        ):
            return ComponentResult(
                component=self.name,
                status=DiagnosticStatus.FAILED,
                details={
                    **base_details,
                    "normalized_mutual_information": (
                        normalized if math.isfinite(normalized) else None
                    ),
                    "nmi_denominator": denominator,
                },
                provenance=provenance,
                reason_code="numeric_computation_failed",
                reason=(
                    "The normalized mutual information calculation did not "
                    "produce a finite value inside [0, 1]."
                ),
            )
        if normalized < 0.0:
            normalized = 0.0
            numeric_zero_clamped = True
        elif normalized > 1.0:
            normalized = 1.0
            numeric_upper_clamped = True

        # Rule 3: a valid extraction with zero lexical hits is the canonical
        # zero.  A READY plan guarantees the zero is verified (an extraction
        # record exists) or explicitly opted into, and the smoothed joint is
        # exactly uniform, so 0.0 is the analytic value rather than a sentinel.
        zero_lexical_hits = counts.total_events == 0
        if zero_lexical_hits:
            if mutual_information != 0.0 or normalized != 0.0:
                numeric_zero_clamped = True
            mutual_information = 0.0
            normalized = 0.0

        observed_cells = [
            {
                "group": group,
                "trait": trait,
                "count": counts.counts[group][trait],
                "pmi": statistics["pointwise"][group][trait],
            }
            for group in counts.group_terms
            for trait in counts.trait_terms
            if counts.counts[group][trait] > 0
        ]
        observed_cells.sort(
            key=lambda cell: (
                -cell["pmi"],
                -cell["count"],
                cell["group"],
                cell["trait"],
            )
        )
        top_pairs = observed_cells[: self.top_pmi_pairs]

        details: dict[str, Any] = {
            "axis": counts.axis,
            "evidence_path": evidence_path,
            "normalized_mutual_information": normalized,
            "mutual_information": mutual_information,
            "entropy_group": entropy_group,
            "entropy_trait": entropy_trait,
            "nmi_denominator": denominator,
            "total_events": counts.total_events,
            "pair_space_size": counts.pair_space_size,
            "observed_cell_count": counts.observed_cell_count,
            "group_term_count": len(counts.group_terms),
            "trait_term_count": len(counts.trait_terms),
            "group_margin_probabilities": dict(statistics["group_probabilities"]),
            "trait_margin_probabilities": dict(statistics["trait_probabilities"]),
            "top_pmi_pairs": top_pairs,
            "top_pmi_pairs_scope": _TOP_PMI_PAIRS_SCOPE,
            "smoothing_alpha": self.smoothing_alpha,
            "smoothing_scheme": _SMOOTHING_SCHEME,
            "log_base": self.log_base.value,
            "information_unit": self.information_unit,
            "pair_space": _PAIR_SPACE,
            "unit": _UNIT,
            "estimand": _ESTIMAND,
            "nmi_definition": _NMI_DEFINITION,
            "paper_alignment": self.paper_alignment,
            "zero_lexical_hits": zero_lexical_hits,
            "numeric_zero_clamped": numeric_zero_clamped,
            "numeric_upper_clamped": numeric_upper_clamped,
        }
        if self.include_count_matrix:
            details["count_matrix"] = {
                group: dict(row) for group, row in counts.counts.items()
            }

        assumptions = [
            LEAKAGE_INTENT_ASSUMPTION,
            COUNTING_BASIS_ASSUMPTION,
            SMOOTHING_ASSUMPTION,
            LOG_BASE_ASSUMPTION,
            f"Counting basis: {counts.counting_basis}",
        ]
        if extractor is not None:
            assumptions.append(extractor.config.alignment_assumption)
        assumptions.append(f"Estimator alignment: {self.paper_alignment}.")
        assumptions.append(f"Design stance: {_enum_value(spec.design_stance)}.")

        return ComponentResult(
            component=self.name,
            status=DiagnosticStatus.READY,
            value=normalized,
            details=details,
            assumptions=tuple(assumptions),
            provenance=provenance,
        )


def audit_leakage(
    evidence: AssociationCounts | TextEvidence,
    spec: DatasetAuditSpec,
    diagnostic: StereotypeLeakage | None = None,
) -> DiagnosticReport:
    """Run the stereotype-leakage component and return a one-part report."""

    if not isinstance(spec, DatasetAuditSpec):
        raise TypeError(f"spec must be DatasetAuditSpec, got {type(spec).__name__}.")

    spec_config = getattr(spec, "leakage_extraction", None)

    if diagnostic is None:
        selected = (
            StereotypeLeakage()
            if spec_config is None
            else StereotypeLeakage(extractor=_resolve_extractor(spec_config))
        )
    else:
        if not isinstance(diagnostic, StereotypeLeakage):
            raise TypeError(
                "diagnostic must be a StereotypeLeakage, "
                f"got {type(diagnostic).__name__}."
            )
        if spec_config is None or diagnostic.extractor is None:
            selected = (
                diagnostic
                if spec_config is None
                else replace(
                    diagnostic,
                    extractor=_resolve_extractor(spec_config),
                )
            )
        else:
            if diagnostic.extractor.config_digest != spec_config.config_digest:
                raise ValueError(
                    "the supplied diagnostic's extraction configuration does "
                    "not match spec.leakage_extraction; supply one or make "
                    "them identical."
                )
            selected = diagnostic

    plan = selected.plan(evidence, spec)
    try:
        result = selected.compute(evidence, spec)
    except Exception as exc:
        if plan.status is not DiagnosticStatus.READY:
            raise
        provenance: dict[str, Any] = {"evidence": _evidence_metadata(evidence)}
        extractor = selected._resolved_extractor(spec)
        if extractor is not None:
            provenance["extraction"] = extractor.config.to_dict()
        result = ComponentResult(
            component=selected.name,
            status=DiagnosticStatus.FAILED,
            details={"exception_type": type(exc).__name__},
            provenance=provenance,
            reason_code="computation_failed",
            reason=(
                "Applicability was established, but the diagnostic computation "
                f"failed with {type(exc).__name__}."
            ),
        )

    ready = result.status is DiagnosticStatus.READY
    warnings: list[str] = []
    if ready:
        warnings.append(LEAKAGE_INTENT_WARNING)
    if ready and result.details.get("zero_lexical_hits"):
        warnings.append(ZERO_HIT_WARNING)
    if ready and spec.design_stance is DesignStance.STRESS_TEST:
        warnings.append(STRESS_TEST_WARNING)
    deduplicated: list[str] = []
    for warning in warnings:
        if warning not in deduplicated:
            deduplicated.append(warning)

    return DiagnosticReport(
        spec=spec,
        components={selected.name: result},
        warnings=tuple(deduplicated),
        provenance={
            "entry_point": "audit_leakage",
            "diagnostic": selected.name,
        },
    )


__all__ = [
    "EXTRACTOR_VERSION",
    "LeakageExtractionConfig",
    "LeakageExtractionRecord",
    "LeakageExtractor",
    "LogBase",
    "StereotypeLeakage",
    "SurfaceCooccurrenceExtractor",
    "TokenMatchAttribute",
    "audit_leakage",
]
