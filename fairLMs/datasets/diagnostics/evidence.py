"""Typed evidence and explicit schema adapters for dataset diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterable,
    Mapping,
    Optional,
    Sequence,
)

from fairLMs.datasets.diagnostics._utils import (
    freeze_json_mapping,
    json_digest,
    normalize_string_sequence,
    require_nonempty_string,
    thaw_json,
)


if TYPE_CHECKING:  # pragma: no cover - imported only by type checkers
    import pandas as pd


def _validate_support(value: Any, *, field_name: str = "support") -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a sequence, not a bare string.")
    try:
        support = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence of category labels.") from exc
    if len(support) < 2:
        raise ValueError(f"{field_name} must contain at least two categories.")
    for label in support:
        require_nonempty_string(label, f"{field_name} label")
    if len(set(support)) != len(support):
        raise ValueError(f"{field_name} must not contain duplicate category labels.")
    return tuple(sorted(support))


def _sorted_labels(values: Iterable[Any]) -> list[Any]:
    """Sort heterogeneous label sets deterministically for error messages."""
    items = list(values)
    if all(isinstance(item, str) for item in items):
        return sorted(items)
    return sorted(items, key=repr)


def _adapter_provenance(
    provenance: Optional[Mapping[str, Any]],
    *,
    adapter: str,
    group_field: str,
    trait_field: Optional[str] = None,
    count_field: Optional[str] = None,
    score_field: Optional[str] = None,
    value_map: Optional[Sequence[Mapping[str, Any]]],
    trait_value_map: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Mapping[str, Any]:
    if provenance is None:
        merged = {}
    elif isinstance(provenance, Mapping):
        merged = dict(provenance)
    else:
        raise TypeError(
            f"provenance must be a mapping, got {type(provenance).__name__}."
        )
    reserved = {"adapter", "field_mapping", "value_map", "value_map_used"}
    if trait_field is not None:
        reserved |= {"trait_value_map", "trait_value_map_used"}
    overlap = reserved.intersection(merged)
    if overlap:
        raise ValueError(
            "provenance uses reserved adapter key(s): " + ", ".join(sorted(overlap))
        )
    field_mapping = {"group": group_field}
    if trait_field is not None:
        field_mapping["trait"] = trait_field
    if count_field is not None:
        field_mapping["count"] = count_field
    if score_field is not None:
        field_mapping["score"] = score_field
    resolved = {
        **merged,
        "adapter": adapter,
        "field_mapping": field_mapping,
        "value_map": value_map,
        "value_map_used": value_map is not None,
    }
    if trait_field is not None:
        resolved["trait_value_map"] = trait_value_map
        resolved["trait_value_map_used"] = trait_value_map is not None
    return resolved


def _mapping_adapter_provenance(
    provenance: Optional[Mapping[str, Any]],
    *,
    adapter: str,
    field_mapping: Mapping[str, str],
    value_map: Optional[Sequence[Mapping[str, Any]]] = None,
    value_map_key: str = "value_map",
) -> Mapping[str, Any]:
    """Build adapter provenance for a container with its own field vocabulary."""
    if provenance is None:
        merged = {}
    elif isinstance(provenance, Mapping):
        merged = dict(provenance)
    else:
        raise TypeError(
            f"provenance must be a mapping, got {type(provenance).__name__}."
        )
    used_key = f"{value_map_key}_used"
    reserved = {"adapter", "field_mapping", value_map_key, used_key}
    overlap = reserved.intersection(merged)
    if overlap:
        raise ValueError(
            "provenance uses reserved adapter key(s): " + ", ".join(sorted(overlap))
        )
    return {
        **merged,
        "adapter": adapter,
        "field_mapping": dict(field_mapping),
        value_map_key: value_map,
        used_key: value_map is not None,
    }


def _paired_adapter_provenance(
    provenance: Optional[Mapping[str, Any]],
    *,
    adapter: str,
    pair_id_field: str,
    condition_field: str,
    score_field: Optional[str] = None,
    condition_map: Optional[Sequence[Mapping[str, Any]]],
    value_field: Optional[str] = None,
    value_role: str = "score",
) -> Mapping[str, Any]:
    if provenance is None:
        merged = {}
    elif isinstance(provenance, Mapping):
        merged = dict(provenance)
    else:
        raise TypeError(
            f"provenance must be a mapping, got {type(provenance).__name__}."
        )
    reserved = {
        "adapter",
        "field_mapping",
        "condition_map",
        "condition_map_used",
    }
    overlap = reserved.intersection(merged)
    if overlap:
        raise ValueError(
            "provenance uses reserved paired-adapter key(s): "
            + ", ".join(sorted(overlap))
        )
    resolved_value_field = score_field if value_field is None else value_field
    if resolved_value_field is None:
        raise ValueError(
            "either score_field or value_field must name the mapped value column."
        )
    return {
        **merged,
        "adapter": adapter,
        "field_mapping": {
            "pair_id": pair_id_field,
            "condition": condition_field,
            value_role: resolved_value_field,
        },
        "condition_map": condition_map,
        "condition_map_used": condition_map is not None,
    }


def _portable_raw_key(value: Any, *, path: str) -> tuple[Any, tuple[int, str]]:
    """Return a portable JSON scalar and a deterministic heterogeneous sort key."""
    if isinstance(value, Enum):
        raise TypeError(
            f"{path} must be a JSON scalar (null, boolean, string, or finite "
            "number), not an enum."
        )
    if value is None:
        return None, (0, "")
    if isinstance(value, bool):
        return value, (1, "1" if value else "0")
    if isinstance(value, Integral):
        normalized = int(value)
        return normalized, (2, str(normalized))
    if isinstance(value, Real):
        try:
            normalized = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path} must be representable as a finite float."
            ) from exc
        if not math.isfinite(normalized):
            raise ValueError(f"{path} must not be NaN or infinity.")
        return normalized, (3, normalized.hex())
    if isinstance(value, str):
        return value, (4, value)
    raise TypeError(
        f"{path} must be a JSON scalar (null, boolean, string, or finite "
        f"number), got {type(value).__name__}."
    )


def _prepare_value_map(
    value_map: Optional[Mapping[Any, str]],
    *,
    support: Sequence[str],
) -> tuple[Optional[Mapping[Any, str]], Optional[tuple[Mapping[str, Any], ...]]]:
    """Validate and defensively copy a raw-to-canonical category mapping."""
    if value_map is None:
        return None, None
    if not isinstance(value_map, Mapping):
        raise TypeError("value_map must be a mapping when provided.")

    lookup: dict[tuple[int, str], str] = {}
    serialized_entries: list[tuple[tuple[int, str], Mapping[str, Any]]] = []
    for raw, canonical in value_map.items():
        portable_raw, sort_key = _portable_raw_key(raw, path="value_map raw key")
        if sort_key in lookup:
            raise ValueError(
                "value_map contains raw keys that collide after JSON-scalar "
                f"normalization: {raw!r}."
            )
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError(
                f"value_map output for raw key {raw!r} must be a non-empty " "string."
            )
        if canonical not in support:
            raise ValueError(
                f"value_map output {canonical!r} for raw key {raw!r} is outside "
                f"the explicit support {list(support)!r}."
            )
        lookup[sort_key] = canonical
        serialized_entries.append(
            (sort_key, {"raw": portable_raw, "canonical": canonical})
        )

    serialized_entries.sort(key=lambda item: item[0])
    return (
        MappingProxyType(lookup),
        tuple(entry for _, entry in serialized_entries),
    )


@dataclass(frozen=True, kw_only=True)
class RepresentationEvidence:
    """Observed category counts for one explicit protected axis."""

    axis: str
    counts: Mapping[str, int]
    source: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )
        if not isinstance(self.counts, Mapping):
            raise TypeError(
                "counts must be a mapping of category label -> integer count, "
                f"got {type(self.counts).__name__}."
            )
        if len(self.counts) < 2:
            raise ValueError("counts must contain at least two categories.")

        counts = {}
        for label, raw in self.counts.items():
            require_nonempty_string(label, "count category")
            if isinstance(raw, bool) or not isinstance(raw, Integral):
                raise TypeError(
                    f"count for {label!r} must be an integer, "
                    f"got {type(raw).__name__}."
                )
            count = int(raw)
            if count < 0:
                raise ValueError(f"count for {label!r} must be non-negative.")
            counts[label] = count
        if sum(counts.values()) <= 0:
            raise ValueError("counts must contain at least one observed item.")
        object.__setattr__(
            self, "counts", MappingProxyType(dict(sorted(counts.items())))
        )
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    @property
    def support(self) -> tuple[str, ...]:
        """Sorted category support, including explicit zero-count cells."""
        return tuple(self.counts)

    @property
    def total(self) -> int:
        """Total number of represented items."""
        return sum(self.counts.values())

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "axis": self.axis,
            "counts": dict(self.counts),
            "source": self.source,
            "provenance": thaw_json(self.provenance),
        }

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        axis: str,
        group_field: str,
        support: Sequence[str],
        source: str,
        value_map: Optional[Mapping[Any, str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "RepresentationEvidence":
        """Count one explicitly mapped category per record.

        The adapter never guesses fields, drops missing values, or infers the
        full support from observed rows.
        """
        support_labels = _validate_support(support)
        group_field = require_nonempty_string(group_field, "group_field")
        if isinstance(records, (str, bytes, Mapping)):
            raise TypeError("records must be an iterable of mapping rows.")
        try:
            iterator = iter(records)
        except TypeError as exc:
            raise TypeError("records must be an iterable of mapping rows.") from exc
        prepared_value_map, serialized_value_map = _prepare_value_map(
            value_map,
            support=support_labels,
        )

        counts = {label: 0 for label in support_labels}
        row_count = 0
        for index, record in enumerate(iterator):
            row_count += 1
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"record at row {index} must be a mapping, "
                    f"got {type(record).__name__}."
                )
            if group_field not in record:
                raise ValueError(
                    f"record at row {index} is missing mapped field "
                    f"{group_field!r}."
                )
            raw = record[group_field]
            if raw is None and (
                prepared_value_map is None or (0, "") not in prepared_value_map
            ):
                raise ValueError(
                    f"record at row {index} has a missing value for "
                    f"{group_field!r}."
                )
            if prepared_value_map is not None:
                _, lookup_key = _portable_raw_key(
                    raw,
                    path=f"record at row {index} value for {group_field!r}",
                )
                present = lookup_key in prepared_value_map
                if not present:
                    raise ValueError(
                        f"record at row {index} has unmapped value {raw!r} for "
                        f"{group_field!r}."
                    )
                label = prepared_value_map[lookup_key]
            else:
                label = raw
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"record at row {index} maps to a non-string or empty "
                    f"category {label!r}."
                )
            if label not in counts:
                raise ValueError(
                    f"record at row {index} maps to {label!r}, which is outside "
                    f"the explicit support {list(support_labels)!r}."
                )
            counts[label] += 1
        if row_count == 0:
            raise ValueError("records must contain at least one row.")

        return cls(
            axis=axis,
            counts=counts,
            source=source,
            provenance=_adapter_provenance(
                provenance,
                adapter="records",
                group_field=group_field,
                value_map=serialized_value_map,
            ),
        )

    @classmethod
    def from_dataframe(
        cls,
        frame: "pd.DataFrame",
        *,
        axis: str,
        group_column: str,
        support: Sequence[str],
        source: str,
        value_map: Optional[Mapping[Any, str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "RepresentationEvidence":
        """Adapt one explicitly named pandas column without guessing schema."""
        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"frame must be a pandas DataFrame, got {type(frame).__name__}."
            )
        group_column = require_nonempty_string(group_column, "group_column")
        if group_column not in frame.columns:
            raise ValueError(f"frame is missing mapped group column {group_column!r}.")
        selected = frame[group_column]
        if not isinstance(selected, pd.Series):
            raise ValueError(
                f"frame mapped group column {group_column!r} must be unique."
            )
        record_evidence = cls.from_records(
            ({group_column: value} for value in selected.tolist()),
            axis=axis,
            group_field=group_column,
            support=support,
            source=source,
            value_map=value_map,
        )
        merged_provenance = _adapter_provenance(
            provenance,
            adapter="dataframe",
            group_field=group_column,
            value_map=record_evidence.provenance["value_map"],
        )
        return cls(
            axis=record_evidence.axis,
            counts=record_evidence.counts,
            source=record_evidence.source,
            provenance=merged_provenance,
        )


def _normalize_scores(value: Any) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("scores must be an ordered sequence of finite real numbers.")
    raw_scores = tuple(value)
    if not raw_scores:
        raise ValueError("scores must contain at least one value.")

    scores = []
    for index, raw in enumerate(raw_scores):
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise TypeError(
                f"score at row {index} must be a real number, "
                f"got {type(raw).__name__}."
            )
        try:
            score = float(raw)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"score at row {index} must be representable as a finite float."
            ) from exc
        if not math.isfinite(score):
            raise ValueError(f"score at row {index} must be finite.")
        scores.append(score)
    return tuple(scores)


def _normalize_score_range(value: Any) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("score_range must be an ordered pair of finite numbers.")
    endpoints = tuple(value)
    if len(endpoints) != 2:
        raise ValueError("score_range must contain exactly two endpoints.")

    normalized = []
    for name, raw in zip(("lower", "upper"), endpoints):
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise TypeError(
                f"score_range {name} endpoint must be a real number, "
                f"got {type(raw).__name__}."
            )
        try:
            endpoint = float(raw)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"score_range {name} endpoint must be representable as a finite float."
            ) from exc
        if not math.isfinite(endpoint):
            raise ValueError(f"score_range {name} endpoint must be finite.")
        normalized.append(endpoint)

    lower, upper = normalized
    if lower > upper:
        raise ValueError(
            "score_range lower endpoint must not exceed the upper endpoint."
        )
    return lower, upper


def _normalize_condition_roles(value: Any) -> tuple[str, str]:
    roles = normalize_string_sequence(
        value,
        field_name="condition_roles",
        allow_empty=False,
    )
    if len(roles) != 2:
        raise ValueError("condition_roles must contain exactly two roles.")
    if roles[0] == roles[1]:
        raise ValueError("condition_roles must contain two distinct roles.")
    return roles


def _normalize_pair_ids(
    value: Any,
    *,
    field_name: str = "pair_id",
    plural_field_name: str = "pair_ids",
) -> tuple[tuple[Any, ...], tuple[tuple[int, str], ...]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            f"{plural_field_name} must be an ordered sequence of JSON scalars."
        )
    raw_pair_ids = tuple(value)
    if not raw_pair_ids:
        raise ValueError(f"{plural_field_name} must contain at least one value.")

    portable_ids = []
    lookup_keys = []
    for index, raw in enumerate(raw_pair_ids):
        portable, lookup_key = _portable_raw_key(
            raw,
            path=f"{field_name} at row {index}",
        )
        if portable is None:
            raise ValueError(f"{field_name} at row {index} must not be missing.")
        if isinstance(portable, bool):
            raise TypeError(f"{field_name} at row {index} must not be a boolean.")
        if isinstance(portable, str) and not portable.strip():
            raise ValueError(f"{field_name} at row {index} must not be empty.")
        portable_ids.append(portable)
        lookup_keys.append(lookup_key)
    return tuple(portable_ids), tuple(lookup_keys)


@dataclass(frozen=True, kw_only=True)
class ScoredGroups:
    """Finite row-level scores paired with one explicitly declared group."""

    axis: str
    groups: Sequence[str]
    scores: Sequence[float]
    score_name: str
    source: str
    score_range: Optional[Sequence[float]] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self,
            "score_name",
            require_nonempty_string(self.score_name, "score_name"),
        )
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )

        groups = normalize_string_sequence(
            self.groups,
            field_name="groups",
            allow_empty=False,
        )
        scores = _normalize_scores(self.scores)
        if len(groups) != len(scores):
            raise ValueError(
                "groups and scores must contain the same number of rows; "
                f"got {len(groups)} groups and {len(scores)} scores."
            )
        support = tuple(sorted(set(groups)))
        if len(support) < 2:
            raise ValueError("groups must contain at least two observed categories.")

        score_range = _normalize_score_range(self.score_range)
        if score_range is not None:
            lower, upper = score_range
            for index, score in enumerate(scores):
                if score < lower or score > upper:
                    raise ValueError(
                        f"score at row {index} ({score!r}) is outside the "
                        f"declared score_range [{lower!r}, {upper!r}]."
                    )

        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "score_range", score_range)
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    @property
    def support(self) -> tuple[str, ...]:
        """Sorted observed group support."""
        return tuple(sorted(set(self.groups)))

    @property
    def total(self) -> int:
        """Number of scored rows."""
        return len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "axis": self.axis,
            "groups": list(self.groups),
            "scores": list(self.scores),
            "score_name": self.score_name,
            "source": self.source,
            "score_range": (
                None if self.score_range is None else list(self.score_range)
            ),
            "support": list(self.support),
            "provenance": thaw_json(self.provenance),
        }

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        axis: str,
        group_field: str,
        score_field: str,
        support: Sequence[str],
        score_name: str,
        source: str,
        value_map: Optional[Mapping[Any, str]] = None,
        score_range: Optional[Sequence[float]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "ScoredGroups":
        """Adapt explicitly mapped rows without coercing or dropping values."""
        support_labels = _validate_support(support)
        group_field = require_nonempty_string(group_field, "group_field")
        score_field = require_nonempty_string(score_field, "score_field")
        if group_field == score_field:
            raise ValueError("group_field and score_field must name distinct fields.")
        if isinstance(records, (str, bytes, Mapping)):
            raise TypeError("records must be an iterable of mapping rows.")
        try:
            iterator = iter(records)
        except TypeError as exc:
            raise TypeError("records must be an iterable of mapping rows.") from exc
        prepared_value_map, serialized_value_map = _prepare_value_map(
            value_map,
            support=support_labels,
        )

        groups = []
        scores = []
        for index, record in enumerate(iterator):
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"record at row {index} must be a mapping, "
                    f"got {type(record).__name__}."
                )
            for field_name in (group_field, score_field):
                if field_name not in record:
                    raise ValueError(
                        f"record at row {index} is missing mapped field "
                        f"{field_name!r}."
                    )

            raw_group = record[group_field]
            if raw_group is None and (
                prepared_value_map is None or (0, "") not in prepared_value_map
            ):
                raise ValueError(
                    f"record at row {index} has a missing value for "
                    f"{group_field!r}."
                )
            if prepared_value_map is not None:
                _, lookup_key = _portable_raw_key(
                    raw_group,
                    path=f"record at row {index} value for {group_field!r}",
                )
                if lookup_key not in prepared_value_map:
                    raise ValueError(
                        f"record at row {index} has unmapped value "
                        f"{raw_group!r} for {group_field!r}."
                    )
                group = prepared_value_map[lookup_key]
            else:
                group = raw_group
            if not isinstance(group, str) or not group.strip():
                raise ValueError(
                    f"record at row {index} maps to a non-string or empty "
                    f"group {group!r}."
                )
            if group not in support_labels:
                raise ValueError(
                    f"record at row {index} maps to {group!r}, which is "
                    f"outside the explicit support {list(support_labels)!r}."
                )

            raw_score = record[score_field]
            if isinstance(raw_score, bool) or not isinstance(raw_score, Real):
                raise TypeError(
                    f"score at row {index} must be a real number, "
                    f"got {type(raw_score).__name__}."
                )
            try:
                score = float(raw_score)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"score at row {index} must be representable as a finite float."
                ) from exc
            if not math.isfinite(score):
                raise ValueError(f"score at row {index} must be finite.")
            groups.append(group)
            scores.append(score)

        if not groups:
            raise ValueError("records must contain at least one row.")
        observed_support = set(groups)
        if observed_support != set(support_labels):
            raise ValueError(
                "every explicit support category must have at least one scored "
                "row; missing categories: "
                f"{sorted(set(support_labels) - observed_support)!r}."
            )

        return cls(
            axis=axis,
            groups=groups,
            scores=scores,
            score_name=score_name,
            source=source,
            score_range=score_range,
            provenance=_adapter_provenance(
                provenance,
                adapter="records",
                group_field=group_field,
                score_field=score_field,
                value_map=serialized_value_map,
            ),
        )

    @classmethod
    def from_dataframe(
        cls,
        frame: "pd.DataFrame",
        *,
        axis: str,
        group_column: str,
        score_column: str,
        support: Sequence[str],
        score_name: str,
        source: str,
        value_map: Optional[Mapping[Any, str]] = None,
        score_range: Optional[Sequence[float]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "ScoredGroups":
        """Adapt two explicit DataFrame columns without guessing or coercion."""
        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"frame must be a pandas DataFrame, got {type(frame).__name__}."
            )
        group_column = require_nonempty_string(group_column, "group_column")
        score_column = require_nonempty_string(score_column, "score_column")
        if group_column == score_column:
            raise ValueError("group_column and score_column must be distinct.")
        for role, column in (("group", group_column), ("score", score_column)):
            if column not in frame.columns:
                raise ValueError(f"frame is missing mapped {role} column {column!r}.")
            if not isinstance(frame[column], pd.Series):
                raise ValueError(
                    f"frame mapped {role} column {column!r} must be unique."
                )

        record_evidence = cls.from_records(
            (
                {group_column: group, score_column: score}
                for group, score in zip(
                    frame[group_column].tolist(),
                    frame[score_column].tolist(),
                )
            ),
            axis=axis,
            group_field=group_column,
            score_field=score_column,
            support=support,
            score_name=score_name,
            source=source,
            value_map=value_map,
            score_range=score_range,
        )
        merged_provenance = _adapter_provenance(
            provenance,
            adapter="dataframe",
            group_field=group_column,
            score_field=score_column,
            value_map=record_evidence.provenance["value_map"],
        )
        return cls(
            axis=record_evidence.axis,
            groups=record_evidence.groups,
            scores=record_evidence.scores,
            score_name=record_evidence.score_name,
            source=record_evidence.source,
            score_range=record_evidence.score_range,
            provenance=merged_provenance,
        )


@dataclass(frozen=True, kw_only=True)
class LabeledScoredGroups:
    """Scored group rows plus the outcome label each row was scored against.

    Composes :class:`ScoredGroups` rather than widening it. ``ScoredGroups``
    froze its contract with the explicit rule that labels and pair geometry do
    not widen the type, so the group/score half of the validation - finite
    scores, equal row counts, at least two observed groups, declared score range
    - is inherited here for free instead of being duplicated or relaxed.

    This is what the group-fairness post-processing methods need: a threshold or
    a calibrator is fitted against *outcomes*, not against scores alone.

    ``positive_label`` is mandatory and never inferred. Equal opportunity is
    defined as an equal true-positive rate **for the positive class**, so which
    label is positive is part of the question being asked, not a property of the
    data. This matches the rest of the library, which never infers a protected
    attribute, a role or a threshold.

    .. note::
       ``positive_label`` presumes a **binary** label. Under multi-class labels
       equal opportunity is defined per class, and this type does not model
       that. Ship binary, and do not pretend otherwise.

    Examples
    --------
    >>> from fairLMs.datasets.diagnostics import LabeledScoredGroups, ScoredGroups
    >>> evidence = LabeledScoredGroups(
    ...     scored=ScoredGroups(
    ...         axis="gender",
    ...         groups=["f", "f", "m", "m"],
    ...         scores=[0.9, 0.2, 0.8, 0.1],
    ...         score_name="p_hire",
    ...         source="unit-test",
    ...         score_range=[0.0, 1.0],
    ...     ),
    ...     labels=["hired", "not_hired", "hired", "not_hired"],
    ...     label_name="outcome",
    ...     positive_label="hired",
    ... )
    >>> evidence.n_rows
    4
    >>> tuple(group for group, _, _ in evidence.iter_groups())
    ('f', 'm')
    """

    scored: ScoredGroups
    labels: Sequence[str]
    label_name: str
    positive_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.scored, ScoredGroups):
            raise TypeError(
                "scored must be a ScoredGroups, got "
                f"{type(self.scored).__name__}."
            )
        object.__setattr__(
            self,
            "label_name",
            require_nonempty_string(self.label_name, "label_name"),
        )
        object.__setattr__(
            self,
            "positive_label",
            require_nonempty_string(self.positive_label, "positive_label"),
        )
        labels = normalize_string_sequence(
            self.labels,
            field_name="labels",
            allow_empty=False,
        )
        if len(labels) != len(self.scored.scores):
            raise ValueError(
                "labels and scores must contain the same number of rows; got "
                f"{len(labels)} labels and {len(self.scored.scores)} scores."
            )
        observed = set(labels)
        if len(observed) < 2:
            raise ValueError(
                "labels must contain at least two distinct outcomes; got only "
                f"{sorted(observed)!r}. A single outcome cannot define a rate."
            )
        if self.positive_label not in observed:
            raise ValueError(
                f"positive_label {self.positive_label!r} does not occur in labels; "
                f"observed labels are {sorted(observed)!r}."
            )
        object.__setattr__(self, "labels", labels)

    # -- convenience views over the composed evidence -----------------------
    @property
    def axis(self) -> str:
        """The protected axis, from the composed :class:`ScoredGroups`."""
        return self.scored.axis

    @property
    def groups(self) -> tuple:
        """Per-row group membership."""
        return tuple(self.scored.groups)

    @property
    def scores(self) -> tuple:
        """Per-row scores."""
        return tuple(self.scored.scores)

    @property
    def support(self) -> tuple:
        """Sorted observed group support."""
        return self.scored.support

    @property
    def label_support(self) -> tuple:
        """Sorted observed label support."""
        return tuple(sorted(set(self.labels)))

    @property
    def n_rows(self) -> int:
        """Number of labelled, scored rows."""
        return len(self.labels)

    def positives(self) -> tuple:
        """Per-row indicator of membership in the declared positive class."""
        return tuple(label == self.positive_label for label in self.labels)

    def iter_groups(self):
        """Yield ``(group, scores, positives)`` per group, in sorted order.

        Groups are emitted in sorted order so a fitted rule is deterministic.
        """
        for group in self.support:
            rows = [i for i, g in enumerate(self.groups) if g == group]
            yield (
                group,
                tuple(self.scores[i] for i in rows),
                tuple(self.labels[i] == self.positive_label for i in rows),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "scored": self.scored.to_dict(),
            "labels": list(self.labels),
            "label_name": self.label_name,
            "positive_label": self.positive_label,
            "label_support": list(self.label_support),
        }


@dataclass(frozen=True, kw_only=True)
class PairedScores:
    """Finite scores for complete two-condition counterfactual pairs."""

    axis: str
    pair_ids: Sequence[Any]
    conditions: Sequence[str]
    scores: Sequence[float]
    condition_roles: Sequence[str]
    score_name: str
    source: str
    pairing_basis: str
    score_range: Optional[Sequence[float]] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self,
            "score_name",
            require_nonempty_string(self.score_name, "score_name"),
        )
        object.__setattr__(
            self,
            "source",
            require_nonempty_string(self.source, "source"),
        )
        object.__setattr__(
            self,
            "pairing_basis",
            require_nonempty_string(self.pairing_basis, "pairing_basis"),
        )

        roles = _normalize_condition_roles(self.condition_roles)
        conditions = normalize_string_sequence(
            self.conditions,
            field_name="conditions",
            allow_empty=False,
        )
        scores = _normalize_scores(self.scores)
        pair_ids, pair_keys = _normalize_pair_ids(self.pair_ids)
        lengths = {len(pair_ids), len(conditions), len(scores)}
        if len(lengths) != 1:
            raise ValueError(
                "pair_ids, conditions, and scores must contain the same number "
                f"of rows; got {len(pair_ids)}, {len(conditions)}, and "
                f"{len(scores)}."
            )

        score_range = _normalize_score_range(self.score_range)
        if score_range is not None:
            lower, upper = score_range
            for index, score in enumerate(scores):
                if score < lower or score > upper:
                    raise ValueError(
                        f"score at row {index} ({score!r}) is outside the "
                        f"declared score_range [{lower!r}, {upper!r}]."
                    )

        pairs: dict[tuple[int, str], dict[str, Any]] = {}
        for index, (pair_id, pair_key, condition, score) in enumerate(
            zip(pair_ids, pair_keys, conditions, scores)
        ):
            if condition not in roles:
                raise ValueError(
                    f"condition at row {index} ({condition!r}) is outside the "
                    f"declared condition_roles {list(roles)!r}."
                )
            pair = pairs.setdefault(
                pair_key,
                {"pair_id": pair_id, "scores": {}},
            )
            if condition in pair["scores"]:
                raise ValueError(
                    f"pair {pair_id!r} contains duplicate condition role "
                    f"{condition!r}."
                )
            pair["scores"][condition] = score

        canonical_pair_ids = []
        canonical_conditions = []
        canonical_scores = []
        for pair_key in sorted(pairs):
            pair = pairs[pair_key]
            missing = [role for role in roles if role not in pair["scores"]]
            if missing:
                raise ValueError(
                    f"pair {pair['pair_id']!r} is missing condition role(s): "
                    f"{missing!r}."
                )
            for role in roles:
                canonical_pair_ids.append(pair["pair_id"])
                canonical_conditions.append(role)
                canonical_scores.append(pair["scores"][role])

        object.__setattr__(self, "pair_ids", tuple(canonical_pair_ids))
        object.__setattr__(self, "conditions", tuple(canonical_conditions))
        object.__setattr__(self, "scores", tuple(canonical_scores))
        object.__setattr__(self, "condition_roles", roles)
        object.__setattr__(self, "score_range", score_range)
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    @property
    def total(self) -> int:
        """Number of scored rows across both condition roles."""
        return len(self.scores)

    @property
    def pair_count(self) -> int:
        """Number of complete validated pairs."""
        return self.total // 2

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "axis": self.axis,
            "pair_ids": list(self.pair_ids),
            "conditions": list(self.conditions),
            "scores": list(self.scores),
            "condition_roles": list(self.condition_roles),
            "score_name": self.score_name,
            "source": self.source,
            "pairing_basis": self.pairing_basis,
            "score_range": (
                None if self.score_range is None else list(self.score_range)
            ),
            "pair_count": self.pair_count,
            "provenance": thaw_json(self.provenance),
        }

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        axis: str,
        pair_id_field: str,
        condition_field: str,
        score_field: str,
        condition_roles: Sequence[str],
        score_name: str,
        source: str,
        pairing_basis: str,
        condition_map: Optional[Mapping[Any, str]] = None,
        score_range: Optional[Sequence[float]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "PairedScores":
        """Adapt explicit pair/condition/score fields without dropping rows."""
        roles = _normalize_condition_roles(condition_roles)
        pair_id_field = require_nonempty_string(pair_id_field, "pair_id_field")
        condition_field = require_nonempty_string(
            condition_field,
            "condition_field",
        )
        score_field = require_nonempty_string(score_field, "score_field")
        mapped_fields = (pair_id_field, condition_field, score_field)
        if len(set(mapped_fields)) != len(mapped_fields):
            raise ValueError(
                "pair_id_field, condition_field, and score_field must be distinct."
            )
        if isinstance(records, (str, bytes, Mapping)):
            raise TypeError("records must be an iterable of mapping rows.")
        try:
            iterator = iter(records)
        except TypeError as exc:
            raise TypeError("records must be an iterable of mapping rows.") from exc

        prepared_condition_map, serialized_condition_map = _prepare_value_map(
            condition_map,
            support=roles,
        )
        pair_ids = []
        conditions = []
        scores = []
        for index, record in enumerate(iterator):
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"record at row {index} must be a mapping, "
                    f"got {type(record).__name__}."
                )
            for field_name in mapped_fields:
                if field_name not in record:
                    raise ValueError(
                        f"record at row {index} is missing mapped field "
                        f"{field_name!r}."
                    )

            raw_condition = record[condition_field]
            if raw_condition is None and (
                prepared_condition_map is None or (0, "") not in prepared_condition_map
            ):
                raise ValueError(
                    f"record at row {index} has a missing value for "
                    f"{condition_field!r}."
                )
            if prepared_condition_map is not None:
                _, lookup_key = _portable_raw_key(
                    raw_condition,
                    path=(f"record at row {index} value for {condition_field!r}"),
                )
                if lookup_key not in prepared_condition_map:
                    raise ValueError(
                        f"record at row {index} has unmapped value "
                        f"{raw_condition!r} for {condition_field!r}."
                    )
                condition = prepared_condition_map[lookup_key]
            else:
                condition = raw_condition

            pair_ids.append(record[pair_id_field])
            conditions.append(condition)
            scores.append(record[score_field])

        if not pair_ids:
            raise ValueError("records must contain at least one row.")
        return cls(
            axis=axis,
            pair_ids=pair_ids,
            conditions=conditions,
            scores=scores,
            condition_roles=roles,
            score_name=score_name,
            source=source,
            pairing_basis=pairing_basis,
            score_range=score_range,
            provenance=_paired_adapter_provenance(
                provenance,
                adapter="records",
                pair_id_field=pair_id_field,
                condition_field=condition_field,
                score_field=score_field,
                condition_map=serialized_condition_map,
            ),
        )

    @classmethod
    def from_dataframe(
        cls,
        frame: "pd.DataFrame",
        *,
        axis: str,
        pair_id_column: str,
        condition_column: str,
        score_column: str,
        condition_roles: Sequence[str],
        score_name: str,
        source: str,
        pairing_basis: str,
        condition_map: Optional[Mapping[Any, str]] = None,
        score_range: Optional[Sequence[float]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "PairedScores":
        """Adapt three explicit DataFrame columns without guessing schema."""
        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"frame must be a pandas DataFrame, got {type(frame).__name__}."
            )
        pair_id_column = require_nonempty_string(
            pair_id_column,
            "pair_id_column",
        )
        condition_column = require_nonempty_string(
            condition_column,
            "condition_column",
        )
        score_column = require_nonempty_string(score_column, "score_column")
        mapped_columns = (pair_id_column, condition_column, score_column)
        if len(set(mapped_columns)) != len(mapped_columns):
            raise ValueError(
                "pair_id_column, condition_column, and score_column must be "
                "distinct."
            )
        for role, column in (
            ("pair ID", pair_id_column),
            ("condition", condition_column),
            ("score", score_column),
        ):
            if column not in frame.columns:
                raise ValueError(f"frame is missing mapped {role} column {column!r}.")
            if not isinstance(frame[column], pd.Series):
                raise ValueError(
                    f"frame mapped {role} column {column!r} must be unique."
                )

        record_evidence = cls.from_records(
            (
                {
                    pair_id_column: pair_id,
                    condition_column: condition,
                    score_column: score,
                }
                for pair_id, condition, score in zip(
                    frame[pair_id_column].tolist(),
                    frame[condition_column].tolist(),
                    frame[score_column].tolist(),
                )
            ),
            axis=axis,
            pair_id_field=pair_id_column,
            condition_field=condition_column,
            score_field=score_column,
            condition_roles=condition_roles,
            score_name=score_name,
            source=source,
            pairing_basis=pairing_basis,
            condition_map=condition_map,
            score_range=score_range,
        )
        merged_provenance = _paired_adapter_provenance(
            provenance,
            adapter="dataframe",
            pair_id_field=pair_id_column,
            condition_field=condition_column,
            score_field=score_column,
            condition_map=record_evidence.provenance["condition_map"],
        )
        return cls(
            axis=record_evidence.axis,
            pair_ids=record_evidence.pair_ids,
            conditions=record_evidence.conditions,
            scores=record_evidence.scores,
            condition_roles=record_evidence.condition_roles,
            score_name=record_evidence.score_name,
            source=record_evidence.source,
            pairing_basis=record_evidence.pairing_basis,
            score_range=record_evidence.score_range,
            provenance=merged_provenance,
        )


def _normalize_text_sequence(value: Any, *, field_name: str) -> tuple[str, ...]:
    """Validate an ordered sequence of non-blank text units.

    Blank units are rejected, never dropped: a silently discarded row would
    make every downstream denominator wrong without leaving a trace.
    """
    texts = normalize_string_sequence(
        value,
        field_name=field_name,
        allow_empty=False,
    )
    singular = field_name.removesuffix("s")
    for index, item in enumerate(texts):
        if not item.strip():
            raise ValueError(f"{singular} at row {index} must not be blank.")
    return texts


def _validate_declared_groups(
    declared: Any,
    observed: Sequence[str],
    *,
    field_name: str = "declared_groups",
) -> tuple[str, ...]:
    """Validate an explicit group support against the observed row labels.

    Groups declared with zero observed rows are kept so that an empty declared
    group stays visible with a count of ``0`` instead of disappearing.
    """
    declared = _validate_support(declared, field_name=field_name)
    allowed = set(declared)
    for index, label in enumerate(observed):
        if label not in allowed:
            raise ValueError(
                f"group at row {index} ({label!r}) is outside the declared "
                f"{field_name} {list(declared)!r}."
            )
    return declared


def _canonicalize_role_rows(
    *,
    keys: Sequence[Any],
    key_field: str,
    roles: Sequence[str],
    role_field: str,
    payloads: Sequence[str],
    payload_field: str,
    declared_roles: Sequence[str],
    require_complete: bool,
    key_label: Optional[str] = None,
    role_label: Optional[str] = None,
) -> tuple[tuple[Any, ...], tuple[str, ...], tuple[str, ...]]:
    """Group role-tagged rows by key and re-emit them in a canonical order.

    Rows are re-emitted sorted by the portable sort key of *keys* with the
    roles of each key in declared order, so option or condition position never
    carries information. Malformed rows are refused, never dropped.
    """
    key_label = key_field if key_label is None else key_label
    role_label = role_field if role_label is None else role_label
    if len({len(keys), len(roles), len(payloads)}) != 1:
        raise ValueError(
            f"{key_field}s, {role_field}s, and {payload_field}s must contain the "
            f"same number of rows; got {len(keys)}, {len(roles)}, and "
            f"{len(payloads)}."
        )

    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    for index, (raw_key, role, payload) in enumerate(zip(keys, roles, payloads)):
        key, sort_key = _portable_raw_key(
            raw_key,
            path=f"{key_field} at row {index}",
        )
        if role not in declared_roles:
            raise ValueError(
                f"{role_label} at row {index} ({role!r}) is outside the declared "
                f"roles {list(declared_roles)!r}."
            )
        entry = grouped.setdefault(sort_key, {"key": key, "payloads": {}})
        if role in entry["payloads"]:
            raise ValueError(
                f"{key_label} {entry['key']!r} contains duplicate "
                f"{role_label} {role!r}."
            )
        entry["payloads"][role] = payload

    canonical_keys: list[Any] = []
    canonical_roles: list[str] = []
    canonical_payloads: list[str] = []
    for sort_key in sorted(grouped):
        entry = grouped[sort_key]
        if require_complete:
            missing = [
                role for role in declared_roles if role not in entry["payloads"]
            ]
            if missing:
                raise ValueError(
                    f"{key_label} {entry['key']!r} is missing "
                    f"{role_label}(s): {missing!r}."
                )
        for role in declared_roles:
            if role not in entry["payloads"]:
                continue
            canonical_keys.append(entry["key"])
            canonical_roles.append(role)
            canonical_payloads.append(entry["payloads"][role])
    return (
        tuple(canonical_keys),
        tuple(canonical_roles),
        tuple(canonical_payloads),
    )


def _record_iterator(records: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(records, (str, bytes, Mapping)):
        raise TypeError("records must be an iterable of mapping rows.")
    try:
        return iter(records)
    except TypeError as exc:
        raise TypeError("records must be an iterable of mapping rows.") from exc


def _require_record_fields(
    record: Any,
    *,
    index: int,
    fields: Sequence[str],
) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(
            f"record at row {index} must be a mapping, "
            f"got {type(record).__name__}."
        )
    for field_name in fields:
        if field_name not in record:
            raise ValueError(
                f"record at row {index} is missing mapped field {field_name!r}."
            )
    return record


def _resolve_mapped_label(
    raw: Any,
    *,
    prepared_value_map: Optional[Mapping[tuple[int, str], str]],
    index: int,
    field_name: str,
    support: Sequence[str],
    label_noun: str,
) -> str:
    """Map one raw cell onto an explicitly declared label without guessing."""
    if raw is None and (
        prepared_value_map is None or (0, "") not in prepared_value_map
    ):
        raise ValueError(
            f"record at row {index} has a missing value for {field_name!r}."
        )
    if prepared_value_map is not None:
        _, lookup_key = _portable_raw_key(
            raw,
            path=f"record at row {index} value for {field_name!r}",
        )
        if lookup_key not in prepared_value_map:
            raise ValueError(
                f"record at row {index} has unmapped value {raw!r} for "
                f"{field_name!r}."
            )
        label = prepared_value_map[lookup_key]
    else:
        label = raw
    if not isinstance(label, str) or not label.strip():
        raise ValueError(
            f"record at row {index} maps to a non-string or empty "
            f"{label_noun} {label!r}."
        )
    if label not in support:
        raise ValueError(
            f"record at row {index} maps to {label!r}, which is outside the "
            f"explicit support {list(support)!r}."
        )
    return label


def _resolve_record_text(raw: Any, *, index: int, field_name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            f"record at row {index} has a non-string or blank value for "
            f"{field_name!r}."
        )
    return raw


def _require_frame(frame: Any, *, columns: Sequence[tuple[str, str]]) -> None:
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"frame must be a pandas DataFrame, got {type(frame).__name__}."
        )
    for role, column in columns:
        if column not in frame.columns:
            raise ValueError(f"frame is missing mapped {role} column {column!r}.")
        if not isinstance(frame[column], pd.Series):
            raise ValueError(f"frame mapped {role} column {column!r} must be unique.")


def _require_count(raw: Any, *, name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, Integral):
        raise TypeError(f"{name} must be an integer, got {type(raw).__name__}.")
    return int(raw)


@dataclass(frozen=True, kw_only=True)
class LeakageExtractionRecord:
    """Proof that a declared text-to-count extraction actually ran.

    This record is what separates a valid zero-hit extraction from an
    unexplained all-zero count matrix, so every field is required.
    """

    extractor_id: str
    extractor_version: str
    config_digest: str
    text_digest: str
    text_unit_count: int
    token_count: int
    matched_group_positions: int
    matched_trait_positions: int
    event_count: int
    window: int
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "extractor_id",
            "extractor_version",
            "config_digest",
            "text_digest",
        ):
            object.__setattr__(
                self, name, require_nonempty_string(getattr(self, name), name)
            )

        text_unit_count = _require_count(self.text_unit_count, name="text_unit_count")
        if text_unit_count < 1:
            raise ValueError(
                "text_unit_count must be at least 1; an extraction that "
                "processed no text is not a valid extraction."
            )
        object.__setattr__(self, "text_unit_count", text_unit_count)

        for name in (
            "token_count",
            "matched_group_positions",
            "matched_trait_positions",
            "event_count",
        ):
            value = _require_count(getattr(self, name), name=name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)

        window = _require_count(self.window, name="window")
        if window < 1:
            raise ValueError("window must be at least 1.")
        object.__setattr__(self, "window", window)

        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "extractor_id": self.extractor_id,
            "extractor_version": self.extractor_version,
            "config_digest": self.config_digest,
            "text_digest": self.text_digest,
            "text_unit_count": self.text_unit_count,
            "token_count": self.token_count,
            "matched_group_positions": self.matched_group_positions,
            "matched_trait_positions": self.matched_trait_positions,
            "event_count": self.event_count,
            "window": self.window,
            "provenance": thaw_json(self.provenance),
        }


@dataclass(frozen=True, kw_only=True)
class AssociationCounts:
    """A complete group-term by trait-term co-occurrence count matrix.

    Every declared cell must be present. Nothing is zero-filled, inferred or
    completed here: an incomplete matrix is refused so that a partial count
    table can never be reported as a valid association statistic. An all-zero
    matrix is allowed at container level and resolved by the diagnostic, which
    requires either an extraction record or an explicit opt-in.
    """

    axis: str
    group_terms: Sequence[str]
    trait_terms: Sequence[str]
    counts: Mapping[str, Mapping[str, int]]
    source: str
    counting_basis: str
    extraction: Optional[LeakageExtractionRecord] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    total_events: int = field(init=False)
    pair_space_size: int = field(init=False)
    observed_cell_count: int = field(init=False)
    group_margins: Mapping[str, int] = field(init=False)
    trait_margins: Mapping[str, int] = field(init=False)
    lexicon_digest: str = field(init=False)
    matrix_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )
        object.__setattr__(
            self,
            "counting_basis",
            require_nonempty_string(self.counting_basis, "counting_basis"),
        )

        group_terms = _validate_support(self.group_terms, field_name="group_terms")
        trait_terms = _validate_support(self.trait_terms, field_name="trait_terms")
        overlap = set(group_terms).intersection(trait_terms)
        if overlap:
            raise ValueError(
                "group_terms and trait_terms must be disjoint; shared term(s): "
                f"{sorted(overlap)!r}."
            )

        if not isinstance(self.counts, Mapping):
            raise TypeError(
                "counts must be a mapping of group term -> trait term -> integer "
                f"count, got {type(self.counts).__name__}."
            )
        observed_rows = set(self.counts)
        missing_rows = set(group_terms) - observed_rows
        extra_rows = observed_rows - set(group_terms)
        if missing_rows or extra_rows:
            raise ValueError(
                "counts must contain exactly one row per group term; missing: "
                f"{_sorted_labels(missing_rows)!r}, unexpected: "
                f"{_sorted_labels(extra_rows)!r}."
            )

        counts: dict[str, Mapping[str, int]] = {}
        group_margins: dict[str, int] = {}
        trait_margins: dict[str, int] = {term: 0 for term in trait_terms}
        total_events = 0
        observed_cell_count = 0
        for group in group_terms:
            row = self.counts[group]
            if not isinstance(row, Mapping):
                raise TypeError(
                    f"counts[{group!r}] must be a mapping of trait term -> "
                    f"integer count, got {type(row).__name__}."
                )
            observed_cells = set(row)
            missing_cells = set(trait_terms) - observed_cells
            extra_cells = observed_cells - set(trait_terms)
            if missing_cells or extra_cells:
                raise ValueError(
                    f"counts[{group!r}] must contain exactly one cell per trait "
                    f"term; missing: {_sorted_labels(missing_cells)!r}, "
                    f"unexpected: {_sorted_labels(extra_cells)!r}."
                )
            cells: dict[str, int] = {}
            row_total = 0
            for trait in trait_terms:
                raw = row[trait]
                if isinstance(raw, bool) or not isinstance(raw, Integral):
                    raise TypeError(
                        f"count for ({group!r}, {trait!r}) must be an integer, "
                        f"got {type(raw).__name__}."
                    )
                count = int(raw)
                if count < 0:
                    raise ValueError(
                        f"count for ({group!r}, {trait!r}) must be non-negative."
                    )
                cells[trait] = count
                row_total += count
                trait_margins[trait] += count
                if count > 0:
                    observed_cell_count += 1
            counts[group] = MappingProxyType(dict(sorted(cells.items())))
            group_margins[group] = row_total
            total_events += row_total

        if self.extraction is not None:
            if not isinstance(self.extraction, LeakageExtractionRecord):
                raise TypeError(
                    "extraction must be a LeakageExtractionRecord or None, got "
                    f"{type(self.extraction).__name__}."
                )
            if self.extraction.event_count != total_events:
                raise ValueError(
                    f"extraction.event_count {self.extraction.event_count!r} "
                    f"does not match the matrix total {total_events!r}."
                )

        object.__setattr__(self, "group_terms", group_terms)
        object.__setattr__(self, "trait_terms", trait_terms)
        object.__setattr__(
            self, "counts", MappingProxyType(dict(sorted(counts.items())))
        )
        object.__setattr__(self, "total_events", total_events)
        object.__setattr__(
            self, "pair_space_size", len(group_terms) * len(trait_terms)
        )
        object.__setattr__(self, "observed_cell_count", observed_cell_count)
        object.__setattr__(
            self,
            "group_margins",
            MappingProxyType(dict(sorted(group_margins.items()))),
        )
        object.__setattr__(
            self,
            "trait_margins",
            MappingProxyType(dict(sorted(trait_margins.items()))),
        )
        lexicon_payload = {
            "group_terms": list(group_terms),
            "trait_terms": list(trait_terms),
        }
        object.__setattr__(self, "lexicon_digest", json_digest(lexicon_payload))
        object.__setattr__(
            self,
            "matrix_digest",
            json_digest(
                {
                    **lexicon_payload,
                    "counts": {
                        group: dict(row) for group, row in counts.items()
                    },
                }
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "axis": self.axis,
            "group_terms": list(self.group_terms),
            "trait_terms": list(self.trait_terms),
            "counts": {group: dict(row) for group, row in self.counts.items()},
            "source": self.source,
            "counting_basis": self.counting_basis,
            "extraction": (
                None if self.extraction is None else self.extraction.to_dict()
            ),
            "total_events": self.total_events,
            "pair_space_size": self.pair_space_size,
            "observed_cell_count": self.observed_cell_count,
            "group_margins": dict(self.group_margins),
            "trait_margins": dict(self.trait_margins),
            "lexicon_digest": self.lexicon_digest,
            "matrix_digest": self.matrix_digest,
            "provenance": thaw_json(self.provenance),
        }

    @classmethod
    def from_pair_counts(
        cls,
        pair_counts: Mapping[tuple[str, str], int],
        *,
        axis: str,
        group_terms: Sequence[str],
        trait_terms: Sequence[str],
        source: str,
        counting_basis: str,
        extraction: Optional[LeakageExtractionRecord] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "AssociationCounts":
        """Densify sparse pair counts over an explicitly declared pair space.

        This is the only sanctioned densification door: the caller declares the
        complete pair space, so filling unlisted cells with ``0`` is a declared
        choice rather than an inference.
        """
        if not isinstance(pair_counts, Mapping):
            raise TypeError(
                "pair_counts must be a mapping of (group_term, trait_term) -> "
                f"integer count, got {type(pair_counts).__name__}."
            )
        group_labels = _validate_support(group_terms, field_name="group_terms")
        trait_labels = _validate_support(trait_terms, field_name="trait_terms")
        dense: dict[str, dict[str, Any]] = {
            group: {trait: 0 for trait in trait_labels} for group in group_labels
        }
        for key, value in pair_counts.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise TypeError(
                    "pair_counts keys must be (group_term, trait_term) tuples, "
                    f"got {type(key).__name__}."
                )
            group, trait = key
            if group not in dense or trait not in trait_labels:
                raise ValueError(
                    f"pair_counts key {key!r} is outside the declared group x "
                    "trait pair space."
                )
            dense[group][trait] = value
        return cls(
            axis=axis,
            group_terms=group_labels,
            trait_terms=trait_labels,
            counts=dense,
            source=source,
            counting_basis=counting_basis,
            extraction=extraction,
            provenance={} if provenance is None else provenance,
        )

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        axis: str,
        group_field: str,
        trait_field: str,
        count_field: str,
        group_terms: Sequence[str],
        trait_terms: Sequence[str],
        source: str,
        counting_basis: str,
        group_value_map: Optional[Mapping[Any, str]] = None,
        trait_value_map: Optional[Mapping[Any, str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "AssociationCounts":
        """Adapt long-format count rows; every declared cell must appear once."""
        group_labels = _validate_support(group_terms, field_name="group_terms")
        trait_labels = _validate_support(trait_terms, field_name="trait_terms")
        group_field = require_nonempty_string(group_field, "group_field")
        trait_field = require_nonempty_string(trait_field, "trait_field")
        count_field = require_nonempty_string(count_field, "count_field")
        mapped_fields = (group_field, trait_field, count_field)
        if len(set(mapped_fields)) != len(mapped_fields):
            raise ValueError(
                "group_field, trait_field, and count_field must be distinct."
            )
        iterator = _record_iterator(records)
        prepared_group_map, serialized_group_map = _prepare_value_map(
            group_value_map,
            support=group_labels,
        )
        prepared_trait_map, serialized_trait_map = _prepare_value_map(
            trait_value_map,
            support=trait_labels,
        )

        cells: dict[tuple[str, str], int] = {}
        for index, record in enumerate(iterator):
            record = _require_record_fields(
                record, index=index, fields=mapped_fields
            )
            group = _resolve_mapped_label(
                record[group_field],
                prepared_value_map=prepared_group_map,
                index=index,
                field_name=group_field,
                support=group_labels,
                label_noun="group term",
            )
            trait = _resolve_mapped_label(
                record[trait_field],
                prepared_value_map=prepared_trait_map,
                index=index,
                field_name=trait_field,
                support=trait_labels,
                label_noun="trait term",
            )
            raw_count = record[count_field]
            if isinstance(raw_count, bool) or not isinstance(raw_count, Integral):
                raise TypeError(
                    f"count at row {index} must be an integer, "
                    f"got {type(raw_count).__name__}."
                )
            count = int(raw_count)
            if count < 0:
                raise ValueError(f"count at row {index} must be non-negative.")
            if (group, trait) in cells:
                raise ValueError(
                    f"record at row {index} duplicates cell "
                    f"({group!r}, {trait!r})."
                )
            cells[(group, trait)] = count

        missing = [
            (group, trait)
            for group in group_labels
            for trait in trait_labels
            if (group, trait) not in cells
        ]
        if missing:
            raise ValueError(
                f"records are missing count rows for cell(s): {sorted(missing)!r}."
            )

        return cls(
            axis=axis,
            group_terms=group_labels,
            trait_terms=trait_labels,
            counts={
                group: {trait: cells[(group, trait)] for trait in trait_labels}
                for group in group_labels
            },
            source=source,
            counting_basis=counting_basis,
            provenance=_adapter_provenance(
                provenance,
                adapter="AssociationCounts.from_records",
                group_field=group_field,
                trait_field=trait_field,
                count_field=count_field,
                value_map=serialized_group_map,
                trait_value_map=serialized_trait_map,
            ),
        )

    @classmethod
    def from_dataframe(
        cls,
        frame: "pd.DataFrame",
        *,
        axis: str,
        group_column: str,
        trait_column: str,
        count_column: str,
        group_terms: Sequence[str],
        trait_terms: Sequence[str],
        source: str,
        counting_basis: str,
        group_value_map: Optional[Mapping[Any, str]] = None,
        trait_value_map: Optional[Mapping[Any, str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "AssociationCounts":
        """Adapt three explicit DataFrame columns without guessing schema."""
        group_column = require_nonempty_string(group_column, "group_column")
        trait_column = require_nonempty_string(trait_column, "trait_column")
        count_column = require_nonempty_string(count_column, "count_column")
        mapped_columns = (group_column, trait_column, count_column)
        if len(set(mapped_columns)) != len(mapped_columns):
            raise ValueError(
                "group_column, trait_column, and count_column must be distinct."
            )
        _require_frame(
            frame,
            columns=(
                ("group", group_column),
                ("trait", trait_column),
                ("count", count_column),
            ),
        )
        record_evidence = cls.from_records(
            (
                {
                    group_column: group,
                    trait_column: trait,
                    count_column: count,
                }
                for group, trait, count in zip(
                    frame[group_column].tolist(),
                    frame[trait_column].tolist(),
                    frame[count_column].tolist(),
                )
            ),
            axis=axis,
            group_field=group_column,
            trait_field=trait_column,
            count_field=count_column,
            group_terms=group_terms,
            trait_terms=trait_terms,
            source=source,
            counting_basis=counting_basis,
            group_value_map=group_value_map,
            trait_value_map=trait_value_map,
        )
        merged_provenance = _adapter_provenance(
            provenance,
            adapter="AssociationCounts.from_dataframe",
            group_field=group_column,
            trait_field=trait_column,
            count_field=count_column,
            value_map=record_evidence.provenance["value_map"],
            trait_value_map=record_evidence.provenance["trait_value_map"],
        )
        return cls(
            axis=record_evidence.axis,
            group_terms=record_evidence.group_terms,
            trait_terms=record_evidence.trait_terms,
            counts=record_evidence.counts,
            source=record_evidence.source,
            counting_basis=record_evidence.counting_basis,
            provenance=merged_provenance,
        )


@dataclass(frozen=True, kw_only=True)
class TextEvidence:
    """Text units for one protected axis, before any extraction has run.

    The corpus itself is never serialized by ``to_dict``: reports stay portable
    and no benchmark text leaks into a diagnostic report.
    """

    axis: str
    texts: Sequence[str]
    source: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    total: int = field(init=False)
    text_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )
        texts = _normalize_text_sequence(self.texts, field_name="texts")
        object.__setattr__(self, "texts", texts)
        object.__setattr__(self, "total", len(texts))
        object.__setattr__(self, "text_digest", json_digest(list(texts)))
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation without the corpus itself."""
        return {
            "axis": self.axis,
            "source": self.source,
            "total": self.total,
            "text_digest": self.text_digest,
            "provenance": thaw_json(self.provenance),
        }

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        axis: str,
        text_field: str,
        source: str,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "TextEvidence":
        """Adapt one explicitly named text field without guessing schema."""
        text_field = require_nonempty_string(text_field, "text_field")
        iterator = _record_iterator(records)
        texts: list[str] = []
        for index, record in enumerate(iterator):
            record = _require_record_fields(
                record, index=index, fields=(text_field,)
            )
            texts.append(
                _resolve_record_text(
                    record[text_field], index=index, field_name=text_field
                )
            )
        if not texts:
            raise ValueError("records must contain at least one row.")
        return cls(
            axis=axis,
            texts=texts,
            source=source,
            provenance=_mapping_adapter_provenance(
                provenance,
                adapter="TextEvidence.from_records",
                field_mapping={"text": text_field},
            ),
        )

    @classmethod
    def from_dataframe(
        cls,
        frame: "pd.DataFrame",
        *,
        axis: str,
        text_column: str,
        source: str,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "TextEvidence":
        """Adapt one explicit DataFrame column without guessing schema."""
        text_column = require_nonempty_string(text_column, "text_column")
        _require_frame(frame, columns=(("text", text_column),))
        record_evidence = cls.from_records(
            ({text_column: value} for value in frame[text_column].tolist()),
            axis=axis,
            text_field=text_column,
            source=source,
        )
        return cls(
            axis=record_evidence.axis,
            texts=record_evidence.texts,
            source=record_evidence.source,
            provenance=_mapping_adapter_provenance(
                provenance,
                adapter="TextEvidence.from_dataframe",
                field_mapping={"text": text_column},
            ),
        )


@dataclass(frozen=True, kw_only=True)
class GroupedTexts:
    """Text units each carrying one explicitly declared group label.

    ``declared_groups`` is required: a declared group with no text must stay
    visible with a count of ``0`` so that a normalized component can block
    explicitly instead of silently losing the group.
    """

    axis: str
    groups: Sequence[str]
    texts: Sequence[str]
    declared_groups: Sequence[str]
    source: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    support: tuple[str, ...] = field(init=False)
    observed_support: tuple[str, ...] = field(init=False)
    group_sample_counts: Mapping[str, int] = field(init=False)
    total: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )
        groups = normalize_string_sequence(
            self.groups,
            field_name="groups",
            allow_empty=False,
        )
        texts = _normalize_text_sequence(self.texts, field_name="texts")
        if len(groups) != len(texts):
            raise ValueError(
                "groups and texts must contain the same number of rows; "
                f"got {len(groups)} groups and {len(texts)} texts."
            )
        declared_groups = _validate_declared_groups(self.declared_groups, groups)

        counts = {label: 0 for label in declared_groups}
        for label in groups:
            counts[label] += 1

        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "texts", texts)
        object.__setattr__(self, "declared_groups", declared_groups)
        object.__setattr__(self, "support", declared_groups)
        object.__setattr__(self, "observed_support", tuple(sorted(set(groups))))
        object.__setattr__(
            self,
            "group_sample_counts",
            MappingProxyType(dict(sorted(counts.items()))),
        )
        object.__setattr__(self, "total", len(texts))
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "axis": self.axis,
            "groups": list(self.groups),
            "texts": list(self.texts),
            "declared_groups": list(self.declared_groups),
            "source": self.source,
            "support": list(self.support),
            "observed_support": list(self.observed_support),
            "group_sample_counts": dict(self.group_sample_counts),
            "total": self.total,
            "provenance": thaw_json(self.provenance),
        }

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        axis: str,
        group_field: str,
        text_field: str,
        declared_groups: Sequence[str],
        source: str,
        value_map: Optional[Mapping[Any, str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "GroupedTexts":
        """Adapt explicit group and text fields without dropping rows."""
        group_labels = _validate_support(
            declared_groups, field_name="declared_groups"
        )
        group_field = require_nonempty_string(group_field, "group_field")
        text_field = require_nonempty_string(text_field, "text_field")
        if group_field == text_field:
            raise ValueError("group_field and text_field must name distinct fields.")
        iterator = _record_iterator(records)
        prepared_value_map, serialized_value_map = _prepare_value_map(
            value_map,
            support=group_labels,
        )

        groups: list[str] = []
        texts: list[str] = []
        for index, record in enumerate(iterator):
            record = _require_record_fields(
                record, index=index, fields=(group_field, text_field)
            )
            groups.append(
                _resolve_mapped_label(
                    record[group_field],
                    prepared_value_map=prepared_value_map,
                    index=index,
                    field_name=group_field,
                    support=group_labels,
                    label_noun="group",
                )
            )
            texts.append(
                _resolve_record_text(
                    record[text_field], index=index, field_name=text_field
                )
            )
        if not groups:
            raise ValueError("records must contain at least one row.")

        return cls(
            axis=axis,
            groups=groups,
            texts=texts,
            declared_groups=group_labels,
            source=source,
            provenance=_mapping_adapter_provenance(
                provenance,
                adapter="GroupedTexts.from_records",
                field_mapping={"group": group_field, "text": text_field},
                value_map=serialized_value_map,
            ),
        )

    @classmethod
    def from_dataframe(
        cls,
        frame: "pd.DataFrame",
        *,
        axis: str,
        group_column: str,
        text_column: str,
        declared_groups: Sequence[str],
        source: str,
        value_map: Optional[Mapping[Any, str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "GroupedTexts":
        """Adapt two explicit DataFrame columns without guessing schema."""
        group_column = require_nonempty_string(group_column, "group_column")
        text_column = require_nonempty_string(text_column, "text_column")
        if group_column == text_column:
            raise ValueError("group_column and text_column must be distinct.")
        _require_frame(
            frame,
            columns=(("group", group_column), ("text", text_column)),
        )
        record_evidence = cls.from_records(
            (
                {group_column: group, text_column: text}
                for group, text in zip(
                    frame[group_column].tolist(),
                    frame[text_column].tolist(),
                )
            ),
            axis=axis,
            group_field=group_column,
            text_field=text_column,
            declared_groups=declared_groups,
            source=source,
            value_map=value_map,
        )
        return cls(
            axis=record_evidence.axis,
            groups=record_evidence.groups,
            texts=record_evidence.texts,
            declared_groups=record_evidence.declared_groups,
            source=record_evidence.source,
            provenance=_mapping_adapter_provenance(
                provenance,
                adapter="GroupedTexts.from_dataframe",
                field_mapping={"group": group_column, "text": text_column},
                value_map=record_evidence.provenance["value_map"],
            ),
        )


@dataclass(frozen=True, kw_only=True)
class PairedTexts:
    """Complete two-condition aligned text pairs for a minimal-pair audit.

    A pair whose two sides are byte-identical is refused: no intervention was
    applied, so it is not a minimal pair. Malformed pairs are refused rather
    than dropped.
    """

    axis: str
    pair_ids: Sequence[Any]
    conditions: Sequence[str]
    texts: Sequence[str]
    condition_roles: Sequence[str]
    pairing_basis: str
    source: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    total: int = field(init=False)
    pair_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )
        object.__setattr__(
            self,
            "pairing_basis",
            require_nonempty_string(self.pairing_basis, "pairing_basis"),
        )

        roles = _normalize_condition_roles(self.condition_roles)
        conditions = normalize_string_sequence(
            self.conditions,
            field_name="conditions",
            allow_empty=False,
        )
        texts = _normalize_text_sequence(self.texts, field_name="texts")
        pair_ids, _ = _normalize_pair_ids(self.pair_ids)
        if len({len(pair_ids), len(conditions), len(texts)}) != 1:
            raise ValueError(
                "pair_ids, conditions, and texts must contain the same number "
                f"of rows; got {len(pair_ids)}, {len(conditions)}, and "
                f"{len(texts)}."
            )

        canonical_ids, canonical_conditions, canonical_texts = (
            _canonicalize_role_rows(
                keys=pair_ids,
                key_field="pair_id",
                roles=conditions,
                role_field="condition",
                payloads=texts,
                payload_field="text",
                declared_roles=roles,
                require_complete=True,
                key_label="pair",
                role_label="condition role",
            )
        )
        for start in range(0, len(canonical_texts), 2):
            if canonical_texts[start] == canonical_texts[start + 1]:
                raise ValueError(
                    f"pair {canonical_ids[start]!r} has identical text for both "
                    "condition roles; no intervention was applied."
                )

        object.__setattr__(self, "pair_ids", canonical_ids)
        object.__setattr__(self, "conditions", canonical_conditions)
        object.__setattr__(self, "texts", canonical_texts)
        object.__setattr__(self, "condition_roles", roles)
        object.__setattr__(self, "total", len(canonical_texts))
        object.__setattr__(self, "pair_count", len(canonical_texts) // 2)
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "axis": self.axis,
            "pair_ids": list(self.pair_ids),
            "conditions": list(self.conditions),
            "texts": list(self.texts),
            "condition_roles": list(self.condition_roles),
            "pairing_basis": self.pairing_basis,
            "source": self.source,
            "pair_count": self.pair_count,
            "provenance": thaw_json(self.provenance),
        }

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        axis: str,
        pair_id_field: str,
        condition_field: str,
        text_field: str,
        condition_roles: Sequence[str],
        pairing_basis: str,
        source: str,
        condition_map: Optional[Mapping[Any, str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "PairedTexts":
        """Adapt explicit pair/condition/text fields without dropping rows."""
        roles = _normalize_condition_roles(condition_roles)
        pair_id_field = require_nonempty_string(pair_id_field, "pair_id_field")
        condition_field = require_nonempty_string(condition_field, "condition_field")
        text_field = require_nonempty_string(text_field, "text_field")
        mapped_fields = (pair_id_field, condition_field, text_field)
        if len(set(mapped_fields)) != len(mapped_fields):
            raise ValueError(
                "pair_id_field, condition_field, and text_field must be distinct."
            )
        iterator = _record_iterator(records)
        prepared_condition_map, serialized_condition_map = _prepare_value_map(
            condition_map,
            support=roles,
        )

        pair_ids: list[Any] = []
        conditions: list[Any] = []
        texts: list[str] = []
        for index, record in enumerate(iterator):
            record = _require_record_fields(
                record, index=index, fields=mapped_fields
            )
            raw_condition = record[condition_field]
            if raw_condition is None and (
                prepared_condition_map is None
                or (0, "") not in prepared_condition_map
            ):
                raise ValueError(
                    f"record at row {index} has a missing value for "
                    f"{condition_field!r}."
                )
            if prepared_condition_map is not None:
                _, lookup_key = _portable_raw_key(
                    raw_condition,
                    path=f"record at row {index} value for {condition_field!r}",
                )
                if lookup_key not in prepared_condition_map:
                    raise ValueError(
                        f"record at row {index} has unmapped value "
                        f"{raw_condition!r} for {condition_field!r}."
                    )
                condition = prepared_condition_map[lookup_key]
            else:
                condition = raw_condition
            pair_ids.append(record[pair_id_field])
            conditions.append(condition)
            texts.append(
                _resolve_record_text(
                    record[text_field], index=index, field_name=text_field
                )
            )
        if not pair_ids:
            raise ValueError("records must contain at least one row.")

        return cls(
            axis=axis,
            pair_ids=pair_ids,
            conditions=conditions,
            texts=texts,
            condition_roles=roles,
            pairing_basis=pairing_basis,
            source=source,
            provenance=_paired_adapter_provenance(
                provenance,
                adapter="PairedTexts.from_records",
                pair_id_field=pair_id_field,
                condition_field=condition_field,
                condition_map=serialized_condition_map,
                value_field=text_field,
                value_role="text",
            ),
        )

    @classmethod
    def from_dataframe(
        cls,
        frame: "pd.DataFrame",
        *,
        axis: str,
        pair_id_column: str,
        condition_column: str,
        text_column: str,
        condition_roles: Sequence[str],
        pairing_basis: str,
        source: str,
        condition_map: Optional[Mapping[Any, str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "PairedTexts":
        """Adapt three explicit DataFrame columns without guessing schema."""
        pair_id_column = require_nonempty_string(pair_id_column, "pair_id_column")
        condition_column = require_nonempty_string(
            condition_column, "condition_column"
        )
        text_column = require_nonempty_string(text_column, "text_column")
        mapped_columns = (pair_id_column, condition_column, text_column)
        if len(set(mapped_columns)) != len(mapped_columns):
            raise ValueError(
                "pair_id_column, condition_column, and text_column must be "
                "distinct."
            )
        _require_frame(
            frame,
            columns=(
                ("pair ID", pair_id_column),
                ("condition", condition_column),
                ("text", text_column),
            ),
        )
        record_evidence = cls.from_records(
            (
                {
                    pair_id_column: pair_id,
                    condition_column: condition,
                    text_column: text,
                }
                for pair_id, condition, text in zip(
                    frame[pair_id_column].tolist(),
                    frame[condition_column].tolist(),
                    frame[text_column].tolist(),
                )
            ),
            axis=axis,
            pair_id_field=pair_id_column,
            condition_field=condition_column,
            text_field=text_column,
            condition_roles=condition_roles,
            pairing_basis=pairing_basis,
            source=source,
            condition_map=condition_map,
        )
        return cls(
            axis=record_evidence.axis,
            pair_ids=record_evidence.pair_ids,
            conditions=record_evidence.conditions,
            texts=record_evidence.texts,
            condition_roles=record_evidence.condition_roles,
            pairing_basis=record_evidence.pairing_basis,
            source=record_evidence.source,
            provenance=_paired_adapter_provenance(
                provenance,
                adapter="PairedTexts.from_dataframe",
                pair_id_field=pair_id_column,
                condition_field=condition_column,
                condition_map=record_evidence.provenance["condition_map"],
                value_field=text_column,
                value_role="text",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class OptionItems:
    """Multiple-choice options carrying explicitly declared, non-positional roles.

    The layout is one row per option, so option position carries no information
    at all. There is no positional constructor and no ``option_0``/``option_1``
    fallback: a stereotype or anti-stereotype role is only ever what the caller
    declared it to be.
    """

    axis: str
    item_ids: Sequence[Any]
    roles: Sequence[str]
    options: Sequence[str]
    declared_roles: Sequence[str]
    question_family: str
    source: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    item_count: int = field(init=False)
    total: int = field(init=False)
    role_counts: Mapping[str, int] = field(init=False)
    roles_by_item: Mapping[str, tuple[str, ...]] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )
        object.__setattr__(
            self,
            "question_family",
            require_nonempty_string(self.question_family, "question_family"),
        )

        declared_roles = normalize_string_sequence(
            self.declared_roles,
            field_name="declared_roles",
            allow_empty=False,
        )
        if len(declared_roles) < 2:
            raise ValueError("declared_roles must contain at least two roles.")
        if len(set(declared_roles)) != len(declared_roles):
            raise ValueError("declared_roles must not contain duplicate roles.")

        item_ids, _ = _normalize_pair_ids(
            self.item_ids,
            field_name="item_id",
            plural_field_name="item_ids",
        )
        roles = normalize_string_sequence(
            self.roles,
            field_name="roles",
            allow_empty=False,
        )
        options = _normalize_text_sequence(self.options, field_name="options")
        if len({len(item_ids), len(roles), len(options)}) != 1:
            raise ValueError(
                "item_ids, roles, and options must contain the same number of "
                f"rows; got {len(item_ids)}, {len(roles)}, and {len(options)}."
            )

        canonical_ids, canonical_roles, canonical_options = _canonicalize_role_rows(
            keys=item_ids,
            key_field="item_id",
            roles=roles,
            role_field="role",
            payloads=options,
            payload_field="option",
            declared_roles=declared_roles,
            require_complete=False,
            key_label="item",
        )

        role_counts = {role: 0 for role in declared_roles}
        roles_by_item: dict[str, tuple[str, ...]] = {}
        seen_items: dict[str, Any] = {}
        for item_id, role in zip(canonical_ids, canonical_roles):
            role_counts[role] += 1
            key = str(item_id)
            if key in seen_items and seen_items[key] != item_id:
                raise ValueError(
                    f"item_ids {seen_items[key]!r} and {item_id!r} collide under "
                    "the JSON-safe string key used by roles_by_item."
                )
            seen_items[key] = item_id
            roles_by_item[key] = roles_by_item.get(key, ()) + (role,)

        object.__setattr__(self, "item_ids", canonical_ids)
        object.__setattr__(self, "roles", canonical_roles)
        object.__setattr__(self, "options", canonical_options)
        object.__setattr__(self, "declared_roles", declared_roles)
        object.__setattr__(self, "item_count", len(roles_by_item))
        object.__setattr__(self, "total", len(canonical_options))
        object.__setattr__(
            self,
            "role_counts",
            MappingProxyType(dict(sorted(role_counts.items()))),
        )
        object.__setattr__(
            self,
            "roles_by_item",
            MappingProxyType(dict(sorted(roles_by_item.items()))),
        )
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "axis": self.axis,
            "item_ids": list(self.item_ids),
            "roles": list(self.roles),
            "options": list(self.options),
            "declared_roles": list(self.declared_roles),
            "question_family": self.question_family,
            "source": self.source,
            "item_count": self.item_count,
            "total": self.total,
            "role_counts": dict(self.role_counts),
            "provenance": thaw_json(self.provenance),
        }

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        axis: str,
        item_id_field: str,
        role_field: str,
        option_field: str,
        declared_roles: Sequence[str],
        question_family: str,
        source: str,
        role_map: Optional[Mapping[Any, str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "OptionItems":
        """Adapt explicit item/role/option fields; roles are never positional."""
        role_labels = normalize_string_sequence(
            declared_roles,
            field_name="declared_roles",
            allow_empty=False,
        )
        item_id_field = require_nonempty_string(item_id_field, "item_id_field")
        role_field = require_nonempty_string(role_field, "role_field")
        option_field = require_nonempty_string(option_field, "option_field")
        mapped_fields = (item_id_field, role_field, option_field)
        if len(set(mapped_fields)) != len(mapped_fields):
            raise ValueError(
                "item_id_field, role_field, and option_field must be distinct."
            )
        iterator = _record_iterator(records)
        prepared_role_map, serialized_role_map = _prepare_value_map(
            role_map,
            support=role_labels,
        )

        item_ids: list[Any] = []
        roles: list[Any] = []
        options: list[str] = []
        for index, record in enumerate(iterator):
            record = _require_record_fields(
                record, index=index, fields=mapped_fields
            )
            raw_role = record[role_field]
            if raw_role is None and (
                prepared_role_map is None or (0, "") not in prepared_role_map
            ):
                raise ValueError(
                    f"record at row {index} has a missing value for "
                    f"{role_field!r}."
                )
            if prepared_role_map is not None:
                _, lookup_key = _portable_raw_key(
                    raw_role,
                    path=f"record at row {index} value for {role_field!r}",
                )
                if lookup_key not in prepared_role_map:
                    raise ValueError(
                        f"record at row {index} has unmapped value {raw_role!r} "
                        f"for {role_field!r}."
                    )
                role = prepared_role_map[lookup_key]
            else:
                role = raw_role
            item_ids.append(record[item_id_field])
            roles.append(role)
            options.append(
                _resolve_record_text(
                    record[option_field], index=index, field_name=option_field
                )
            )
        if not item_ids:
            raise ValueError("records must contain at least one row.")

        return cls(
            axis=axis,
            item_ids=item_ids,
            roles=roles,
            options=options,
            declared_roles=role_labels,
            question_family=question_family,
            source=source,
            provenance=_mapping_adapter_provenance(
                provenance,
                adapter="OptionItems.from_records",
                field_mapping={
                    "item_id": item_id_field,
                    "role": role_field,
                    "option": option_field,
                },
                value_map=serialized_role_map,
                value_map_key="role_map",
            ),
        )

    @classmethod
    def from_dataframe(
        cls,
        frame: "pd.DataFrame",
        *,
        axis: str,
        item_id_column: str,
        role_column: str,
        option_column: str,
        declared_roles: Sequence[str],
        question_family: str,
        source: str,
        role_map: Optional[Mapping[Any, str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "OptionItems":
        """Adapt three explicit DataFrame columns without guessing schema."""
        item_id_column = require_nonempty_string(item_id_column, "item_id_column")
        role_column = require_nonempty_string(role_column, "role_column")
        option_column = require_nonempty_string(option_column, "option_column")
        mapped_columns = (item_id_column, role_column, option_column)
        if len(set(mapped_columns)) != len(mapped_columns):
            raise ValueError(
                "item_id_column, role_column, and option_column must be distinct."
            )
        _require_frame(
            frame,
            columns=(
                ("item ID", item_id_column),
                ("role", role_column),
                ("option", option_column),
            ),
        )
        record_evidence = cls.from_records(
            (
                {
                    item_id_column: item_id,
                    role_column: role,
                    option_column: option,
                }
                for item_id, role, option in zip(
                    frame[item_id_column].tolist(),
                    frame[role_column].tolist(),
                    frame[option_column].tolist(),
                )
            ),
            axis=axis,
            item_id_field=item_id_column,
            role_field=role_column,
            option_field=option_column,
            declared_roles=declared_roles,
            question_family=question_family,
            source=source,
            role_map=role_map,
        )
        return cls(
            axis=record_evidence.axis,
            item_ids=record_evidence.item_ids,
            roles=record_evidence.roles,
            options=record_evidence.options,
            declared_roles=record_evidence.declared_roles,
            question_family=record_evidence.question_family,
            source=record_evidence.source,
            provenance=_mapping_adapter_provenance(
                provenance,
                adapter="OptionItems.from_dataframe",
                field_mapping={
                    "item_id": item_id_column,
                    "role": role_column,
                    "option": option_column,
                },
                value_map=record_evidence.provenance["role_map"],
                value_map_key="role_map",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class TemplateGroups:
    """Template identities assigned to explicitly declared groups.

    ``template_identity_rule`` states what makes two rows the same template, so
    a template count is never a bare number of unexplained provenance.
    """

    axis: str
    groups: Sequence[str]
    template_ids: Sequence[Any]
    declared_groups: Sequence[str]
    template_identity_rule: str
    source: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    support: tuple[str, ...] = field(init=False)
    observed_support: tuple[str, ...] = field(init=False)
    group_instance_counts: Mapping[str, int] = field(init=False)
    group_unique_template_counts: Mapping[str, int] = field(init=False)
    total: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", require_nonempty_string(self.axis, "axis"))
        object.__setattr__(
            self, "source", require_nonempty_string(self.source, "source")
        )
        object.__setattr__(
            self,
            "template_identity_rule",
            require_nonempty_string(
                self.template_identity_rule, "template_identity_rule"
            ),
        )

        groups = normalize_string_sequence(
            self.groups,
            field_name="groups",
            allow_empty=False,
        )
        template_ids, template_keys = _normalize_pair_ids(
            self.template_ids,
            field_name="template_id",
            plural_field_name="template_ids",
        )
        if len(groups) != len(template_ids):
            raise ValueError(
                "groups and template_ids must contain the same number of rows; "
                f"got {len(groups)} groups and {len(template_ids)} template_ids."
            )
        declared_groups = _validate_declared_groups(self.declared_groups, groups)

        instance_counts = {label: 0 for label in declared_groups}
        unique_keys: dict[str, set[tuple[int, str]]] = {
            label: set() for label in declared_groups
        }
        for label, template_key in zip(groups, template_keys):
            instance_counts[label] += 1
            unique_keys[label].add(template_key)

        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "template_ids", template_ids)
        object.__setattr__(self, "declared_groups", declared_groups)
        object.__setattr__(self, "support", declared_groups)
        object.__setattr__(self, "observed_support", tuple(sorted(set(groups))))
        object.__setattr__(
            self,
            "group_instance_counts",
            MappingProxyType(dict(sorted(instance_counts.items()))),
        )
        object.__setattr__(
            self,
            "group_unique_template_counts",
            MappingProxyType(
                dict(sorted((label, len(keys)) for label, keys in unique_keys.items()))
            ),
        )
        object.__setattr__(self, "total", len(template_ids))
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        return {
            "axis": self.axis,
            "groups": list(self.groups),
            "template_ids": list(self.template_ids),
            "declared_groups": list(self.declared_groups),
            "template_identity_rule": self.template_identity_rule,
            "source": self.source,
            "support": list(self.support),
            "observed_support": list(self.observed_support),
            "group_instance_counts": dict(self.group_instance_counts),
            "group_unique_template_counts": dict(self.group_unique_template_counts),
            "total": self.total,
            "provenance": thaw_json(self.provenance),
        }

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        axis: str,
        group_field: str,
        template_id_field: str,
        declared_groups: Sequence[str],
        template_identity_rule: str,
        source: str,
        value_map: Optional[Mapping[Any, str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "TemplateGroups":
        """Adapt explicit group and template-identity fields without guessing."""
        group_labels = _validate_support(
            declared_groups, field_name="declared_groups"
        )
        group_field = require_nonempty_string(group_field, "group_field")
        template_id_field = require_nonempty_string(
            template_id_field, "template_id_field"
        )
        if group_field == template_id_field:
            raise ValueError(
                "group_field and template_id_field must name distinct fields."
            )
        iterator = _record_iterator(records)
        prepared_value_map, serialized_value_map = _prepare_value_map(
            value_map,
            support=group_labels,
        )

        groups: list[str] = []
        template_ids: list[Any] = []
        for index, record in enumerate(iterator):
            record = _require_record_fields(
                record, index=index, fields=(group_field, template_id_field)
            )
            groups.append(
                _resolve_mapped_label(
                    record[group_field],
                    prepared_value_map=prepared_value_map,
                    index=index,
                    field_name=group_field,
                    support=group_labels,
                    label_noun="group",
                )
            )
            template_ids.append(record[template_id_field])
        if not groups:
            raise ValueError("records must contain at least one row.")

        return cls(
            axis=axis,
            groups=groups,
            template_ids=template_ids,
            declared_groups=group_labels,
            template_identity_rule=template_identity_rule,
            source=source,
            provenance=_mapping_adapter_provenance(
                provenance,
                adapter="TemplateGroups.from_records",
                field_mapping={
                    "group": group_field,
                    "template_id": template_id_field,
                },
                value_map=serialized_value_map,
            ),
        )

    @classmethod
    def from_dataframe(
        cls,
        frame: "pd.DataFrame",
        *,
        axis: str,
        group_column: str,
        template_id_column: str,
        declared_groups: Sequence[str],
        template_identity_rule: str,
        source: str,
        value_map: Optional[Mapping[Any, str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "TemplateGroups":
        """Adapt two explicit DataFrame columns without guessing schema."""
        group_column = require_nonempty_string(group_column, "group_column")
        template_id_column = require_nonempty_string(
            template_id_column, "template_id_column"
        )
        if group_column == template_id_column:
            raise ValueError("group_column and template_id_column must be distinct.")
        _require_frame(
            frame,
            columns=(
                ("group", group_column),
                ("template ID", template_id_column),
            ),
        )
        record_evidence = cls.from_records(
            (
                {group_column: group, template_id_column: template_id}
                for group, template_id in zip(
                    frame[group_column].tolist(),
                    frame[template_id_column].tolist(),
                )
            ),
            axis=axis,
            group_field=group_column,
            template_id_field=template_id_column,
            declared_groups=declared_groups,
            template_identity_rule=template_identity_rule,
            source=source,
            value_map=value_map,
        )
        return cls(
            axis=record_evidence.axis,
            groups=record_evidence.groups,
            template_ids=record_evidence.template_ids,
            declared_groups=record_evidence.declared_groups,
            template_identity_rule=record_evidence.template_identity_rule,
            source=record_evidence.source,
            provenance=_mapping_adapter_provenance(
                provenance,
                adapter="TemplateGroups.from_dataframe",
                field_mapping={
                    "group": group_column,
                    "template_id": template_id_column,
                },
                value_map=record_evidence.provenance["value_map"],
            ),
        )


@dataclass(frozen=True, kw_only=True)
class DatasetEvidence:
    """A collection of typed, axis-keyed evidence views for one target.

    This is a container, not a facade: it re-models nothing, derives nothing,
    computes nothing, and never converts one view into another. Each view is
    built by its own container's adapter and passed in explicitly.
    """

    target_name: str
    representation: Mapping[str, RepresentationEvidence] = field(default_factory=dict)
    association_counts: Mapping[str, AssociationCounts] = field(default_factory=dict)
    texts: Mapping[str, TextEvidence] = field(default_factory=dict)
    grouped_texts: Mapping[str, GroupedTexts] = field(default_factory=dict)
    paired_texts: Mapping[str, PairedTexts] = field(default_factory=dict)
    option_items: Mapping[str, OptionItems] = field(default_factory=dict)
    template_groups: Mapping[str, TemplateGroups] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    axes: tuple[str, ...] = field(init=False)
    available_views: tuple[str, ...] = field(init=False)

    VIEW_NAMES: ClassVar[tuple[str, ...]] = (
        "association_counts",
        "grouped_texts",
        "option_items",
        "paired_texts",
        "representation",
        "template_groups",
        "texts",
    )

    _VIEW_TYPES: ClassVar[Mapping[str, type]] = MappingProxyType(
        {
            "association_counts": AssociationCounts,
            "grouped_texts": GroupedTexts,
            "option_items": OptionItems,
            "paired_texts": PairedTexts,
            "representation": RepresentationEvidence,
            "template_groups": TemplateGroups,
            "texts": TextEvidence,
        }
    )

    def __post_init__(self) -> None:
        axes: set[str] = set()
        available: list[str] = []
        for view in self.VIEW_NAMES:
            expected = self._VIEW_TYPES[view]
            supplied = getattr(self, view)
            if not isinstance(supplied, Mapping):
                raise TypeError(
                    f"{view} must be a mapping of axis -> {expected.__name__}."
                )
            validated: dict[str, Any] = {}
            for axis, value in supplied.items():
                require_nonempty_string(axis, f"{view} axis")
                if isinstance(value, (ScoredGroups, PairedScores)):
                    raise TypeError(
                        f"{view}[{axis!r}] holds row-level scorer results; audit "
                        "them with audit_scores(...), not audit_dataset(...)."
                    )
                if not isinstance(value, expected):
                    raise TypeError(
                        f"{view}[{axis!r}] must be a {expected.__name__}, "
                        f"got {type(value).__name__}."
                    )
                if value.axis != axis:
                    raise ValueError(
                        f"{view} mapping key {axis!r} does not match "
                        f"evidence.axis {value.axis!r}."
                    )
                validated[axis] = value
            object.__setattr__(
                self, view, MappingProxyType(dict(sorted(validated.items())))
            )
            if validated:
                available.append(view)
                axes.update(validated)

        if not available:
            raise ValueError("DatasetEvidence must contain at least one evidence view.")

        object.__setattr__(
            self,
            "target_name",
            require_nonempty_string(self.target_name, "target_name"),
        )
        object.__setattr__(self, "axes", tuple(sorted(axes)))
        object.__setattr__(self, "available_views", tuple(sorted(available)))
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    def views_for(self, axis: str) -> Mapping[str, Any]:
        """Return the view name -> evidence object mapping for one axis."""
        axis = require_nonempty_string(axis, "axis")
        selected = {
            view: getattr(self, view)[axis]
            for view in self.VIEW_NAMES
            if axis in getattr(self, view)
        }
        return MappingProxyType(dict(sorted(selected.items())))

    def to_dict(self) -> dict[str, Any]:
        """Return a new JSON-safe representation."""
        payload: dict[str, Any] = {"target_name": self.target_name}
        for view in self.VIEW_NAMES:
            payload[view] = {
                axis: value.to_dict() for axis, value in getattr(self, view).items()
            }
        payload["axes"] = list(self.axes)
        payload["available_views"] = list(self.available_views)
        payload["provenance"] = thaw_json(self.provenance)
        return payload


__all__ = [
    "AssociationCounts",
    "DatasetEvidence",
    "GroupedTexts",
    "LabeledScoredGroups",
    "LeakageExtractionRecord",
    "OptionItems",
    "PairedScores",
    "PairedTexts",
    "RepresentationEvidence",
    "ScoredGroups",
    "TemplateGroups",
    "TextEvidence",
]
