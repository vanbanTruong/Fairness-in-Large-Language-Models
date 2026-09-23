"""Internal validation and JSON-safe immutability helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping, Type, TypeVar

_EnumT = TypeVar("_EnumT", bound=Enum)


def require_nonempty_string(value: Any, field_name: str) -> str:
    """Return *value* after validating that it is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def normalize_enum(value: Any, enum_type: Type[_EnumT], field_name: str) -> _EnumT:
    """Normalize a string-valued enum while preserving actionable errors."""
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        available = ", ".join(repr(item.value) for item in enum_type)
        raise ValueError(
            f"{field_name} must be one of: {available}; got {value!r}."
        ) from exc


def normalize_string_sequence(
    value: Any,
    *,
    field_name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    """Validate an ordered string sequence and return an immutable snapshot.

    Sets, mappings, generators, and other merely iterable objects are rejected:
    accepting them would make serialized array order unstable or surprising.
    """

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be an ordered sequence of strings.")
    items = tuple(value)
    if not allow_empty and not items:
        raise ValueError(f"{field_name} must not be empty.")
    singular = field_name.removesuffix("s")
    for item in items:
        require_nonempty_string(item, singular)
    return items


def freeze_json(value: Any, *, path: str = "value") -> Any:
    """Defensively copy *value* into immutable, strictly JSON-safe objects.

    Unknown objects are rejected instead of being stringified. This prevents
    paths, exceptions, model objects, or other process-local state from leaking
    into a supposedly portable diagnostic report.
    """
    if isinstance(value, Enum):
        return freeze_json(value.value, path=path)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return number
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{path} must use string mapping keys for JSON; "
                    f"got {type(key).__name__} ({key!r})."
                )
            frozen[key] = freeze_json(item, path=f"{path}[{key!r}]")
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must contain only JSON-compatible values; "
        f"got {type(value).__name__}."
    )


def freeze_json_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    """Validate and freeze a JSON object."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping, got {type(value).__name__}.")
    frozen = freeze_json(value, path=path)
    assert isinstance(frozen, Mapping)
    return frozen


def json_digest(payload: Any, *, path: str = "digest_payload") -> str:
    """Return the sha256 hex digest of a canonical JSON encoding of *payload*.

    Non-JSON-safe payloads are rejected by :func:`freeze_json`, so a digest can
    never silently cover a stringified object.
    """
    text = json.dumps(
        thaw_json(freeze_json(payload, path=path)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def thaw_json(value: Any) -> Any:
    """Return new ordinary dict/list containers suitable for ``json.dumps``."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value
