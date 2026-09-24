"""Internal helpers for migrating callers off the legacy ``**kwargs`` API.

Every metric family follows the same pattern: accept a typed container as the
second positional argument, but keep the historical ``T1_terms=`` /
``prompts=`` / ``y_true=`` keyword style working for one deprecation cycle.
These helpers keep that shim uniform (and in one place) instead of copy-pasted
across eleven modules.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional, Sequence, Tuple

__all__ = ["warn_legacy", "take", "unwrap", "as_examples", "require_mapping_keys"]


def warn_legacy(metric: str, keys: Sequence[str], replacement: str) -> None:
    """Emit the standard deprecation notice for legacy keyword usage."""
    warnings.warn(
        f"Passing {', '.join(sorted(k for k in keys if k))} to {metric}.compute() "
        f"is deprecated and will be removed in a future release. Pass "
        f"{replacement} as the second argument instead, e.g. "
        f"{metric}().compute(model, {replacement}(...)).",
        DeprecationWarning,
        stacklevel=4,
    )


def take(legacy: dict, *aliases: str) -> Tuple[Any, Optional[str]]:
    """Return ``(value, key_used)`` for the first alias present and not None."""
    for alias in aliases:
        if legacy.get(alias) is not None:
            return legacy[alias], alias
    return None, None


def unwrap(data: Any, *skip_types: type) -> Any:
    """Resolve a dataset-like object to its examples.

    Containers listed in ``skip_types`` (and mappings) pass through untouched so
    a typed container is never mistaken for a dataset.
    """
    if data is None:
        return None
    if isinstance(data, (Mapping, *skip_types)):
        return data
    if hasattr(data, "load") and callable(data.load):
        return data.load()
    return data


def as_examples(data: Any, metric: str, what: str) -> list:
    """Coerce ``data`` to a non-empty list of examples, or raise."""
    if data is None:
        raise ValueError(f"{metric} requires {what} as the second argument.")
    try:
        examples = list(data)
    except TypeError as exc:
        raise TypeError(
            f"{metric}: expected {what}, got {type(data).__name__} which is not "
            f"iterable."
        ) from exc
    if not examples:
        raise ValueError(f"{metric}: {what} is empty.")
    return examples


def require_mapping_keys(examples: Sequence[Any], metric: str, *keys: str) -> None:
    """Validate that dict-shaped examples carry the keys the metric reads.

    Validate every row before scoring, including non-mapping rows.
    """
    for index, row in enumerate(examples):
        if not isinstance(row, Mapping):
            raise TypeError(
                f"{metric}: example {index} must be a mapping, got {type(row).__name__}."
            )
        missing = [key for key in keys if key not in row]
        if missing:
            raise ValueError(
                f"{metric}: example {index} is missing key(s) {', '.join(missing)}. "
                f"Each example needs {', '.join(keys)}."
            )
