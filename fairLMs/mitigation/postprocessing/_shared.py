"""Declarations and per-group guards shared by the post-processing mitigators."""

from __future__ import annotations

from typing import Sequence

__all__ = [
    "_ALL_ARCHITECTURES",
    "_require_both_classes",
    "_require_rows",
]

_ALL_ARCHITECTURES = ("encoder_only", "decoder_only", "encoder_decoder")


def _require_rows(group: str, n: int, minimum: int, mitigator: str) -> None:
    """Refuse a group too small to fit against, rather than quietly skipping it."""
    if n < minimum:
        raise ValueError(
            f"{mitigator}: group {group!r} has {n} row(s), fewer than the {minimum} "
            f"needed to fit against. Supply more rows for this group or drop it "
            f"from the evidence deliberately; it will not be skipped silently."
        )


def _require_both_classes(
    group: str, positives: Sequence[bool], mitigator: str
) -> None:
    n_pos = sum(positives)
    if n_pos == 0 or n_pos == len(positives):
        raise ValueError(
            f"{mitigator}: group {group!r} has only one outcome class "
            f"({n_pos} positive of {len(positives)}). A per-group rule cannot be "
            f"fitted against a constant outcome."
        )
