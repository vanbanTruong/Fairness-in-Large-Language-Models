"""Resolve the small dataset resources distributed with FairLMs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"


def package_root() -> Path:
    return _PACKAGE_ROOT


def resource_root() -> Path:
    return _RESOURCE_ROOT


def resolve_crows_pairs_csv(path: Optional[PathLike] = None) -> Path:
    """Return the bundled CrowS-Pairs CSV or a caller-supplied copy."""
    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"CrowS-Pairs CSV not found: {resolved}")
        return resolved

    canonical = _RESOURCE_ROOT / "crows_pairs" / "crows_pairs_anonymized.csv"
    if not canonical.is_file():
        raise FileNotFoundError(
            f"Bundled CrowS-Pairs CSV not found: {canonical}. Reinstall FairLMs "
            "or pass path= to CrowSPairs."
        )
    return canonical
