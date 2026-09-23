"""Resolve bundled data files and optional artifact output directories."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

PathLike = Union[str, Path]

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = _PACKAGE_ROOT / "data"
_ARTIFACTS_ROOT = _PACKAGE_ROOT / "artifacts"

# Legacy leaf locations kept as fallbacks so existing scripts keep working
# while callers migrate to the shared loaders.
_LEGACY_CROWS_CANDIDATES: Sequence[Path] = (
    _PACKAGE_ROOT
    / "definition"
    / "encoder_only"
    / "intrinsic_bias"
    / "probability_based"
    / "pseudo_log_likelihood_metrics"
    / "cps"
    / "crows_pairs_anonymized.csv",
    _PACKAGE_ROOT
    / "definition"
    / "decoder_only"
    / "extrinsic_bias"
    / "demographic_representation"
    / "dnp"
    / "data"
    / "crows_pairs_anonymized.csv",
)

_LEGACY_BBQ_DIRS: Sequence[Path] = (
    _PACKAGE_ROOT / "definition" / "encoder_only" / "extrinsic_bias" / "equal_opportunity" / "data",
    _PACKAGE_ROOT / "definition" / "encoder_only" / "extrinsic_bias" / "fair_inference" / "data",
    _PACKAGE_ROOT / "definition" / "encoder_only" / "extrinsic_bias" / "context_based_disparity" / "data",
    _PACKAGE_ROOT / "definition" / "decoder_only" / "extrinsic_bias" / "demographic_representation" / "dnp" / "data",
    _PACKAGE_ROOT / "definition" / "decoder_only" / "extrinsic_bias" / "demographic_representation" / "drd" / "data",
)


def package_root() -> Path:
    return _PACKAGE_ROOT


def data_root() -> Path:
    return _DATA_ROOT


def artifacts_root() -> Path:
    _ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    return _ARTIFACTS_ROOT


def _first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_crows_pairs_csv(path: Optional[PathLike] = None) -> Path:
    """Return the CrowS-Pairs CSV path (canonical package data preferred)."""
    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"CrowS-Pairs CSV not found: {resolved}")
        return resolved

    canonical = _DATA_ROOT / "crows_pairs" / "crows_pairs_anonymized.csv"
    found = _first_existing((canonical, *_LEGACY_CROWS_CANDIDATES))
    if found is None:
        raise FileNotFoundError(
            "CrowS-Pairs CSV not found. Expected at "
            f"{canonical} (or a legacy metric leaf copy)."
        )
    return found


def resolve_bbq_dir(path: Optional[PathLike] = None) -> Path:
    """Return a directory containing BBQ ``*.jsonl`` category files."""
    if path is not None:
        resolved = Path(path)
        if not resolved.is_dir():
            raise FileNotFoundError(f"BBQ data directory not found: {resolved}")
        return resolved

    canonical = _DATA_ROOT / "bbq"
    found = _first_existing((canonical, *_LEGACY_BBQ_DIRS))
    if found is None:
        raise FileNotFoundError(
            "BBQ data directory not found. Expected at "
            f"{canonical} (or a legacy metric leaf data/ folder)."
        )
    return found


def bbq_category_files(
    data_dir: Optional[PathLike] = None,
    categories: Optional[Sequence[str]] = None,
) -> list[Path]:
    """List BBQ jsonl files for the requested categories (default: core nine)."""
    root = resolve_bbq_dir(data_dir)
    default = (
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Religion",
        "SES",
        "Sexual_orientation",
    )
    names = categories if categories is not None else default
    paths: list[Path] = []
    missing: list[str] = []
    for name in names:
        stem = name.replace(".jsonl", "")
        candidate = root / f"{stem}.jsonl"
        if candidate.exists():
            paths.append(candidate)
        else:
            missing.append(stem)
    if not paths:
        raise FileNotFoundError(
            f"No BBQ category files found under {root}. Missing: {missing}"
        )
    return paths
