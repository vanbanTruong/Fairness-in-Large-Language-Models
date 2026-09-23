"""I/O helpers for metric runners."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

import pandas as pd

from fairLMs.utils.paths import artifacts_root

PathLike = Union[str, Path]


def results_to_csv(
    results: Iterable[Mapping],
    filename: str,
    output_dir: Optional[PathLike] = None,
) -> str:
    """Write rows to CSV under ``output_dir`` (defaults to package artifacts/)."""
    out_dir = Path(output_dir) if output_dir is not None else artifacts_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    pd.DataFrame(list(results)).to_csv(out_path, index=False)
    return str(out_path)
