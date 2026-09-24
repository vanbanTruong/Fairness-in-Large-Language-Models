"""I/O helpers for metric runners."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

import pandas as pd

PathLike = Union[str, Path]


def results_to_csv(
    results: Iterable[Mapping],
    filename: str,
    output_dir: Optional[PathLike] = None,
) -> str:
    """Write rows to CSV under ``output_dir``.

    The default is ``./fairlms-results`` in the caller's working directory.
    Installed package directories may be read-only and must never receive run
    artifacts.
    """
    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path.cwd() / "fairlms-results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    pd.DataFrame(list(results)).to_csv(out_path, index=False)
    return str(out_path)
