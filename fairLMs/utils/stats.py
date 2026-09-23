"""Small statistical helpers used across metric runners."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_confidence_interval(scores, confidence=0.95):
    """Return (low, high) t-interval for the mean of ``scores``."""
    n = len(scores)
    if n < 2:
        return (float("nan"), float("nan"))
    lo, hi = stats.t.interval(
        confidence,
        df=n - 1,
        loc=np.mean(scores),
        scale=stats.sem(scores),
    )
    return float(lo), float(hi)
