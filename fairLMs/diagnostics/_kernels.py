"""Private numeric kernels shared by more than one diagnostic module.

Every helper here is deliberately small, pure, and free of any dependency on
the public diagnostic containers, so that the one-way import direction
``_utils -> _kernels -> evidence -> spec -> base -> leakage / construction``
stays intact.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - imported only by type checkers
    from fairLMs.diagnostics.leakage import LogBase


def stable_log_ratio(
    numerator: float,
    denominator: float,
    *,
    base: "LogBase",
) -> float:
    """Preserve canonical arithmetic, with a safe extreme-value fallback.

    ``base`` is declared rather than assumed because mutual information and
    pointwise mutual information are not invariant to the logarithm base. The
    enum is annotated under ``TYPE_CHECKING`` and dispatched on ``base.value``
    at runtime so this module never imports ``leakage``.

    The function is *total*: additive smoothing keeps both operands strictly
    positive in the reals, but not in IEEE-754 doubles, where a marginal
    product can flush to zero (underflow) or a smoothed total can become
    ``inf`` and drive every cell to zero (overflow). Rather than raising
    ``ZeroDivisionError`` from inside a pure kernel -- which would reach the
    caller as a bare exception carrying none of the marginals, entropies or
    estimator settings -- a degenerate operand pair yields the corresponding
    non-finite value, so the caller's own numeric guards decide the reason
    code.
    """
    log = math.log2 if getattr(base, "value", base) == "base_2" else math.log
    if (
        numerator > 0.0
        and denominator > 0.0
        and math.isfinite(numerator)
        and math.isfinite(denominator)
    ):
        ratio = numerator / denominator
        if ratio > 0.0 and math.isfinite(ratio):
            return log(ratio)
        # Both operands are finite and strictly positive here, so both logs
        # are defined and this fallback cannot raise.
        return log(numerator) - log(denominator)
    if (
        math.isnan(numerator)
        or math.isnan(denominator)
        or numerator < 0.0
        or denominator < 0.0
    ):
        return math.nan
    if denominator == 0.0:
        # ``log(x / 0)``: +inf for positive mass, undefined for 0 / 0.
        return math.inf if numerator > 0.0 else math.nan
    if numerator == 0.0:
        return -math.inf
    if numerator == math.inf:
        return math.nan if denominator == math.inf else math.inf
    return -math.inf  # finite positive mass over an infinite denominator


def max_pairwise_gap(
    values: Mapping[str, float],
) -> tuple[float, Optional[tuple[str, str]]]:
    """Return the widest absolute gap and the pair that realizes it.

    The comparison is strict, so the lexicographically first maximal pair
    wins and the reported pair is stable across runs.
    """
    labels = sorted(values)
    if len(labels) < 2:
        return 0.0, None
    best_gap = 0.0
    best_pair: Optional[tuple[str, str]] = None
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            gap = abs(float(values[left]) - float(values[right]))
            if best_pair is None or gap > best_gap:
                best_gap = gap
                best_pair = (left, right)
    return best_gap, best_pair


def token_levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    """Return the token-level edit distance between two token sequences."""
    previous = list(range(len(right) + 1))
    for row, left_token in enumerate(left, start=1):
        current = [row]
        for column, right_token in enumerate(right, start=1):
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            substitution = previous[column - 1] + (
                0 if left_token == right_token else 1
            )
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


__all__ = ["max_pairwise_gap", "stable_log_ratio", "token_levenshtein"]
