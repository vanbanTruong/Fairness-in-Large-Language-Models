"""Private warning and assumption strings shared by more than one module.

``representativeness._STRESS_TEST_WARNING`` is deliberately **not** replaced by
the constant below: its wording is reference-distribution specific and is
pinned by existing tests.
"""

from __future__ import annotations

from typing import Final

STRESS_TEST_WARNING: Final[str] = (
    "The audit has a stress-test design stance: a measured disparity or "
    "association may be the benchmark's intended signal and must not be "
    "treated as an automatic fairness failure."
)
DESCRIPTIVE_INTERPRETATION: Final[str] = (
    "Dataset diagnostics are descriptive evidence about corpus composition, "
    "not an automatic fairness pass/fail judgment; their meaning depends on "
    "the benchmark design stance."
)


__all__ = ["DESCRIPTIVE_INTERPRETATION", "STRESS_TEST_WARNING"]
