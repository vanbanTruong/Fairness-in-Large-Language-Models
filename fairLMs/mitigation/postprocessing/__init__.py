"""Post-processing mitigators: black-box, fitted on outputs.

These touch no model. They fit a decision rule on scores that have already been
produced, which is why they apply to every architecture and need only black-box
access. All three return a **serializable** rule, so the fitted object can be
stored, reviewed and applied later without the model that produced the scores.

One mitigator per module, with the per-group guards they share in
:mod:`~._shared`:

==============================  ==================================
``score_calibration``           :mod:`~.score_calibration`
``group_aware_thresholding``    :mod:`~.group_aware_thresholding`
``output_reranking``            :mod:`~.output_reranking`
==============================  ==================================
"""

from __future__ import annotations

from ._shared import _ALL_ARCHITECTURES, _require_both_classes, _require_rows
from .output_reranking import OutputReranking, _rerank_one
from .group_aware_thresholding import (
    GroupAwareThresholding,
    _candidate_thresholds,
    _rates,
)
from .score_calibration import (
    ScoreCalibration,
    _apply_isotonic,
    _apply_platt,
    _fit_isotonic,
    _fit_platt,
)

__all__ = [
    "GroupAwareThresholding",
    "OutputReranking",
    "ScoreCalibration",
]
