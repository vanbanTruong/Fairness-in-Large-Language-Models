from __future__ import annotations

import numpy as np
from collections import namedtuple
from typing import Any, Sequence

GapGYResult = namedtuple(
    "GapGYResult",
    ["tpr_g1", "tpr_g2", "gap", "g1", "g2", "y", "n_g1", "n_g2"],
)


def _validate_inputs(y_true: Sequence, y_pred: Sequence, groups: Sequence):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    if not (y_true.shape == y_pred.shape == groups.shape):
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, "
            f"y_pred={y_pred.shape}, groups={groups.shape}"
        )
    return y_true, y_pred, groups


def gap_g_y(
    y_true: Sequence,
    y_pred: Sequence,
    groups: Sequence,
    g1: Any,
    g2: Any,
    y: Any = 1,
) -> GapGYResult:
    
    y_true, y_pred, groups = _validate_inputs(y_true, y_pred, groups)

    def tpr_for(g):
        mask = (groups == g) & (y_true == y)
        if mask.sum() == 0:
            return float("nan"), 0
        return float(np.mean(y_pred[mask] == y)), int(mask.sum())

    tpr1, n1 = tpr_for(g1)
    tpr2, n2 = tpr_for(g2)
    gap = (tpr1 - tpr2) if (np.isfinite(tpr1) and np.isfinite(tpr2)) else float("nan")

    return GapGYResult(tpr_g1=tpr1, tpr_g2=tpr2, gap=gap,
                        g1=g1, g2=g2, y=y, n_g1=n1, n_g2=n2)