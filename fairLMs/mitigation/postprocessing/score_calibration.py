"""Score calibration: Platt or isotonic, fitted per group."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from fairLMs.diagnostics.evidence import LabeledScoredGroups
from fairLMs.mitigation.base import MitigationResult, Mitigator
from fairLMs.mitigation.postprocessing._shared import (
    _ALL_ARCHITECTURES,
    _require_both_classes,
    _require_rows,
)

__all__ = ["ScoreCalibration"]


def _fit_platt(scores: Sequence[float], positives: Sequence[bool]) -> Dict[str, float]:
    """Fit a 1-D logistic ``sigmoid(a * s + b)`` by Newton-Raphson.

    Implemented directly rather than via scikit-learn so that calibration works
    in the base install. Newton on a 2-parameter convex problem converges in a
    handful of iterations and needs no optimizer dependency.
    """
    y = [1.0 if p else 0.0 for p in positives]
    a, b = 0.0, 0.0
    for _ in range(100):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for s, target in zip(scores, y):
            p = 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, a * s + b))))
            residual = p - target
            w = p * (1.0 - p)
            g_a += residual * s
            g_b += residual
            h_aa += w * s * s
            h_ab += w * s
            h_bb += w
        # Ridge term keeps the Hessian invertible when a group is separable.
        h_aa += 1e-9
        h_bb += 1e-9
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-15:
            break
        step_a = (h_bb * g_a - h_ab * g_b) / det
        step_b = (h_aa * g_b - h_ab * g_a) / det
        a -= step_a
        b -= step_b
        if max(abs(step_a), abs(step_b)) < 1e-10:
            break
    return {"a": a, "b": b}


def _fit_isotonic(
    scores: Sequence[float], positives: Sequence[bool]
) -> Dict[str, List[float]]:
    """Fit a monotone step function by pool-adjacent-violators (PAVA).

    Rows sharing a score are pooled into a single block **before** the merge.
    Isotonic regression is a function of the score, so tied scores have to
    receive one fitted value. Running PAVA over the raw rows instead leaves
    equal scores carrying different values, and which one a later lookup lands
    on then depends on the order the rows happened to arrive in - the same
    evidence, reordered, would calibrate differently and the stored rule would
    not be reproducible from it.

    The returned rule therefore carries one entry per **distinct** score, which
    also makes the lookup in :func:`_apply_isotonic` unambiguous.
    """
    pooled: Dict[float, List[float]] = {}
    for score, is_positive in zip(scores, positives):
        pooled.setdefault(float(score), []).append(1.0 if is_positive else 0.0)

    xs = sorted(pooled)
    # Each block carries the mean outcome, the number of rows behind it (the
    # merge weight), and how many distinct scores it spans (so the fitted list
    # stays aligned one-to-one with `xs`).
    values: List[float] = []
    weights: List[float] = []
    spans: List[int] = []
    for x in xs:
        rows = pooled[x]
        values.append(sum(rows) / len(rows))
        weights.append(float(len(rows)))
        spans.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            v2, w2, s2 = values.pop(), weights.pop(), spans.pop()
            v1, w1, s1 = values.pop(), weights.pop(), spans.pop()
            values.append((v1 * w1 + v2 * w2) / (w1 + w2))
            weights.append(w1 + w2)
            spans.append(s1 + s2)

    fitted: List[float] = []
    for value, span in zip(values, spans):
        fitted.extend([value] * span)
    return {"x": xs, "y": fitted}


def _apply_platt(rule: Dict[str, float], score: float) -> float:
    z = max(-500.0, min(500.0, rule["a"] * score + rule["b"]))
    return 1.0 / (1.0 + math.exp(-z))


def _apply_isotonic(rule: Dict[str, List[float]], score: float) -> float:
    xs, ys = rule["x"], rule["y"]
    if score <= xs[0]:
        return ys[0]
    if score >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] < score:
            lo = mid + 1
        else:
            hi = mid
    return ys[lo]


class ScoreCalibration(Mitigator):
    """Platt or isotonic calibration fitted **per group**.

    A single global calibrator leaves each group's scores miscalibrated in a
    different direction; fitting per group is what makes a downstream threshold
    mean the same thing for everyone.

    Parameters
    ----------
    method:
        ``"platt"`` for a logistic fit, ``"isotonic"`` for a monotone step fit.
    min_rows_per_group:
        Refuse any group with fewer rows than this. Never skip it silently.

    Examples
    --------
    >>> from fairLMs.diagnostics import LabeledScoredGroups, ScoredGroups
    >>> from fairLMs.mitigation import ScoreCalibration
    >>> evidence = LabeledScoredGroups(
    ...     scored=ScoredGroups(
    ...         axis="gender",
    ...         groups=["f", "f", "f", "m", "m", "m"],
    ...         scores=[0.2, 0.6, 0.9, 0.1, 0.5, 0.8],
    ...         score_name="p_hire", source="doctest", score_range=[0.0, 1.0],
    ...     ),
    ...     labels=["no", "yes", "yes", "no", "no", "yes"],
    ...     label_name="outcome", positive_label="yes",
    ... )
    >>> outcome = ScoreCalibration().apply(None, evidence)
    >>> outcome.category
    'post'
    >>> tuple(sorted(outcome.result["groups"]))
    ('f', 'm')
    """

    name = "score_calibration"
    category = "post"
    access = "black_box"
    architectures = _ALL_ARCHITECTURES
    requires = frozenset()  # operates on scores, not on a model
    accepts = (LabeledScoredGroups,)

    def __init__(self, method: str = "platt", min_rows_per_group: int = 3):
        self.method = method
        self.min_rows_per_group = min_rows_per_group

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if self.method not in ("platt", "isotonic"):
            raise ValueError(
                f"method must be 'platt' or 'isotonic'; got {self.method!r}."
            )
        fitted = {}
        for group, scores, positives in evidence.iter_groups():
            _require_rows(group, len(scores), self.min_rows_per_group, self.name)
            _require_both_classes(group, positives, self.name)
            fitted[group] = (
                _fit_platt(scores, positives)
                if self.method == "platt"
                else _fit_isotonic(scores, positives)
            )
        return self._result(
            {"method": self.method, "groups": fitted},
            axis=evidence.axis,
            n_rows=evidence.n_rows,
            positive_label=evidence.positive_label,
            groups=list(evidence.support),
        )

    @staticmethod
    def transform(rule: Dict[str, Any], group: str, score: float) -> float:
        """Apply a fitted rule from :attr:`MitigationResult.result` to one score."""
        if group not in rule["groups"]:
            raise KeyError(
                f"no calibrator was fitted for group {group!r}; fitted groups are "
                f"{sorted(rule['groups'])}."
            )
        per_group = rule["groups"][group]
        if rule["method"] == "platt":
            return _apply_platt(per_group, score)
        return _apply_isotonic(per_group, score)
