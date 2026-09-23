"""Group-aware thresholding: per-group decision thresholds matching rates."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from fairLMs.diagnostics.evidence import LabeledScoredGroups
from fairLMs.mitigation.base import MitigationResult, Mitigator
from fairLMs.mitigation.postprocessing._shared import (
    _ALL_ARCHITECTURES,
    _require_both_classes,
    _require_rows,
)

__all__ = ["GroupAwareThresholding"]


def _rates(scores: Sequence[float], positives: Sequence[bool], threshold: float):
    """Return ``(tpr, fpr)`` at *threshold*, predicting positive when ``s >= t``."""
    tp = fp = pos = neg = 0
    for score, is_positive in zip(scores, positives):
        predicted = score >= threshold
        if is_positive:
            pos += 1
            tp += predicted
        else:
            neg += 1
            fp += predicted
    return (tp / pos if pos else 0.0), (fp / neg if neg else 0.0)


def _candidate_thresholds(scores: Sequence[float]) -> List[float]:
    """Every threshold that produces a distinct split, plus one above the top."""
    unique = sorted(set(float(s) for s in scores))
    return unique + [unique[-1] + 1.0]


class GroupAwareThresholding(Mitigator):
    """Per-group decision thresholds meeting equal opportunity or equalized odds.

    Searches each group's achievable ``(tpr, fpr)`` operating points and picks
    the per-group thresholds whose rates agree most closely across groups.

    ``equal_opportunity`` matches true-positive rates only; ``equalized_odds``
    matches true-positive and false-positive rates jointly. Which is the right
    criterion is a question about the deployment, so it is a declared parameter
    with no default that pretends otherwise.

    Reports the residual gap it actually achieved: with finite data the rates
    rarely match exactly, and rounding that away would overstate the result.

    **Equalising the rates is not sufficient on its own.** Two operating points
    equalise every rate perfectly and are useless: rejecting everyone (all rates
    zero) and accepting everyone (all rates one). A search that minimises only
    the gap finds one of them and reports a perfect score, because a gap of zero
    is exactly what it was asked for. Among the points that tie on the gap this
    therefore maximises Youden's J, ``mean(TPR) - mean(FPR)``, which is zero for
    both degenerate points and positive for any genuinely discriminating rule.
    The achieved value is reported as ``achieved_utility`` and the rule is
    recorded in provenance, so the trade-off stays visible rather than implied.

    Parameters
    ----------
    criterion:
        ``"equal_opportunity"`` or ``"equalized_odds"``.
    min_rows_per_group:
        Refuse any group with fewer rows than this.

    Examples
    --------
    >>> from fairLMs.diagnostics import LabeledScoredGroups, ScoredGroups
    >>> from fairLMs.mitigation import GroupAwareThresholding
    >>> evidence = LabeledScoredGroups(
    ...     scored=ScoredGroups(
    ...         axis="gender",
    ...         groups=["f", "f", "f", "f", "m", "m", "m", "m"],
    ...         scores=[0.1, 0.4, 0.6, 0.9, 0.3, 0.5, 0.7, 0.95],
    ...         score_name="p_hire", source="doctest", score_range=[0.0, 1.0],
    ...     ),
    ...     labels=["no", "no", "yes", "yes", "no", "no", "yes", "yes"],
    ...     label_name="outcome", positive_label="yes",
    ... )
    >>> rule = GroupAwareThresholding().apply(None, evidence).result
    >>> tuple(sorted(rule["thresholds"]))
    ('f', 'm')
    >>> rule["achieved_tpr_gap"] <= 1.0
    True
    """

    name = "group_aware_thresholding"
    category = "post"
    access = "black_box"
    architectures = _ALL_ARCHITECTURES
    requires = frozenset()
    accepts = (LabeledScoredGroups,)

    def __init__(
        self,
        criterion: str = "equal_opportunity",
        min_rows_per_group: int = 3,
    ):
        self.criterion = criterion
        self.min_rows_per_group = min_rows_per_group

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if self.criterion not in ("equal_opportunity", "equalized_odds"):
            raise ValueError(
                "criterion must be 'equal_opportunity' or 'equalized_odds'; got "
                f"{self.criterion!r}."
            )

        # Per group: every achievable (threshold, tpr, fpr) operating point.
        options: Dict[str, List[Tuple[float, float, float]]] = {}
        for group, scores, positives in evidence.iter_groups():
            _require_rows(group, len(scores), self.min_rows_per_group, self.name)
            _require_both_classes(group, positives, self.name)
            options[group] = [
                (t, *_rates(scores, positives, t))
                for t in _candidate_thresholds(scores)
            ]

        groups = sorted(options)
        # Sweep a shared target rate and let each group pick its closest point.
        # Linear in the number of candidate points per group, rather than the
        # exponential cost of searching threshold combinations directly.
        best: Optional[Dict[str, Any]] = None
        targets = sorted({point[1] for pts in options.values() for point in pts})
        for target_tpr in targets:
            chosen = {}
            for group in groups:
                # Closest to the shared target TPR, then the *lowest* FPR among
                # the points that reach it. Several thresholds usually deliver
                # the same TPR; picking by threshold alone would silently take
                # the one that also admits the most false positives.
                chosen[group] = min(
                    options[group],
                    key=lambda p: (abs(p[1] - target_tpr), p[2], p[0]),
                )
            tprs = [chosen[g][1] for g in groups]
            fprs = [chosen[g][2] for g in groups]
            tpr_gap = max(tprs) - min(tprs)
            fpr_gap = max(fprs) - min(fprs)
            cost = tpr_gap + (fpr_gap if self.criterion == "equalized_odds" else 0.0)
            utility = (sum(tprs) - sum(fprs)) / len(groups)
            # Minimise the gap first, then break ties on utility. Rounding keeps
            # float noise from deciding the ordering; the thresholds themselves
            # are the final tie-break so the result is deterministic.
            key = (
                round(cost, 12),
                -round(utility, 12),
                tuple(chosen[g][0] for g in groups),
            )
            if best is None or key < best["key"]:
                best = {
                    "key": key,
                    "thresholds": {g: chosen[g][0] for g in groups},
                    "tpr": {g: chosen[g][1] for g in groups},
                    "fpr": {g: chosen[g][2] for g in groups},
                    "tpr_gap": tpr_gap,
                    "fpr_gap": fpr_gap,
                    "utility": utility,
                }

        assert best is not None  # at least two groups are guaranteed upstream
        return self._result(
            {
                "criterion": self.criterion,
                "thresholds": best["thresholds"],
                "tpr": best["tpr"],
                "fpr": best["fpr"],
                "achieved_tpr_gap": best["tpr_gap"],
                "achieved_fpr_gap": best["fpr_gap"],
                "achieved_utility": best["utility"],
                "positive_label": evidence.positive_label,
            },
            axis=evidence.axis,
            n_rows=evidence.n_rows,
            groups=groups,
            selection_rule=(
                "minimise the rate gap, then maximise Youden's J "
                "(mean TPR - mean FPR) among the operating points that tie"
            ),
        )

    @staticmethod
    def decide(rule: Dict[str, Any], group: str, score: float) -> bool:
        """Apply a fitted threshold rule to one score."""
        if group not in rule["thresholds"]:
            raise KeyError(
                f"no threshold was fitted for group {group!r}; fitted groups are "
                f"{sorted(rule['thresholds'])}."
            )
        return score >= rule["thresholds"][group]
