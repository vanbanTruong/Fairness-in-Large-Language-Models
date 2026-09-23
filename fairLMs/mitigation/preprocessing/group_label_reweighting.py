"""Group/label reweighting: per-row weights toward a declared target joint."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from fairLMs.mitigation.base import MitigationResult, Mitigator
from fairLMs.mitigation.evidence import GroupLabeledRecords
from fairLMs.mitigation.preprocessing._shared import _ALL_ARCHITECTURES

__all__ = ["GroupLabelReweighting"]


class GroupLabelReweighting(Mitigator):
    r"""Per-row weights ``w = pi(y, a) / P_hat(y, a)`` toward a declared target.

    ``P_hat`` is the empirical joint of outcome and group in the supplied rows.
    ``pi`` is the target joint. The default target is the **independence**
    product ``P(y) * P(a)``, the standard reweighting choice: it removes the
    label-group association while leaving both marginals as observed.

    Parameters
    ----------
    target:
        ``"independence"`` for ``P(y) P(a)``, or ``"uniform"`` for an equal mass
        in every observed ``(y, a)`` cell.

    Examples
    --------
    >>> from fairLMs.mitigation import GroupLabelReweighting, GroupLabeledRecords
    >>> records = GroupLabeledRecords(
    ...     axis="gender",
    ...     groups=["f", "f", "m", "m"],
    ...     labels=["no", "no", "yes", "no"],
    ...     label_name="outcome", source="doctest",
    ... )
    >>> weights = GroupLabelReweighting().apply(None, records).result["weights"]
    >>> len(weights)
    4
    """

    name = "group_label_reweighting"
    category = "pre"
    access = "black_box"
    architectures = _ALL_ARCHITECTURES
    requires = frozenset()
    accepts = (GroupLabeledRecords,)

    def __init__(self, target: str = "independence"):
        self.target = target

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if self.target not in ("independence", "uniform"):
            raise ValueError(
                f"target must be 'independence' or 'uniform'; got {self.target!r}."
            )
        n = evidence.n_rows
        joint: Dict[Tuple[str, str], int] = {}
        label_counts: Dict[str, int] = {}
        group_counts: Dict[str, int] = {}
        for label, group in zip(evidence.labels, evidence.groups):
            joint[(label, group)] = joint.get((label, group), 0) + 1
            label_counts[label] = label_counts.get(label, 0) + 1
            group_counts[group] = group_counts.get(group, 0) + 1

        if self.target == "independence":
            pi = {
                cell: (label_counts[cell[0]] / n) * (group_counts[cell[1]] / n)
                for cell in joint
            }
        else:
            pi = {cell: 1.0 / len(joint) for cell in joint}

        weights = [
            pi[(label, group)] / (joint[(label, group)] / n)
            for label, group in zip(evidence.labels, evidence.groups)
        ]
        return self._result(
            {
                "weights": weights,
                "target": self.target,
                "cells": {
                    f"{label}|{group}": {
                        "observed": count / n,
                        "target": pi[(label, group)],
                        "weight": pi[(label, group)] / (count / n),
                    }
                    for (label, group), count in sorted(joint.items())
                },
            },
            axis=evidence.axis,
            label_name=evidence.label_name,
            n_rows=n,
            source=evidence.source,
        )
