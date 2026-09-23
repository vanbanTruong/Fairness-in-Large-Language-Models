"""Group-regularized objective: a differentiable TPR/FPR-gap surrogate."""

from __future__ import annotations

from typing import Any, Sequence

from fairLMs.mitigation.base import MitigationResult, Mitigator
from fairLMs.mitigation.evidence import GroupLabeledRecords

__all__ = ["GroupRegularizedObjective"]


class GroupRegularizedObjective(Mitigator):
    """Differentiable TPR/FPR-gap surrogate added to the task loss.

    The hard rate gap is piecewise constant and has no useful gradient, so the
    surrogate replaces the indicator with the predicted positive probability.
    That makes the gap differentiable while keeping its meaning: the mean
    predicted score on the positive class, compared across groups.

    Says *objective* rather than the book's "group-regularized fine-tuning"
    because this ships a loss term, not a trainer.

    Returns ``(probabilities, labels, groups) -> loss``.

    Parameters
    ----------
    criterion:
        ``"equal_opportunity"`` penalises the TPR gap; ``"equalized_odds"``
        penalises TPR and FPR gaps jointly.

    Examples
    --------
    >>> import torch
    >>> from fairLMs.mitigation import GroupLabeledRecords, GroupRegularizedObjective
    >>> records = GroupLabeledRecords(
    ...     axis="gender", groups=["f", "f", "m", "m"],
    ...     labels=["yes", "no", "yes", "no"],
    ...     label_name="outcome", source="doctest",
    ... )
    >>> loss_term = GroupRegularizedObjective().apply(None, records).result
    >>> probabilities = torch.tensor([0.8, 0.2, 0.8, 0.2])
    >>> groups, positives = ["f", "f", "m", "m"], [True, False, True, False]
    >>> round(float(loss_term(probabilities, positives, groups)), 6)  # no gap
    0.0
    """

    name = "group_regularized_objective"
    category = "in"
    access = "white_box"
    architectures = ("encoder_only",)
    # Empty by design: this mitigator builds a loss term and reads nothing
    # from a model itself. The white-box demand of in-processing - the
    # training loop needs parameters and gradients - is carried by `access`,
    # which is checked whenever a model is supplied. Declaring capabilities
    # here would instead force a model to be passed just to build a loss.
    requires = frozenset()
    accepts = (GroupLabeledRecords,)

    def __init__(self, criterion: str = "equal_opportunity"):
        self.criterion = criterion

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        import torch

        if self.criterion not in ("equal_opportunity", "equalized_odds"):
            raise ValueError(
                "criterion must be 'equal_opportunity' or 'equalized_odds'; got "
                f"{self.criterion!r}."
            )
        criterion = self.criterion

        def component(
            probabilities: Any, positives: Sequence[bool], groups: Sequence[str]
        ) -> Any:
            """Soft TPR (and optionally FPR) gap across groups."""
            if not (len(positives) == len(groups) == probabilities.shape[0]):
                raise ValueError(
                    f"{GroupRegularizedObjective.name}: probabilities, positives "
                    f"and groups must align; got {probabilities.shape[0]}, "
                    f"{len(positives)} and {len(groups)}."
                )
            gap = probabilities.new_zeros(())
            for wanted in (
                (True,) if criterion == "equal_opportunity" else (True, False)
            ):
                rates = []
                for group in sorted(set(groups)):
                    rows = [
                        i
                        for i, (g, p) in enumerate(zip(groups, positives))
                        if g == group and bool(p) is wanted
                    ]
                    if not rows:
                        raise ValueError(
                            f"{GroupRegularizedObjective.name}: group {group!r} has "
                            f"no {'positive' if wanted else 'negative'} rows in this "
                            f"batch, so its rate is undefined. Use a group-stratified "
                            f"sampler rather than letting the term silently skip it."
                        )
                    rates.append(probabilities[rows].mean())
                stacked = torch.stack(rates)
                gap = gap + (stacked.max() - stacked.min())
            return gap

        return self._result(
            component,
            axis=evidence.axis,
            criterion=criterion,
            groups=sorted(set(evidence.groups)),
            surrogate="mean predicted probability replaces the rate indicator",
            returns="loss component; the caller owns the training loop",
        )
