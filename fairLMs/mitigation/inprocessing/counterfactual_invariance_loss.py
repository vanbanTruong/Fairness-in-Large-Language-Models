"""Counterfactual invariance: symmetric KL between ``x`` and ``tau(x)``."""

from __future__ import annotations

from typing import Any

from fairLMs.metrics.data import PromptPairs
from fairLMs.mitigation.base import MitigationResult, Mitigator

__all__ = ["CounterfactualInvarianceLoss"]


class CounterfactualInvarianceLoss(Mitigator):
    """Symmetric KL between predictions on ``x`` and ``tau(x)``.

    Penalises any change in the predictive distribution under a counterfactual
    rewrite. Symmetric rather than one-directional so that neither branch is
    privileged as the reference.

    Returns ``(logits_factual, logits_counterfactual) -> loss``.

    Parameters
    ----------
    reduction:
        ``"mean"`` or ``"sum"`` over the batch.

    Examples
    --------
    >>> import torch
    >>> from fairLMs.metrics import PromptPairs
    >>> from fairLMs.mitigation import CounterfactualInvarianceLoss
    >>> pairs = PromptPairs(["he is a nurse"], ["she is a nurse"])
    >>> loss_term = CounterfactualInvarianceLoss().apply(None, pairs).result
    >>> identical = torch.tensor([[1.0, 2.0]])
    >>> round(float(loss_term(identical, identical)), 6)   # invariant: no penalty
    0.0
    >>> float(loss_term(identical, torch.tensor([[2.0, 1.0]]))) > 0
    True
    """

    name = "counterfactual_invariance_loss"
    category = "in"
    access = "white_box"
    architectures = ("encoder_only",)
    # Empty by design: this mitigator builds a loss term and reads nothing
    # from a model itself. The white-box demand of in-processing - the
    # training loop needs parameters and gradients - is carried by `access`,
    # which is checked whenever a model is supplied. Declaring capabilities
    # here would instead force a model to be passed just to build a loss.
    requires = frozenset()
    accepts = (PromptPairs,)

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        import torch

        if self.reduction not in ("mean", "sum"):
            raise ValueError(
                f"reduction must be 'mean' or 'sum'; got {self.reduction!r}."
            )
        reduction = self.reduction

        def component(logits_factual: Any, logits_counterfactual: Any) -> Any:
            """Symmetric KL between the two branches' predictive distributions."""
            if logits_factual.shape != logits_counterfactual.shape:
                raise ValueError(
                    f"{CounterfactualInvarianceLoss.name}: the two branches must "
                    f"have the same shape; got {tuple(logits_factual.shape)} and "
                    f"{tuple(logits_counterfactual.shape)}."
                )
            log_p = torch.log_softmax(logits_factual, dim=-1)
            log_q = torch.log_softmax(logits_counterfactual, dim=-1)
            p, q = log_p.exp(), log_q.exp()
            per_row = ((p - q) * (log_p - log_q)).sum(dim=-1)
            return per_row.mean() if reduction == "mean" else per_row.sum()

        return self._result(
            component,
            n_pairs=len(evidence),
            reduction=reduction,
            returns="loss component; the caller owns the training loop",
        )
