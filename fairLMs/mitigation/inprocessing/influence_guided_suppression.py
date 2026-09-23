"""Influence-guided suppression: the IF-Guide objective on precomputed scores."""

from __future__ import annotations

from typing import Any, Sequence

from fairLMs.mitigation.base import MitigationResult, Mitigator
from fairLMs.mitigation.evidence import InfluenceScoredCorpus

__all__ = ["InfluenceGuidedSuppression"]


class InfluenceGuidedSuppression(Mitigator):
    r"""Suppression term ``-lambda * sum_i |I_i| * loss_i`` over flagged instances.

    A deliberate reduction of IF-Guide (Coalson et al., 2025): this ships the
    **objective only** and consumes **precomputed** influence scores. Estimating
    influence - Hessian inverse, EK-FAC, anything else - is an optional backend
    and never a core dependency, so the component is testable on scores the
    caller supplies and the influence machinery is not reimplemented here.

    Returns ``(per_example_loss, indices) -> term``. The term is **negative**, so
    composing it the documented way, ``loss = task_loss + component(...)``,
    performs gradient *ascent* on the flagged rows in proportion to their
    absolute influence: the model is pushed away from fitting them. Adding a
    positive multiple of their loss does the exact opposite - it makes the
    optimizer fit the harmful examples *harder* - so the sign is fixed here
    rather than left to the caller to get right.

    .. warning::
       Ascent on a loss is unbounded below. With a large ``lambda_``, or with no
       task loss of comparable scale to balance it, the objective diverges
       instead of converging. Keep ``lambda_`` small relative to the task loss
       and watch the flagged rows' loss rather than letting it grow without
       limit.

    Parameters
    ----------
    lambda_:
        Suppression strength.
    normalize:
        Divide influence scores by their maximum absolute value, so ``lambda_``
        means the same thing across corpora with different influence scales.

    Examples
    --------
    >>> import torch
    >>> from fairLMs.mitigation import (
    ...     InfluenceGuidedSuppression, InfluenceScoredCorpus)
    >>> corpus = InfluenceScoredCorpus(
    ...     n_examples=4, flagged=[1, 3], influence=[2.0, 1.0],
    ...     source="doctest-precomputed",
    ... )
    >>> term = InfluenceGuidedSuppression().apply(None, corpus).result
    >>> losses = torch.tensor([1.0, 1.0, 1.0, 1.0])
    >>> float(term(losses, [0, 1, 2, 3]))   # -(1.0 for row 1 + 0.5 for row 3)
    -1.5
    """

    name = "influence_guided_suppression"
    category = "in"
    access = "white_box"
    architectures = ("decoder_only",)
    # Empty by design: this mitigator builds a loss term and reads nothing
    # from a model itself. The white-box demand of in-processing - the
    # training loop needs parameters and gradients - is carried by `access`,
    # which is checked whenever a model is supplied. Declaring capabilities
    # here would instead force a model to be passed just to build a loss.
    requires = frozenset()
    accepts = (InfluenceScoredCorpus,)

    def __init__(self, lambda_: float = 1.0, normalize: bool = True):
        self.lambda_ = lambda_
        self.normalize = normalize

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        import torch

        weights = [abs(value) for value in evidence.influence]
        largest = max(weights) if weights else 0.0
        if self.normalize and largest > 0:
            weights = [w / largest for w in weights]
        lookup = dict(zip(evidence.flagged, weights))
        lambda_ = self.lambda_
        n_examples = evidence.n_examples

        def component(per_example_loss: Any, indices: Sequence[int]) -> Any:
            """Influence-weighted ascent term over the flagged rows of a batch."""
            if len(indices) != per_example_loss.shape[0]:
                raise ValueError(
                    f"{InfluenceGuidedSuppression.name}: got {len(indices)} indices "
                    f"for {per_example_loss.shape[0]} per-example losses."
                )
            out_of_range = [i for i in indices if not 0 <= i < n_examples]
            if out_of_range:
                raise ValueError(
                    f"{InfluenceGuidedSuppression.name}: indices "
                    f"{out_of_range[:5]!r} are outside the corpus range "
                    f"[0, {n_examples})."
                )
            scale = torch.tensor(
                [lookup.get(int(i), 0.0) for i in indices],
                dtype=per_example_loss.dtype,
                device=per_example_loss.device,
            )
            # Negative: added to the task loss this ascends the flagged rows'
            # loss. A positive sign would up-weight them and train the model to
            # fit the harmful examples, which is the opposite of suppression.
            return -lambda_ * (scale * per_example_loss).sum()

        return self._result(
            component,
            n_examples=evidence.n_examples,
            n_flagged=len(evidence.flagged),
            lambda_=lambda_,
            normalized=bool(self.normalize),
            influence_source=evidence.source,
            direction=(
                "negative term; added to the task loss it ascends the flagged "
                "rows' loss, suppressing them rather than fitting them"
            ),
            scope=(
                "objective only; influence scores are precomputed by the caller, "
                "the estimation half of IF-Guide is out of scope"
            ),
            returns="loss component; the caller owns the training loop",
        )
