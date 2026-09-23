"""Adversarial debiasing: gradient-reversal adversary on the protected attribute."""

from __future__ import annotations

from typing import Any, Sequence

from fairLMs.mitigation.base import MitigationResult, Mitigator
from fairLMs.mitigation.evidence import AttributeLabeledVectors, GroupLabeledRecords

__all__ = ["AdversarialDebiasing"]


class AdversarialDebiasing(Mitigator):
    """Gradient-reversal adversary predicting the protected attribute.

    Elazar and Goldberg (2018). The adversary is trained to read the protected
    attribute out of pooled hidden states; a gradient-reversal layer means the
    encoder is simultaneously trained to make that impossible.

    Returns ``(pooled, attribute_targets) -> loss``, already gradient-reversed,
    so adding it to the task loss trains the encoder adversarially. The adversary
    module is reachable as ``result.adversary`` for a separate optimizer, which
    is the usual way this is trained.

    Parameters
    ----------
    lambda_:
        Gradient-reversal strength.
    hidden_size:
        Width of the adversary's hidden layer.
    seed:
        Seed for adversary initialization.

    Examples
    --------
    >>> import torch
    >>> from fairLMs.mitigation import AdversarialDebiasing, AttributeLabeledVectors
    >>> evidence = AttributeLabeledVectors(
    ...     axis="gender",
    ...     vectors=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
    ...     labels=["f", "f", "m", "m"], source="doctest",
    ... )
    >>> loss_term = AdversarialDebiasing().apply(None, evidence).result
    >>> pooled = torch.tensor(evidence.vectors, dtype=torch.float32)
    >>> float(loss_term(pooled, ["f", "f", "m", "m"])) > 0
    True
    """

    name = "adversarial_debiasing"
    category = "in"
    access = "white_box"
    architectures = ("encoder_only", "encoder_decoder")
    # Empty by design: this mitigator builds a loss term and reads nothing
    # from a model itself. The white-box demand of in-processing - the
    # training loop needs parameters and gradients - is carried by `access`,
    # which is checked whenever a model is supplied. Declaring capabilities
    # here would instead force a model to be passed just to build a loss.
    requires = frozenset()
    accepts = (AttributeLabeledVectors, GroupLabeledRecords)

    def __init__(self, lambda_: float = 1.0, hidden_size: int = 128, seed: int = 0):
        self.lambda_ = lambda_
        self.hidden_size = hidden_size
        self.seed = seed

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        import torch

        classes = sorted(set(evidence.labels))
        if len(classes) < 2:
            raise ValueError(
                f"{self.name}: the adversary needs at least two attribute values; "
                f"got {classes!r}."
            )
        n_features = (
            evidence.n_features
            if isinstance(evidence, AttributeLabeledVectors)
            else None
        )
        if n_features is None:
            raise ValueError(
                f"{self.name}: the adversary's input width must be known. Supply "
                f"AttributeLabeledVectors so n_features is declared."
            )

        # Seed the adversary's initialization without disturbing the caller's
        # RNG. A bare `torch.manual_seed` here reseeds the *global* generator,
        # and the caller owns the training loop: their shuffling, dropout and
        # parameter init must not shift because a loss term was constructed.
        # fork_rng saves and restores the global state, so the init stays
        # reproducible from `seed` while the side effect is contained.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            adversary = torch.nn.Sequential(
                torch.nn.Linear(n_features, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_size, len(classes)),
            )
        index = {label: i for i, label in enumerate(classes)}
        lambda_ = self.lambda_

        class _Reverse(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.view_as(x)

            @staticmethod
            def backward(ctx, grad):
                return -lambda_ * grad

        def component(pooled: Any, attributes: Sequence[str]) -> Any:
            """Adversarial attribute-prediction loss, gradient-reversed."""
            unknown = sorted(set(attributes) - set(index))
            if unknown:
                raise ValueError(
                    f"{AdversarialDebiasing.name}: unseen attribute value(s) "
                    f"{unknown!r}; the adversary was built for {classes!r}."
                )
            if len(attributes) != pooled.shape[0]:
                raise ValueError(
                    f"{AdversarialDebiasing.name}: got {len(attributes)} attribute "
                    f"labels for {pooled.shape[0]} pooled rows."
                )
            targets = torch.tensor([index[a] for a in attributes], device=pooled.device)
            logits = adversary(_Reverse.apply(pooled))
            return torch.nn.functional.cross_entropy(logits, targets)

        component.adversary = adversary
        component.classes = tuple(classes)
        return self._result(
            component,
            axis=evidence.axis,
            attribute_values=list(classes),
            n_features=n_features,
            lambda_=self.lambda_,
            returns="loss component; the caller owns the training loop",
        )
