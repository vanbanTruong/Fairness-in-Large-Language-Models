"""In-processing mitigators: white-box loss components.

Every mitigator here returns a **callable loss component, not a trained model**.
There is no trainer in the core package: the caller owns the optimizer, the
schedule and the data loader, and composes the returned term into their own
objective. This keeps the library out of the business of reimplementing a
training loop that every user already has.

The returned callables take and return ``torch`` tensors and are differentiable,
so ``loss = task_loss + lambda * component(...)`` works directly. ``torch`` is
imported lazily inside each callable, so importing this module stays cheap.

One mitigator per module:

====================================  ======================================
``adversarial_debiasing``             :mod:`~.adversarial_debiasing`
``counterfactual_invariance_loss``    :mod:`~.counterfactual_invariance_loss`
``group_regularized_objective``       :mod:`~.group_regularized_objective`
``influence_guided_suppression``      :mod:`~.influence_guided_suppression`
====================================  ======================================
"""

from __future__ import annotations

from .adversarial_debiasing import AdversarialDebiasing
from .counterfactual_invariance_loss import CounterfactualInvarianceLoss
from .group_regularized_objective import GroupRegularizedObjective
from .influence_guided_suppression import InfluenceGuidedSuppression

__all__ = [
    "AdversarialDebiasing",
    "CounterfactualInvarianceLoss",
    "GroupRegularizedObjective",
    "InfluenceGuidedSuppression",
]
