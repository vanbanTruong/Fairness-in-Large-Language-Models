"""Registry of the shipped bias mitigators.

Mirrors :mod:`fairLMs.datasets.diagnostics.registry`. Membership here is the library's
statement that a method is *available*: the extension points named in the paper
but not implemented - RLHF, DPO, Constitutional AI, UniDetox, the
influence-estimation half of IF-Guide, GeDi, RAD, DExperts, FairSteer, ARGRE,
LSDM, FairMed, gender-constrained beam search, model-based rewriting - must
never appear in this mapping.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Type

from .base import Mitigator
from .inprocessing import (
    AdversarialDebiasing,
    CounterfactualInvarianceLoss,
    GroupRegularizedObjective,
    InfluenceGuidedSuppression,
)
from .intraprocessing import (
    IterativeNullspaceProjection,
    SelfDebiasing,
    SubspaceProjection,
)
from .postprocessing import (
    GroupAwareThresholding,
    OutputReranking,
    ScoreCalibration,
)
from .preprocessing import (
    CounterfactualDataAugmentation,
    DebiasingPrompt,
    GroupLabelReweighting,
    IdentityTermAugmentation,
)

_COMPONENTS = (
    # pre (4)
    CounterfactualDataAugmentation,
    GroupLabelReweighting,
    IdentityTermAugmentation,
    DebiasingPrompt,
    # in (4)
    AdversarialDebiasing,
    CounterfactualInvarianceLoss,
    GroupRegularizedObjective,
    InfluenceGuidedSuppression,
    # intra (3)
    SubspaceProjection,
    IterativeNullspaceProjection,
    SelfDebiasing,
    # post (3)
    ScoreCalibration,
    GroupAwareThresholding,
    OutputReranking,
)

MITIGATOR_REGISTRY: Mapping[str, Type[Mitigator]] = MappingProxyType(
    {component.name: component for component in _COMPONENTS}
)


def list_mitigators() -> list[str]:
    """Return the registered mitigator names in deterministic order."""
    return sorted(MITIGATOR_REGISTRY)


def get_mitigator(name: str) -> Mitigator:
    """Instantiate the mitigator registered under *name*."""
    try:
        mitigator_type = MITIGATOR_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown mitigator {name!r}. Available: {', '.join(list_mitigators())}"
        ) from exc
    return mitigator_type()


def list_by_category(category: str) -> list[str]:
    """Return the registered mitigators in one intervention category."""
    from .base import CATEGORIES

    if category not in CATEGORIES:
        raise ValueError(
            f"category must be one of {list(CATEGORIES)}; got {category!r}."
        )
    return sorted(
        name for name, cls in MITIGATOR_REGISTRY.items() if cls.category == category
    )


__all__ = [
    "MITIGATOR_REGISTRY",
    "get_mitigator",
    "list_by_category",
    "list_mitigators",
]
