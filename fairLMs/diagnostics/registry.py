"""Registry helpers for dataset diagnostics."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Type

from .base import DatasetDiagnostic
from .construction import (
    DependencyDepthDisparity,
    FramingDisparity,
    GrammarConsistency,
    LengthDisparity,
    MinimalPairResidual,
    OptionLengthBias,
    SemanticEquivalence,
    TemplateImbalance,
)
from .leakage import StereotypeLeakage
from .representativeness import RepresentativenessBias
from .scoring import (
    ScorerCounterfactualSensitivity,
    ScorerMeanGap,
    ScorerRateGap,
    ScorerWasserstein1Gap,
)

# Every registered diagnostic is constructible with zero arguments, which
# ``get_diagnostic`` requires. Several of them then report ``blocked`` for
# their own missing configuration -- ``b_min`` without an identity mask,
# ``b_opt`` without an option-role contrast, ``b_frame`` without a frame
# predicate, ``b_leak`` on text evidence without an extraction configuration.
# That is the designed outcome, not a defect: the configuration is an
# estimand declaration and is never inferred.
#
# The three backend-dependent construction slots -- ``b_equiv``, ``b_gram``
# and ``b_diff_dep`` -- follow the same rule. Each is a real class that takes
# an optional ``backend`` (an ``EmbeddingBackend``, ``GrammarCheckerBackend``
# or ``DependencyParserBackend``; reference implementations live in
# ``fairLMs.diagnostics.backends``). Without a backend the slot is ``blocked``
# with a reason code naming the missing backend, but only after the shared
# applicability precedence: when the slot was not requested, the target kind
# is unsupported or its evidence view is absent it is ``not_applicable`` for
# that ordinary reason instead. ``BACKEND_CONSTRUCTION_SLOTS`` and
# ``CONSTRUCTION_BACKEND_REQUIREMENTS``, not the reported status, enumerate
# the slots that need a backend.
DIAGNOSTIC_REGISTRY: Mapping[str, Type[DatasetDiagnostic]] = MappingProxyType(
    {
        DependencyDepthDisparity.name: DependencyDepthDisparity,
        FramingDisparity.name: FramingDisparity,
        GrammarConsistency.name: GrammarConsistency,
        LengthDisparity.name: LengthDisparity,
        MinimalPairResidual.name: MinimalPairResidual,
        OptionLengthBias.name: OptionLengthBias,
        RepresentativenessBias.name: RepresentativenessBias,
        ScorerCounterfactualSensitivity.name: ScorerCounterfactualSensitivity,
        ScorerMeanGap.name: ScorerMeanGap,
        ScorerRateGap.name: ScorerRateGap,
        ScorerWasserstein1Gap.name: ScorerWasserstein1Gap,
        SemanticEquivalence.name: SemanticEquivalence,
        StereotypeLeakage.name: StereotypeLeakage,
        TemplateImbalance.name: TemplateImbalance,
    }
)


def list_diagnostics() -> list[str]:
    """Return the registered diagnostic names in deterministic order."""

    return sorted(DIAGNOSTIC_REGISTRY)


def get_diagnostic(name: str) -> DatasetDiagnostic:
    """Instantiate the diagnostic registered under ``name``."""

    try:
        diagnostic_type = DIAGNOSTIC_REGISTRY[name]
    except KeyError as exc:
        alternatives = ", ".join(list_diagnostics())
        raise KeyError(
            f"Unknown diagnostic {name!r}. Available: {alternatives}"
        ) from exc
    return diagnostic_type()


__all__ = ["DIAGNOSTIC_REGISTRY", "get_diagnostic", "list_diagnostics"]
