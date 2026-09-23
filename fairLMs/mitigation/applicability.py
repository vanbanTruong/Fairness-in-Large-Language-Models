"""The shared applicability matcher, re-exported for the mitigation layer.

The implementation lives at :mod:`fairLMs.applicability`, at the package root,
because :mod:`fairLMs.metrics` matches through the *same* matcher and must not
import the mitigation layer. This module exists so that the mitigation package
presents the matcher alongside the components that declare into it.
"""

from fairLMs.applicability import (
    ACCESS_LEVELS,
    ARCHITECTURES,
    CAPABILITIES,
    TASK_PROFILES,
    AccessLevel,
    ApplicabilityError,
    ModelProfile,
    access_rank,
    check_applicability,
    describe_model,
    validate_declaration,
)

__all__ = [
    "ACCESS_LEVELS",
    "ARCHITECTURES",
    "CAPABILITIES",
    "TASK_PROFILES",
    "AccessLevel",
    "ApplicabilityError",
    "ModelProfile",
    "access_rank",
    "check_applicability",
    "describe_model",
    "validate_declaration",
]
