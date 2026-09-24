"""Intra-processing mitigators: gray-box edits to a loaded model.

Every mitigator here returns a :class:`~fairLMs.definitions.models.base.ModelAdapter`, so supported metrics can evaluate the edited behavior through the same API.
Capabilities describe the edited outputs, not capabilities of the base model.

.. warning::
   **Report removal relative to the probe family used.** A projection removes
   the attribute a *linear* probe could read. Gonen and Goldberg showed that
   this can hide bias rather than remove it: the geometry survives in clusters a
   linear probe no longer detects. Nothing here should be described as removing
   bias in absolute terms, and the provenance of every result records the probe
   family so the claim stays attached to its evidence.

One mitigator per module, with the projection machinery the two projection-based
methods share in :mod:`~._projection`:

======================================  ==========================================
``subspace_projection``                 :mod:`~.subspace_projection`
``iterative_nullspace_projection``      :mod:`~.iterative_nullspace_projection`
``self_debiasing``                      :mod:`~.self_debiasing`
======================================  ==========================================
"""

from __future__ import annotations

from ._projection import (
    ProjectedModelAdapter,
    _declared_layer,
    _embedding_module,
    _nullspace_projection,
)
from .iterative_nullspace_projection import (
    IterativeNullspaceProjection,
    _fit_linear_probe,
    _probe_accuracy,
)
from .self_debiasing import SelfDebiasedModelAdapter, SelfDebiasing
from .subspace_projection import SubspaceProjection, _pair_vectors

__all__ = [
    "IterativeNullspaceProjection",
    "ProjectedModelAdapter",
    "SelfDebiasedModelAdapter",
    "SelfDebiasing",
    "SubspaceProjection",
]
