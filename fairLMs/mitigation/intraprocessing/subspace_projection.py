"""Subspace projection: estimate a bias basis from paired evidence, project it out."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from fairLMs.metrics.data import GroupWordPairs, PromptPairs
from fairLMs.mitigation.base import MitigationResult, Mitigator
from fairLMs.mitigation.evidence import AttributeLabeledVectors
from fairLMs.mitigation.intraprocessing._projection import (
    ProjectedModelAdapter,
    _declared_layer,
    _nullspace_projection,
)

__all__ = ["SubspaceProjection"]


class SubspaceProjection(Mitigator):
    """Estimate a bias basis from paired evidence, then project it out.

    The basis is the top ``n_components`` uncentered singular directions of the
    within-pair difference vectors, and the returned adapter applies
    ``P_perp = I - B (B^T B)^-1 B^T`` to hidden states.

    Keeps its own name rather than the book's "projection-based debiasing",
    which also covers INLP - a separate component here.

    Parameters
    ----------
    n_components:
        Size of the estimated bias subspace.
    encode:
        Callable ``(model, texts) -> array`` producing representations. Required
        when the evidence is textual pairs; unused when vectors are supplied.

    Examples
    --------
    >>> import numpy as np
    >>> from fairLMs.applicability import AccessLevel, ModelProfile
    >>> from fairLMs.mitigation import AttributeLabeledVectors, SubspaceProjection
    >>> from fairLMs.models.base import ModelAdapter
    >>> class Stub(ModelAdapter):
    ...     name = "stub"
    ...     task = "encoder"
    ...     def load(self): raise NotImplementedError
    >>> evidence = AttributeLabeledVectors(
    ...     axis="gender",
    ...     vectors=[[1.0, 0.0], [2.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
    ...     labels=["f", "f", "m", "m"], source="doctest",
    ...     representation_layer="input_embeddings", pooling="token",
    ...     pair_ids=["p1", "p2", "p1", "p2"],
    ... )
    >>> outcome = SubspaceProjection().apply(Stub(), evidence)
    >>> outcome.category
    'intra'
    >>> from fairLMs.models.base import ModelAdapter
    >>> isinstance(outcome.result, ModelAdapter)      # re-usable by any metric
    True
    >>> np.round(outcome.result.projection, 6)        # first axis removed
    array([[0., 0.],
           [0., 1.]])
    """

    name = "subspace_projection"
    category = "intra"
    access = "gray_box"
    architectures = ("encoder_only", "encoder_decoder")
    requires = frozenset({"hidden_states"})
    accepts = (PromptPairs, GroupWordPairs, AttributeLabeledVectors)

    def __init__(
        self, n_components: int = 1, encode: Any = None, layer: Optional[str] = None
    ):
        self.n_components = n_components
        self.encode = encode
        self.layer = layer

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if self.n_components < 1:
            raise ValueError(
                f"n_components must be at least 1; got {self.n_components!r}."
            )
        layer = _declared_layer(self, evidence)
        left, right = _pair_vectors(self, model, evidence)
        differences = left - right
        if differences.shape[0] < self.n_components:
            raise ValueError(
                f"{self.name}: {differences.shape[0]} pair(s) cannot support an "
                f"{self.n_components}-dimensional bias subspace."
            )

        # Within-pair centering yields +difference/2 and -difference/2.
        # SVD of raw differences has that same span; centering the difference
        # rows again would incorrectly erase a shared protected direction.
        _, singular, right_vectors = np.linalg.svd(differences, full_matrices=False)
        rank = np.linalg.matrix_rank(differences)
        if rank < self.n_components:
            raise ValueError(
                f"Paired differences have rank {rank}; cannot estimate {self.n_components} components."
            )
        basis = right_vectors[: self.n_components].T
        projection = _nullspace_projection(basis)

        total = float((singular**2).sum())
        explained = (
            [float(s**2 / total) for s in singular[: self.n_components]]
            if total > 0
            else [0.0] * self.n_components
        )
        adapter = ProjectedModelAdapter(
            model,
            projection,
            method=self.name,
            axis=getattr(evidence, "axis", "declared-pairs"),
            probe_family="linear (SVD of within-pair differences)",
            layer=layer,
        )
        return self._result(
            adapter,
            representation_layer=layer,
            pooling=getattr(evidence, "pooling", None),
            n_components=self.n_components,
            n_pairs=int(differences.shape[0]),
            n_features=int(differences.shape[1]),
            explained_variance_ratio=explained,
            probe_family="linear (PCA of paired differences)",
            removal_claim=(
                "Removes the subspace a linear probe of this family reads. Not "
                "evidence of removal in absolute terms: projection can hide "
                "rather than remove bias (Gonen and Goldberg, 2019)."
            ),
        )


def _pair_vectors(mitigator: Mitigator, model: Any, evidence: Any):
    """Return aligned ``(left, right)`` representation arrays from paired evidence."""
    if isinstance(evidence, AttributeLabeledVectors):
        classes = sorted(set(evidence.labels))
        if len(classes) != 2:
            raise ValueError(
                f"{mitigator.name}: paired estimation needs exactly two attribute "
                f"values; got {classes!r}."
            )
        vectors = np.asarray(evidence.vectors, dtype=float)
        if evidence.pair_ids is None:
            raise ValueError(
                "SubspaceProjection needs explicit pair_ids for attribute-labeled vectors; group row order is not pairing evidence."
            )
        ids = list(dict.fromkeys(evidence.pair_ids))
        left, right = [], []
        for pair_id in ids:
            rows = [i for i, value in enumerate(evidence.pair_ids) if value == pair_id]
            if len(rows) != 2 or {evidence.labels[i] for i in rows} != set(classes):
                raise ValueError(
                    "Each pair_id must identify exactly two rows, one from each attribute value."
                )
            left.append(
                vectors[next(i for i in rows if evidence.labels[i] == classes[0])]
            )
            right.append(
                vectors[next(i for i in rows if evidence.labels[i] == classes[1])]
            )
        return np.asarray(left), np.asarray(right)

    if isinstance(evidence, PromptPairs):
        left_texts, right_texts = list(evidence.factual), list(evidence.counterfactual)
    else:  # GroupWordPairs
        left_texts, right_texts = list(evidence.group_1), list(evidence.group_2)

    if mitigator.encode is None:
        raise ValueError(
            f"{mitigator.name}: textual pair evidence needs an encoder. Pass "
            f"encode=(model, texts) -> array, or supply "
            f"AttributeLabeledVectors with representations already computed."
        )
    left = np.asarray(mitigator.encode(model, left_texts), dtype=float)
    right = np.asarray(mitigator.encode(model, right_texts), dtype=float)
    if left.ndim != 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("encode must return finite 2-D representation arrays.")
    if left.shape != right.shape:
        raise ValueError(
            f"{mitigator.name}: encode returned mismatched shapes {left.shape} and "
            f"{right.shape} for the two sides of the pair set."
        )
    return left, right
