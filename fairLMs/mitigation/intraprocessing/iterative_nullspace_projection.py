"""INLP: iteratively fit a linear attribute probe and project onto its nullspace."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from fairLMs.mitigation.base import MitigationResult, Mitigator
from fairLMs.mitigation.evidence import AttributeLabeledVectors
from fairLMs.mitigation.intraprocessing._projection import (
    ProjectedModelAdapter,
    _declared_layer,
    _nullspace_projection,
)

__all__ = ["IterativeNullspaceProjection"]


class IterativeNullspaceProjection(Mitigator):
    """Iteratively fit a linear attribute probe and project onto its nullspace.

    INLP (Ravfogel et al., 2020). Each round fits a linear classifier for the
    protected attribute, projects the representation onto that classifier's
    nullspace, and repeats until the probe is at chance or the iteration budget
    is spent. The composed projection is applied by the returned adapter.

    Stops early when probe accuracy reaches chance, and **reports the accuracy
    it actually reached**: claiming removal without the probe curve would be
    exactly the overstatement the literature warns about.

    Parameters
    ----------
    n_iterations:
        Maximum probe/project rounds.
    tolerance:
        Stop once probe accuracy is within this of the majority-class rate.
    seed:
        Seed for the probe's optimizer.

    Examples
    --------
    >>> from fairLMs.mitigation import (
    ...     AttributeLabeledVectors, IterativeNullspaceProjection)
    >>> from fairLMs.definitions.models.base import ModelAdapter
    >>> class Stub(ModelAdapter):
    ...     name = "stub"
    ...     task = "encoder"
    ...     def load(self): raise NotImplementedError
    >>> evidence = AttributeLabeledVectors(
    ...     axis="gender",
    ...     vectors=[[2.0, 0.1], [1.8, -0.1], [-2.0, 0.1], [-1.9, -0.2]],
    ...     labels=["f", "f", "m", "m"], source="doctest",
    ...     representation_layer="input_embeddings", pooling="token",
    ...     pair_ids=["p1", "p2", "p1", "p2"],
    ... )
    >>> outcome = IterativeNullspaceProjection(n_iterations=4).apply(
    ...     Stub(), evidence)
    >>> isinstance(outcome.result, ModelAdapter)
    True
    >>> outcome.provenance["probe_family"]
    'linear (logistic regression)'
    >>> 0.0 <= outcome.provenance["final_probe_accuracy"] <= 1.0
    True
    """

    name = "iterative_nullspace_projection"
    category = "intra"
    access = "gray_box"
    architectures = ("encoder_only", "encoder_decoder")
    requires = frozenset({"hidden_states"})
    accepts = (AttributeLabeledVectors,)

    def __init__(
        self,
        n_iterations: int = 8,
        tolerance: float = 0.02,
        seed: int = 0,
        layer: Optional[str] = None,
        validation_fraction: float = 0.25,
    ):
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.seed = seed
        self.layer = layer
        self.validation_fraction = validation_fraction

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if (
            isinstance(self.n_iterations, bool)
            or not isinstance(self.n_iterations, int)
            or self.n_iterations < 1
        ):
            raise ValueError(
                f"n_iterations must be at least 1; got {self.n_iterations!r}."
            )
        vectors = np.array(evidence.vectors, dtype=float)
        classes = sorted(set(evidence.labels))
        if len(classes) != 2:
            raise ValueError(
                f"{self.name}: INLP fits a binary attribute probe; got "
                f"{len(classes)} attribute values {classes!r}."
            )
        targets = np.array(
            [1.0 if label == classes[1] else 0.0 for label in evidence.labels]
        )
        from sklearn.model_selection import train_test_split

        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one.")
        if not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative.")
        if min(sum(targets == value) for value in (0.0, 1.0)) < 2:
            raise ValueError(
                "INLP needs at least two rows per group for held-out probe evaluation."
            )
        layer = _declared_layer(self, evidence)
        validation_size = max(2, int(np.ceil(len(targets) * self.validation_fraction)))
        if validation_size > len(targets) - 2:
            raise ValueError(
                "validation_fraction leaves insufficient probe training rows."
            )
        train, validation = train_test_split(
            np.arange(len(targets)),
            test_size=validation_size,
            stratify=targets,
            random_state=self.seed,
        )
        chance = float(
            max((targets[validation] == 0).mean(), (targets[validation] == 1).mean())
        )
        n_features = vectors.shape[1]
        composed = np.eye(n_features)
        directions, history = [], []
        for _ in range(self.n_iterations):
            current = vectors @ composed
            weights = _fit_linear_probe(current[train], targets[train], seed=self.seed)
            accuracy = _probe_accuracy(
                current[validation], targets[validation], weights
            )
            history.append(accuracy)
            if accuracy <= chance + self.tolerance:
                break
            direction = composed @ weights
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                break
            directions.append(direction / norm)
            # One orthogonal projection onto the intersection of nullspaces.
            composed = _nullspace_projection(np.stack(directions, axis=1))

        final_vectors = vectors @ composed
        final_weights = _fit_linear_probe(
            final_vectors[train], targets[train], seed=self.seed
        )
        final_accuracy = _probe_accuracy(
            final_vectors[validation], targets[validation], final_weights
        )
        adapter = ProjectedModelAdapter(
            model,
            composed,
            method=self.name,
            axis=evidence.axis,
            probe_family="linear (logistic regression)",
            layer=layer,
        )
        return self._result(
            adapter,
            axis=evidence.axis,
            representation_layer=layer,
            pooling=getattr(evidence, "pooling", None),
            n_iterations_run=len(directions),
            n_iterations_max=self.n_iterations,
            probe_accuracy_history=[float(a) for a in history],
            final_probe_accuracy=float(final_accuracy),
            probe_evaluation="held-out validation; used for stopping, not an independent test set",
            n_probe_train=len(train),
            n_probe_validation=len(validation),
            majority_class_rate=chance,
            reached_chance=bool(final_accuracy <= chance + self.tolerance),
            probe_family="linear (logistic regression)",
            removal_claim=(
                "Probe performance on the final projected representation is reported; "
                "this is not evidence of absolute bias removal (Gonen and Goldberg, 2019)."
            ),
        )


def _fit_linear_probe(
    features: np.ndarray, targets: np.ndarray, *, seed: int, max_iter: int = 2000
) -> np.ndarray:
    """Fit a converged no-intercept logistic attribute probe.

    INLP is only as strong as this probe. A direction is projected out *because*
    a probe could read the attribute along it, so an **underfit** probe makes
    the method quietly do nothing while reporting chance accuracy - the exact
    overstatement the module warning is about, in the worst direction. A fixed
    gradient-descent schedule underfits whenever the representation scale is
    small: an attribute that is perfectly separable at unit scale becomes
    invisible once the informative feature is small next to the others. The
    probe is therefore solved properly rather than stepped a fixed number of
    times.

    ``C`` is large, so the fit is all but unregularized: the direction is set by
    the data geometry rather than by a shrinkage prior, which is what makes it
    insensitive to the overall magnitude of the layer. The features are used
    **as given**, on purpose. Per-feature rescaling looks like the scale-robust
    choice and is wrong here: hidden-state dimensions are commensurable, all
    living in the one space the projection acts on, so dividing each by its own
    spread inflates a low-variance noise dimension into an apparent signal and
    the recovered direction points at noise. Euclidean geometry in this space is
    the thing being relied on, so it is left intact.

    Only the direction is used; the caller normalizes it before projecting.

    Held-out accuracy remains a diagnostic, not a fairness certificate. A
    genuinely weak signal - an informative dimension buried orders of magnitude
    below the others - still reads at chance on held-out rows, and stopping
    without projecting is then the right answer rather than a failure to fit.
    """
    from sklearn.linear_model import LogisticRegression

    features = np.asarray(features, dtype=float)
    labels = (np.asarray(targets, dtype=float) > 0.5).astype(int)
    probe = LogisticRegression(
        fit_intercept=False,
        C=1e4,
        max_iter=max_iter,
        random_state=seed,
    )
    probe.fit(features, labels)
    return np.asarray(probe.coef_[0], dtype=float)


def _probe_accuracy(
    features: np.ndarray, targets: np.ndarray, weights: np.ndarray
) -> float:
    predictions = (features @ weights) > 0
    return float((predictions == (targets > 0.5)).mean())
