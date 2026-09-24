"""Projection machinery shared by the two projection-based intra-processing methods.

:class:`~fairLMs.mitigation.intraprocessing.subspace_projection.SubspaceProjection`
and
:class:`~fairLMs.mitigation.intraprocessing.iterative_nullspace_projection.IterativeNullspaceProjection`
differ in how they *estimate* a direction to remove; everything about applying
one - resolving the intervention site, building the orthogonal projector, and
wrapping the result as a :class:`~fairLMs.definitions.models.base.ModelAdapter` - is common
and lives here.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from fairLMs.definitions.models.base import LoadedModel, ModelAdapter

__all__ = [
    "ProjectedModelAdapter",
    "_declared_layer",
    "_embedding_module",
    "_nullspace_projection",
]


class ProjectedModelAdapter(ModelAdapter):
    """A :class:`ModelAdapter` that applies a projection to hidden states.

    Installs a forward hook at an explicitly named representation layer.
    The base and edited adapters share the loaded model: while the hook is
    installed, callers of the base see the projection too. Use ``remove()``
    before a baseline; ``compare_before_after`` manages the hook lifecycle.
    Do not use a shared base concurrently while the hook is active.
    """

    def __init__(
        self,
        base: ModelAdapter,
        projection: np.ndarray,
        *,
        method: str,
        axis: str,
        probe_family: str,
        layer: str,
    ):
        if not isinstance(layer, str) or not layer.strip():
            raise ValueError("A named representation layer is required.")
        self.layer = layer
        self.base = base
        self.projection = np.array(projection, dtype=float, copy=True)
        if (
            self.projection.ndim != 2
            or self.projection.shape[0] != self.projection.shape[1]
            or not np.isfinite(self.projection).all()
        ):
            raise ValueError("projection must be a finite square matrix.")
        self.method = method
        self.axis = axis
        self.probe_family = probe_family
        self.name = f"{getattr(base, 'name', type(base).__name__)}+{method}"
        # Mirror the tag the applicability matcher reads, so the mitigated model
        # profiles exactly as the model it wraps.
        self.task = getattr(base, "task", None)
        self._loaded: Optional[LoadedModel] = None

    def load(self) -> LoadedModel:
        if self._loaded is not None:
            return self._loaded

        import torch

        loaded = self.base.load()
        matrix = torch.tensor(self.projection, dtype=torch.float32)

        def project_tensor(tensor):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
                raise TypeError(
                    f"Layer {self.layer!r} does not expose token/row representations."
                )
            if tensor.shape[-1] != matrix.shape[0]:
                raise ValueError(
                    f"Projection width {matrix.shape[0]} does not match layer {self.layer!r} width {tensor.shape[-1]}."
                )
            return tensor @ matrix.to(device=tensor.device, dtype=tensor.dtype)

        def project(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                return project_tensor(output)
            if isinstance(output, tuple) and output:
                return (project_tensor(output[0]),) + tuple(output[1:])
            if isinstance(output, dict) and "last_hidden_state" in output:
                import copy

                result = copy.copy(output)
                result["last_hidden_state"] = project_tensor(
                    output["last_hidden_state"]
                )
                if output.get("hidden_states") is not None:
                    result["hidden_states"] = tuple(output["hidden_states"][:-1]) + (
                        result["last_hidden_state"],
                    )
                return result
            raise TypeError(
                f"Layer {self.layer!r} returned an unsupported representation type {type(output).__name__}."
            )

        module = _embedding_module(loaded.model, self.layer)
        weight = getattr(module, "weight", None)
        if (
            isinstance(weight, torch.Tensor)
            and weight.ndim == 2
            and self.layer == "input_embeddings"
            and weight.shape[1] != matrix.shape[0]
        ):
            raise ValueError(
                "Projection dimension does not match the declared input_embeddings layer."
            )
        self._handle = module.register_forward_hook(project)
        self._loaded = LoadedModel(
            name=self.name,
            tokenizer=loaded.tokenizer,
            model=loaded.model,
            device=loaded.device,
            task=loaded.task,
        )
        return self._loaded

    def remove(self) -> None:
        """Detach the projection hook, restoring the underlying model."""
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.remove()
            self._handle = None
        self._loaded = None

    def __repr__(self) -> str:
        return (
            f"ProjectedModelAdapter(base={self.base!r}, method={self.method!r}, "
            f"axis={self.axis!r})"
        )


def _embedding_module(model: Any, layer: str):
    """Resolve the explicitly declared representation intervention site."""
    if layer == "input_embeddings":
        getter = getattr(model, "get_input_embeddings", None)
        module = getter() if callable(getter) else None
    else:
        try:
            module = model.get_submodule(layer)
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"Model has no representation layer {layer!r}.") from exc
    if module is None:
        raise TypeError(f"Cannot locate representation layer {layer!r}.")
    return module


def _declared_layer(mitigator, evidence):
    supplied = getattr(evidence, "representation_layer", None)
    configured = mitigator.layer
    if supplied is not None and configured is not None and supplied != configured:
        raise ValueError(
            "Evidence representation_layer differs from the requested intervention layer."
        )
    layer = configured or supplied
    if not isinstance(layer, str) or not layer.strip():
        raise ValueError(
            "Declare the representation layer via layer= or evidence.representation_layer; it is never inferred from vector width."
        )
    return layer


def _nullspace_projection(basis: np.ndarray) -> np.ndarray:
    r"""Return ``P_perp = I - B (B^T B)^-1 B^T`` for column basis ``B``.

    Uses the pseudo-inverse, so a rank-deficient or near-collinear basis yields
    the projection onto the span actually spanned instead of raising on a
    singular Gram matrix.
    """
    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2:
        raise ValueError(f"basis must be 2-D (n_features, k); got {basis.shape}.")
    identity = np.eye(basis.shape[0])
    return identity - basis @ np.linalg.pinv(basis.T @ basis) @ basis.T
