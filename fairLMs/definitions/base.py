"""Base contracts for sklearn-style fairness metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import functools
import json
import math

from fairLMs.definitions.core.params import ParameterizedComponent


@dataclass
class MetricResult:
    """Standard return type for every :meth:`FairnessMetric.compute` call."""

    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    by_category: Optional[Dict[str, Any]] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.details.get(
            "status", "ready" if math.isfinite(self.score) else "undefined"
        )

    def to_dict(self) -> dict:
        from fairLMs.definitions.core.provenance import json_safe

        return json_safe(
            {
                "score": self.score,
                "status": self.status,
                "details": self.details,
                "by_category": self.by_category,
                "provenance": self.provenance,
            }
        )

    def to_json(self, *, indent=2) -> str:
        return json.dumps(
            self.to_dict(), allow_nan=False, sort_keys=True, indent=indent
        )

    def __float__(self) -> float:
        return float(self.score)

    def __repr__(self) -> str:
        parts = [f"score={self.score!r}"]
        if self.by_category is not None:
            parts.append(f"by_category={self.by_category!r}")
        if self.details:
            parts.append(f"details_keys={list(self.details)}")
        return f"MetricResult({', '.join(parts)})"


class FairnessMetric(ParameterizedComponent, ABC):
    """Abstract base class: every metric exposes ``compute``.

    Follows scikit-learn's estimator conventions, inherited from
    :class:`~fairLMs.definitions.core.params.ParameterizedComponent`: ``__init__`` takes
    configuration only and stores it verbatim, data goes to :meth:`compute`, and
    ``get_params`` / ``set_params`` come for free.

    A metric also **declares what it needs** - architectures, capabilities and
    evidence containers - so an unsatisfiable pairing is refused by name rather
    than failing deep inside a forward pass. See :mod:`fairLMs.definitions.core.applicability`.
    """

    def __init_subclass__(cls, **kwargs):
        """Attach metadata to successful public calls, preserving signatures."""
        super().__init_subclass__(**kwargs)
        compute = cls.__dict__.get("compute")
        if compute is None or getattr(compute, "__isabstractmethod__", False):
            return

        @functools.wraps(compute)
        def recorded(self, *args, **kw):
            result = compute(self, *args, **kw)
            if isinstance(result, MetricResult):
                from fairLMs._version import __version__
                from fairLMs.definitions.core.provenance import (
                    evidence_provenance,
                    json_safe,
                    model_provenance,
                )

                model = args[0] if args else kw.get("model")
                data = args[1] if len(args) > 1 else kw.get("data", kw.get("dataset"))
                result.provenance.update(
                    {
                        "library_version": __version__,
                        "metric": self.name,
                        "configuration": json_safe(self.get_params()),
                        "model": model_provenance(model),
                        "evidence": evidence_provenance(data),
                        "scoring_protocol": {
                            "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
                            "version": __version__,
                            "definition": result.details.get("scoring_protocol"),
                            "aggregation": result.details.get("aggregation"),
                        },
                    }
                )
                if "completion_model" in result.details:
                    result.provenance["model"]["effective_model"] = result.details[
                        "completion_model"
                    ]
                overrides = {k: v for k, v in kw.items() if k in self.get_params()}
                if overrides:
                    result.provenance["compute_overrides"] = json_safe(overrides)
            return result

        cls.compute = recorded

    name: str = "metric"
    bias_type: str = ""  # "intrinsic" | "extrinsic"
    architectures: Tuple[str, ...] = ()

    #: Capabilities the model must expose, from
    #: :data:`fairLMs.definitions.core.applicability.CAPABILITIES`. Empty means the metric reads
    #: no model output: it scores precomputed predictions.
    requires: frozenset = frozenset()

    #: Evidence container types :meth:`compute` consumes as ``data``. Empty
    #: means no declared constraint.
    accepts: Tuple[type, ...] = ()

    #: The ``task`` a Hugging Face checkpoint must be loaded with for this
    #: metric to read the quantity it is defined on: one of ``mlm``,
    #: ``encoder``, ``sequence_classification``, ``seq2seq``, ``causal``.
    #: Checked by :func:`fairLMs.definitions.resolve.check_task` when the metric
    #: resolves a model. ``None`` means the metric imposes no requirement:
    #: it scores precomputed predictions, calls an API, or accepts any head.
    required_task: Optional[str] = None

    @abstractmethod
    def compute(
        self,
        model: Any = None,
        data: Any = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Evaluate this metric.

        Parameters
        ----------
        model:
            A :class:`~fairLMs.definitions.models.HuggingFaceModel`,
            :class:`~fairLMs.definitions.models.LoadedModel`,
            :class:`~fairLMs.definitions.models.OpenAIModel`, or (for metrics that score
            precomputed predictions) ``None``.
        data:
            The inputs this metric scores. Metrics that consume a corpus accept
            a :class:`~fairLMs.datasets.FairnessDataset` or sequence of
            examples; metrics that need structured word/context sets accept a
            typed container from :mod:`fairLMs.definitions.data`.
        """
