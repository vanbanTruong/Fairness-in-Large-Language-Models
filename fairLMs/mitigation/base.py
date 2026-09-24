"""Base contracts for bias mitigators.

A mitigator adopts the same contract as a metric: it declares an intervention
category, an access level, supported architectures, capabilities and evidence
containers, and applicability is decided from those declarations by the one
shared matcher in :mod:`fairLMs.definitions.core.applicability`.

The four intervention categories differ in *what they hand back*, which is the
only place the uniform contract bends:

=========  ==================================================================
``pre``    transformed evidence, or per-row weights, plus transform provenance
``in``     a loss/regularizer callable composable with a normal training loop
``intra``  a :class:`~fairLMs.definitions.models.base.ModelAdapter` wrapping the original
``post``   a fitted decision rule: calibrator, thresholds, or reranker
=========  ==================================================================

The ``intra`` row is load-bearing. Because the result *is* an adapter, any of
the registered metrics re-evaluates the mitigated model with no modification.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from fairLMs.definitions.core.applicability import AccessLevel, check_applicability
from fairLMs.datasets.diagnostics._utils import (
    freeze_json_mapping,
    require_nonempty_string,
    thaw_json,
)
from fairLMs.definitions.models.base import ModelAdapter
from fairLMs.definitions.core.params import ParameterizedComponent

__all__ = [
    "CATEGORIES",
    "MITIGATION_SCHEMA_VERSION",
    "AccessLevel",
    "Mitigator",
    "MitigationResult",
]

MITIGATION_SCHEMA_VERSION = "1.0"

#: Intervention categories, in the order the book chapter presents them.
CATEGORIES: Tuple[str, ...] = ("pre", "in", "intra", "post")


@dataclass(frozen=True, kw_only=True)
class MitigationResult:
    """What a mitigator returns: a payload, tagged by category, plus provenance.

    Deliberately **not** float-convertible. Unlike
    :class:`~fairLMs.definitions.MetricResult` this is not a scalar, and giving it a
    ``__float__`` would invite it to be averaged into a summary number that has
    no defensible meaning.

    ``result`` is validated against ``category`` so that the guarantee each
    category makes is actually enforced rather than merely documented.
    """

    mitigator: str
    category: str
    result: Any
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mitigator", require_nonempty_string(self.mitigator, "mitigator")
        )
        if self.category not in CATEGORIES:
            raise ValueError(
                f"category must be one of {list(CATEGORIES)}; got {self.category!r}."
            )
        if self.result is None:
            raise ValueError(
                f"{self.mitigator}: result must not be None. A mitigator that "
                f"cannot produce a payload must raise, not return an empty one."
            )
        if self.category == "intra" and not isinstance(self.result, ModelAdapter):
            raise TypeError(
                f"{self.mitigator}: an intra-processing result must be a "
                f"ModelAdapter so that any metric re-evaluates the mitigated "
                f"model unchanged; got {type(self.result).__name__}."
            )
        if self.category == "in" and not callable(self.result):
            raise TypeError(
                f"{self.mitigator}: an in-processing result must be a callable "
                f"loss component composable with a training loop; got "
                f"{type(self.result).__name__}."
            )
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, path="provenance"),
        )

    @property
    def is_serializable(self) -> bool:
        """Whether ``result`` survives :meth:`to_dict` as data rather than a stub."""
        return _payload(self.result)[1]

    def to_dict(self) -> dict:
        """Return a JSON-safe report.

        Everything except the live payload always round-trips. A payload that is
        a model or a closure cannot be JSON, and is recorded as a typed stub
        rather than stringified: a report that quietly contained
        ``"<function _loss at 0x7f...>"`` would look like data and be worthless.
        """
        payload, serializable = _payload(self.result)
        return {
            "schema_version": MITIGATION_SCHEMA_VERSION,
            "mitigator": self.mitigator,
            "category": self.category,
            "result": payload,
            "result_type": type(self.result).__name__,
            "result_serializable": serializable,
            "provenance": thaw_json(self.provenance),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Serialize deterministically using strict standard JSON numbers."""
        return json.dumps(
            self.to_dict(), allow_nan=False, indent=indent, sort_keys=True
        )

    def __repr__(self) -> str:
        return (
            f"MitigationResult(mitigator={self.mitigator!r}, "
            f"category={self.category!r}, result={type(self.result).__name__})"
        )


def _payload(result: Any) -> tuple:
    """Return ``(json_safe_payload, serializable)`` for a result payload."""
    if hasattr(result, "to_dict") and callable(result.to_dict):
        try:
            return thaw_json(result.to_dict()), True
        except Exception:  # pragma: no cover - defensive
            return None, False
    try:
        return thaw_json(freeze_json_mapping({"v": result}, path="result")["v"]), True
    except (TypeError, ValueError):
        return None, False


class Mitigator(ParameterizedComponent, ABC):
    """Abstract base class: every mitigator exposes ``apply``.

    Follows the same scikit-learn conventions as
    :class:`~fairLMs.definitions.FairnessMetric`: ``__init__`` takes configuration
    only and stores each argument verbatim, **data goes to :meth:`apply`, never
    to ``__init__``**, and ``get_params`` / ``set_params`` come for free.

    Subclasses implement :meth:`_apply`. :meth:`apply` runs the shared
    applicability matcher first, so no subclass re-checks it and none can forget.
    """

    #: Registry key.
    name: str = "mitigator"

    #: One of :data:`CATEGORIES`.
    category: str = ""

    #: The **minimum** access level the method needs.
    access: AccessLevel = AccessLevel.BLACK_BOX

    #: Supported model families.
    architectures: Tuple[str, ...] = ()

    #: Capabilities the model must expose, from
    #: :data:`fairLMs.definitions.core.applicability.CAPABILITIES`.
    requires: frozenset = frozenset()

    #: Evidence container types :meth:`apply` consumes.
    accepts: Tuple[type, ...] = ()

    _compute_name = "apply"

    def apply(self, model: Any = None, evidence: Any = None) -> MitigationResult:
        """Check applicability, then run the intervention.

        Parameters
        ----------
        model:
            The model to mitigate, or ``None`` for the components that operate
            purely on evidence (all of pre- and post-processing).
        evidence:
            A container from this mitigator's ``accepts``.

        Raises
        ------
        fairLMs.definitions.core.applicability.ApplicabilityError
            When the model or evidence cannot satisfy this mitigator's
            declarations. The message names the specific missing capability,
            container, architecture or access level.
        """
        check_applicability(self, model, evidence)
        return self._apply(model, evidence)

    @abstractmethod
    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        """Run the intervention. Applicability is already checked."""

    # -- helpers shared by the concrete mitigators --------------------------
    def _result(self, result: Any, **provenance: Any) -> MitigationResult:
        """Build a :class:`MitigationResult` tagged with this mitigator's config."""
        return MitigationResult(
            mitigator=self.name,
            category=self.category,
            result=result,
            provenance={**self._provenance(), **provenance},
        )

    def _provenance(self) -> dict:
        """Config, declarations and library version, for the audit trail."""
        from fairLMs._version import __version__

        return {
            "mitigator": self.name,
            "category": self.category,
            "access": AccessLevel(self.access).value,
            "config": {k: _jsonable(v) for k, v in sorted(self.get_params().items())},
            "fairLMs_version": __version__,
        }


def _jsonable(value: Any) -> Any:
    """Render a config value for provenance without letting an object leak in."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    # A callable or an array: record that one was supplied and what it was,
    # which is what an audit needs, without pretending to serialize it.
    return f"<{type(value).__name__}>"
