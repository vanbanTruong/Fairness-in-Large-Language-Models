"""Capability declarations and the applicability matcher shared by all components.

A fairLMs component - a metric or a mitigator - declares what it needs from a
model rather than assuming it. A deployment either satisfies the declaration or
is refused by name, before any compute happens.

Four declarations, four checks:

===================  ==========================================================
``requires``         capabilities the model must expose, from :data:`CAPABILITIES`
``accepts``          evidence container types the component consumes
``architectures``    model families, from :data:`ARCHITECTURES`
``access``           minimum :class:`AccessLevel` the deployment must grant
===================  ==========================================================

The matcher refuses **only** when a declaration exists and is violated. An empty
declaration means "no declared constraint" and never refuses, which is what lets
this mechanism be added to an existing component without changing its behaviour.
It is equally silent when the *model* side is unknown: a bare
``(tokenizer, model)`` tuple carries no task, and refusing on missing information
would break that escape hatch. This mirrors
:func:`fairLMs.definitions.resolve.check_task`, which established the pattern.

This module lives in the definitions core rather than inside mitigation because
metrics use the same matcher and must not import the mitigation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Tuple

__all__ = [
    "ACCESS_LEVELS",
    "ARCHITECTURES",
    "CAPABILITIES",
    "AccessLevel",
    "ApplicabilityError",
    "ModelProfile",
    "TASK_PROFILES",
    "access_rank",
    "check_applicability",
    "describe_model",
    "validate_declaration",
]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
#: What a model can be asked to produce. These strings are public API: a refusal
#: message names them, and users match on them. Each is something at least one
#: shipped component actually reads - the vocabulary describes the code, it does
#: not aspire beyond it.
CAPABILITIES: frozenset = frozenset(
    {
        # Per-layer activations and pooled sentence embeddings. WEAT/SEAT/CEAT
        # read these; every projection-based mitigator edits them.
        "hidden_states",
        # Per-head attention weights. AULA reweights pseudo-log-likelihood by
        # them; NIE mediates through them.
        "attentions",
        # A differentiable backward pass over parameters. GBE differentiates a
        # contrastive loss; every in-processing objective composes into one.
        "gradients",
        # Vocabulary logits at a [MASK] position, i.e. an MLM head.
        "masked_token_scores",
        # Logits from a sequence-classification head.
        "sequence_logits",
        # Per-token log probabilities for scoring a fixed continuation.
        "token_logprobs",
        # Sampling free text from a prompt.
        "free_generation",
        # A local tokenizer object paired with the weights. Twenty-four shipped
        # metrics call ``resolve.get_tokenizer_model`` and then tokenize
        # themselves, so a deployment that answers only over the wire cannot
        # serve them however many logprobs it returns. Declaring it is what lets
        # the matcher refuse those pairings by name instead of letting them die
        # on a missing ``.tokenizer`` attribute deep inside the metric.
        "local_tokenizer",
        "completions_api",
    }
)

#: Model families, matching :attr:`fairLMs.definitions.FairnessMetric.architectures`.
ARCHITECTURES: Tuple[str, ...] = ("encoder_only", "decoder_only", "encoder_decoder")


class AccessLevel(str, Enum):
    """What a deployment exposes, as the paper and book chapter define it."""

    #: Inputs and outputs only.
    BLACK_BOX = "black_box"
    #: Activations and logits, but no weight updates.
    GRAY_BOX = "gray_box"
    #: Parameters and gradients.
    WHITE_BOX = "white_box"


#: Increasing order of access. ``black_box < gray_box < white_box``.
ACCESS_LEVELS: Tuple[AccessLevel, ...] = (
    AccessLevel.BLACK_BOX,
    AccessLevel.GRAY_BOX,
    AccessLevel.WHITE_BOX,
)

_ACCESS_RANK: Mapping[AccessLevel, int] = {
    level: rank for rank, level in enumerate(ACCESS_LEVELS)
}


def access_rank(level: Any) -> int:
    """Return the ordinal of *level* in ``black_box < gray_box < white_box``."""
    return _ACCESS_RANK[_as_access_level(level, "access")]


def _as_access_level(value: Any, field_name: str) -> AccessLevel:
    if isinstance(value, AccessLevel):
        return value
    try:
        return AccessLevel(value)
    except (TypeError, ValueError) as exc:
        available = ", ".join(repr(level.value) for level in ACCESS_LEVELS)
        raise ValueError(
            f"{field_name} must be one of: {available}; got {value!r}."
        ) from exc


class ApplicabilityError(TypeError):
    """A component was paired with a model or evidence it declared it cannot use.

    Subclasses :class:`TypeError` so that it joins the refusal family already
    raised by :func:`fairLMs.definitions.resolve.check_task`: callers that guard a
    metric call with ``except TypeError`` keep working.
    """


# ---------------------------------------------------------------------------
# Model side
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelProfile:
    """What a particular deployment of a model offers."""

    architecture: str
    capabilities: frozenset
    access: AccessLevel
    task: Optional[str] = None

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {list(ARCHITECTURES)}; "
                f"got {self.architecture!r}."
            )
        object.__setattr__(
            self,
            "capabilities",
            _validate_capabilities(self.capabilities, "capabilities"),
        )
        object.__setattr__(self, "access", _as_access_level(self.access, "access"))

    def to_dict(self) -> dict:
        """Return a JSON-safe description, for provenance."""
        return {
            "architecture": self.architecture,
            "capabilities": sorted(self.capabilities),
            "access": self.access.value,
            "task": self.task,
        }


def _validate_capabilities(value: Any, field_name: str) -> frozenset:
    if isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a set of capability strings, not a bare string."
        )
    try:
        names = frozenset(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of strings.") from exc
    unknown = sorted(name for name in names if name not in CAPABILITIES)
    if unknown:
        raise ValueError(
            f"{field_name} names unknown capabilities: {', '.join(map(repr, unknown))}. "
            f"Known capabilities: {', '.join(sorted(CAPABILITIES))}."
        )
    return names


#: Derived from the ``task`` tag every adapter already carries, rather than from
#: a parallel declaration that could drift away from it. A locally loaded
#: checkpoint grants white-box access: the weights are in the process.
TASK_PROFILES: Mapping[str, ModelProfile] = {
    "mlm": ModelProfile(
        architecture="encoder_only",
        capabilities=frozenset(
            {
                "local_tokenizer",
                "hidden_states",
                "attentions",
                "gradients",
                "masked_token_scores",
            }
        ),
        access=AccessLevel.WHITE_BOX,
        task="mlm",
    ),
    "encoder": ModelProfile(
        architecture="encoder_only",
        capabilities=frozenset(
            {"local_tokenizer", "hidden_states", "attentions", "gradients"}
        ),
        access=AccessLevel.WHITE_BOX,
        task="encoder",
    ),
    "sequence_classification": ModelProfile(
        architecture="encoder_only",
        capabilities=frozenset(
            {
                "local_tokenizer",
                "hidden_states",
                "attentions",
                "gradients",
                "sequence_logits",
            }
        ),
        access=AccessLevel.WHITE_BOX,
        task="sequence_classification",
    ),
    "seq2seq": ModelProfile(
        architecture="encoder_decoder",
        capabilities=frozenset(
            {
                "local_tokenizer",
                "hidden_states",
                "attentions",
                "gradients",
                "token_logprobs",
                "free_generation",
            }
        ),
        access=AccessLevel.WHITE_BOX,
        task="seq2seq",
    ),
    "causal": ModelProfile(
        architecture="decoder_only",
        capabilities=frozenset(
            {
                "local_tokenizer",
                "hidden_states",
                "attentions",
                "gradients",
                "token_logprobs",
                "free_generation",
            }
        ),
        access=AccessLevel.WHITE_BOX,
        task="causal",
    ),
    # The row that makes refusal meaningful: no `hidden_states`, no
    # `attentions`, no `gradients`, so anything reading activations is refused
    # by name instead of failing inside a forward pass that will never run.
    #
    # `token_logprobs` is present because the completions endpoint returns them
    # and two shipped metrics read them: `bias_amplifier` requests
    # ``logprobs=1, echo=True`` to score a continuation, and
    # `counterfactual_fairness` requests ``logprobs=TOP_LOGPROBS`` for a
    # next-token distribution. Declaring the API generation-only would refuse
    # two metrics that demonstrably work against it today.
    "openai": ModelProfile(
        architecture="decoder_only",
        capabilities=frozenset(
            {"free_generation", "token_logprobs", "completions_api"}
        ),
        access=AccessLevel.BLACK_BOX,
        task="openai",
    ),
}


def describe_model(model: Any) -> Optional[ModelProfile]:
    """Return what *model* offers, or ``None`` when that cannot be known.

    Recognises, in order: an explicit :class:`ModelProfile`; an object declaring
    its own ``profile``; an adapter or loaded bundle carrying a known ``task``.
    Everything else - raw ``(tokenizer, model)`` tuples, bare ``nn.Module``
    objects, ``None`` - is unknown, and unknown is not refused.
    """
    if model is None:
        return None
    if isinstance(model, ModelProfile):
        return model

    # An escape hatch for adapters outside the built-in task vocabulary: declare
    # a profile and the matcher honours it verbatim.
    declared = getattr(model, "profile", None)
    if isinstance(declared, ModelProfile):
        return declared

    task = getattr(model, "task", None)
    if isinstance(task, str):
        return TASK_PROFILES.get(task)
    return None


# ---------------------------------------------------------------------------
# Component side
# ---------------------------------------------------------------------------
def validate_declaration(component: Any) -> None:
    """Raise when *component* declares something outside the vocabulary.

    Catches a typo in a capability string at conformance-test time rather than
    letting it silently never match.
    """
    label = _component_label(component)

    requires = getattr(component, "requires", frozenset())
    _validate_capabilities(requires, f"{label}.requires")

    architectures = getattr(component, "architectures", ())
    if isinstance(architectures, str):
        raise TypeError(
            f"{label}.architectures must be a tuple of strings, not a bare string."
        )
    unknown = sorted(a for a in architectures if a not in ARCHITECTURES)
    if unknown:
        raise ValueError(
            f"{label}.architectures names unknown families: "
            f"{', '.join(map(repr, unknown))}. Known: {', '.join(ARCHITECTURES)}."
        )

    accepts = getattr(component, "accepts", ())
    if isinstance(accepts, type):
        raise TypeError(
            f"{label}.accepts must be a tuple of types, not a bare type. "
            f"Write ``accepts = ({accepts.__name__},)``."
        )
    for entry in accepts:
        if not isinstance(entry, type):
            raise TypeError(f"{label}.accepts must contain types; got {entry!r}.")

    access = getattr(component, "access", None)
    if access is not None:
        _as_access_level(access, f"{label}.access")


def check_applicability(
    component: Any,
    model: Any = None,
    evidence: Any = None,
) -> Optional[ModelProfile]:
    """Refuse an unsatisfiable pairing before any compute happens.

    Returns the resolved :class:`ModelProfile`, or ``None`` when the model side
    is unknown. Raises :class:`ApplicabilityError` naming the specific missing
    capability, container, architecture or access level - never a bare
    "not applicable".
    """
    label = _component_label(component)
    profile = describe_model(model)

    _check_capabilities(component, model, profile, label)
    _check_evidence(component, evidence, label)
    _check_architecture(component, profile, label)
    _check_access(component, profile, label)
    return profile


def _check_capabilities(
    component: Any, model: Any, profile: Optional[ModelProfile], label: str
) -> None:
    requires = frozenset(getattr(component, "requires", frozenset()))
    if not requires:
        return
    if model is None:
        raise ApplicabilityError(
            f"{label} requires a model exposing {_names(requires)}, but no model "
            f"was given. Pass a loaded model as the first argument."
        )
    if profile is None:
        # Unknown deployment: the (tokenizer, model) escape hatch. Stay silent
        # rather than refuse on missing information.
        return
    missing = requires - profile.capabilities
    if missing:
        raise ApplicabilityError(
            f"{label} requires {_names(missing)}; {_model_label(model, profile)} "
            f"provides only {_names(profile.capabilities)}. "
            f"{_capability_hint(missing)}"
        )


def _check_evidence(component: Any, evidence: Any, label: str) -> None:
    accepts = tuple(getattr(component, "accepts", ()))
    if not accepts or evidence is None:
        # No declaration, or evidence resolved elsewhere in the call.
        return
    if not isinstance(evidence, accepts):
        accepted = ", ".join(sorted(cls.__name__ for cls in accepts))
        raise ApplicabilityError(
            f"{label} accepts {accepted}; got {type(evidence).__name__}."
        )


def _check_architecture(
    component: Any, profile: Optional[ModelProfile], label: str
) -> None:
    architectures = tuple(getattr(component, "architectures", ()))
    if not architectures or profile is None:
        return
    if profile.architecture not in architectures:
        raise ApplicabilityError(
            f"{label} supports {', '.join(architectures)}; the given model is "
            f"{profile.architecture}"
            + (f" (task={profile.task!r})." if profile.task else ".")
        )


def _check_access(component: Any, profile: Optional[ModelProfile], label: str) -> None:
    required = getattr(component, "access", None)
    if required is None or profile is None:
        return
    required = _as_access_level(required, f"{label}.access")
    if access_rank(profile.access) < access_rank(required):
        raise ApplicabilityError(
            f"{label} needs {required.value} access; the given model grants only "
            f"{profile.access.value}. "
            f"{_ACCESS_HINT[required]}"
        )


_ACCESS_HINT: Mapping[AccessLevel, str] = {
    AccessLevel.GRAY_BOX: "It needs activations or logits, which this deployment "
    "does not expose.",
    AccessLevel.WHITE_BOX: "It needs parameters and gradients, which this "
    "deployment does not expose.",
    AccessLevel.BLACK_BOX: "",
}


def _capability_hint(missing: Iterable[str]) -> str:
    missing = set(missing)
    if missing <= {"free_generation"}:
        return "Use a generative deployment."
    if "gradients" in missing:
        return "Load the checkpoint locally: gradients need the weights in-process."
    return (
        "Load the checkpoint locally with HuggingFaceModel(name, task=...) so the "
        "quantity is observable."
    )


def _names(capabilities: Iterable[str]) -> str:
    return ", ".join(f"`{name}`" for name in sorted(capabilities)) or "(nothing)"


def _component_label(component: Any) -> str:
    if isinstance(component, str):
        return component
    if isinstance(component, type):
        return component.__name__
    return type(component).__name__


def _model_label(model: Any, profile: ModelProfile) -> str:
    name = getattr(model, "name", None)
    base = type(model).__name__
    if isinstance(name, str) and name and name != base:
        return f"{base}({name!r})"
    return base
