"""Self-debiasing: self-diagnosis prompt, then rescale the next-token distribution."""

from __future__ import annotations

from typing import Any, Optional

from fairLMs.mitigation.base import MitigationResult, Mitigator
from fairLMs.mitigation.evidence import PromptSpec
from fairLMs.definitions.models.base import LoadedModel, ModelAdapter

__all__ = ["SelfDebiasedModelAdapter", "SelfDebiasing"]


class SelfDebiasedModelAdapter(ModelAdapter):
    """An adapter exposing edited generation through its loaded model as well.

    Only generation is declared: unchanged internal activations/logits must not
    be reported as if they had been edited by this intervention.
    """

    def __init__(self, base, spec, *, decay, max_new_tokens, epsilon=0.01, seed=0):
        from fairLMs.definitions.core.applicability import ModelProfile, AccessLevel

        self.base, self.spec = base, spec
        self.decay, self.max_new_tokens = decay, max_new_tokens
        self.epsilon, self.seed = epsilon, seed
        self.name = f"{getattr(base, 'name', type(base).__name__)}+self_debiasing"
        self.task = "causal"
        self.profile = ModelProfile(
            architecture="decoder_only",
            capabilities=frozenset({"free_generation", "local_tokenizer"}),
            access=AccessLevel.GRAY_BOX,
            task="causal",
        )
        self._loaded = None

    def load(self):
        if self._loaded is None:
            from fairLMs.mitigation._generation import SelfDebiasedDecoder

            loaded = self.base.load()
            edited = SelfDebiasedDecoder(
                loaded.model,
                loaded.tokenizer,
                self.spec,
                decay=self.decay,
                epsilon=self.epsilon,
                max_new_tokens=self.max_new_tokens,
                seed=self.seed,
            )
            self._loaded = LoadedModel(
                name=self.name,
                tokenizer=loaded.tokenizer,
                model=edited,
                device=loaded.device,
                task=self.task,
                profile=self.profile,
            )
        return self._loaded

    def generate(self, prompt: str, **kwargs):
        loaded = self.load()
        inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)
        result = loaded.model.generate(**inputs, **kwargs)
        return loaded.tokenizer.decode(result[0], skip_special_tokens=True)

    def __repr__(self):
        return f"SelfDebiasedModelAdapter(base={self.base!r}, decay={self.decay!r})"


class SelfDebiasing(Mitigator):
    """Self-diagnosis prompt, then rescale the next-token distribution.

    Schick et al. (2021). The model is asked to describe the attribute itself,
    and tokens the diagnosis pass favours are damped in the plain pass. Needs no
    weight update, which is what makes it intra-processing rather than
    in-processing.

    Parameters
    ----------
    decay:
        Strength of the damping applied to diagnosed tokens.
    max_new_tokens:
        Default generation length for the returned adapter.

    Examples
    --------
    >>> from fairLMs.mitigation import PromptSpec, SelfDebiasing
    >>> from fairLMs.definitions.models.base import ModelAdapter
    >>> class Stub(ModelAdapter):
    ...     name = "stub"
    ...     task = "causal"
    ...     def load(self): raise NotImplementedError
    >>> spec = PromptSpec(
    ...     templates=["The following text is biased. {query}"],
    ...     attribute="gender",
    ... )
    >>> outcome = SelfDebiasing().apply(Stub(), spec)
    >>> isinstance(outcome.result, ModelAdapter)
    True
    >>> outcome.result.task            # profiles as the model it wraps
    'causal'
    """

    name = "self_debiasing"
    category = "intra"
    access = "gray_box"
    architectures = ("decoder_only",)
    requires = frozenset({"token_logprobs", "free_generation"})
    accepts = (PromptSpec,)

    def __init__(
        self,
        decay: float = 50.0,
        max_new_tokens: int = 20,
        epsilon: float = 0.01,
        seed: Optional[int] = 0,
    ):
        self.decay = decay
        self.max_new_tokens = max_new_tokens
        self.epsilon = epsilon
        self.seed = seed

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        import math

        if not math.isfinite(self.decay) or self.decay < 0:
            raise ValueError(
                f"decay must be non-negative and finite; got {self.decay!r}."
            )
        if not 0 < self.epsilon <= 1:
            raise ValueError("epsilon must be in (0, 1].")
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer.")
        adapter = SelfDebiasedModelAdapter(
            model,
            evidence,
            decay=self.decay,
            max_new_tokens=self.max_new_tokens,
            epsilon=self.epsilon,
            seed=self.seed,
        )
        return self._result(
            adapter,
            attribute=evidence.attribute,
            n_templates=len(evidence.templates),
            templates=list(evidence.templates),
            removal_claim=(
                "Damps tokens the model's own diagnosis pass favours. Acts on the "
                "output distribution only; the underlying representations are "
                "unchanged."
            ),
        )
