"""Debiasing prompt: prepend instruction templates to queries."""

from __future__ import annotations

from typing import Any

from fairLMs.mitigation.base import MitigationResult, Mitigator
from fairLMs.mitigation.evidence import PromptSpec

__all__ = ["DebiasingPrompt"]


class DebiasingPrompt(Mitigator):
    """Prepend system/task instruction templates to queries.

    Pre-processing, but **generative models only**: an instruction has nowhere
    to go in a masked-token or classification forward pass. Declared as
    decoder-only and encoder-decoder, and refused elsewhere.

    Examples
    --------
    >>> from fairLMs.mitigation import DebiasingPrompt, PromptSpec
    >>> spec = PromptSpec(
    ...     templates=["Answer without stereotyping. {query}"],
    ...     queries=["Describe a nurse."],
    ... )
    >>> prompts, = DebiasingPrompt().apply(None, spec).result["prompts"]
    >>> tuple(prompts)
    ('Answer without stereotyping. Describe a nurse.',)
    """

    name = "debiasing_prompt"
    category = "pre"
    access = "black_box"
    # Generative-only, and that restriction lives here rather than in
    # `requires`: this rewrites query strings and reads nothing from a model, so
    # declaring a capability would falsely demand one be supplied.
    architectures = ("decoder_only", "encoder_decoder")
    requires = frozenset()
    accepts = (PromptSpec,)

    def __init__(self) -> None:
        pass

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if not evidence.queries:
            raise ValueError(
                f"{self.name}: PromptSpec carries no queries to rewrite. Build "
                f"PromptSpec(templates=[...], queries=[...])."
            )
        prompts = [list(evidence.render(query)) for query in evidence.queries]
        return self._result(
            {
                "prompts": prompts,
                "templates": list(evidence.templates),
                "queries": list(evidence.queries),
            },
            n_queries=len(evidence.queries),
            n_templates=len(evidence.templates),
            attribute=evidence.attribute,
        )
