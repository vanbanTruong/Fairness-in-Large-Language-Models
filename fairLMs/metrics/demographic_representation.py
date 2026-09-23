"""Demographic representation metrics: DNP, DRD.

Both prompt a decoder model and measure how demographic words are distributed in
what it produces, so ``data`` is a
:class:`~fairLMs.metrics.data.DemographicPrompts`.
"""

from __future__ import annotations

from typing import Any

from fairLMs.definition.decoder_only.extrinsic_bias.demographic_representation.dnp.dnp import (
    compute_dnp,
)
from fairLMs.definition.decoder_only.extrinsic_bias.demographic_representation.drd.drd import (
    compute_drd,
)
from fairLMs.metrics._compat import as_examples, take, unwrap, warn_legacy
from fairLMs.metrics.base import FairnessMetric, MetricResult
from fairLMs.metrics.data import DemographicPrompts
from fairLMs.metrics.resolve import get_tokenizer_model


def _prompts_from(data: Any, metric: str) -> list:
    examples = as_examples(data, metric, "a sequence of prompts")
    return [ex["prompt"] if isinstance(ex, dict) else ex for ex in examples]


class _DemographicPromptMetric(FairnessMetric):
    """Shared legacy handling for DNP and DRD. Abstract: subclasses add compute."""

    bias_type = "extrinsic"
    architectures = ("decoder_only",)

    def _prepare(self, data: Any, legacy: dict, *allowed: str) -> DemographicPrompts:
        data = unwrap(
            data if data is not None else legacy.pop("dataset", None),
            DemographicPrompts,
        )

        if not isinstance(data, DemographicPrompts):
            prompts, k0 = take(legacy, "prompts")
            stereo, k1 = take(legacy, "stereo_words")
            counter, k2 = take(legacy, "counter_words")
            neutral, k3 = take(legacy, "neutral_words")
            if data is not None and prompts is None:
                prompts = _prompts_from(data, type(self).__name__)
                k0 = None
            if prompts is None or stereo is None or counter is None:
                raise ValueError(
                    f"{type(self).__name__} requires prompts plus demographic word "
                    f"lists. Pass a DemographicPrompts as the second argument, e.g. "
                    f"compute(model, DemographicPrompts(prompts, "
                    f'["he"], ["she"]))'
                )
            warn_legacy(type(self).__name__, [k0, k1, k2, k3], "DemographicPrompts")
            for key in (k0, k1, k2, k3):
                legacy.pop(key, None)
            data = DemographicPrompts(prompts, stereo, counter, neutral)

        self._reject_unknown_kwargs(legacy, *allowed)
        return data


class DemographicNextTokenProportion(_DemographicPromptMetric):
    """Normalized next-token probability mass per demographic group (DNP).

    Needs ``neutral_words`` on the :class:`DemographicPrompts` because the score
    is normalised against a neutral baseline. Reports ``mean_pd``.
    """

    name = "demographic_next_token_proportion"
    required_task = "causal"
    requires = frozenset({"token_logprobs", "local_tokenizer"})
    accepts = (DemographicPrompts,)

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        prompts = self._prepare(data, legacy)
        prompts.require_neutral("DemographicNextTokenProportion")

        tok, hf_model, _ = get_tokenizer_model(model, tokenizer, metric=self)
        mean_ps, mean_psp, mean_pd, rows = compute_dnp(
            hf_model,
            tok,
            list(prompts.prompts),
            list(prompts.stereotype_words),
            list(prompts.counter_words),
            list(prompts.neutral_words),
        )
        return MetricResult(
            score=float(mean_pd),
            details={
                "mean_ps": mean_ps,
                "mean_psp": mean_psp,
                "mean_pd": mean_pd,
                "rows": rows,
                "n_prompts": len(prompts.prompts),
            },
        )


class DemographicRepresentationDivergence(_DemographicPromptMetric):
    """Demographic representation disparity in free generations (DRD).

    Parameters
    ----------
    max_new_tokens:
        Generation budget per prompt.
    """

    name = "demographic_representation_divergence"
    required_task = "causal"
    requires = frozenset({"free_generation", "local_tokenizer"})
    accepts = (DemographicPrompts,)

    def __init__(self, *, max_new_tokens: int = 50):
        self.max_new_tokens = max_new_tokens

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        prompts = self._prepare(data, legacy, "max_new_tokens")

        tok, hf_model, _ = get_tokenizer_model(model, tokenizer, metric=self)
        drd, n_s, n_sp, rows = compute_drd(
            hf_model,
            tok,
            list(prompts.prompts),
            list(prompts.stereotype_words),
            list(prompts.counter_words),
            max_new_tokens=legacy.get("max_new_tokens", self.max_new_tokens),
        )
        return MetricResult(
            score=float(drd),
            details={
                "n_stereotype_total": n_s,
                "n_counter_total": n_sp,
                "status": "ready" if n_s + n_sp else "insufficient_evidence",
                "reason": (
                    None
                    if n_s + n_sp
                    else "No demographic terms matched the generations."
                ),
                "mention_coverage": sum(bool(r["n_s"] + r["n_sp"]) for r in rows)
                / len(rows),
                "rows": rows,
                "n_prompts": len(prompts.prompts),
            },
        )
