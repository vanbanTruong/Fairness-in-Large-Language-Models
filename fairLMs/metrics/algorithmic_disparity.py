"""Algorithmic disparity metrics: LFP, MCD.

Both score generated translations of a sentence list, so ``data`` is simply a
sequence of sentences (or a dataset whose examples carry ``sentence`` / ``text``
/ ``premise``).
"""

from __future__ import annotations

from typing import Any, List

from fairLMs.definition.encoder_decoder.intrinsic_bias.algorithmic_disparity.lfp.lfp import (
    compute_lfp,
)
from fairLMs.definition.encoder_decoder.intrinsic_bias.algorithmic_disparity.mcd.mcd import (
    compute_mcd,
)
from fairLMs.metrics._compat import as_examples, take, unwrap, warn_legacy
from fairLMs.metrics.base import FairnessMetric, MetricResult
from fairLMs.metrics.resolve import get_tokenizer_model
from fairLMs.metrics.data import RECORD_CORPUS


def _sentences(data: Any, metric: str) -> List[str]:
    examples = as_examples(data, metric, "a sequence of sentences")
    if isinstance(examples[0], str):
        return list(examples)
    out = []
    for ex in examples:
        if not isinstance(ex, dict):
            raise TypeError(
                f"{metric}: expected strings or dicts, got {type(ex).__name__}."
            )
        value = ex.get("sentence") or ex.get("text") or ex.get("premise")
        if value is None:
            raise ValueError(
                f"{metric}: example dict has no 'sentence'/'text'/'premise' key; "
                f"got {sorted(ex)}."
            )
        out.append(value)
    return out


class _TranslationCorpusMetric(FairnessMetric):
    """Shared plumbing for metrics that generate from a sentence list.

    Abstract: subclasses supply ``compute``.
    """

    bias_type = "intrinsic"
    architectures = ("encoder_decoder",)

    def __init__(self, *, max_new_tokens: int = 128):
        self.max_new_tokens = max_new_tokens

    def _prepare(self, data: Any, legacy: dict) -> List[str]:
        data = unwrap(data if data is not None else legacy.pop("dataset", None))
        if data is None:
            data, key = take(legacy, "sentences")
            if data is None:
                raise ValueError(
                    f"{type(self).__name__} requires sentences. Pass them as the "
                    f"second argument, e.g. compute(model, ['The doctor is here.'])."
                )
            warn_legacy(type(self).__name__, [key], "the sentence list")
            legacy.pop(key, None)
        self._reject_unknown_kwargs(legacy, "max_new_tokens")
        return _sentences(data, type(self).__name__)


class LexicalFrequencyProportion(_TranslationCorpusMetric):
    """Lexical frequency profile of generated translations (LFP).

    Reports ``pb1`` as the score; ``pb2`` and ``pb3`` are in ``details``.
    """

    name = "lexical_frequency_proportion"
    required_task = "seq2seq"
    requires = frozenset({"free_generation", "local_tokenizer"})
    accepts = RECORD_CORPUS

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        sentences = self._prepare(data, legacy)
        tok, hf_model, _ = get_tokenizer_model(model, tokenizer, metric=self)
        pb1, pb2, pb3, rows = compute_lfp(
            hf_model,
            tok,
            sentences,
            max_new_tokens=legacy.get("max_new_tokens", self.max_new_tokens),
        )
        return MetricResult(
            score=float(pb1),
            details={
                "pb1": pb1,
                "pb2": pb2,
                "pb3": pb3,
                "rows": rows,
                "n_sentences": len(sentences),
            },
        )


class MorphologicalChoiceDivergence(_TranslationCorpusMetric):
    """Morphological complexity disparity across translations (MCD).

    Reports ``mean_h`` as the score; ``mean_d`` is in ``details``.
    """

    name = "morphological_choice_divergence"
    required_task = "seq2seq"
    requires = frozenset({"free_generation", "local_tokenizer"})
    accepts = RECORD_CORPUS

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        sentences = self._prepare(data, legacy)
        tok, hf_model, _ = get_tokenizer_model(model, tokenizer, metric=self)
        mean_h, mean_d, rows = compute_mcd(
            hf_model,
            tok,
            sentences,
            max_new_tokens=legacy.get("max_new_tokens", self.max_new_tokens),
        )
        return MetricResult(
            score=float(mean_h),
            details={
                "mean_h": mean_h,
                "mean_d": mean_d,
                "rows": rows,
                "n_sentences": len(sentences),
            },
        )
