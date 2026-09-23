"""Pseudo-log-likelihood metrics: PLL, CPS, AUL, AULA, CAT.

All five score a masked LM's preference between paired sentences. The four
pair-based metrics take the pairs directly as ``data`` (a
:class:`~fairLMs.datasets.FairnessDataset` or a sequence of dicts with
``stereotype`` / ``anti_stereotype`` keys); CAT takes
:class:`~fairLMs.metrics.data.SentenceTriples`.
"""

from __future__ import annotations

from typing import Any, Optional

from fairLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.aul.aul import (  # noqa: E501
    compute_aul,
)
from fairLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.aula.aula import (  # noqa: E501
    compute_aula,
)
from fairLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cat.cat import (  # noqa: E501
    compute_ss,
)
from fairLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cps.cps import (  # noqa: E501
    compute_cps,
)
from fairLMs.definition.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.pll.pll import (  # noqa: E501
    compute_pll,
)
from fairLMs.metrics._compat import (
    as_examples,
    require_mapping_keys,
    take,
    unwrap,
    warn_legacy,
)
from fairLMs.metrics.base import FairnessMetric, MetricResult
from fairLMs.metrics.data import RECORD_CORPUS, SentenceTriples
from fairLMs.metrics.resolve import get_tokenizer_model

_PAIR_KEYS = ("stereotype", "anti_stereotype")


class _PairMetric(FairnessMetric):
    """Shared plumbing for the four pair-preference metrics."""

    bias_type = "intrinsic"
    architectures = ("encoder_only",)
    _compute_fn = None
    _extra_allowed: tuple = ()

    def _pairs(self, data: Any, legacy: dict) -> list:
        data = unwrap(data if data is not None else legacy.pop("dataset", None))
        if data is None:
            data, key = take(legacy, "sentence_pairs", "pairs")
            if data is not None:
                warn_legacy(type(self).__name__, [key], "the pairs")
                legacy.pop(key, None)
        pairs = as_examples(
            data,
            type(self).__name__,
            "a dataset or sequence of stereotype/anti-stereotype pairs",
        )
        require_mapping_keys(pairs, type(self).__name__, *_PAIR_KEYS)
        for index, pair in enumerate(pairs):
            if any(
                not isinstance(pair[key], str) or not pair[key].strip()
                for key in _PAIR_KEYS
            ):
                raise ValueError(
                    f"{type(self).__name__}: pair {index} needs non-empty sentence strings."
                )
        return pairs

    def _run(self, model, pairs, tokenizer, **compute_kwargs) -> MetricResult:
        tok, hf_model, _ = get_tokenizer_model(model, tokenizer, metric=self)
        score, accuracy, per_bias_type = type(self)._compute_fn(
            tok, hf_model, pairs, **compute_kwargs
        )
        return MetricResult(
            score=float(score),
            details={"accuracy": accuracy, "n_pairs": len(pairs)},
            by_category=dict(per_bias_type),
        )

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        pairs = self._pairs(data, legacy)
        self._reject_unknown_kwargs(legacy, *self._extra_allowed)
        return self._run(model, pairs, tokenizer, **self._compute_kwargs(legacy))

    def _compute_kwargs(self, legacy: dict) -> dict:
        return {}


class CrowSPairsScore(_PairMetric):
    """Pseudo-log-likelihood CrowS-Pairs Score (Nangia et al., 2020).

    ``data`` yields dicts with ``stereotype``, ``anti_stereotype`` and
    optionally ``bias_type``. See :class:`~fairLMs.datasets.CrowSPairs`.

    >>> CrowSPairsScore().compute(model, CrowSPairs(n_max=50))   # doctest: +SKIP
    """

    name = "crows_pairs_score"
    required_task = "mlm"
    requires = frozenset({"masked_token_scores", "local_tokenizer"})
    accepts = RECORD_CORPUS
    _compute_fn = staticmethod(compute_cps)

    def __init__(self, *, batch_size: int = 16):
        self.batch_size = batch_size

    def _compute_kwargs(self, legacy: dict) -> dict:
        return {"batch_size": self.batch_size}


class PseudoLogLikelihoodScore(_PairMetric):
    """Full-sentence pseudo log-likelihood preference score."""

    name = "pseudo_log_likelihood_score"
    required_task = "mlm"
    requires = frozenset({"masked_token_scores", "local_tokenizer"})
    accepts = RECORD_CORPUS
    _compute_fn = staticmethod(compute_pll)

    def __init__(self, *, batch_size: int = 16):
        self.batch_size = batch_size

    def _compute_kwargs(self, legacy: dict) -> dict:
        return {"batch_size": self.batch_size}


class AllUnmaskedLikelihoodScore(_PairMetric):
    """All Unmasked Likelihood (AUL) bias score.

    Parameters
    ----------
    use_attention:
        Weight per-token log-probabilities by mean attention (turning AUL into
        AULA). Requires a model exposing ``.bert`` or ``.roberta``.
    """

    name = "all_unmasked_likelihood_score"
    required_task = "mlm"
    requires = frozenset({"masked_token_scores", "local_tokenizer"})
    accepts = RECORD_CORPUS
    _compute_fn = staticmethod(compute_aul)
    _extra_allowed = ("use_attention",)

    def __init__(self, *, use_attention: bool = False):
        self.use_attention = use_attention

    def _compute_kwargs(self, legacy: dict) -> dict:
        return {"use_attention": legacy.get("use_attention", self.use_attention)}


class AllUnmaskedLikelihoodAttentionScore(_PairMetric):
    """Attention-weighted AUL (AULA)."""

    name = "all_unmasked_likelihood_attention_score"
    required_task = "mlm"
    requires = frozenset({"masked_token_scores", "attentions", "local_tokenizer"})
    accepts = RECORD_CORPUS
    _compute_fn = staticmethod(compute_aula)
    _extra_allowed = ("use_attention",)

    def _compute_kwargs(self, legacy: dict) -> dict:
        return {"use_attention": legacy.get("use_attention", True)}


class ContextAssociationTestScore(FairnessMetric):
    """StereoSet-style SS / LMS / iCAT over sentence triples.

    ``data`` is a :class:`~fairLMs.metrics.data.SentenceTriples`, or any
    dataset / sequence of dicts with ``stereotype``, ``anti_stereotype`` and
    ``unrelated`` keys (which is coerced for you).

    The reported ``score`` is **iCAT**; ``details`` carries ``ss`` and ``lms``.
    """

    name = "context_association_test"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)
    required_task = "mlm"
    requires = frozenset({"masked_token_scores", "local_tokenizer"})
    accepts = (SentenceTriples,)

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = unwrap(
            data if data is not None else legacy.pop("dataset", None), SentenceTriples
        )

        if data is None:
            stereo, k1 = take(legacy, "stereo_sentences")
            anti, k2 = take(legacy, "anti_sentences")
            related, k3 = take(legacy, "related_sentences")
            if None in (stereo, anti, related):
                raise ValueError(
                    "ContextAssociationTestScore requires sentence triples. Pass a "
                    "SentenceTriples as the second argument, e.g. "
                    "ContextAssociationTestScore().compute(model, "
                    "SentenceTriples(stereo, anti, unrelated))."
                )
            warn_legacy("ContextAssociationTestScore", [k1, k2, k3], "SentenceTriples")
            for key in (k1, k2, k3):
                legacy.pop(key, None)
            data = SentenceTriples(stereo, anti, related)

        self._reject_unknown_kwargs(legacy)

        if not isinstance(data, SentenceTriples):
            data = SentenceTriples.from_examples(
                as_examples(
                    data,
                    "ContextAssociationTestScore",
                    "a SentenceTriples or sequence of triples",
                )
            )

        tok, hf_model, _ = get_tokenizer_model(model, tokenizer, metric=self)
        ss, lms, icat, rows = compute_ss(
            hf_model,
            tok,
            list(data.stereotype),
            list(data.anti_stereotype),
            list(data.unrelated),
            contexts=data.contexts,
        )
        return MetricResult(
            score=float(icat),
            details={
                "ss": ss,
                "lms": lms,
                "icat": float(icat),
                "rows": rows,
                "n": len(data),
                "scoring_protocol": "sum of candidate-token masked PLL; condition on scoring_context when supplied",
                "aggregation": "micro over input triples (not the official StereoSet per-target macro aggregation)",
                "lms_comparisons_per_triple": 2,
            },
        )
