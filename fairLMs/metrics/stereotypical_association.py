"""Decoder stereotypical-association metrics: SLL, CA."""

from __future__ import annotations

from typing import Any

import numpy as np

from fairLMs.definition.decoder_only.intrinsic_bias.stereotypical_association.ca.ca import (
    compute_ca,
)
from fairLMs.definition.decoder_only.intrinsic_bias.stereotypical_association.sll.sll import (
    compute_sll,
)
from fairLMs.metrics._compat import take, unwrap, warn_legacy
from fairLMs.metrics.base import FairnessMetric, MetricResult
from fairLMs.metrics.data import ConceptSpec, OccupationTriples
from fairLMs.metrics.resolve import get_tokenizer_model


class StereotypicalLogLikelihood(FairnessMetric):
    """Stereotypical log-likelihood gaps across occupation templates (SLL).

    ``data`` is an :class:`~fairLMs.metrics.data.OccupationTriples` of
    ``(occupation, stereotype_word, counter_word)``.

    The score is the mean absolute gap across the NV / CV / IV template variants,
    each of which appears in ``details``.
    """

    name = "stereotypical_log_likelihood"
    bias_type = "intrinsic"
    architectures = ("decoder_only",)
    required_task = "causal"
    requires = frozenset({"token_logprobs", "local_tokenizer"})
    accepts = (OccupationTriples,)

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = unwrap(
            data if data is not None else legacy.pop("dataset", None), OccupationTriples
        )

        if data is None:
            triples, key = take(legacy, "occupation_pairs", "occupation_triples")
            if triples is None:
                raise ValueError(
                    "StereotypicalLogLikelihood requires occupation triples. Pass an "
                    "OccupationTriples as the second argument, e.g. compute(model, "
                    'OccupationTriples([("nurse", "woman", "man")])).'
                )
            warn_legacy("StereotypicalLogLikelihood", [key], "OccupationTriples")
            legacy.pop(key, None)
            data = OccupationTriples(triples)

        self._reject_unknown_kwargs(legacy)
        if not isinstance(data, OccupationTriples):
            data = OccupationTriples(data)

        tok, hf_model, device = get_tokenizer_model(model, tokenizer, metric=self)
        scores = compute_sll(hf_model, tok, device, list(data.triples))
        primary = float(np.mean([abs(v) for v in scores.values()]))
        details = dict(scores)
        details["n_occupations"] = len(data)
        return MetricResult(score=primary, details=details)


class CooccurrenceAssociation(FairnessMetric):
    """Concept-group association via generation and total-variation distance (CA).

    ``data`` is a :class:`~fairLMs.metrics.data.ConceptSpec` carrying the
    concepts, a ``{concept}`` prompt template, and the group term lists counted
    in the generated text.

    A score of ``nan`` means no group term appeared in any generation. Raise
    ``n_samples`` or widen the term lists (``details['n_skipped']`` reports how
    many concepts were skipped).

    Parameters
    ----------
    n_samples:
        Generations sampled per concept.
    """

    name = "cooccurrence_association"
    bias_type = "intrinsic"
    architectures = ("decoder_only",)
    required_task = "causal"
    requires = frozenset({"free_generation", "local_tokenizer"})
    accepts = (ConceptSpec,)

    def __init__(self, *, n_samples: int = 20):
        self.n_samples = n_samples

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = unwrap(
            data if data is not None else legacy.pop("dataset", None), ConceptSpec
        )

        if data is None:
            concepts, k1 = take(legacy, "concepts")
            template, k2 = take(legacy, "prompt_template")
            terms, k3 = take(legacy, "group_terms")
            if None in (concepts, template, terms):
                raise ValueError(
                    "CooccurrenceAssociation requires concepts, a prompt template and "
                    "group terms. Pass a ConceptSpec as the second argument, e.g. "
                    'compute(model, ConceptSpec(["career"], '
                    '"People interested in {concept} are usually", '
                    '{"men": ["he"], "women": ["she"]})).'
                )
            warn_legacy("CooccurrenceAssociation", [k1, k2, k3], "ConceptSpec")
            for key in (k1, k2, k3):
                legacy.pop(key, None)
            data = ConceptSpec(concepts, template, terms)

        self._reject_unknown_kwargs(legacy, "n_samples")
        if not isinstance(data, ConceptSpec):
            raise TypeError(
                f"CooccurrenceAssociation expects a ConceptSpec as data, got "
                f"{type(data).__name__}."
            )

        n_samples = legacy.get("n_samples", self.n_samples)
        tok, hf_model, _ = get_tokenizer_model(model, tokenizer, metric=self)
        mean_tvd, n_valid, n_skipped = compute_ca(
            hf_model,
            tok,
            list(data.concepts),
            data.prompt_template,
            dict(data.group_terms),
            n_samples=n_samples,
        )
        return MetricResult(
            score=float(mean_tvd),
            details={
                "n_valid": n_valid,
                "n_skipped": n_skipped,
                "n_samples": n_samples,
                "n_concepts": len(data.concepts),
            },
        )
