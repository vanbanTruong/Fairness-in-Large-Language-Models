"""Encoder-decoder stereotypical-association metrics: SD, SVA."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from fairLMs.definitions.encoder_decoder.intrinsic_bias.stereotypical_association.sd.sd import (
    LABEL_DOMAIN_FOR,
    compute_sd,
)
from fairLMs.definitions.encoder_decoder.intrinsic_bias.stereotypical_association.sva.sva import (
    compute_sva,
)
from fairLMs.definitions._compat import take, unwrap, warn_legacy
from fairLMs.definitions.base import FairnessMetric, MetricResult
from fairLMs.definitions.data import LabeledSentences, StereotypeLabelled, WordSets
from fairLMs.definitions.resolve import get_tokenizer_model


class StereotypicalDivergence(FairnessMetric):
    """Stereotype vs anti-stereotype task-performance divergence (SD).

    ``data`` is a :class:`~fairLMs.definitions.data.StereotypeLabelled` pairing two
    labelled sentence sets.

    The task is a French-translation cue test: the model is scored on whether it
    prefers a gendered (or age-marked) French continuation for each source
    sentence, and the two sets' accuracies are compared. ``labels`` must
    therefore come from the scorer's own vocabulary: ``"male"`` / ``"female"``
    for the default, ``"young"`` / ``"old"`` for ``age_accuracy``.

    Parameters
    ----------
    max_new_tokens:
        Generation budget per sentence.
    metric_fn:
        Scorer comparing a prediction to its gold label. ``None`` uses
        ``pronoun_accuracy``. The two built-ins (``pronoun_accuracy``,
        ``age_accuracy``) each imply a prediction routine; supplying any other
        callable requires ``predict_fn`` as well.
    predict_fn:
        ``callable(model, tokenizer, sentence) -> str`` producing the label that
        ``metric_fn`` grades. Needed only for a custom ``metric_fn``.
    """

    name = "stereotypical_divergence"
    bias_type = "intrinsic"
    architectures = ("encoder_decoder",)
    required_task = "seq2seq"
    requires = frozenset({"token_logprobs", "local_tokenizer"})
    accepts = (StereotypeLabelled,)

    def __init__(
        self,
        *,
        max_new_tokens: int = 128,
        metric_fn: Optional[Callable] = None,
        predict_fn: Optional[Callable] = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.metric_fn = metric_fn
        self.predict_fn = predict_fn

    @staticmethod
    def _check_label_domain(metric_fn, predict_fn, *label_sets) -> None:
        """Refuse a label vocabulary the scorer cannot grade.

        Both built-in scorers return 0.5 for a gold label they do not
        recognise, so a wholly wrong label vocabulary yields a clean-looking
        ``m_stereo == m_anti == 0.5`` and a divergence of exactly 0.0, a
        non-result indistinguishable from a real finding of parity. Individual
        unknown labels are still allowed through as chance, which is what 0.5
        is for; only a complete mismatch is an error.
        """
        if predict_fn is not None:
            return  # custom pairing: the label domain is the caller's to define
        name = getattr(metric_fn, "__name__", None) if metric_fn else "pronoun_accuracy"
        domain = LABEL_DOMAIN_FOR.get(name)
        if domain is None:
            return
        labels = [str(label) for labels in label_sets for label in labels]
        if labels and not any(label in domain for label in labels):
            raise ValueError(
                f"StereotypicalDivergence with metric_fn={name!r} scores labels "
                f"from {domain}, but none of the {len(labels)} labels supplied "
                f"is one of those (saw {sorted(set(labels))[:5]}). Every row "
                f"would score 0.5 and the divergence would be 0.0 regardless "
                f"of the model."
            )

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = unwrap(
            data if data is not None else legacy.pop("dataset", None), StereotypeLabelled
        )

        if data is None:
            ss, k1 = take(legacy, "stereo_sentences")
            sl, k2 = take(legacy, "stereo_labels")
            asents, k3 = take(legacy, "anti_sentences")
            al, k4 = take(legacy, "anti_labels")
            if None in (ss, sl, asents, al):
                raise ValueError(
                    "StereotypicalDivergence requires two labelled sentence sets. "
                    "Pass a StereotypeLabelled as the second argument, e.g. "
                    "compute(model, StereotypeLabelled("
                    "LabeledSentences(stereo, stereo_labels), "
                    "LabeledSentences(anti, anti_labels)))."
                )
            warn_legacy("StereotypicalDivergence", [k1, k2, k3, k4], "StereotypeLabelled")
            for key in (k1, k2, k3, k4):
                legacy.pop(key, None)
            data = StereotypeLabelled(
                LabeledSentences(ss, sl), LabeledSentences(asents, al)
            )

        self._reject_unknown_kwargs(
            legacy, "max_new_tokens", "metric_fn", "predict_fn"
        )
        if not isinstance(data, StereotypeLabelled):
            raise TypeError(
                f"StereotypicalDivergence expects a StereotypeLabelled as data, got "
                f"{type(data).__name__}."
            )

        metric_fn = legacy.get("metric_fn", self.metric_fn)
        predict_fn = legacy.get("predict_fn", self.predict_fn)
        self._check_label_domain(
            metric_fn,
            predict_fn,
            data.stereotype.labels,
            data.anti_stereotype.labels,
        )
        sd_kwargs = {"max_new_tokens": legacy.get("max_new_tokens", self.max_new_tokens)}
        if metric_fn is not None:
            sd_kwargs["metric_fn"] = metric_fn
        if predict_fn is not None:
            sd_kwargs["predict_fn"] = predict_fn

        tok, hf_model, _ = get_tokenizer_model(model, tokenizer, metric=self)
        m_stereo, m_anti, delta_s, rows = compute_sd(
            hf_model,
            tok,
            list(data.stereotype.sentences),
            list(data.stereotype.labels),
            list(data.anti_stereotype.sentences),
            list(data.anti_stereotype.labels),
            **sd_kwargs,
        )
        return MetricResult(
            score=float(delta_s),
            details={
                "m_stereo": m_stereo,
                "m_anti": m_anti,
                "delta_s": float(delta_s),
                "rows": rows,
            },
        )


class StereotypicalValueAttribution(FairnessMetric):
    """Shapley attribution of stereotype bias to attention heads (SVA).

    ``data`` is a :class:`~fairLMs.definitions.data.WordSets` whose ``target_1`` /
    ``target_2`` hold the stereotypical and anti-stereotypical sentences.
    Attribute roles are unused, so pass the same sentences again if you have no
    attribute sets.

    The stereotype ``direction`` is derived from the sentences when not supplied,
    which also guarantees it matches the model's hidden size; previously the
    caller had to supply a correctly-sized vector by hand.

    Parameters
    ----------
    n_samples:
        Monte-Carlo permutations for the Shapley estimate.
    top_pct:
        Fraction of heads whose attribution mass is reported as the score.
    direction:
        Optional precomputed unit direction of length ``d_model``.
    n_layers, n_heads:
        Optional overrides; derived from ``model.config`` when omitted.
    """

    name = "stereotypical_value_attribution"
    bias_type = "intrinsic"
    architectures = ("encoder_decoder",)
    required_task = "seq2seq"
    requires = frozenset({"hidden_states", "attentions", "local_tokenizer"})
    accepts = (WordSets,)

    def __init__(
        self,
        *,
        n_samples: int = 15,
        top_pct: float = 0.10,
        direction: Any = None,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
    ):
        self.n_samples = n_samples
        self.top_pct = top_pct
        self.direction = direction
        self.n_layers = n_layers
        self.n_heads = n_heads

    @staticmethod
    def _derive_shape(config, n_layers, n_heads):
        if n_layers is None:
            n_layers = getattr(config, "num_layers", None) or getattr(
                config, "num_hidden_layers", None
            )
        if n_heads is None:
            n_heads = getattr(config, "num_heads", None) or getattr(
                config, "num_attention_heads", None
            )
        if n_layers is None or n_heads is None:
            raise ValueError(
                "StereotypicalValueAttribution could not derive n_layers/n_heads "
                "from model.config; pass them to the constructor."
            )
        return int(n_layers), int(n_heads)

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = unwrap(data if data is not None else legacy.pop("dataset", None), WordSets)

        if data is None:
            stereo, k1 = take(legacy, "stereo_sents")
            anti, k2 = take(legacy, "anti_sents")
            if None in (stereo, anti):
                raise ValueError(
                    "StereotypicalValueAttribution requires stereotypical and "
                    "anti-stereotypical sentences. Pass a WordSets as the second "
                    "argument, e.g. compute(model, WordSets(stereo, anti, stereo, anti))."
                )
            warn_legacy("StereotypicalValueAttribution", [k1, k2], "WordSets")
            legacy.pop(k1, None)
            legacy.pop(k2, None)
            data = WordSets(stereo, anti, stereo, anti)

        self._reject_unknown_kwargs(
            legacy, "n_samples", "top_pct", "direction", "n_layers", "n_heads"
        )
        if not isinstance(data, WordSets):
            raise TypeError(
                f"StereotypicalValueAttribution expects a WordSets as data, got "
                f"{type(data).__name__}."
            )

        tok, hf_model, _ = get_tokenizer_model(model, tokenizer, metric=self)
        n_layers, n_heads = self._derive_shape(
            hf_model.config,
            legacy.get("n_layers", self.n_layers),
            legacy.get("n_heads", self.n_heads),
        )

        stereo_sents = list(data.target_1)
        anti_sents = list(data.target_2)

        direction = legacy.get("direction", self.direction)
        if direction is None:
            from fairLMs.definitions.encoder_decoder.intrinsic_bias.stereotypical_association.sva.sva import (  # noqa: E501
                compute_stereotype_direction,
            )

            direction = compute_stereotype_direction(
                hf_model, tok, stereo_sents, anti_sents
            )
            derived_direction = True
        else:
            direction = np.asarray(direction, dtype=float)
            derived_direction = False
            expected = getattr(hf_model.config, "d_model", None) or getattr(
                hf_model.config, "hidden_size", None
            )
            if expected is not None and direction.shape[-1] != expected:
                raise ValueError(
                    f"direction has length {direction.shape[-1]} but the model's "
                    f"hidden size is {expected}. Omit direction= to derive it from "
                    f"the sentences."
                )

        sva, phi = compute_sva(
            hf_model,
            tok,
            stereo_sents,
            anti_sents,
            np.asarray(direction),
            n_layers,
            n_heads,
            n_samples=legacy.get("n_samples", self.n_samples),
            top_pct=legacy.get("top_pct", self.top_pct),
        )
        return MetricResult(
            score=float(sva),
            details={
                "phi": phi,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "direction_derived": derived_direction,
            },
        )
