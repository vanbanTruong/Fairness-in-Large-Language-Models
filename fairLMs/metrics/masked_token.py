"""Masked-token metrics: DisCo, LPBS, CBS."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from fairLMs.definition.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.cbs.cbs import (  # noqa: E501
    compute_cbs,
)
from fairLMs.definition.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.disco.disco import (  # noqa: E501
    compute_disco,
)
from fairLMs.definition.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.lpbs.lpbs import (  # noqa: E501
    DEFAULT_TEMPLATE,
    compute_lpbs,
)
from fairLMs.metrics._compat import as_examples, take, unwrap, warn_legacy
from fairLMs.metrics.base import FairnessMetric, MetricResult
from fairLMs.metrics.data import ContrastSpec, GroupWordPairs, RECORD_CORPUS
from fairLMs.metrics.resolve import get_tokenizer_model


class DiscoveryOfCorrelationsScore(FairnessMetric):
    """DisCo: top-k prediction divergence across demographic word pairs.

    ``data`` is a :class:`~fairLMs.metrics.data.GroupWordPairs`. The two word
    lists are compared **pairwise**, so they must be the same length.

    Parameters
    ----------
    k:
        Number of top fill-mask predictions compared per template.
    n_bootstrap:
        Bootstrap resamples for the confidence interval.
    seed:
        Bootstrap RNG seed.
    templates:
        Fill-mask templates. ``None`` uses the built-in set.
    """

    name = "discovery_of_correlations"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)
    required_task = "mlm"
    requires = frozenset({"masked_token_scores", "local_tokenizer"})
    accepts = (GroupWordPairs,)

    def __init__(
        self,
        *,
        k: int = 3,
        n_bootstrap: int = 1000,
        seed: int = 42,
        templates: Optional[Sequence[str]] = None,
    ):
        self.k = k
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.templates = templates

    def _as_pipeline(self, model, tokenizer):
        """Return a fill-mask pipeline, building one from an MLM if needed."""
        if model is None:
            raise ValueError(
                "DiscoveryOfCorrelationsScore requires a fill-mask pipeline or a "
                "masked LM as `model`."
            )
        # An already-constructed transformers pipeline is callable and has .tokenizer
        if callable(model) and hasattr(model, "tokenizer"):
            return model
        from transformers import pipeline

        tok, hf_model, device = get_tokenizer_model(model, tokenizer, metric=self)
        return pipeline(
            "fill-mask",
            model=hf_model,
            tokenizer=tok,
            device=0 if str(device).startswith("cuda") else -1,
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
            data if data is not None else legacy.pop("dataset", None), GroupWordPairs
        )

        if data is None:
            g1, k1 = take(legacy, "group1_words")
            g2, k2 = take(legacy, "group2_words")
            if None in (g1, g2):
                raise ValueError(
                    "DiscoveryOfCorrelationsScore requires two aligned word lists. "
                    "Pass a GroupWordPairs as the second argument, e.g. "
                    "DiscoveryOfCorrelationsScore().compute(model, "
                    'GroupWordPairs(["he"], ["she"])).'
                )
            warn_legacy("DiscoveryOfCorrelationsScore", [k1, k2], "GroupWordPairs")
            legacy.pop(k1, None)
            legacy.pop(k2, None)
            data = GroupWordPairs(g1, g2)

        self._reject_unknown_kwargs(
            legacy, "k", "n_bootstrap", "seed", "templates", "pipe"
        )
        pipe_override, _ = take(legacy, "pipe")

        if not isinstance(data, GroupWordPairs):
            raise TypeError(
                f"DiscoveryOfCorrelationsScore expects a GroupWordPairs as data, "
                f"got {type(data).__name__}."
            )

        pipe = self._as_pipeline(pipe_override or model, tokenizer)
        k = legacy.get("k", self.k)
        n_bootstrap = legacy.get("n_bootstrap", self.n_bootstrap)
        seed = legacy.get("seed", self.seed)
        templates = legacy.get("templates", self.templates)

        disco, ci_low, ci_high = compute_disco(
            pipe,
            list(data.group_1),
            list(data.group_2),
            templates=templates,
            k=k,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        return MetricResult(
            score=float(disco),
            details={
                "ci_low": ci_low,
                "ci_high": ci_high,
                "k": k,
                "n_bootstrap": n_bootstrap,
                "n_pairs": len(data.group_1),
            },
        )


class LogProbabilityBiasScore(FairnessMetric):
    """Kurita et al. (2019) log-probability bias score for masked LMs.

    ``data`` is a sequence of attribute words (e.g. professions), or any dataset
    whose examples carry ``profession_name`` / ``attribute`` / ``word``.

    Parameters
    ----------
    gender_words:
        The two demographic tokens to contrast.
    template:
        Template with a ``GGG`` group slot and an ``XXX`` attribute slot.
    gender_comes_first:
        Whether the group slot precedes the attribute slot in ``template``.
    """

    name = "log_probability_bias_score"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)
    required_task = "mlm"
    requires = frozenset({"masked_token_scores", "local_tokenizer"})
    accepts = RECORD_CORPUS

    def __init__(
        self,
        *,
        gender_words: Tuple[str, str] = ("he", "she"),
        template: str = DEFAULT_TEMPLATE,
        gender_comes_first: bool = True,
    ):
        self.gender_words = gender_words
        self.template = template
        self.gender_comes_first = gender_comes_first

    @staticmethod
    def _attribute_words(examples) -> list:
        if examples and isinstance(examples[0], dict):
            words = [
                ex.get("profession_name") or ex.get("attribute") or ex.get("word")
                for ex in examples
            ]
            words = [w for w in words if w]
            if not words:
                raise ValueError(
                    "LogProbabilityBiasScore: no attribute words found. Examples "
                    "need a 'profession_name', 'attribute' or 'word' key."
                )
            return words
        return list(examples)

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = unwrap(data if data is not None else legacy.pop("dataset", None))

        if data is None:
            data, key = take(legacy, "attribute_words")
            if data is None:
                raise ValueError(
                    "LogProbabilityBiasScore requires attribute words. Pass them as "
                    "the second argument, e.g. LogProbabilityBiasScore().compute("
                    'model, ["nurse", "engineer"]).'
                )
            warn_legacy("LogProbabilityBiasScore", [key], "the attribute word list")
            legacy.pop(key, None)

        self._reject_unknown_kwargs(
            legacy, "gender_words", "template", "gender_comes_first"
        )
        gender_words = legacy.get("gender_words", self.gender_words)
        template = legacy.get("template", self.template)
        gender_comes_first = legacy.get("gender_comes_first", self.gender_comes_first)

        words = self._attribute_words(
            as_examples(data, "LogProbabilityBiasScore", "a sequence of attribute words")
        )

        tok, hf_model, _ = get_tokenizer_model(model, tokenizer, metric=self)
        outcomes, mean_lpbs, std_lpbs, prop = compute_lpbs(
            tok,
            hf_model,
            gender_words,
            words,
            template=template,
            gender_comes_first=gender_comes_first,
        )
        return MetricResult(
            score=float(mean_lpbs),
            details={
                "std": std_lpbs,
                "proportion_favoring_group1": prop,
                "outcomes": outcomes,
                "n_attributes": len(words),
                "gender_words": tuple(gender_words),
                "template": template,
            },
        )


class ContrastBasedScore(FairnessMetric):
    """Contrast-based masked preference score (CBS).

    ``data`` is a :class:`~fairLMs.metrics.data.ContrastSpec` carrying the group
    terms, ``(negative, positive, stereo_group)`` triples, and templates.

    The reported ``score`` is the **confirmatory** CBS for the declared stereo
    groups when any are declared, else the mean exploratory CBS across groups.

    Parameters
    ----------
    n_bootstrap:
        Bootstrap resamples for confidence intervals.
    n_perm:
        Permutations for the multiplicity-corrected null.
    seed:
        RNG seed.
    group_placeholder, attr_placeholder:
        Substrings in ``templates`` replaced by the group term and attribute.
    """

    name = "contrast_based_score"
    bias_type = "intrinsic"
    architectures = ("encoder_only",)
    required_task = "mlm"
    requires = frozenset({"masked_token_scores", "local_tokenizer"})
    accepts = (ContrastSpec,)

    def __init__(
        self,
        *,
        n_bootstrap: int = 1000,
        n_perm: int = 1000,
        seed: int = 42,
        group_placeholder: str = "{N}",
        attr_placeholder: str = "{A}",
    ):
        self.n_bootstrap = n_bootstrap
        self.n_perm = n_perm
        self.seed = seed
        self.group_placeholder = group_placeholder
        self.attr_placeholder = attr_placeholder

    def compute(
        self,
        model: Any = None,
        data: Any = None,
        *,
        tokenizer: Any = None,
        **legacy: Any,
    ) -> MetricResult:
        data = unwrap(
            data if data is not None else legacy.pop("dataset", None), ContrastSpec
        )

        if data is None:
            terms, k1 = take(legacy, "group_terms")
            pairs, k2 = take(legacy, "contrast_pairs")
            templates, k3 = take(legacy, "templates")
            if None in (terms, pairs, templates):
                raise ValueError(
                    "ContrastBasedScore requires group terms, contrast triples and "
                    "templates. Pass a ContrastSpec as the second argument, e.g. "
                    "ContrastBasedScore().compute(model, ContrastSpec("
                    '["he", "she"], [("dumb", "smart", "she")], ["{N} is {A}."])).'
                )
            warn_legacy("ContrastBasedScore", [k1, k2, k3], "ContrastSpec")
            for key in (k1, k2, k3):
                legacy.pop(key, None)
            data = ContrastSpec(terms, pairs, templates)

        self._reject_unknown_kwargs(
            legacy,
            "n_bootstrap",
            "n_perm",
            "seed",
            "group_placeholder",
            "attr_placeholder",
        )
        if not isinstance(data, ContrastSpec):
            raise TypeError(
                f"ContrastBasedScore expects a ContrastSpec as data, got "
                f"{type(data).__name__}."
            )

        tok, hf_model, _ = get_tokenizer_model(model, tokenizer, metric=self)
        per_group, info = compute_cbs(
            tok,
            hf_model,
            list(data.group_terms),
            list(data.contrast_pairs),
            list(data.templates),
            n_bootstrap=legacy.get("n_bootstrap", self.n_bootstrap),
            n_perm=legacy.get("n_perm", self.n_perm),
            seed=legacy.get("seed", self.seed),
            group_placeholder=legacy.get("group_placeholder", self.group_placeholder),
            attr_placeholder=legacy.get("attr_placeholder", self.attr_placeholder),
        )

        stereo = info.get("stereo") if isinstance(info, dict) else None
        if isinstance(stereo, dict) and stereo.get("cbs") is not None:
            score = float(stereo["cbs"])
            basis = "confirmatory"
        else:
            scores = [
                v["cbs"]
                for v in (per_group or {}).values()
                if isinstance(v, dict) and v.get("cbs") is not None
            ]
            score = float(sum(scores) / len(scores)) if scores else float("nan")
            basis = "exploratory_mean"
        return MetricResult(
            score=score,
            details={"per_group": per_group, "info": info, "score_basis": basis},
        )
