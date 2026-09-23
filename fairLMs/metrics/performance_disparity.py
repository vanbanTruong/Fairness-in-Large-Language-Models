"""Performance disparity metrics: AD, BA, SNS."""

from __future__ import annotations

from typing import Any, Callable

from fairLMs.definition.decoder_only.extrinsic_bias.performance_disparity.ad.ad import (
    compute_ad,
)
from fairLMs.definition.decoder_only.extrinsic_bias.performance_disparity.ba.ba import (
    compute_ba,
)
from fairLMs.definition.decoder_only.extrinsic_bias.performance_disparity.sns.sns import (
    compute_sns,
)
from fairLMs.metrics._compat import as_examples, take, unwrap, warn_legacy
from fairLMs.metrics.base import FairnessMetric, MetricResult
from fairLMs.metrics.data import GroupProperties, QuerySpec, ScorePair
from fairLMs.metrics.resolve import get_openai_bundle


class AccuracyDisparity(FairnessMetric):
    """Accuracy gap between a stereotyped set and its counter-stereotyped twin.

    ``data`` is a :class:`~fairLMs.metrics.data.ScorePair` of per-item scores.
    No model is used; a plain-function equivalent lives in
    :mod:`fairLMs.metrics.functional`.
    """

    name = "accuracy_disparity"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)
    requires = frozenset(set())
    accepts = (ScorePair,)

    def compute(
        self, model: Any = None, data: Any = None, **legacy: Any
    ) -> MetricResult:
        data = unwrap(
            data if data is not None else legacy.pop("dataset", None), ScorePair
        )

        if not isinstance(data, ScorePair):
            s, k1 = take(legacy, "scores_s")
            sp, k2 = take(legacy, "scores_sp")
            if s is not None and sp is not None:
                warn_legacy("AccuracyDisparity", [k1, k2], "ScorePair")
                legacy.pop(k1, None)
                legacy.pop(k2, None)
                data = ScorePair(s, sp)
            elif data is not None:
                data = ScorePair.from_examples(
                    as_examples(
                        data,
                        "AccuracyDisparity",
                        "a ScorePair or sequence of (stereotype, counter) scores",
                    )
                )
            else:
                raise ValueError(
                    "AccuracyDisparity requires two score lists. Pass a ScorePair as "
                    "the second argument, e.g. compute(None, ScorePair(s, sp))."
                )

        self._reject_unknown_kwargs(legacy)
        acc_s, acc_sp, ad = compute_ad(
            list(data.stereotype), list(data.counter_stereotype)
        )
        return MetricResult(
            score=float(ad),
            details={
                "accuracy_stereotype": acc_s,
                "accuracy_counter": acc_sp,
                "n": (len(data.stereotype), len(data.counter_stereotype)),
            },
        )


class BiasAmplifierScore(FairnessMetric):
    """BiasAsker absolute & relative bias (BA). Requires ``OPENAI_API_KEY``.

    ``data`` is a :class:`~fairLMs.metrics.data.GroupProperties`. Reports
    absolute bias ``ab`` as the score; relative bias ``rb`` is in ``details``.

    Parameters
    ----------
    completion_model:
        Completions model used for the forced-choice scoring. This is now
        actually forwarded; it previously had no effect.
    """

    name = "bias_amplifier"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)
    requires = frozenset({"token_logprobs", "completions_api"})
    accepts = (GroupProperties,)

    def __init__(self, *, completion_model: str | None = None):
        self.completion_model = completion_model

    def compute(
        self, model: Any = None, data: Any = None, **legacy: Any
    ) -> MetricResult:
        data = unwrap(
            data if data is not None else legacy.pop("dataset", None), GroupProperties
        )

        if not isinstance(data, GroupProperties):
            groups, k1 = take(legacy, "groups")
            props, k2 = take(legacy, "properties")
            ab_t, k3 = take(legacy, "ab_template")
            rb_t, k4 = take(legacy, "rb_template")
            if None in (groups, props, ab_t, rb_t):
                raise ValueError(
                    "BiasAmplifierScore requires groups, properties and two prompt "
                    "templates. Pass a GroupProperties as the second argument."
                )
            warn_legacy("BiasAmplifierScore", [k1, k2, k3, k4], "GroupProperties")
            for key in (k1, k2, k3, k4):
                legacy.pop(key, None)
            data = GroupProperties(groups, props, ab_t, rb_t)

        # max_new_tokens was accepted but never used by compute_ba.
        if "max_new_tokens" in legacy:
            warn_legacy("BiasAmplifierScore", ["max_new_tokens"], "GroupProperties")
            legacy.pop("max_new_tokens")
        self._reject_unknown_kwargs(legacy, "completion_model")

        bundle = get_openai_bundle(model, metric=self)
        completion_model = (
            legacy.get("completion_model", self.completion_model) or bundle.model
        )
        ab, rb, ab_rows, rb_rows = compute_ba(
            bundle.client,
            list(data.groups),
            list(data.properties),
            data.ab_template,
            data.rb_template,
            model=completion_model,
        )
        return MetricResult(
            score=float(ab),
            details={
                "ab": ab,
                "rb": rb,
                "ab_rows": ab_rows,
                "rb_rows": rb_rows,
                "completion_model": completion_model,
            },
        )


class SensitiveNameSimilarity(FairnessMetric):
    """Sensitive-to-neutral response similarity (SNS).

    ``data`` is a :class:`~fairLMs.metrics.data.QuerySpec`. ``model`` must be a
    callable taking a prompt string and returning the model's response.

    Reports ``snsr`` (similarity ratio) as the score; ``snsv`` (variance) is in
    ``details``.

    Parameters
    ----------
    k:
        Top-k responses compared per query.
    """

    name = "sensitive_name_similarity"
    bias_type = "extrinsic"
    architectures = ("decoder_only",)
    requires = frozenset({"free_generation"})
    accepts = (QuerySpec,)

    def __init__(self, *, k: int = 5):
        self.k = k

    def compute(
        self, model: Any = None, data: Any = None, **legacy: Any
    ) -> MetricResult:
        data = unwrap(
            data if data is not None else legacy.pop("dataset", None), QuerySpec
        )

        if not isinstance(data, QuerySpec):
            queries, k1 = take(legacy, "queries")
            neutral_fn, k2 = take(legacy, "neutral_prompt_fn")
            group_fn, k3 = take(legacy, "group_prompt_fn")
            values, k4 = take(legacy, "group_values")
            if None in (queries, neutral_fn, group_fn, values):
                raise ValueError(
                    "SensitiveNameSimilarity requires queries, two prompt builders "
                    "and group values. Pass a QuerySpec as the second argument."
                )
            warn_legacy("SensitiveNameSimilarity", [k1, k2, k3, k4], "QuerySpec")
            for key in (k1, k2, k3, k4):
                legacy.pop(key, None)
            data = QuerySpec(queries, neutral_fn, group_fn, values)

        self._reject_unknown_kwargs(legacy, "k", "call_model")
        call_model: Callable = legacy.get("call_model") or model
        if not callable(call_model):
            raise TypeError(
                "SensitiveNameSimilarity needs a callable that maps a prompt string "
                "to a response string as `model`."
            )

        k = legacy.get("k", self.k)
        snsr, snsv, df = compute_sns(
            call_model,
            list(data.queries),
            data.neutral_prompt_fn,
            data.group_prompt_fn,
            list(data.group_values),
            k=k,
        )
        return MetricResult(
            score=float(snsr),
            details={"snsr": snsr, "snsv": snsv, "table": df, "k": k},
        )
