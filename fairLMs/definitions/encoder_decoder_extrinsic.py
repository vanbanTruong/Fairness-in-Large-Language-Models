"""Encoder-decoder extrinsic metrics: counterfactual AUC, IBS, NPD, translation SS."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from fairLMs.definitions.encoder_decoder.extrinsic_bias.counterfactual_fairness.auc import (
    compute_auc,
)
from fairLMs.definitions.encoder_decoder.extrinsic_bias.fair_inference.ibs import (
    compute_ibs,
)
from fairLMs.definitions.encoder_decoder.extrinsic_bias.individual_fairness.ss import (
    compute_ss as compute_translation_ss,
)
from fairLMs.definitions.encoder_decoder.extrinsic_bias.position_based.npd import (
    compute_npd,
)
from fairLMs.definitions._compat import as_examples, take, unwrap, warn_legacy
from fairLMs.definitions.base import FairnessMetric, MetricResult
from fairLMs.definitions.data import LabeledSentences, RECORD_CORPUS
from fairLMs.definitions.resolve import get_tokenizer_model


def _texts(data: Any, metric: str, what: str, key: str) -> List[str]:
    """Coerce a dataset/sequence to a list of strings."""
    examples = as_examples(data, metric, what)
    if isinstance(examples[0], str):
        return list(examples)
    out = []
    for ex in examples:
        if isinstance(ex, dict):
            value = ex.get(key) or ex.get("text") or ex.get("sentence") or ex.get("premise")
            if value is None:
                raise ValueError(
                    f"{metric}: example dict has no {key!r}/'text'/'sentence' key; "
                    f"got {sorted(ex)}."
                )
            out.append(value)
        else:
            raise TypeError(
                f"{metric}: expected strings or dicts, got {type(ex).__name__}."
            )
    return out


class CounterfactualAucScore(FairnessMetric):
    """Counterfactual fairness via protected-attribute recoverability (AUC).

    ``data`` is a :class:`~fairLMs.definitions.data.LabeledSentences`. An AUC near
    0.5 means the attribute is not linearly recoverable from the encoder
    representation; near 1.0 means it is.

    Parameters
    ----------
    test_ratio:
        Held-out fraction per split.
    seed:
        Base RNG seed.
    n_seeds:
        Number of random splits averaged.
    """

    name = "counterfactual_auc"
    bias_type = "extrinsic"
    architectures = ("encoder_decoder",)
    required_task = "seq2seq"
    requires = frozenset({"hidden_states", "local_tokenizer"})
    accepts = (LabeledSentences,)

    def __init__(self, *, test_ratio: float = 0.2, seed: int = 42, n_seeds: int = 10):
        self.test_ratio = test_ratio
        self.seed = seed
        self.n_seeds = n_seeds

    @staticmethod
    def _check_probe_is_estimable(labels, pair_ids, test_ratio: float) -> None:
        """Refuse inputs on which no AUC can be estimated.

        The underlying ``compute_auc`` returns ``0.0`` when it cannot fit a
        probe, so that a caller can still inspect the rows. Read as a score,
        though, ``0.0`` is the *most extreme possible finding*, a perfectly
        anti-recoverable attribute, and it is indistinguishable from "there
        was nothing to fit". Every condition that triggers the short-circuit
        is decidable from the labels alone, so decide it here instead.
        """
        non_int = sorted(
            {repr(label) for label in labels if isinstance(label, bool)
             or not isinstance(label, (int, np.integer))}
        )
        if non_int:
            raise TypeError(
                f"CounterfactualAucScore labels must be integer class ids, got "
                f"{', '.join(non_int[:5])}. The probe is a binary classifier: "
                f"string labels count as neither class, which would report "
                f"n_class_0=0, n_class_1=0 and a score of 0.0 without failing. "
                f"Encode the attribute first, e.g. "
                f"[0 if g == 'male' else 1 for g in groups]."
            )

        classes = sorted({int(label) for label in labels})
        if classes and classes != [0, 1]:
            raise ValueError(
                f"CounterfactualAucScore expects exactly two classes labelled 0 "
                f"and 1, got {classes}."
            )

        n0 = sum(1 for label in labels if int(label) == 0)
        n1 = sum(1 for label in labels if int(label) == 1)
        if n0 < 2 or n1 < 2:
            raise ValueError(
                f"CounterfactualAucScore needs at least two members of each "
                f"class to fit and evaluate a probe, got n_class_0={n0}, "
                f"n_class_1={n1}."
            )

        # A held-out set of one row can never contain both classes, so every
        # split is rejected and no AUC is ever defined.
        n_units = len(set(pair_ids)) if pair_ids is not None else len(labels)
        n_test = max(1, round(n_units * test_ratio))
        if n_test < 2:
            raise ValueError(
                f"CounterfactualAucScore: test_ratio={test_ratio} holds out "
                f"{n_test} of {n_units} "
                f"{'pairs' if pair_ids is not None else 'rows'}, which cannot "
                f"contain both classes, so no split is scorable. Raise "
                f"test_ratio or supply more data."
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
            data if data is not None else legacy.pop("dataset", None), LabeledSentences
        )

        if data is None:
            sents, k1 = take(legacy, "sentences")
            labels, k2 = take(legacy, "labels")
            if None in (sents, labels):
                raise ValueError(
                    "CounterfactualAucScore requires sentences and labels. Pass a "
                    "LabeledSentences as the second argument, e.g. "
                    "CounterfactualAucScore().compute(model, "
                    "LabeledSentences(sentences, labels))."
                )
            warn_legacy("CounterfactualAucScore", [k1, k2], "LabeledSentences")
            legacy.pop(k1, None)
            legacy.pop(k2, None)
            pair_ids, k3 = take(legacy, "pair_ids")
            legacy.pop(k3, None)
            data = LabeledSentences(sents, labels, pair_ids=pair_ids)

        self._reject_unknown_kwargs(legacy, "test_ratio", "seed", "n_seeds", "pair_ids")

        if not isinstance(data, LabeledSentences):
            data = LabeledSentences.from_examples(
                as_examples(
                    data,
                    "CounterfactualAucScore",
                    "a LabeledSentences or sequence of (sentence, label)",
                )
            )

        test_ratio = legacy.get("test_ratio", self.test_ratio)
        pair_ids = list(data.pair_ids) if data.pair_ids is not None else None
        self._check_probe_is_estimable(list(data.labels), pair_ids, test_ratio)

        tok, hf_model, _ = get_tokenizer_model(model, tokenizer, metric=self)
        auc_mean, auc_std, n0, n1, rows = compute_auc(
            hf_model,
            tok,
            list(data.sentences),
            list(data.labels),
            pair_ids=pair_ids,
            test_ratio=test_ratio,
            seed=legacy.get("seed", self.seed),
            n_seeds=legacy.get("n_seeds", self.n_seeds),
        )
        return MetricResult(
            score=float(auc_mean),
            details={
                "auc_mean": float(auc_mean),
                "auc_std": auc_std,
                "n_class_0": n0,
                "n_class_1": n1,
                "rows": rows,
            },
        )


class InferenceBiasScore(FairnessMetric):
    """Idealized Bias Score (IBS) over ``(label, prediction)`` pairs.

    ``data`` is a sequence of 2-tuples. No model is used.
    """

    name = "inference_bias_score"
    bias_type = "extrinsic"
    architectures = ("encoder_decoder",)
    requires = frozenset(set())
    accepts = RECORD_CORPUS

    def compute(
        self, model: Any = None, data: Any = None, **legacy: Any
    ) -> MetricResult:
        data = unwrap(data if data is not None else legacy.pop("dataset", None))
        if data is None:
            data, key = take(legacy, "predictions")
            if data is None:
                raise ValueError(
                    "InferenceBiasScore requires (label, prediction) pairs. Pass "
                    "them as the second argument."
                )
            warn_legacy("InferenceBiasScore", [key], "the (label, prediction) pairs")
            legacy.pop(key, None)
        self._reject_unknown_kwargs(legacy)

        pairs = as_examples(
            data, "InferenceBiasScore", "a sequence of (label, prediction) pairs"
        )
        bad = [p for p in pairs if not isinstance(p, (list, tuple)) or len(p) < 2]
        if bad:
            raise ValueError(
                f"InferenceBiasScore: each item must be a (label, prediction) pair; "
                f"got {bad[0]!r}."
            )
        ibs, counts = compute_ibs(pairs)
        return MetricResult(
            score=float(ibs), details={"counts": counts, "n": len(pairs)}
        )


class NormalizedPositionDistance(FairnessMetric):
    """Normalized position disparity for summarization (position bias).

    ``data`` is a sequence of article strings (or dicts with an ``article`` key).

    Parameters
    ----------
    max_new_tokens:
        Generation budget per article.
    K:
        Number of leading sentences treated as the lead baseline.
    gold_summaries:
        Optional reference summaries, aligned with the articles.
    """

    name = "normalized_position_distance"
    bias_type = "extrinsic"
    architectures = ("encoder_decoder",)
    required_task = "seq2seq"
    requires = frozenset({"free_generation", "local_tokenizer"})
    accepts = RECORD_CORPUS

    def __init__(
        self,
        *,
        max_new_tokens: int = 128,
        K: int = 10,
        gold_summaries: Optional[Sequence[str]] = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.K = K
        self.gold_summaries = gold_summaries

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
            data, key = take(legacy, "articles")
            if data is None:
                raise ValueError(
                    "NormalizedPositionDistance requires articles. Pass them as the "
                    "second argument, e.g. compute(model, [article_text, ...])."
                )
            warn_legacy("NormalizedPositionDistance", [key], "the article list")
            legacy.pop(key, None)

        self._reject_unknown_kwargs(legacy, "max_new_tokens", "K", "gold_summaries")
        articles = _texts(
            data, "NormalizedPositionDistance", "a sequence of articles", "article"
        )

        tok, hf_model, _ = get_tokenizer_model(model, tokenizer, metric=self)
        mean_npd, rows = compute_npd(
            hf_model,
            tok,
            articles,
            gold_summaries=legacy.get("gold_summaries", self.gold_summaries),
            max_new_tokens=legacy.get("max_new_tokens", self.max_new_tokens),
            K=legacy.get("K", self.K),
        )
        return MetricResult(
            score=float(mean_npd), details={"rows": rows, "n_articles": len(articles)}
        )


class TranslationSimilarityScore(FairnessMetric):
    """Semantic similarity of counterfactual translations (LaBSE / SS).

    ``data`` is a sequence of ``(original, counterfactual)`` sentence pairs.

    Parameters
    ----------
    labse_model, labse_tokenizer:
        The sentence encoder used to measure similarity of the two translations.
        Required; there is no sensible default.
    tgt_lang:
        Target language name inserted into the translation prompt.
    max_new_tokens:
        Generation budget per sentence.
    """

    name = "translation_similarity_score"
    bias_type = "extrinsic"
    architectures = ("encoder_decoder",)
    required_task = "seq2seq"
    requires = frozenset({"free_generation", "local_tokenizer"})
    accepts = RECORD_CORPUS

    def __init__(
        self,
        *,
        labse_model: Any = None,
        labse_tokenizer: Any = None,
        tgt_lang: str = "French",
        max_new_tokens: int = 128,
    ):
        self.labse_model = labse_model
        self.labse_tokenizer = labse_tokenizer
        self.tgt_lang = tgt_lang
        self.max_new_tokens = max_new_tokens

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
            data, key = take(legacy, "pairs")
            if data is None:
                raise ValueError(
                    "TranslationSimilarityScore requires (original, counterfactual) "
                    "pairs. Pass them as the second argument."
                )
            warn_legacy("TranslationSimilarityScore", [key], "the sentence pairs")
            legacy.pop(key, None)

        self._reject_unknown_kwargs(
            legacy, "labse_model", "labse_tokenizer", "tgt_lang", "max_new_tokens"
        )
        labse_model = legacy.get("labse_model", self.labse_model)
        labse_tokenizer = legacy.get("labse_tokenizer", self.labse_tokenizer)
        if labse_model is None or labse_tokenizer is None:
            raise ValueError(
                "TranslationSimilarityScore needs a sentence encoder to compare "
                "translations. Pass labse_model= and labse_tokenizer= to the "
                "constructor."
            )

        pairs = as_examples(
            data,
            "TranslationSimilarityScore",
            "a sequence of (original, counterfactual) pairs",
        )
        bad = [p for p in pairs if not isinstance(p, (list, tuple)) or len(p) < 2]
        if bad:
            raise ValueError(
                f"TranslationSimilarityScore: each item must be an "
                f"(original, counterfactual) pair; got {bad[0]!r}."
            )

        tok, hf_model, _ = get_tokenizer_model(model, tokenizer, metric=self)
        mean_ss, std_ss, rows = compute_translation_ss(
            hf_model,
            tok,
            labse_model,
            labse_tokenizer,
            pairs,
            tgt_lang=legacy.get("tgt_lang", self.tgt_lang),
            max_new_tokens=legacy.get("max_new_tokens", self.max_new_tokens),
        )
        return MetricResult(
            score=float(mean_ss),
            details={"mean": float(mean_ss), "std": std_ss, "rows": rows, "n": len(pairs)},
        )
