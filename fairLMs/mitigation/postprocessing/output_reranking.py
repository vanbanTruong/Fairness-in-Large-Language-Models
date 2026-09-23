"""Output reranking: reorder candidates by declared quality and bias scorers."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from fairLMs.mitigation.base import MitigationResult, Mitigator
from fairLMs.mitigation.evidence import GENERATOR_RANK, CandidateSets

__all__ = ["OutputReranking"]

#: The objective every fitted rule records, so the sign convention is readable
#: off a stored rule rather than inferred from the class that produced it.
OBJECTIVE = "argmax lambda * q(y) + (1 - lambda) * f(y)"


def _score(
    fn: Callable[[Any, Any], float],
    query: Any,
    candidate: Any,
    *,
    owner: str,
    role: str,
) -> float:
    """Call one declared scorer and refuse anything that is not a real number."""
    value = fn(query, candidate)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{owner}: {role} must return a real number for "
            f"(query={query!r}, candidate={candidate!r}); got "
            f"{type(value).__name__}."
        )
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(
            f"{owner}: {role} returned a non-finite score for "
            f"(query={query!r}, candidate={candidate!r})."
        )
    return value


def _rerank_one(
    query: Any,
    candidates: Sequence[Any],
    *,
    scorer: Any,
    quality: Optional[Callable[[Any, Any], float]],
    lambda_: float,
    owner: str,
) -> Tuple[list, list, list]:
    """Reorder one candidate list; the single definition of the ordering.

    Implements ``argmax over candidates of lambda * q(y) + (1 - lambda) * f(y)``,
    with ``f`` the declared bias scorer and ``q`` the declared quality scorer.
    Shared by :meth:`OutputReranking._apply` and
    :meth:`OutputReranking.rerank` so a fitted rule cannot drift from the
    ordering that produced it.

    Returns ``(reordered_candidates, bias_scores, quality_scores)``, each aligned
    with the returned order.
    """
    n = len(candidates)
    scored = []
    for rank, candidate in enumerate(candidates):
        bias = _score(scorer, query, candidate, owner=owner, role="scorer")
        if quality is None:
            # No declared q: the generator's own ordering is the quality
            # signal, as a descending score in [0, 1].
            q = 1.0 - (rank / (n - 1)) if n > 1 else 1.0
        else:
            q = _score(quality, query, candidate, owner=owner, role="quality")
        # Both terms enter additively and as declared: the library never flips
        # the sign of a caller's f, because it cannot know its scale.
        combined = lambda_ * q + (1.0 - lambda_) * bias
        scored.append((combined, rank, candidate, bias, q))
    # Ties keep the generator's original order.
    scored.sort(key=lambda item: (-item[0], item[1]))
    return (
        [item[2] for item in scored],
        [item[3] for item in scored],
        [item[4] for item in scored],
    )


class OutputReranking(Mitigator):
    """Reorder candidate generations by **declared** quality and bias scorers.

    Selects, over the candidates generated for one query,

        y* = argmax over y in {y_1, ..., y_k} of
             lambda * q(y) + (1 - lambda) * f(y)

    and returns the full ordering that objective induces, not only its argmax.
    ``f`` is the declared **bias** scorer and ``q`` the declared **quality**
    scorer; both arrive on :class:`~fairLMs.mitigation.CandidateSets`, never
    from the library. Reranking trades the generator's own ordering for one the
    caller is willing to defend, and a built-in default would hide that choice.

    ``lambda_=1`` ranks purely by ``q`` and ``lambda_=0`` purely by ``f``. When
    ``CandidateSets.quality`` is omitted, ``q`` falls back to the generator's own
    ordering as a descending score in ``[0, 1]``, so ``lambda_=1`` then
    reproduces the original order exactly; the fallback is recorded as
    ``quality_name='generator_rank'`` so a result never hides which ``q``
    produced it.

    .. important::

       ``f`` is a **bias** score, and both terms are **added**, so the objective
       prefers candidates with *higher* ``f``. A caller who wants bias penalised
       must declare ``f`` oriented so that higher is better -- a negated bias
       score, for instance. The library neither flips nor rescales a declared
       scorer: it cannot know the caller's scale, and silently negating ``f``
       would rerank by a notion of bias the caller never declared. Putting ``q``
       and ``f`` on comparable scales is likewise the caller's decision, since
       normalising them would change the trade-off ``lambda_`` expresses.

    Parameters
    ----------
    lambda_:
        ``lambda``: the weight on ``q``, with ``1 - lambda_`` on ``f``, in
        ``[0, 1]``.

    Examples
    --------
    ``f`` is declared so that the gendered continuation scores *lower*, and
    ``lambda_=0`` ranks by ``f`` alone:

    >>> from fairLMs.mitigation import CandidateSets, OutputReranking
    >>> evidence = CandidateSets(
    ...     queries=["the nurse said"],
    ...     candidates=[["she smiled", "they smiled"]],
    ...     scorer=lambda q, c: -1.0 if "she" in c else 0.0,
    ...     scorer_name="doctest-gendered-pronoun",
    ... )
    >>> outcome = OutputReranking(lambda_=0.0).apply(None, evidence)
    >>> ranking, = outcome.result["rankings"]
    >>> tuple(ranking)
    ('they smiled', 'she smiled')

    A declared ``q`` replaces the generator's ordering, and ``lambda_`` trades
    the two off. Here ``q`` prefers the gendered continuation strongly enough to
    outweigh ``f`` at ``lambda_=0.6``:

    >>> evidence = CandidateSets(
    ...     queries=["the nurse said"],
    ...     candidates=[["she smiled", "they smiled"]],
    ...     scorer=lambda q, c: -1.0 if "she" in c else 0.0,
    ...     scorer_name="doctest-gendered-pronoun",
    ...     quality=lambda q, c: 1.0 if "she" in c else 0.0,
    ...     quality_name="doctest-quality",
    ... )
    >>> outcome = OutputReranking(lambda_=0.6).apply(None, evidence)
    >>> ranking, = outcome.result["rankings"]
    >>> tuple(ranking)
    ('she smiled', 'they smiled')
    >>> outcome = OutputReranking(lambda_=0.4).apply(None, evidence)
    >>> ranking, = outcome.result["rankings"]
    >>> tuple(ranking)
    ('they smiled', 'she smiled')
    """

    name = "output_reranking"
    category = "post"
    access = "black_box"
    # Generative-only, and that restriction lives here rather than in
    # `requires`: reranking reorders candidates the caller already generated and
    # reads nothing from a model, so declaring a capability would falsely demand
    # one be supplied.
    architectures = ("decoder_only", "encoder_decoder")
    requires = frozenset()
    accepts = (CandidateSets,)

    # Neither term is invisible by default: the endpoints, which drop `f` or `q`
    # entirely, are choices the caller has to make explicitly.
    def __init__(self, lambda_: float = 0.5):
        self.lambda_ = lambda_

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1]; got {self.lambda_!r}.")

        quality_name = evidence.quality_name or GENERATOR_RANK
        rankings, bias_scores, quality_scores = [], [], []
        for query, candidates in zip(evidence.queries, evidence.candidates):
            order, biases, qualities = _rerank_one(
                query,
                candidates,
                scorer=evidence.scorer,
                quality=evidence.quality,
                lambda_=self.lambda_,
                owner=self.name,
            )
            rankings.append(order)
            bias_scores.append(biases)
            quality_scores.append(qualities)

        return self._result(
            {
                "rankings": rankings,
                "bias_scores": bias_scores,
                "quality_scores": quality_scores,
                "scorer_name": evidence.scorer_name,
                "quality_name": quality_name,
                "lambda": self.lambda_,
                # The sign convention travels with the serialized rule, so a
                # rule read back later cannot be misread as penalizing `f`.
                "objective": OBJECTIVE,
            },
            n_queries=evidence.n_queries,
            scorer_name=evidence.scorer_name,
            quality_name=quality_name,
        )

    @staticmethod
    def rerank(
        rule: Dict[str, Any],
        query: Any,
        candidates: Sequence[Any],
        *,
        scorer: Any,
        quality: Optional[Callable[[Any, Any], float]] = None,
    ) -> Tuple[list, list, list]:
        """Apply a fitted rule from :attr:`MitigationResult.result` to new candidates.

        Mirrors :meth:`ScoreCalibration.transform` and
        :meth:`GroupAwareThresholding.decide`: the fitted object is a rule, and
        this re-applies it. Returns ``(reordered_candidates, bias_scores,
        quality_scores)``.

        The two scorers are passed in rather than stored, because a scorer is a
        live callable and a rule has to stay serializable. ``rule['scorer_name']``
        and ``rule['quality_name']`` record which ``f`` and ``q`` produced the
        fit, so the caller can check they are re-applying the same pair; a
        mismatch is refused rather than silently reordering under a different
        objective.
        """
        for key in ("lambda", "scorer_name", "quality_name"):
            if key not in rule:
                raise KeyError(
                    f"rule is missing {key!r}; pass MitigationResult.result from "
                    "OutputReranking."
                )
        _check_declared(rule["scorer_name"], scorer, role="scorer", notion="bias")
        fitted_quality = rule["quality_name"]
        if fitted_quality == GENERATOR_RANK:
            if quality is not None:
                raise ValueError(
                    "rule was fitted with no quality scorer (q was the "
                    "generator's own ordering) but one was supplied; "
                    "re-applying a rule under a different q would reorder by a "
                    "different objective."
                )
        elif quality is None:
            raise ValueError(
                f"rule was fitted with quality scorer {fitted_quality!r} but "
                "none was supplied; pass quality= so the rule is re-applied "
                "under the q it was fitted with."
            )
        else:
            _check_declared(fitted_quality, quality, role="quality", notion="quality")
        return _rerank_one(
            query,
            candidates,
            scorer=scorer,
            quality=quality,
            lambda_=float(rule["lambda"]),
            owner="OutputReranking.rerank",
        )


def _check_declared(fitted_name: Any, fn: Any, *, role: str, notion: str) -> None:
    """Refuse a callable that is not the one the rule was fitted with.

    Lambdas carry no usable name, so they are let through rather than reported
    as a mismatch the caller cannot fix.
    """
    declared = getattr(fn, "__name__", None) or type(fn).__name__
    if fitted_name in (declared, "<lambda>", None) or declared == "<lambda>":
        return
    raise ValueError(
        f"rule was fitted with {role} {fitted_name!r} but {declared!r} was "
        f"supplied; re-applying a rule under a different {role} would reorder "
        f"by a different notion of {notion}."
    )
