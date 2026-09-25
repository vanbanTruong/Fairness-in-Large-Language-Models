"""Post-processing components: known values and mathematical invariants."""

import math

import pytest

from fairLMs.datasets.diagnostics import LabeledScoredGroups, ScoredGroups
from fairLMs.mitigation import (
    CandidateSets,
    OutputReranking,
    GroupAwareThresholding,
    ScoreCalibration,
)


def _evidence(groups, scores, labels, *, positive="yes"):
    return LabeledScoredGroups(
        scored=ScoredGroups(
            axis="gender",
            groups=groups,
            scores=scores,
            score_name="p",
            source="unit-test",
        ),
        labels=labels,
        label_name="outcome",
        positive_label=positive,
    )


SEPARABLE = _evidence(
    groups=["f"] * 4 + ["m"] * 4,
    scores=[0.1, 0.2, 0.8, 0.9, 0.15, 0.25, 0.85, 0.95],
    labels=["no", "no", "yes", "yes", "no", "no", "yes", "yes"],
)

#: 'm' scores are shifted up by 0.5 relative to 'f'. Both groups separate their
#: own classes perfectly, so the correct per-group thresholds are recoverable
#: exactly and the expected answer can be asserted rather than approximated.
SHIFTED = _evidence(
    groups=["f"] * 4 + ["m"] * 4,
    scores=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
    labels=["no", "no", "yes", "yes", "no", "no", "yes", "yes"],
)


class TestScoreCalibration:
    def test_fits_one_rule_per_group(self):
        result = ScoreCalibration().apply(None, SEPARABLE)
        assert result.category == "post"
        assert sorted(result.result["groups"]) == ["f", "m"]

    def test_platt_is_monotone_increasing_in_the_score(self):
        rule = ScoreCalibration(method="platt").apply(None, SEPARABLE).result
        values = [
            ScoreCalibration.transform(rule, "f", s)
            for s in [0.0, 0.25, 0.5, 0.75, 1.0]
        ]
        assert values == sorted(values)
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_isotonic_is_monotone_non_decreasing(self):
        rule = ScoreCalibration(method="isotonic").apply(None, SEPARABLE).result
        values = [
            ScoreCalibration.transform(rule, "m", s)
            for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ]
        assert values == sorted(values)

    def test_isotonic_reproduces_a_perfectly_ordered_group(self):
        # Known value: with labels already sorted by score, PAVA is the identity
        # and the fitted step function returns exactly 0 then 1.
        rule = ScoreCalibration(method="isotonic").apply(None, SEPARABLE).result
        assert ScoreCalibration.transform(rule, "f", 0.1) == 0.0
        assert ScoreCalibration.transform(rule, "f", 0.9) == 1.0

    def test_platt_separates_the_two_classes(self):
        rule = ScoreCalibration(method="platt").apply(None, SEPARABLE).result
        low = ScoreCalibration.transform(rule, "f", 0.1)
        high = ScoreCalibration.transform(rule, "f", 0.9)
        assert low < 0.5 < high

    def test_an_undersized_group_raises_rather_than_being_skipped(self):
        evidence = _evidence(
            groups=["f", "f", "f", "f", "m", "m"],
            scores=[0.1, 0.2, 0.8, 0.9, 0.3, 0.7],
            labels=["no", "no", "yes", "yes", "no", "yes"],
        )
        with pytest.raises(ValueError, match="fewer than the 3"):
            ScoreCalibration(min_rows_per_group=3).apply(None, evidence)

    def test_a_single_class_group_raises(self):
        evidence = _evidence(
            groups=["f"] * 4 + ["m"] * 4,
            scores=[0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4],
            labels=["no", "no", "yes", "yes", "no", "no", "no", "no"],
        )
        with pytest.raises(ValueError, match="only one outcome class"):
            ScoreCalibration().apply(None, evidence)

    def test_an_unknown_method_is_refused(self):
        with pytest.raises(ValueError, match="must be 'platt' or 'isotonic'"):
            ScoreCalibration(method="sigmoid").apply(None, SEPARABLE)

    def test_result_serializes_and_roundtrips(self):
        import json

        result = ScoreCalibration().apply(None, SEPARABLE)
        assert result.is_serializable
        restored = json.loads(result.to_json())
        assert restored["category"] == "post"
        assert restored["provenance"]["config"]["method"] == "platt"

    def test_transform_refuses_an_unfitted_group(self):
        rule = ScoreCalibration().apply(None, SEPARABLE).result
        with pytest.raises(KeyError, match="no calibrator was fitted"):
            ScoreCalibration.transform(rule, "nonbinary", 0.5)


class TestGroupAwareThresholding:
    def test_fits_one_threshold_per_group(self):
        rule = GroupAwareThresholding().apply(None, SEPARABLE).result
        assert sorted(rule["thresholds"]) == ["f", "m"]

    def test_equal_opportunity_closes_the_tpr_gap_on_shifted_scores(self):
        # 'm' scores are shifted up by 0.5; a single global threshold would
        # advantage them. Per-group thresholds should recover equal TPR.
        rule = GroupAwareThresholding().apply(None, SHIFTED).result
        assert rule["achieved_tpr_gap"] == pytest.approx(0.0)
        assert rule["thresholds"]["f"] != rule["thresholds"]["m"]
        # Closing the gap is not enough on its own: assert the rule is also
        # useful, or "reject everyone" would satisfy the line above.
        assert all(tpr > 0.0 for tpr in rule["tpr"].values())

    def test_the_shift_is_recovered_exactly(self):
        # Known value. The shift is 0.5 and both groups separate perfectly, so
        # the correct per-group thresholds admit every positive and no negative.
        rule = GroupAwareThresholding().apply(None, SHIFTED).result
        assert rule["thresholds"] == {"f": 0.3, "m": 0.8}
        assert rule["tpr"] == {"f": 1.0, "m": 1.0}
        assert rule["fpr"] == {"f": 0.0, "m": 0.0}

    @pytest.mark.parametrize("criterion", ["equal_opportunity", "equalized_odds"])
    def test_the_degenerate_equalizers_are_not_returned(self, criterion):
        """Rejecting or accepting everyone equalises every rate, uselessly.

        Both have a gap of exactly zero, so a search that minimises only the
        gap terminates on one of them and reports a perfect score. This is the
        regression guard for that: a real bug, caught by a documented example
        that printed thresholds above every observed score.
        """
        rule = GroupAwareThresholding(criterion).apply(None, SHIFTED).result
        rates = list(rule["tpr"].values()) + list(rule["fpr"].values())
        assert not all(r == 0.0 for r in rates), "returned the reject-all rule"
        assert not all(r == 1.0 for r in rates), "returned the accept-all rule"
        # Youden's J is 0 for both degenerate points and positive otherwise.
        assert rule["achieved_utility"] > 0.0

    def test_no_threshold_sits_outside_the_observed_scores(self):
        # A threshold above every score selects nobody; below every score
        # selects everybody. Either means the search escaped the useful range.
        rule = GroupAwareThresholding().apply(None, SHIFTED).result
        for group, scores, _ in SHIFTED.iter_groups():
            assert min(scores) <= rule["thresholds"][group] <= max(scores)

    def test_utility_is_reported_and_the_rule_is_recorded(self):
        outcome = GroupAwareThresholding().apply(None, SHIFTED)
        assert outcome.result["achieved_utility"] == pytest.approx(1.0)
        assert "Youden" in outcome.provenance["selection_rule"]

    def test_reports_the_gap_it_actually_achieved(self):
        rule = GroupAwareThresholding().apply(None, SEPARABLE).result
        assert 0.0 <= rule["achieved_tpr_gap"] <= 1.0
        assert 0.0 <= rule["achieved_fpr_gap"] <= 1.0
        # The reported rates must be consistent with the reported gap.
        tprs = list(rule["tpr"].values())
        assert rule["achieved_tpr_gap"] == pytest.approx(max(tprs) - min(tprs))

    def test_equalized_odds_also_accounts_for_the_false_positive_gap(self):
        eo = GroupAwareThresholding("equal_opportunity").apply(None, SEPARABLE).result
        odds = GroupAwareThresholding("equalized_odds").apply(None, SEPARABLE).result
        assert odds["criterion"] == "equalized_odds"
        # Equalized odds minimises a superset of the objective, so it can never
        # do better on the combined gap than it does itself.
        combined_eo = eo["achieved_tpr_gap"] + eo["achieved_fpr_gap"]
        combined_odds = odds["achieved_tpr_gap"] + odds["achieved_fpr_gap"]
        assert combined_odds <= combined_eo + 1e-12

    def test_decide_applies_the_fitted_threshold(self):
        rule = GroupAwareThresholding().apply(None, SEPARABLE).result
        threshold = rule["thresholds"]["f"]
        assert GroupAwareThresholding.decide(rule, "f", threshold) is True
        assert GroupAwareThresholding.decide(rule, "f", threshold - 0.01) is False

    def test_an_unknown_criterion_is_refused(self):
        with pytest.raises(ValueError, match="must be 'equal_opportunity'"):
            GroupAwareThresholding(criterion="parity").apply(None, SEPARABLE)

    def test_result_serializes(self):
        result = GroupAwareThresholding().apply(None, SEPARABLE)
        assert result.is_serializable
        assert result.to_dict()["result"]["criterion"] == "equal_opportunity"


class TestOutputReranking:
    def _sets(self, scorer=None):
        # `f` is oriented as the objective requires: higher is better, so a
        # gendered continuation scores below a neutral one.
        return CandidateSets(
            queries=["the nurse said"],
            candidates=[["she smiled", "they smiled", "he smiled"]],
            scorer=scorer
            or (lambda q, c: -1.0 if c.startswith(("she", "he")) else 0.0),
            scorer_name="unit-test-gendered",
        )

    def _scored_sets(self, quality, bias):
        return CandidateSets(
            queries=["the nurse said"],
            candidates=[["she smiled", "they smiled", "he smiled"]],
            scorer=bias,
            scorer_name="unit-test-bias",
            quality=quality,
            quality_name="unit-test-quality",
        )

    def test_lambda_one_ranks_purely_by_the_declared_quality(self):
        result = OutputReranking(lambda_=1.0).apply(
            None,
            self._scored_sets(
                quality=lambda q, c: {"she smiled": 0.1, "they smiled": 0.9}.get(
                    c, 0.5
                ),
                # f alone would put every candidate level; only q can order them.
                bias=lambda q, c: 1.0,
            ),
        )
        assert result.result["rankings"][0] == [
            "they smiled",
            "he smiled",
            "she smiled",
        ]

    def test_the_objective_trades_quality_against_bias(self):
        # q prefers "she smiled" by 0.5; f puts it 1.0 below the others. The
        # objective is lambda * q + (1 - lambda) * f, so at lambda=0.6
        # "she smiled" scores 0.6 - 0.4 = 0.2 against 0.3 for "they smiled",
        # and at lambda=0.7 it scores 0.7 - 0.3 = 0.4 against 0.35.
        sets = self._scored_sets(
            quality=lambda q, c: 1.0 if c == "she smiled" else 0.5,
            bias=lambda q, c: -1.0 if c == "she smiled" else 0.0,
        )

        def _top(lambda_):
            outcome = OutputReranking(lambda_=lambda_).apply(None, sets)
            return outcome.result["rankings"][0][0]

        assert _top(0.6) == "they smiled"
        assert _top(0.7) == "she smiled"

    def test_the_objective_is_recorded_on_the_fitted_rule(self):
        rule = OutputReranking(lambda_=0.25).apply(None, self._sets()).result
        assert rule["objective"] == "argmax lambda * q(y) + (1 - lambda) * f(y)"
        assert rule["lambda"] == 0.25

    def test_the_quality_scorer_is_recorded_in_provenance(self):
        result = OutputReranking().apply(
            None, self._scored_sets(quality=lambda q, c: 0.0, bias=lambda q, c: 0.0)
        )
        assert result.provenance["quality_name"] == "unit-test-quality"
        assert result.result["quality_name"] == "unit-test-quality"

    def test_an_absent_quality_scorer_is_recorded_as_the_generator_ordering(self):
        result = OutputReranking(lambda_=1.0).apply(None, self._sets())
        assert result.result["quality_name"] == "generator_rank"
        # Aligned with the returned order, which lambda=1 leaves untouched.
        assert result.result["quality_scores"][0] == [1.0, 0.5, 0.0]

    def test_a_non_numeric_quality_result_is_refused(self):
        with pytest.raises(TypeError, match="quality must return a real number"):
            OutputReranking().apply(
                None,
                self._scored_sets(quality=lambda q, c: "good", bias=lambda q, c: 0.0),
            )

    def test_a_non_finite_quality_result_is_refused(self):
        with pytest.raises(ValueError, match="quality returned a non-finite"):
            OutputReranking().apply(
                None,
                self._scored_sets(quality=lambda q, c: math.inf, bias=lambda q, c: 0.0),
            )

    def test_lambda_one_reproduces_the_original_order_exactly(self):
        # With no declared q, lambda=1 ranks by the generator's own ordering.
        # The invariant that makes lambda interpretable.
        result = OutputReranking(lambda_=1.0).apply(None, self._sets())
        assert result.result["rankings"][0] == [
            "she smiled",
            "they smiled",
            "he smiled",
        ]

    def test_lambda_zero_ranks_purely_by_the_declared_scorer(self):
        result = OutputReranking(lambda_=0.0).apply(None, self._sets())
        # The only unbiased candidate is promoted to the front.
        assert result.result["rankings"][0][0] == "they smiled"

    def test_a_positively_oriented_bias_scorer_promotes_the_biased_candidate(self):
        """The documented consequence of adding `f` rather than subtracting it.

        Nothing in the library flips the sign of a declared `f`, so a caller who
        hands it a raw bias score gets the biased candidates ranked first. This
        pins that behaviour down rather than leaving it to be discovered.
        """
        result = OutputReranking(lambda_=0.0).apply(
            None,
            self._sets(scorer=lambda q, c: 1.0 if c.startswith(("she", "he")) else 0.0),
        )
        assert result.result["rankings"][0][-1] == "they smiled"

    def test_ties_keep_the_generators_original_order(self):
        result = OutputReranking(lambda_=0.0).apply(
            None, self._sets(scorer=lambda q, c: 0.0)
        )
        assert result.result["rankings"][0] == [
            "she smiled",
            "they smiled",
            "he smiled",
        ]

    def test_reranking_is_a_permutation_of_the_candidates(self):
        sets = self._sets()
        result = OutputReranking().apply(None, sets)
        assert sorted(result.result["rankings"][0]) == sorted(sets.candidates[0])

    def test_the_scorer_is_recorded_in_provenance(self):
        result = OutputReranking().apply(None, self._sets())
        assert result.provenance["scorer_name"] == "unit-test-gendered"

    def test_a_non_numeric_scorer_result_is_refused(self):
        with pytest.raises(TypeError, match="must return a real number"):
            OutputReranking().apply(None, self._sets(scorer=lambda q, c: "biased"))

    def test_a_non_finite_scorer_result_is_refused(self):
        with pytest.raises(ValueError, match="non-finite"):
            OutputReranking().apply(None, self._sets(scorer=lambda q, c: math.inf))

    def test_a_lambda_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            OutputReranking(lambda_=2.0).apply(None, self._sets())

    def test_an_encoder_only_model_is_refused_as_non_generative(self):
        from fairLMs.definitions.core.applicability import TASK_PROFILES

        with pytest.raises(TypeError, match="decoder_only"):
            OutputReranking().apply(TASK_PROFILES["mlm"], self._sets())


# --- regression: all three post-processors return a re-appliable rule --------


def _bias(query, candidate):
    """Declared bias scorer: longer candidates are treated as more biased.

    Oriented as the objective requires -- it is added, not subtracted -- so more
    bias means a lower score.
    """
    return -len(str(candidate)) / 10.0


def test_every_post_processor_exposes_an_applier():
    """The paper claims post-processing returns a fitted calibration, threshold
    or reranking rule. Reranking used to return only the reordered output."""
    from fairLMs.mitigation import MITIGATOR_REGISTRY

    assert hasattr(MITIGATOR_REGISTRY["score_calibration"], "transform")
    assert hasattr(MITIGATOR_REGISTRY["group_aware_thresholding"], "decide")
    assert hasattr(MITIGATOR_REGISTRY["output_reranking"], "rerank")


def test_a_fitted_reranking_rule_reproduces_its_own_fit():
    evidence = CandidateSets(
        queries=["q1"],
        candidates=[["aaaa", "bb", "cccccc"]],
        scorer=_bias,
        scorer_name="_bias",
    )
    result = OutputReranking(lambda_=0.3).apply(None, evidence)
    rule = result.result

    replayed, biases, qualities = OutputReranking.rerank(
        rule, "q1", ["aaaa", "bb", "cccccc"], scorer=_bias
    )
    assert replayed == rule["rankings"][0]
    assert biases == rule["bias_scores"][0]
    assert qualities == rule["quality_scores"][0]


def test_a_fitted_rule_applies_to_unseen_candidates():
    evidence = CandidateSets(
        queries=["q1"],
        candidates=[["aaaa", "bb"]],
        scorer=_bias,
        scorer_name="_bias",
    )
    rule = OutputReranking(lambda_=0.0).apply(None, evidence).result

    # lambda=0 orders purely by descending f, i.e. by ascending length.
    order, _, _ = OutputReranking.rerank(
        rule, "unseen", ["ccccccc", "d", "ee"], scorer=_bias
    )
    assert order == ["d", "ee", "ccccccc"]


def test_reapplying_a_rule_under_a_different_scorer_is_refused():
    evidence = CandidateSets(
        queries=["q1"],
        candidates=[["aaaa", "bb"]],
        scorer=_bias,
        scorer_name="_bias",
    )
    rule = OutputReranking().apply(None, evidence).result

    def _other(query, candidate):
        return 0.0

    with pytest.raises(ValueError, match="different notion of bias"):
        OutputReranking.rerank(rule, "q1", ["aaaa", "bb"], scorer=_other)


def _quality(query, candidate):
    """Declared quality scorer: shorter candidates are treated as better."""
    return 1.0 - len(str(candidate)) / 10.0


def test_a_fitted_rule_carries_its_quality_scorer_through_reapplication():
    evidence = CandidateSets(
        queries=["q1"],
        candidates=[["aaaa", "bb", "cccccc"]],
        scorer=_bias,
        scorer_name="_bias",
        quality=_quality,
        quality_name="_quality",
    )
    rule = OutputReranking(lambda_=0.5).apply(None, evidence).result

    order, _, qualities = OutputReranking.rerank(
        rule, "q1", ["aaaa", "bb", "cccccc"], scorer=_bias, quality=_quality
    )
    assert order == rule["rankings"][0]
    assert qualities == rule["quality_scores"][0]

    with pytest.raises(ValueError, match="different notion of quality"):
        OutputReranking.rerank(rule, "q1", ["aaaa", "bb"], scorer=_bias, quality=_bias)
    with pytest.raises(ValueError, match="none was supplied"):
        OutputReranking.rerank(rule, "q1", ["aaaa", "bb"], scorer=_bias)


def test_supplying_a_quality_scorer_a_rule_was_not_fitted_with_is_refused():
    evidence = CandidateSets(
        queries=["q1"], candidates=[["aaaa", "bb"]], scorer=_bias, scorer_name="_bias"
    )
    rule = OutputReranking().apply(None, evidence).result

    with pytest.raises(ValueError, match="fitted with no quality scorer"):
        OutputReranking.rerank(
            rule, "q1", ["aaaa", "bb"], scorer=_bias, quality=_quality
        )


def test_a_quality_scorer_without_a_name_is_refused():
    with pytest.raises(ValueError, match="quality_name must be a non-empty string"):
        CandidateSets(
            queries=["q1"],
            candidates=[["aaaa", "bb"]],
            scorer=_bias,
            scorer_name="_bias",
            quality=_quality,
        )


def test_the_generator_rank_quality_name_is_reserved():
    with pytest.raises(ValueError, match="is reserved"):
        CandidateSets(
            queries=["q1"],
            candidates=[["aaaa", "bb"]],
            scorer=_bias,
            scorer_name="_bias",
            quality=_quality,
            quality_name="generator_rank",
        )


def test_a_quality_name_without_a_scorer_is_refused():
    with pytest.raises(ValueError, match="quality_name was given without quality"):
        CandidateSets(
            queries=["q1"],
            candidates=[["aaaa", "bb"]],
            scorer=_bias,
            scorer_name="_bias",
            quality_name="_quality",
        )
