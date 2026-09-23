"""The sklearn.metrics-style function API must agree with the class API."""

import pytest

from fairLMs.metrics import (
    AccuracyDisparity,
    ContextBasedDisparityScore,
    EqualOpportunityGap,
    FairInferenceScore,
    GroupPredictions,
    InferenceBiasScore,
    ScorePair,
    accuracy_disparity,
    context_based_disparity,
    equal_opportunity_gap,
    fair_inference_score,
    inference_bias_score,
)

Y_TRUE = [1, 1, 1, 1, 0, 0, 1, 1]
Y_PRED = [1, 0, 1, 1, 0, 1, 1, 0]
GROUPS = ["A", "A", "A", "B", "A", "B", "B", "B"]
NLI = [
    {"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1},
    {"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1},
]
BBQ = [
    {"cond": "disambig", "output": "unknown", "expected": "target"},
    {"cond": "ambig", "output": "unknown", "expected": "unknown"},
]


def test_equal_opportunity_gap_matches_class():
    assert equal_opportunity_gap(Y_TRUE, Y_PRED, GROUPS, g1="A", g2="B") == pytest.approx(
        EqualOpportunityGap(g1="A", g2="B")
        .compute(None, GroupPredictions(Y_TRUE, Y_PRED, GROUPS))
        .score
    )


def test_accuracy_disparity_matches_class():
    s, sp = [1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0]
    assert accuracy_disparity(s, sp) == pytest.approx(
        AccuracyDisparity().compute(None, ScorePair(s, sp)).score
    )


def test_inference_bias_score_matches_class():
    preds = [(1, 1), (1, 0), (0, 0), (0, 1)]
    assert inference_bias_score(preds) == pytest.approx(
        InferenceBiasScore().compute(None, preds).score
    )


def test_fair_inference_score_matches_class():
    assert fair_inference_score(NLI) == pytest.approx(
        FairInferenceScore().compute(None, NLI).score
    )


def test_context_based_disparity_matches_class():
    assert context_based_disparity(BBQ) == pytest.approx(
        ContextBasedDisparityScore().compute(None, BBQ).score
    )
    assert context_based_disparity(BBQ, score="s_amb") == pytest.approx(
        ContextBasedDisparityScore(score="s_amb").compute(None, BBQ).score
    )


def test_functions_return_bare_floats():
    """Like sklearn.metrics, these return a float, not a result object."""
    for value in (
        equal_opportunity_gap(Y_TRUE, Y_PRED, GROUPS, g1="A", g2="B"),
        accuracy_disparity([1.0, 0.0], [0.0, 1.0]),
        inference_bias_score([(1, 1), (0, 0)]),
        fair_inference_score(NLI),
        context_based_disparity(BBQ),
    ):
        assert type(value) is float


def test_equal_opportunity_infers_groups_when_not_given():
    explicit = equal_opportunity_gap([1, 1, 0], [1, 0, 0], ["A", "B", "A"], g1="A", g2="B")
    inferred = equal_opportunity_gap([1, 1, 0], [1, 0, 0], ["A", "B", "A"])
    assert explicit == inferred


def test_equal_opportunity_rejects_absent_group():
    with pytest.raises(ValueError, match="does not appear"):
        equal_opportunity_gap([1, 0], [1, 0], ["A", "A"], g1="A", g2="Z")


def test_context_based_disparity_rejects_bad_score_name():
    with pytest.raises(ValueError, match="s_dis"):
        context_based_disparity(BBQ, score="nonsense")
