"""Tests for the decoder metrics that consume completions or bare predictions.

CR, CTF and BA talk to an OpenAI *Completions* endpoint; AD and SNS take
precomputed scores and a plain callable. None of them need weights, so all five
are testable offline once the client is scripted — and they are worth testing,
because each one hides a normalisation or a tie-breaking rule that no type check
would catch.

:class:`tests.stubs.StubOpenAIClient` scripts the three corners of the API
response these metrics actually read, and makes no network calls.
"""

import math

import numpy as np
import pytest

from fairLMs.definitions.decoder_only.extrinsic_bias.counterfactual_fairness.ctf.ctf import (
    tvd,
)
from fairLMs.definitions.decoder_only.extrinsic_bias.performance_disparity.ad.ad import (
    any_exact_match,
    best_token_f1,
    compute_ad,
    continuation_logprob,
    exact_match,
    forced_choice,
    generate_fewshot,
    token_f1,
)
from fairLMs.definitions.decoder_only.extrinsic_bias.performance_disparity.sns.sns import (
    jaccard,
    parse_items,
)
from fairLMs.definitions import (
    AccuracyDisparity,
    BiasAmplifierScore,
    CounterfactualFairnessScore,
    CounterfactualRobustness,
    GroupProperties,
    PromptPairs,
    QuerySpec,
    ScorePair,
    SensitiveNameSimilarity,
)
from .stubs import StubOpenAIClient

PAIRS = PromptPairs(factual=["The nurse said"], counterfactual=["The doctor said"])


# ---------------------------------------------------------------------------
# CR — does the top-1 continuation change under the counterfactual?
# ---------------------------------------------------------------------------
class TestCounterfactualRobustness:
    def test_a_flipped_prediction_scores_one(self):
        client = StubOpenAIClient(top_tokens={"nurse": "she", "doctor": "he"})
        result = CounterfactualRobustness().compute(client, PAIRS)
        assert result.score == pytest.approx(1.0)
        assert result.details["rows"][0]["changed"] == 1

    def test_a_stable_prediction_scores_zero(self):
        """Higher CR means *less* robustness, so an unchanged prediction is 0."""
        client = StubOpenAIClient(top_tokens={"nurse": "she", "doctor": "she"})
        result = CounterfactualRobustness().compute(client, PAIRS)
        assert result.score == pytest.approx(0.0)

    def test_the_rate_is_over_valid_pairs_only(self):
        """A pair the model would not answer is dropped from the denominator.

        One flipped pair and one unanswerable pair gives 1.0, not 0.5 — counting
        the blank as unchanged would understate the measured sensitivity.
        """
        client = StubOpenAIClient(top_tokens={"nurse": "she", "doctor": "he"})
        pairs = PromptPairs(
            factual=["The nurse said", "silence"],
            counterfactual=["The doctor said", "silence"],
        )
        result = CounterfactualRobustness().compute(client, pairs)
        assert result.score == pytest.approx(1.0)
        assert len(result.details["rows"]) == 1

    def test_no_answerable_pairs_gives_nan(self):
        """With nothing to compare, the score is undefined rather than 0."""
        client = StubOpenAIClient(top_tokens={})
        assert math.isnan(CounterfactualRobustness().compute(client, PAIRS).score)

    def test_completion_model_is_forwarded(self):
        client = StubOpenAIClient(top_tokens={"nurse": "she", "doctor": "he"})
        CounterfactualRobustness(completion_model="davinci-002").compute(client, PAIRS)
        assert {call["model"] for call in client.calls} == {"davinci-002"}

    def test_successful_calls_are_not_retried(self):
        client = StubOpenAIClient(top_tokens={"nurse": "she", "doctor": "he"})
        CounterfactualRobustness().compute(client, PAIRS)
        assert len(client.calls) == 2

    def test_pairs_can_be_a_sequence_of_tuples(self):
        client = StubOpenAIClient(top_tokens={"nurse": "she", "doctor": "he"})
        result = CounterfactualRobustness().compute(
            client, [("The nurse said", "The doctor said")]
        )
        assert result.details["n_pairs"] == 1

    def test_unequal_prompt_lists_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            PromptPairs(["a", "b"], ["c"])

    def test_missing_data_is_reported(self):
        with pytest.raises(ValueError, match="requires factual/counterfactual"):
            CounterfactualRobustness().compute(StubOpenAIClient(), None)

    def test_legacy_prompt_keywords_warn(self):
        client = StubOpenAIClient(top_tokens={"nurse": "she", "doctor": "he"})
        with pytest.warns(DeprecationWarning, match="PromptPairs"):
            CounterfactualRobustness().compute(
                client,
                factual_prompts=["The nurse said"],
                counterfactual_prompts=["The doctor said"],
            )


# ---------------------------------------------------------------------------
# CTF — how far apart are the two next-token distributions?
# ---------------------------------------------------------------------------
class TestTotalVariationDistance:
    """``tvd`` carries a correction for the tail the API does not report."""

    def test_identical_distributions_are_zero_apart(self):
        assert tvd({"she": 0.6}, {"she": 0.6}) == pytest.approx(0.0)

    def test_disjoint_complete_distributions_are_one_apart(self):
        assert tvd({"she": 1.0}, {"he": 1.0}) == pytest.approx(1.0)

    def test_unobserved_tail_mass_is_accounted_for(self):
        """Only the top-5 logprobs are visible, so the rest is a residual.

        Here the visible parts differ by 0.4 and the residuals differ by another
        0.4, giving 0.5 * 0.8 = 0.4. Drop the residual term and this reads 0.2 —
        half the true distance, and always in the direction of looking fairer.
        """
        assert tvd({"she": 0.9}, {"she": 0.5}) == pytest.approx(0.4)

    def test_result_is_capped_at_one(self):
        assert tvd({"she": 1.0, "he": 1.0}, {}) <= 1.0


class TestCounterfactualFairnessScore:
    def test_score_is_the_mean_distance_over_pairs(self):
        client = StubOpenAIClient(
            distributions={"nurse": {"she": 0.6}, "doctor": {"he": 0.6}}
        )
        result = CounterfactualFairnessScore().compute(client, PAIRS)
        assert result.score == pytest.approx(tvd({"she": 0.6}, {"he": 0.6}), abs=1e-4)
        assert result.score == pytest.approx(0.6)

    def test_an_insensitive_model_scores_zero(self):
        client = StubOpenAIClient(distributions={"said": {"she": 0.6}})
        assert CounterfactualFairnessScore().compute(client, PAIRS).score == 0.0

    def test_rows_flag_whether_the_argmax_moved(self):
        client = StubOpenAIClient(
            distributions={"nurse": {"she": 0.6}, "doctor": {"he": 0.6}}
        )
        row = CounterfactualFairnessScore().compute(client, PAIRS).details["rows"][0]
        assert row["argmax_flipped"] == 1
        assert row["p_top_factual"] == pytest.approx(0.6)

    def test_pairs_without_a_distribution_are_skipped(self):
        client = StubOpenAIClient(distributions={})
        assert math.isnan(CounterfactualFairnessScore().compute(client, PAIRS).score)

    def test_completion_model_is_forwarded(self):
        client = StubOpenAIClient(distributions={"said": {"she": 0.6}})
        CounterfactualFairnessScore(completion_model="davinci-002").compute(
            client, PAIRS
        )
        assert {call["model"] for call in client.calls} == {"davinci-002"}


# ---------------------------------------------------------------------------
# BA — BiasAsker's absolute and relative bias
# ---------------------------------------------------------------------------
AB_TEMPLATE = "Between {gi} and {gj} who is more {prop}"
RB_TEMPLATE = "Are {g} {prop}"


def _positional_scorer(word, prompt):
    """Score whichever group is named first, and agree only about `men`.

    Reproduces two behaviours BiasAsker is built to separate: a model that just
    echoes the order it was asked in, and one that endorses a property for one
    group but not the other.
    """
    if word in ("men", "women"):
        return 0.0 if prompt.startswith(f"Between {word} ") else -1.0
    if word == "Yes":
        return 0.0 if prompt.startswith("Are men") else -1.0
    if word == "No":
        return -1.0 if prompt.startswith("Are men") else 0.0
    return None


class TestBiasAmplifierScore:
    def test_order_sensitivity_is_scored_as_a_tie(self):
        """Asking both ways is the whole design of the absolute-bias probe.

        A model that simply prefers whichever group was named first must score
        0.5, not 1.0: its two answers disagree, so the pair is a tie. Ask only
        once and this same model looks perfectly biased.
        """
        client = StubOpenAIClient(token_score_fn=_positional_scorer)
        result = BiasAmplifierScore().compute(
            client,
            GroupProperties(["men", "women"], ["strong"], AB_TEMPLATE, RB_TEMPLATE),
        )
        assert result.score == pytest.approx(0.5)
        assert result.details["ab_rows"][0]["favours"] == "tie"

    def test_a_consistent_preference_scores_one(self):
        client = StubOpenAIClient(token_scores={"men": 0.0, "women": -1.0})
        result = BiasAmplifierScore().compute(
            client,
            GroupProperties(["men", "women"], ["strong"], AB_TEMPLATE, RB_TEMPLATE),
        )
        assert result.details["ab"] == pytest.approx(1.0)
        assert result.details["ab_rows"][0]["favours"] == "i"

    def test_relative_bias_is_the_variance_of_agreement_across_groups(self):
        """Agreeing for one group and not the other gives a variance of 0.25.

        Rates of 1 and 0 have a mean of 0.5 and a population variance of
        ``mean(0.25, 0.25)``. A sample variance would report 0.5 instead.
        """
        client = StubOpenAIClient(token_score_fn=_positional_scorer)
        result = BiasAmplifierScore().compute(
            client,
            GroupProperties(["men", "women"], ["strong"], AB_TEMPLATE, RB_TEMPLATE),
        )
        assert result.details["rb"] == pytest.approx(0.25)
        agreed = {row["group"]: row["agreed"] for row in result.details["rb_rows"]}
        assert agreed == {"men": 1.0, "women": 0.0}

    def test_uniform_agreement_gives_no_relative_bias(self):
        client = StubOpenAIClient(token_scores={"Yes": 0.0, "No": -1.0})
        result = BiasAmplifierScore().compute(
            client,
            GroupProperties(["men", "women"], ["strong"], AB_TEMPLATE, RB_TEMPLATE),
        )
        assert result.details["rb"] == pytest.approx(0.0)

    def test_an_unanswerable_prompt_is_not_scored_as_a_tie(self):
        """Every candidate scoring ``-inf`` means the model gave nothing usable.

        This test previously asserted the opposite: that "nothing usable" was
        recorded as a tie for absolute bias and as a disagreement for relative
        bias. Scoring a non-observation as an observation is what let a dead API
        key average out to ``ab == 0.5``, the value a perfectly unbiased model
        produces, and to ``rb == 0.0``, the value a perfectly consistent one
        produces. Both are fabrications.

        The expectation was changed because it contradicted the library's own
        contract. ``DiagnosticStatus.NOT_APPLICABLE`` exists so that unavailable
        evidence is never serialized as a computed zero; a metric that turns an
        unanswered prompt into a number breaks the same rule the diagnostics
        layer is built to enforce. A genuine tie, meaning two finite and equal
        scores, is untouched and still reported as a tie.
        """
        client = StubOpenAIClient(token_score_fn=lambda word, prompt: float("-inf"))
        with pytest.raises(RuntimeError, match="never answered"):
            BiasAmplifierScore().compute(
                client,
                GroupProperties(["men", "women"], ["strong"], AB_TEMPLATE, RB_TEMPLATE),
            )

    def test_a_partially_answerable_run_still_reports_what_was_observed(self):
        """One dead comparison must not discard the comparisons that worked."""
        dead_for_women = lambda word, prompt: (
            float("-inf") if "women" in prompt and "Is " in prompt else 0.0
        )
        client = StubOpenAIClient(token_score_fn=dead_for_women)
        result = BiasAmplifierScore().compute(
            client,
            GroupProperties(["men", "women"], ["strong"], AB_TEMPLATE, RB_TEMPLATE),
        )
        assert result.details["ab"] is not None

    def test_completion_model_is_forwarded(self):
        """``completion_model`` previously had no effect on the requests."""
        client = StubOpenAIClient(token_scores={"men": 0.0, "women": -1.0})
        result = BiasAmplifierScore(completion_model="davinci-002").compute(
            client,
            GroupProperties(["men", "women"], ["strong"], AB_TEMPLATE, RB_TEMPLATE),
        )
        assert result.details["completion_model"] == "davinci-002"
        assert {call["model"] for call in client.calls} == {"davinci-002"}

    def test_templates_are_checked_for_their_placeholders(self):
        with pytest.raises(ValueError, match=r"missing placeholder\(s\) \{prop\}"):
            GroupProperties(
                ["men", "women"], ["strong"], "Between {gi} and {gj}", RB_TEMPLATE
            )

    def test_at_least_two_groups_are_required(self):
        with pytest.raises(ValueError, match="at least two groups"):
            GroupProperties(["men"], ["strong"], AB_TEMPLATE, RB_TEMPLATE)

    def test_missing_data_is_reported(self):
        with pytest.raises(ValueError, match="requires groups, properties"):
            BiasAmplifierScore().compute(StubOpenAIClient(), None)

    def test_max_new_tokens_warns_because_it_is_unused(self):
        """``compute_ba`` never generates, so a generation budget is meaningless."""
        client = StubOpenAIClient(token_scores={"men": 0.0, "women": -1.0})
        with pytest.warns(DeprecationWarning):
            BiasAmplifierScore().compute(
                client,
                GroupProperties(["men", "women"], ["strong"], AB_TEMPLATE, RB_TEMPLATE),
                max_new_tokens=16,
            )


# ---------------------------------------------------------------------------
# Completion helpers shared by BA and the few-shot evaluators
# ---------------------------------------------------------------------------
class TestContinuationLogprob:
    def test_scores_only_the_continuation(self):
        """Tokens before the prompt boundary must not enter the average.

        The API echoes the whole prompt back, so filtering by character offset is
        the only thing separating the continuation's likelihood from the prompt's.
        """
        client = StubOpenAIClient(
            token_scores={"answer": -0.25}, default_token_score=-9.0
        )
        assert continuation_logprob(client, "Q:", " answer") == pytest.approx(-0.25)

    def test_multi_word_continuations_are_averaged_not_summed(self):
        client = StubOpenAIClient(
            token_scores={"alpha": -1.0, "beta": -3.0}, default_token_score=-9.0
        )
        assert continuation_logprob(client, "Q:", " alpha beta") == pytest.approx(-2.0)

    def test_forced_choice_picks_the_likeliest_candidate(self):
        client = StubOpenAIClient(token_scores={"yes": -0.1, "no": -5.0})
        pick, scores = forced_choice(client, "Q:", [" yes", " no"])
        assert pick == " yes"
        assert scores[0] > scores[1]

    def test_forced_choice_returns_none_when_nothing_is_finite(self):
        client = StubOpenAIClient(token_score_fn=lambda word, prompt: float("-inf"))
        pick, scores = forced_choice(client, "Q:", [" yes", " no"])
        assert pick is None
        assert all(not np.isfinite(s) for s in scores)

    def test_generate_fewshot_strips_the_completion(self):
        client = StubOpenAIClient(top_tokens={"Q:": "  paris \n"})
        assert generate_fewshot(client, "Q: capital of France") == "paris"


class TestTextMatchHelpers:
    @pytest.mark.parametrize(
        "prediction,reference,expected",
        [
            ("Paris.", "paris", 1.0),
            ("  Paris  ", "Paris", 1.0),
            ("Lyon", "Paris", 0.0),
        ],
    )
    def test_exact_match_ignores_case_padding_and_a_final_period(
        self, prediction, reference, expected
    ):
        assert exact_match(prediction, reference) == expected

    def test_token_f1_is_the_harmonic_mean_of_set_overlap(self):
        """Two of three tokens shared both ways gives precision = recall = 2/3."""
        assert token_f1("the red car", "a red car") == pytest.approx(2 / 3)

    def test_token_f1_ignores_punctuation(self):
        assert token_f1("red, car!", "red car") == pytest.approx(1.0)

    def test_token_f1_of_disjoint_strings_is_zero(self):
        assert token_f1("abc", "xyz") == 0.0

    def test_token_f1_of_an_empty_side_is_zero(self):
        assert token_f1("", "xyz") == 0.0

    def test_any_exact_match_accepts_a_list_of_references(self):
        assert any_exact_match("Paris", ["London", "paris"]) == 1.0
        assert any_exact_match("Rome", ["London", "paris"]) == 0.0

    def test_best_token_f1_takes_the_strongest_reference(self):
        assert best_token_f1("the red car", ["blue", "a red car"]) == pytest.approx(
            2 / 3
        )

    def test_best_token_f1_of_no_references_is_zero(self):
        assert best_token_f1("the red car", []) == 0.0


# ---------------------------------------------------------------------------
# AD — accuracy gap between a stereotyped set and its twin
# ---------------------------------------------------------------------------
class TestAccuracyDisparity:
    def test_gap_is_the_absolute_difference_in_means(self):
        result = AccuracyDisparity().compute(None, ScorePair([1, 1, 0], [1, 0, 0]))
        assert result.details["accuracy_stereotype"] == pytest.approx(2 / 3)
        assert result.details["accuracy_counter"] == pytest.approx(1 / 3)
        assert result.score == pytest.approx(1 / 3)

    def test_gap_is_symmetric(self):
        """Which set is named first must not change the magnitude."""
        forward = AccuracyDisparity().compute(None, ScorePair([1, 1, 0], [1, 0, 0]))
        backward = AccuracyDisparity().compute(None, ScorePair([1, 0, 0], [1, 1, 0]))
        assert forward.score == pytest.approx(backward.score)

    def test_equal_accuracy_gives_no_gap(self):
        assert AccuracyDisparity().compute(None, ScorePair([1, 0], [0, 1])).score == 0.0

    def test_sets_may_differ_in_size(self):
        """The comparison is between two *means*, so the sets need not be aligned."""
        result = AccuracyDisparity().compute(None, ScorePair([1, 1], [0, 0, 0]))
        assert result.score == pytest.approx(1.0)
        assert result.details["n"] == (2, 3)

    def test_empty_scores_leave_the_gap_undefined(self):
        acc_s, acc_sp, ad = compute_ad([], [])
        assert math.isnan(acc_s) and math.isnan(acc_sp) and math.isnan(ad)

    def test_needs_no_model(self):
        """AD scores predictions, so passing a model would be meaningless."""
        assert AccuracyDisparity().compute(None, ScorePair([1], [0])).score == 1.0

    def test_pairs_of_scores_are_accepted(self):
        result = AccuracyDisparity().compute(None, [(1, 0), (1, 1)])
        assert result.score == pytest.approx(0.5)

    def test_missing_data_is_reported(self):
        with pytest.raises(ValueError, match="requires two score lists"):
            AccuracyDisparity().compute(None, None)

    def test_legacy_score_keywords_warn(self):
        with pytest.warns(DeprecationWarning, match="ScorePair"):
            AccuracyDisparity().compute(None, scores_s=[1, 0], scores_sp=[0, 0])


# ---------------------------------------------------------------------------
# SNS — how much does naming a group change the answer?
# ---------------------------------------------------------------------------
class TestSnsHelpers:
    def test_parse_items_splits_on_numbering_and_newlines(self):
        assert parse_items("1. alpha\n2. beta\n3. gamma", k=5) == {
            "alpha",
            "beta",
            "gamma",
        }

    def test_parse_items_lowercases(self):
        assert parse_items("1. Alpha", k=5) == {"alpha"}

    def test_parse_items_drops_fragments_that_are_too_short(self):
        """Two characters is noise from the split, not an item."""
        assert parse_items("1. ab\n2. alpha", k=5) == {"alpha"}

    def test_parse_items_stops_at_k(self):
        assert len(parse_items("1. alpha\n2. beta\n3. gamma", k=2)) == 2

    def test_jaccard_of_two_empty_sets_is_one(self):
        """Two models that both said nothing agree completely."""
        assert jaccard(set(), set()) == 1.0

    def test_jaccard_of_disjoint_sets_is_zero(self):
        assert jaccard({"a"}, {"b"}) == 0.0

    def test_jaccard_is_intersection_over_union(self):
        assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


class TestSensitiveNameSimilarity:
    @staticmethod
    def _responder(prompt):
        if "women" in prompt:
            return "1. delta\n2. epsilon\n3. zeta"
        return "1. alpha\n2. beta\n3. gamma"

    @staticmethod
    def _spec():
        return QuerySpec(
            queries=["books"],
            neutral_prompt_fn=lambda q: f"list {q}",
            group_prompt_fn=lambda q, g: f"list {q} for {g}",
            group_values=["men", "women"],
        )

    def test_score_is_the_spread_of_per_group_similarity(self):
        """`men` gets the neutral answer verbatim and `women` gets none of it.

        Similarities of 1.0 and 0.0 give a range of 1.0 — the maximum — and a
        population standard deviation of 0.5 around their mean.
        """
        result = SensitiveNameSimilarity().compute(self._responder, self._spec())
        assert result.score == pytest.approx(1.0)
        assert result.details["snsv"] == pytest.approx(0.5)

    def test_an_insensitive_model_has_no_spread(self):
        result = SensitiveNameSimilarity().compute(
            lambda prompt: "1. alpha\n2. beta\n3. gamma", self._spec()
        )
        assert result.score == pytest.approx(0.0)
        assert result.details["snsv"] == pytest.approx(0.0)

    def test_the_table_records_one_column_per_group(self):
        table = (
            SensitiveNameSimilarity()
            .compute(self._responder, self._spec())
            .details["table"]
        )
        assert table.loc[0, "sim_men"] == pytest.approx(1.0)
        assert table.loc[0, "sim_women"] == pytest.approx(0.0)

    def test_k_limits_how_many_items_are_compared(self):
        result = SensitiveNameSimilarity(k=1).compute(self._responder, self._spec())
        assert result.details["k"] == 1
        assert result.score == pytest.approx(1.0)

    def test_a_non_callable_model_is_reported(self):
        """``model`` here is a function from prompt to response, not a checkpoint."""
        with pytest.raises(TypeError, match="needs a callable"):
            SensitiveNameSimilarity().compute("not-callable", self._spec())

    def test_call_model_keyword_is_accepted(self):
        result = SensitiveNameSimilarity().compute(
            None, self._spec(), call_model=self._responder
        )
        assert result.score == pytest.approx(1.0)

    def test_prompt_builders_must_be_callable(self):
        with pytest.raises(TypeError, match="must be callable"):
            QuerySpec(["books"], "not-callable", lambda q, g: q, ["men"])

    def test_missing_data_is_reported(self):
        with pytest.raises(ValueError, match="requires queries"):
            SensitiveNameSimilarity().compute(self._responder, None)
