"""Tests for the masked-token family: DisCo, LPBS, CBS.

These three ask a masked LM to fill a slot and then compare the answers across
demographic groups. DisCo needs only top-k fills, LPBS needs four probabilities
per attribute word, and CBS needs a contrast of raw log-probabilities — so all
three are reachable with the stub's fixed distribution, and all three have
expected values that are arithmetic rather than recordings.

The stub's tables associate ``nurse`` with ``she`` and ``doctor`` with ``he``, so
each metric has a *known direction* to find, which is what makes a sign error
visible here.
"""

import math

import numpy as np
import pytest
import torch

from fairLMs.definitions.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.disco.disco import (  # noqa: E501
    compute_disco_multi_k,
    get_ranked_predictions,
)
from fairLMs.definitions import (
    ContrastBasedScore,
    ContrastSpec,
    DiscoveryOfCorrelationsScore,
    GroupWordPairs,
    LogProbabilityBiasScore,
)
from .stubs import CONTENT_TOKENS, VOCAB, expected_log_softmax, expected_logits

TEMPLATE = "{X} is [MASK]."


def _top_k_tokens(context, k):
    """The tokens the stub ranks highest given a one-word context."""
    logits = expected_logits([context])
    ranked = sorted(VOCAB, key=lambda t: -logits[VOCAB[t]].item())
    return set(ranked[:k])


def _disco(pairs, k):
    """DisCo is the mean *non*-overlap of the two groups' top-k fills, as a percent."""
    overlaps = [
        len(_top_k_tokens(w1, k) & _top_k_tokens(w2, k)) / k for w1, w2 in pairs
    ]
    return round((1.0 - sum(overlaps) / len(overlaps)) * 100.0, 2)


class TestDiscoveryOfCorrelationsScore:
    def test_score_is_the_mean_non_overlap(self, stub_pipeline):
        """With k=3 the two groups agree on two of three fills.

        ``he`` prefers (he, doctor, she); ``she`` prefers (nurse, he, she). The
        intersection is {he, she}, so the overlap is 2/3 and DisCo is 33.33.
        """
        result = DiscoveryOfCorrelationsScore(k=3, n_bootstrap=0).compute(
            stub_pipeline, GroupWordPairs(["he"], ["she"]), templates=[TEMPLATE]
        )
        assert result.score == pytest.approx(_disco([("he", "she")], 3))
        assert result.score == pytest.approx(33.33)

    def test_disjoint_top_one_gives_a_maximal_score(self, stub_pipeline):
        """``he`` and ``she`` have different favourites, so k=1 leaves no overlap."""
        result = DiscoveryOfCorrelationsScore(k=1, n_bootstrap=0).compute(
            stub_pipeline, GroupWordPairs(["he"], ["she"]), templates=[TEMPLATE]
        )
        assert result.score == pytest.approx(100.0)

    def test_covering_the_whole_vocabulary_gives_zero(self, stub_pipeline):
        """At k = |vocabulary| the two sets must coincide, so DisCo is 0 by construction.

        A metric that measured overlap instead of non-overlap would report 100
        here and 0 in the k=1 case above.
        """
        result = DiscoveryOfCorrelationsScore(
            k=len(CONTENT_TOKENS), n_bootstrap=0
        ).compute(
            stub_pipeline, GroupWordPairs(["he"], ["she"]), templates=[TEMPLATE]
        )
        assert result.score == pytest.approx(0.0)

    def test_k_is_reported_and_honoured(self, stub_pipeline):
        result = DiscoveryOfCorrelationsScore(k=2, n_bootstrap=0).compute(
            stub_pipeline, GroupWordPairs(["he"], ["she"]), templates=[TEMPLATE]
        )
        assert result.details["k"] == 2
        assert result.score == pytest.approx(_disco([("he", "she")], 2))

    def test_a_single_pair_yields_no_confidence_interval(self, stub_pipeline):
        """The interval is a *cluster* bootstrap over pairs; one pair cannot support it."""
        result = DiscoveryOfCorrelationsScore(k=3, n_bootstrap=500).compute(
            stub_pipeline, GroupWordPairs(["he"], ["she"]), templates=[TEMPLATE]
        )
        assert result.details["ci_low"] is None
        assert result.details["ci_high"] is None

    def test_several_pairs_yield_an_interval_around_the_point_estimate(
        self, stub_pipeline
    ):
        data = GroupWordPairs(["he", "nurse"], ["she", "doctor"])
        result = DiscoveryOfCorrelationsScore(
            k=3, n_bootstrap=200, seed=0
        ).compute(stub_pipeline, data, templates=[TEMPLATE])
        low, high = result.details["ci_low"], result.details["ci_high"]
        assert low is not None and high is not None
        assert low <= result.score <= high
        assert result.details["n_pairs"] == 2

    def test_seed_makes_the_interval_reproducible(self, stub_pipeline):
        data = GroupWordPairs(["he", "nurse"], ["she", "doctor"])
        kwargs = dict(k=3, n_bootstrap=200, seed=7)
        first = DiscoveryOfCorrelationsScore(**kwargs).compute(
            stub_pipeline, data, templates=[TEMPLATE]
        )
        second = DiscoveryOfCorrelationsScore(**kwargs).compute(
            stub_pipeline, data, templates=[TEMPLATE]
        )
        assert first.details["ci_low"] == second.details["ci_low"]

    def test_default_templates_are_used_when_none_given(self, stub_pipeline):
        result = DiscoveryOfCorrelationsScore(k=3, n_bootstrap=0).compute(
            stub_pipeline, GroupWordPairs(["he"], ["she"])
        )
        assert 0.0 <= result.score <= 100.0

    def test_template_without_a_mask_is_rejected(self, stub_pipeline):
        with pytest.raises(ValueError, match=r"must contain \{X\} and \[MASK\]"):
            DiscoveryOfCorrelationsScore(n_bootstrap=0).compute(
                stub_pipeline, GroupWordPairs(["he"], ["she"]), templates=["{X} is nice."]
            )

    def test_template_without_a_group_slot_is_rejected(self, stub_pipeline):
        with pytest.raises(ValueError, match=r"must contain \{X\} and \[MASK\]"):
            DiscoveryOfCorrelationsScore(n_bootstrap=0).compute(
                stub_pipeline, GroupWordPairs(["he"], ["she"]), templates=["[MASK] is nice."]
            )

    def test_identical_groups_are_rejected_at_construction(self):
        with pytest.raises(ValueError, match="nothing to contrast"):
            GroupWordPairs(["he"], ["he"])

    def test_unequal_groups_are_rejected_at_construction(self):
        with pytest.raises(ValueError, match="must be the same length"):
            GroupWordPairs(["he", "she"], ["she"])

    def test_missing_model_is_reported(self):
        with pytest.raises(ValueError, match="requires a fill-mask pipeline"):
            DiscoveryOfCorrelationsScore().compute(None, GroupWordPairs(["he"], ["she"]))

    def test_wrong_container_is_reported(self, stub_pipeline):
        with pytest.raises(TypeError, match="expects a GroupWordPairs"):
            DiscoveryOfCorrelationsScore().compute(stub_pipeline, ["he", "she"])

    def test_missing_data_is_reported(self, stub_pipeline):
        with pytest.raises(ValueError, match="two aligned word lists"):
            DiscoveryOfCorrelationsScore().compute(stub_pipeline, None)

    def test_legacy_group_keywords_warn(self, stub_pipeline):
        with pytest.warns(DeprecationWarning, match="GroupWordPairs"):
            result = DiscoveryOfCorrelationsScore(k=3, n_bootstrap=0).compute(
                stub_pipeline,
                group1_words=["he"],
                group2_words=["she"],
                templates=[TEMPLATE],
            )
        assert result.score == pytest.approx(33.33)

    def test_pipe_override_is_accepted(self, stub_pipeline):
        """``pipe=`` lets a caller supply the pipeline separately from ``model``."""
        result = DiscoveryOfCorrelationsScore(k=3, n_bootstrap=0).compute(
            None, GroupWordPairs(["he"], ["she"]), pipe=stub_pipeline, templates=[TEMPLATE]
        )
        assert result.score == pytest.approx(33.33)


class TestDiscoAcrossSeveralK:
    """``compute_disco_multi_k`` sweeps k from one fetch instead of many.

    Nothing in the library calls it, but it is the function a paper replication
    reaches for, and it duplicates the scoring rule rather than sharing it — so
    it can drift from ``compute_disco`` silently. These tests hold the two to
    the same answers.
    """

    def test_ranked_predictions_are_returned_highest_first(self, stub_pipeline):
        ranked = get_ranked_predictions(stub_pipeline, TEMPLATE.replace("{X}", "he"), n=4)
        tokens = [token for token, _ in ranked]
        scores = [score for _, score in ranked]
        assert len(ranked) == 4
        assert scores == sorted(scores, reverse=True)
        assert set(tokens) == _top_k_tokens("he", 4)

    def test_each_k_agrees_with_the_single_k_computation(self, stub_pipeline):
        out = compute_disco_multi_k(
            stub_pipeline,
            ["he"],
            ["she"],
            [TEMPLATE],
            k_values=[1, 3],
            n_fetch=5,
            n_bootstrap=0,
        )
        assert out[1][0] == pytest.approx(_disco([("he", "she")], 1))
        assert out[3][0] == pytest.approx(_disco([("he", "she")], 3))
        assert out[1][0] == pytest.approx(100.0)
        assert out[3][0] == pytest.approx(33.33)

    def test_confidence_intervals_need_more_than_one_pair(self, stub_pipeline):
        single = compute_disco_multi_k(
            stub_pipeline, ["he"], ["she"], [TEMPLATE], k_values=[3], n_fetch=5
        )
        assert single[3][1:] == (None, None)

        several = compute_disco_multi_k(
            stub_pipeline,
            ["he", "he", "she"],
            ["she", "nurse", "doctor"],
            [TEMPLATE],
            k_values=[3],
            n_fetch=5,
            n_bootstrap=64,
        )
        low, high = several[3][1:]
        assert low is not None and high is not None
        assert low <= several[3][0] <= high

    def test_asking_for_more_than_was_fetched_is_an_error(self, stub_pipeline):
        with pytest.raises(ValueError, match="n_fetch=3 < max"):
            compute_disco_multi_k(
                stub_pipeline, ["he"], ["she"], [TEMPLATE], k_values=[5], n_fetch=3
            )


# ---------------------------------------------------------------------------
# LPBS
# ---------------------------------------------------------------------------
def _probs(context_tokens):
    return torch.softmax(expected_logits(context_tokens), dim=-1)


def _lpbs(attribute, male="he", female="she"):
    """Kurita et al.'s prior-corrected score, re-derived from the tables.

    The correction is the point of the metric: each group's probability in the
    attribute context is divided by its probability in an attribute-free prior
    context, so a group that is simply more frequent overall does not register as
    biased.
    """
    attr = _probs([attribute])
    prior = _probs([])
    ratio = lambda p_attr, p_prior: math.log((p_attr + 1e-10) / (p_prior + 1e-10))
    return ratio(
        attr[VOCAB[male]].item(), prior[VOCAB[male]].item()
    ) - ratio(attr[VOCAB[female]].item(), prior[VOCAB[female]].item())


# A one-attribute run has no sample variance, so ``np.std(ddof=1)`` warns. That
# behaviour is pinned by `test_single_attribute_leaves_dispersion_undefined`
# below; the rest of these tests deliberately use a single attribute to isolate a
# sign, and should not each re-report the same warning.
@pytest.mark.filterwarnings("ignore:Degrees of freedom:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
class TestLogProbabilityBiasScore:
    def test_score_is_the_mean_prior_corrected_bias(self, stub_bundle):
        attributes = ["nurse", "doctor"]
        expected = float(np.mean([_lpbs(a) for a in attributes]))
        result = LogProbabilityBiasScore().compute(stub_bundle, attributes)
        assert result.score == pytest.approx(expected, abs=1e-6)

    def test_sign_follows_the_stereotypical_association(self, stub_bundle):
        """`nurse` must favour `she` (negative) and `doctor` must favour `he` (positive).

        This is the assertion that a swapped subtraction breaks. The magnitudes
        would survive it; the signs would not.
        """
        nurse = LogProbabilityBiasScore().compute(stub_bundle, ["nurse"])
        doctor = LogProbabilityBiasScore().compute(stub_bundle, ["doctor"])
        assert nurse.score < 0 < doctor.score

    def test_swapping_the_group_words_negates_the_score(self, stub_bundle):
        forward = LogProbabilityBiasScore(gender_words=("he", "she")).compute(
            stub_bundle, ["nurse"]
        )
        reversed_ = LogProbabilityBiasScore(gender_words=("she", "he")).compute(
            stub_bundle, ["nurse"]
        )
        assert forward.score == pytest.approx(-reversed_.score, abs=1e-6)

    def test_prior_correction_actually_changes_the_number(self, stub_bundle):
        """The corrected score must differ from the raw probability difference.

        ``gender_fill_bias`` is the uncorrected quantity; if the correction were
        dropped the two would move together.
        """
        outcomes = LogProbabilityBiasScore().compute(
            stub_bundle, ["nurse"]
        ).details["outcomes"]
        assert outcomes[0]["gender_fill_bias"] != pytest.approx(
            outcomes[0]["gender_fill_bias_prior_corrected"]
        )

    def test_target_fill_bias_compares_the_attribute_across_groups(self, stub_bundle):
        """``target_fill_bias`` reads the other direction: P(attribute | group).

        ``nurse`` is likelier after ``she`` than after ``he``, so this is negative.
        """
        outcomes = LogProbabilityBiasScore().compute(
            stub_bundle, ["nurse"]
        ).details["outcomes"]
        expected = math.log(
            (_probs(["he"])[VOCAB["nurse"]].item() + 1e-10)
            / (_probs(["she"])[VOCAB["nurse"]].item() + 1e-10)
        )
        assert outcomes[0]["target_fill_bias"] == pytest.approx(expected, abs=1e-6)
        assert outcomes[0]["target_fill_bias"] < 0

    def test_details_report_the_dispersion_and_the_split(self, stub_bundle):
        attributes = ["nurse", "doctor"]
        scores = [_lpbs(a) for a in attributes]
        result = LogProbabilityBiasScore().compute(stub_bundle, attributes)
        assert result.details["std"] == pytest.approx(float(np.std(scores, ddof=1)))
        assert result.details["proportion_favoring_group1"] == pytest.approx(0.5)
        assert result.details["n_attributes"] == 2

    def test_single_attribute_leaves_dispersion_undefined(self, stub_bundle):
        """One attribute word has no sample variance, so ``std`` is ``nan``, not 0.

        ``np.std(ddof=1)`` warns and yields ``nan`` here. Pinned because the
        score itself is perfectly well defined, so a caller could reasonably
        expect ``details["std"]`` to be usable and quietly propagate the ``nan``.
        """
        with pytest.warns(RuntimeWarning):
            result = LogProbabilityBiasScore().compute(stub_bundle, ["nurse"])
        assert math.isnan(result.details["std"])
        assert not math.isnan(result.score)

    def test_gender_comes_first_is_accepted(self, stub_bundle):
        """Both settings are honoured; the stub is order-blind so they agree.

        The stub scores a slot against the *bag* of surrounding tokens, so which
        slot comes first cannot change a probability here. The value of the test
        is that the flag reaches ``compute_lpbs`` rather than raising.
        """
        first = LogProbabilityBiasScore(gender_comes_first=True).compute(
            stub_bundle, ["nurse"]
        )
        last = LogProbabilityBiasScore(gender_comes_first=False).compute(
            stub_bundle, ["nurse"]
        )
        assert first.score == pytest.approx(last.score)

    def test_custom_template_is_used_and_reported(self, stub_bundle):
        result = LogProbabilityBiasScore(template="GGG works as a XXX").compute(
            stub_bundle, ["nurse"]
        )
        assert result.details["template"] == "GGG works as a XXX"
        assert result.score == pytest.approx(_lpbs("nurse"), abs=1e-6)

    def test_attribute_words_are_read_from_dict_examples(self, stub_bundle):
        examples = [{"profession_name": "nurse"}, {"profession_name": "doctor"}]
        result = LogProbabilityBiasScore().compute(stub_bundle, examples)
        assert result.details["n_attributes"] == 2

    def test_dict_examples_without_a_usable_key_are_reported(self, stub_bundle):
        with pytest.raises(ValueError, match="no attribute words found"):
            LogProbabilityBiasScore().compute(stub_bundle, [{"unrelated": "nurse"}])

    def test_three_gender_words_are_rejected(self, stub_bundle):
        with pytest.raises(ValueError, match="exactly two tokens"):
            LogProbabilityBiasScore(
                gender_words=("he", "she", "they")
            ).compute(stub_bundle, ["nurse"])

    def test_identical_gender_words_are_rejected(self, stub_bundle):
        with pytest.raises(ValueError, match="two distinct tokens"):
            LogProbabilityBiasScore(gender_words=("he", "he")).compute(
                stub_bundle, ["nurse"]
            )

    def test_missing_data_is_reported(self, stub_bundle):
        with pytest.raises(ValueError, match="requires attribute words"):
            LogProbabilityBiasScore().compute(stub_bundle, None)

    def test_legacy_attribute_words_keyword_warns(self, stub_bundle):
        with pytest.warns(DeprecationWarning, match="attribute word list"):
            LogProbabilityBiasScore().compute(
                stub_bundle, attribute_words=["nurse"]
            )


# ---------------------------------------------------------------------------
# CBS
# ---------------------------------------------------------------------------
def _raw_log_prob(term, attribute):
    """CBS uses *raw* log-probabilities, so the base rate cancels in the contrast."""
    probs = _probs([attribute])
    return math.log(probs[VOCAB[term]].item() + 1e-10)


def _contrast(term, negative, positive):
    return _raw_log_prob(term, negative) - _raw_log_prob(term, positive)


CBS_TEMPLATES = ["{N} is {A}."]


class TestContrastBasedScore:
    def test_winner_of_a_cell_is_the_group_the_negative_word_pulls_hardest(
        self, stub_bundle
    ):
        """`nurse` over `doctor` should favour `she`, so `she` wins that cell.

        The contrast subtracts a positive control from a negative one, which is
        what removes each group's overall frequency from the comparison.
        """
        assert _contrast("she", "nurse", "doctor") > _contrast("he", "nurse", "doctor")

        spec = ContrastSpec(["he", "she"], [("nurse", "doctor", "she")], CBS_TEMPLATES)
        result = ContrastBasedScore(n_bootstrap=25, n_perm=25).compute(
            stub_bundle, spec
        )
        cells = result.details["info"]["cells"]
        assert cells[0]["winner"] == "she"
        assert cells[0]["contrasts"]["she"] == pytest.approx(
            _contrast("she", "nurse", "doctor"), abs=1e-6
        )

    def test_confirmatory_score_is_the_declared_hit_rate(self, stub_bundle):
        """Both declarations match the stub's associations, so CBS is 100 %.

        ``stereo_group`` is external ground truth fixed before the model is run,
        which is what gives this number a meaningful baseline.
        """
        spec = ContrastSpec(
            ["he", "she"],
            [("nurse", "doctor", "she"), ("doctor", "nurse", "he")],
            CBS_TEMPLATES,
        )
        result = ContrastBasedScore(n_bootstrap=25, n_perm=25).compute(
            stub_bundle, spec
        )
        assert result.details["score_basis"] == "confirmatory"
        assert result.score == pytest.approx(100.0)
        assert result.details["info"]["stereo"]["n_declared"] == 2

    def test_misdeclaring_both_groups_scores_zero(self, stub_bundle):
        """Swap the ground truth and the confirmatory test must bottom out.

        A confirmatory statistic that ignored ``stereo_group`` and reported the
        top group instead would still say 100 here.
        """
        spec = ContrastSpec(
            ["he", "she"],
            [("nurse", "doctor", "he"), ("doctor", "nurse", "she")],
            CBS_TEMPLATES,
        )
        result = ContrastBasedScore(n_bootstrap=25, n_perm=25).compute(
            stub_bundle, spec
        )
        assert result.score == pytest.approx(0.0)

    def test_declared_margin_is_positive_when_the_declaration_is_right(
        self, stub_bundle
    ):
        """``margin`` subtracts the per-cell attribute main effect.

        Every group's raw contrast moves together when the negative word is
        simply less likely than the positive one, so the margin — not the raw
        contrast — is the confirmatory effect size.
        """
        spec = ContrastSpec(
            ["he", "she"],
            [("nurse", "doctor", "she"), ("doctor", "nurse", "he")],
            CBS_TEMPLATES,
        )
        stereo = ContrastBasedScore(n_bootstrap=25, n_perm=25).compute(
            stub_bundle, spec
        ).details["info"]["stereo"]
        assert stereo["margin"] > 0

    def test_undeclared_pairs_fall_back_to_the_exploratory_mean(self, stub_bundle):
        """With no ground truth there is no confirmatory test to run.

        ``she`` wins the single cell and ``he`` wins none, so the mean of the two
        per-group win rates is 50.
        """
        spec = ContrastSpec(["he", "she"], [("nurse", "doctor", None)], CBS_TEMPLATES)
        result = ContrastBasedScore(n_bootstrap=25, n_perm=25).compute(
            stub_bundle, spec
        )
        assert result.details["score_basis"] == "exploratory_mean"
        assert result.details["info"]["stereo"] is None
        assert result.score == pytest.approx(50.0)

    def test_baseline_is_one_over_the_number_of_groups(self, stub_bundle):
        spec = ContrastSpec(
            ["he", "she", "nurse", "doctor"],
            [("nurse", "doctor", None)],
            CBS_TEMPLATES,
        )
        result = ContrastBasedScore(n_bootstrap=25, n_perm=25).compute(
            stub_bundle, spec
        )
        assert result.details["info"]["baseline_pct"] == pytest.approx(25.0)

    def test_permutation_null_reports_a_corrected_p_value(self, stub_bundle):
        spec = ContrastSpec(
            ["he", "she"],
            [("nurse", "doctor", "she"), ("doctor", "nurse", "he")],
            CBS_TEMPLATES,
        )
        max_null = ContrastBasedScore(n_bootstrap=25, n_perm=50, seed=3).compute(
            stub_bundle, spec
        ).details["info"]["max_null"]
        assert max_null["n_perm"] == 50
        assert 0.0 < max_null["p_cbs"] <= 1.0
        assert max_null["top_group_cbs"] in {"he", "she"}

    def test_seed_makes_the_bootstrap_reproducible(self, stub_bundle):
        spec = ContrastSpec(["he", "she"], [("nurse", "doctor", "she")], CBS_TEMPLATES)
        kwargs = dict(n_bootstrap=40, n_perm=40, seed=5)
        first = ContrastBasedScore(**kwargs).compute(stub_bundle, spec)
        second = ContrastBasedScore(**kwargs).compute(stub_bundle, spec)
        assert (
            first.details["per_group"]["she"]["cbs_ci"]
            == second.details["per_group"]["she"]["cbs_ci"]
        )

    def test_custom_placeholders_are_honoured(self, stub_bundle):
        spec = ContrastSpec(["he", "she"], [("nurse", "doctor", "she")], ["<G> is <A>."])
        result = ContrastBasedScore(
            n_bootstrap=25, n_perm=25, group_placeholder="<G>", attr_placeholder="<A>"
        ).compute(stub_bundle, spec)
        assert result.score == pytest.approx(100.0)

    def test_declaring_a_group_outside_the_term_list_is_rejected(self):
        with pytest.raises(ValueError, match="not in group_terms"):
            ContrastSpec(["he"], [("nurse", "doctor", "she")], CBS_TEMPLATES)

    def test_contrast_pairs_must_be_triples(self):
        with pytest.raises(ValueError, match="must be a 3-tuple"):
            ContrastSpec(["he"], [("nurse", "doctor")], CBS_TEMPLATES)

    def test_wrong_container_is_reported(self, stub_bundle):
        with pytest.raises(TypeError, match="expects a ContrastSpec"):
            ContrastBasedScore().compute(stub_bundle, ["he", "she"])

    def test_missing_data_is_reported(self, stub_bundle):
        with pytest.raises(ValueError, match="requires group terms"):
            ContrastBasedScore().compute(stub_bundle, None)

    def test_legacy_keywords_warn(self, stub_bundle):
        with pytest.warns(DeprecationWarning, match="ContrastSpec"):
            result = ContrastBasedScore(n_bootstrap=25, n_perm=25).compute(
                stub_bundle,
                group_terms=["he", "she"],
                contrast_pairs=[("nurse", "doctor", "she")],
                templates=CBS_TEMPLATES,
            )
        assert result.score == pytest.approx(100.0)
