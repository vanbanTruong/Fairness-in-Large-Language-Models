"""Encoder-decoder metrics against hand-computed values.

Seven metrics live behind an ``AutoModelForSeq2SeqLM``, and every one of them
reaches through to either a *generation* or a *forced-choice loss*. Neither is
something a test can assert about without a checkpoint — which is why this
family sat at 17-28%.

The two stubs in :mod:`tests.stubs` replace exactly those two channels:

``StubSeq2SeqLM(generations=..., by_prompt=...)``
    Replays scripted text, so LFP's frequency bands, MCD's stem entropies and
    NPD's position histograms all become counts over a string the test wrote.
``StubSeq2SeqLM(scripted_losses=...)``
    Returns a loss per ``forward`` call in the order the metric makes them,
    which is what SD's pronoun choice and IBS's PMI-corrected NLI need. The
    token-bag loss cannot serve here: SD's cues are French and IBS's are
    ``Yes``/``No``/``Maybe``, so all of them collapse to ``[UNK]``.

Generated text has to survive a decode for the first channel to mean anything,
so those tests pair the model with :class:`~tests.stubs.RoundTripTokenizer`
rather than the five-word :class:`~tests.stubs.StubTokenizer`.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from fairLMs.definitions.encoder_decoder.extrinsic_bias.counterfactual_fairness.auc import (
    _contains_any,
    _swap_gender,
    _swap_nationality,
    compute_auc,
)
from fairLMs.definitions.encoder_decoder.extrinsic_bias.fair_inference.ibs import (
    compute_ibs,
    predict_nli,
)
from fairLMs.definitions.encoder_decoder.extrinsic_bias.individual_fairness.ss import (
    _get_labse_embedding,
    compute_ss,
)
from fairLMs.definitions.encoder_decoder.extrinsic_bias.individual_fairness.ss import (
    _swap_gender as ss_swap_gender,
)
from fairLMs.definitions.encoder_decoder.extrinsic_bias.individual_fairness.ss import (
    _swap_nationality as ss_swap_nationality,
)
from fairLMs.definitions.encoder_decoder.extrinsic_bias.position_based.npd import (
    _segment_distribution,
    _split_sentences,
    compute_npd,
)
from fairLMs.definitions.encoder_decoder.intrinsic_bias.algorithmic_disparity.lfp.lfp import (
    _classify_word,
    _tokenize_words,
    compute_lfp,
)
from fairLMs.definitions.encoder_decoder.intrinsic_bias.algorithmic_disparity.mcd.mcd import (
    _stem,
    compute_mcd,
)
from fairLMs.definitions.encoder_decoder.intrinsic_bias.stereotypical_association.sd.sd import (
    age_accuracy,
    compute_sd,
    predict_age,
    predict_gender,
    pronoun_accuracy,
)
from fairLMs.definitions.encoder_decoder.intrinsic_bias.stereotypical_association.sva.sva import (
    compute_bias_score,
    compute_stereotype_direction,
    compute_sva,
)
from fairLMs.definitions import (
    CounterfactualAucScore,
    InferenceBiasScore,
    LexicalFrequencyProportion,
    MorphologicalChoiceDivergence,
    NormalizedPositionDistance,
    StereotypicalDivergence,
    StereotypicalValueAttribution,
    TranslationSimilarityScore,
)
from fairLMs.definitions.data import LabeledSentences, StereotypeLabelled, WordSets
from .stubs import (
    RoundTripTokenizer,
    StubSentenceEncoder,
    StubSeq2SeqLM,
    StubTokenizer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
# "he" and "she" pull the stub encoder's embedding in different directions, so
# a label that tracks the pronoun is linearly recoverable and a label that does
# not is not. That is the whole of what AUC and SVA measure.
HE_SENTENCES = ["he doctor", "he engineer", "he nurse", "he doctor engineer"]
SHE_SENTENCES = ["she doctor", "she engineer", "she nurse", "she doctor engineer"]


def scripted(generations, **kwargs):
    """A seq2seq stub wired to a round-tripping tokenizer, and that tokenizer.

    Returned as a pair because the metric and the model have to agree on the
    tokenizer: the metric encodes the prompt and decodes the generation, and
    the decode is only faithful if it is the same vocabulary that encoded it.
    """
    tokenizer = RoundTripTokenizer()
    model = StubSeq2SeqLM(generations=generations, tokenizer=tokenizer, **kwargs)
    return model, tokenizer


@pytest.fixture
def seq2seq_bundle():
    return StubTokenizer(), StubSeq2SeqLM()


# ---------------------------------------------------------------------------
# Shared counterfactual text helpers
# ---------------------------------------------------------------------------
class TestCounterfactualSwaps:
    """``_swap_gender`` / ``_swap_nationality``, duplicated in auc.py and ss.py.

    Both copies are exercised, because they are genuinely separate code: a fix
    applied to one would not reach the other.
    """

    def test_whole_word_matching_ignores_substrings(self):
        assert _contains_any("he left", ["he"]) is True
        # "the" contains "he", and "shelf" contains "he" — neither is the pronoun.
        assert _contains_any("the shelf", ["he"]) is False

    def test_matching_is_case_insensitive(self):
        assert _contains_any("He arrived", ["he"]) is True

    def test_male_text_swaps_to_female(self):
        swapped, changed = _swap_gender("He fixed his own bike himself.")
        assert changed is True
        assert swapped == "she fixed her own bike herself."

    def test_female_text_swaps_to_male(self):
        swapped, changed = _swap_gender("She lost her keys; they were hers.")
        assert changed is True
        assert swapped == "he lost him keys; they were his."

    def test_mixed_gender_text_is_left_alone(self):
        # Swapping a sentence that already names both would destroy the
        # distinction, so the swap declines rather than guessing.
        original = "He told her the news."
        swapped, changed = _swap_gender(original)
        assert changed is False
        assert swapped == original

    def test_genderless_text_is_left_alone(self):
        swapped, changed = _swap_gender("The doctor arrived.")
        assert changed is False
        assert swapped == "The doctor arrived."

    def test_nationality_swaps_in_both_directions(self):
        assert _swap_nationality("An American engineer.")[0] == "An chinese engineer."
        assert _swap_nationality("A Chinese engineer.")[0] == "A american engineer."

    def test_nationality_swap_declines_when_both_groups_present(self):
        original = "An American and a Chinese engineer."
        swapped, changed = _swap_nationality(original)
        assert changed is False
        assert swapped == original

    def test_nationality_swap_declines_when_neither_group_present(self):
        swapped, changed = _swap_nationality("An engineer.")
        assert changed is False
        assert swapped == "An engineer."

    def test_the_ss_copies_behave_identically(self):
        for text in (
            "He fixed his own bike himself.",
            "She lost her keys; they were hers.",
            "He told her the news.",
            "An American engineer.",
        ):
            assert ss_swap_gender(text) == _swap_gender(text)
            assert ss_swap_nationality(text) == _swap_nationality(text)


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------
class TestCounterfactualAuc:
    """AUC is a *recoverability* probe: 0.5 is fair, 1.0 is fully recoverable."""

    def test_a_label_that_tracks_the_pronoun_is_perfectly_recoverable(
        self, seq2seq_bundle
    ):
        tokenizer, model = seq2seq_bundle
        auc, std, n0, n1, rows = compute_auc(
            model,
            tokenizer,
            HE_SENTENCES + SHE_SENTENCES,
            [0] * 4 + [1] * 4,
            test_ratio=0.5,
            n_seeds=4,
        )
        assert auc == 1.0
        assert std == 0.0
        assert (n0, n1) == (4, 4)
        assert len(rows) == 8

    def test_a_label_uncorrelated_with_the_text_is_not_recoverable(
        self, seq2seq_bundle
    ):
        tokenizer, model = seq2seq_bundle
        # Every sentence is the same string, so the encoder hands the classifier
        # eight identical rows and it can do no better than chance.
        auc, _, n0, n1, _ = compute_auc(
            model,
            tokenizer,
            ["he doctor"] * 8,
            [0, 1] * 4,
            test_ratio=0.5,
            n_seeds=4,
        )
        assert auc == 0.5
        assert (n0, n1) == (4, 4)

    def test_rows_record_the_embedding_norm_and_a_truncated_input(
        self, seq2seq_bundle
    ):
        tokenizer, model = seq2seq_bundle
        long_sentence = "he doctor " * 40
        _, _, _, _, rows = compute_auc(
            model, tokenizer, [long_sentence, "she nurse"], [0, 1], n_seeds=1
        )
        assert rows[0]["index"] == 0
        assert rows[0]["pair_id"] == 0  # falls back to the row index
        assert len(rows[0]["input"]) == 80
        assert rows[0]["emb_norm"] > 0.0
        assert rows[1]["label"] == 1

    def test_pair_ids_are_recorded_and_used_as_split_groups(self, seq2seq_bundle):
        tokenizer, model = seq2seq_bundle
        pair_ids = [0, 0, 1, 1, 2, 2, 3, 3]
        auc, _, _, _, rows = compute_auc(
            model,
            tokenizer,
            ["he doctor", "she doctor"] * 4,
            [0, 1] * 4,
            pair_ids=pair_ids,
            test_ratio=0.5,
            n_seeds=4,
        )
        # Grouping by pair keeps a counterfactual twin out of the training set,
        # but the pronoun still gives the label away.
        assert auc == 1.0
        assert [r["pair_id"] for r in rows] == pair_ids

    def test_a_class_with_fewer_than_two_members_short_circuits(self, seq2seq_bundle):
        tokenizer, model = seq2seq_bundle
        auc, std, n0, n1, rows = compute_auc(
            model, tokenizer, HE_SENTENCES[:3], [0, 0, 1], n_seeds=2
        )
        # Cannot fit and evaluate a classifier on a single positive, so the
        # metric reports 0.0 rather than an arbitrary number — and still hands
        # back the rows, so a caller can see why.
        assert (auc, std) == (0.0, 0.0)
        assert (n0, n1) == (2, 1)
        assert len(rows) == 3

    def test_no_usable_split_short_circuits(self, seq2seq_bundle):
        tokenizer, model = seq2seq_bundle
        # A 5% test ratio leaves a single held-out row, which can never contain
        # both classes, so every split is rejected and no AUC is defined.
        auc, std, _, _, _ = compute_auc(
            model,
            tokenizer,
            HE_SENTENCES + SHE_SENTENCES,
            [0] * 4 + [1] * 4,
            test_ratio=0.05,
            n_seeds=3,
        )
        assert (auc, std) == (0.0, 0.0)

    def test_mismatched_lengths_are_rejected(self, seq2seq_bundle):
        tokenizer, model = seq2seq_bundle
        with pytest.raises(AssertionError, match="must match"):
            compute_auc(model, tokenizer, ["he doctor"], [0, 1])

    def test_the_metric_reports_the_mean_auc_as_its_score(self, seq2seq_bundle):
        result = CounterfactualAucScore(test_ratio=0.5, n_seeds=4).compute(
            seq2seq_bundle,
            LabeledSentences(HE_SENTENCES + SHE_SENTENCES, [0] * 4 + [1] * 4),
        )
        assert result.score == 1.0
        assert result.details["auc_mean"] == 1.0
        assert result.details["n_class_0"] == 4
        assert result.details["n_class_1"] == 4
        assert len(result.details["rows"]) == 8

    def test_a_sequence_of_pairs_is_accepted_in_place_of_labelled_sentences(
        self, seq2seq_bundle
    ):
        examples = list(zip(HE_SENTENCES + SHE_SENTENCES, [0] * 4 + [1] * 4))
        result = CounterfactualAucScore(test_ratio=0.5, n_seeds=2).compute(
            seq2seq_bundle, examples
        )
        assert result.score == 1.0

    def test_legacy_sentences_and_labels_keywords_still_work(self, seq2seq_bundle):
        with pytest.warns(DeprecationWarning):
            result = CounterfactualAucScore(test_ratio=0.5, n_seeds=2).compute(
                seq2seq_bundle,
                sentences=HE_SENTENCES + SHE_SENTENCES,
                labels=[0] * 4 + [1] * 4,
            )
        assert result.score == 1.0

    def test_missing_data_is_an_error(self, seq2seq_bundle):
        with pytest.raises(ValueError, match="requires sentences and labels"):
            CounterfactualAucScore().compute(seq2seq_bundle)

    def test_unknown_keywords_are_rejected(self, seq2seq_bundle):
        with pytest.raises(TypeError, match="n_seed"):
            CounterfactualAucScore().compute(
                seq2seq_bundle,
                LabeledSentences(HE_SENTENCES, [0, 0, 1, 1]),
                n_seed=2,
            )

    # -- the metric refuses what compute_auc merely short-circuits -----------
    #
    # compute_auc returns 0.0 and hands back the rows so a caller can see why.
    # As a *score*, 0.0 is the most extreme possible finding (perfectly
    # anti-recoverable), so the metric decides these cases up front instead.

    def test_string_labels_are_refused_rather_than_scored_as_zero(
        self, seq2seq_bundle
    ):
        with pytest.raises(TypeError, match="must be integer class ids"):
            CounterfactualAucScore().compute(
                seq2seq_bundle,
                LabeledSentences(
                    HE_SENTENCES + SHE_SENTENCES,
                    ["male"] * 4 + ["female"] * 4,
                ),
            )

    def test_non_binary_classes_are_refused(self, seq2seq_bundle):
        with pytest.raises(ValueError, match="exactly two classes"):
            CounterfactualAucScore().compute(
                seq2seq_bundle,
                LabeledSentences(HE_SENTENCES + SHE_SENTENCES, [0, 1, 2, 0] * 2),
            )

    def test_a_class_with_fewer_than_two_members_is_refused(self, seq2seq_bundle):
        with pytest.raises(ValueError, match="at least two members of each"):
            CounterfactualAucScore().compute(
                seq2seq_bundle, LabeledSentences(HE_SENTENCES[:3], [0, 0, 1])
            )

    def test_a_test_ratio_that_cannot_hold_both_classes_is_refused(
        self, seq2seq_bundle
    ):
        with pytest.raises(ValueError, match="cannot\n?\\s*contain both classes|cannot"):
            CounterfactualAucScore(test_ratio=0.05).compute(
                seq2seq_bundle,
                LabeledSentences(HE_SENTENCES + SHE_SENTENCES, [0] * 4 + [1] * 4),
            )

    def test_a_genuine_zero_is_still_reportable(self, seq2seq_bundle):
        # The guards are all decidable from the labels, so they never intercept
        # a real AUC — including one that legitimately lands at an extreme.
        result = CounterfactualAucScore(test_ratio=0.5, n_seeds=4).compute(
            seq2seq_bundle,
            LabeledSentences(HE_SENTENCES + SHE_SENTENCES, [0] * 4 + [1] * 4),
        )
        assert result.score == 1.0
        assert result.details["n_class_0"] == 4


# ---------------------------------------------------------------------------
# IBS
# ---------------------------------------------------------------------------
class TestInferenceBiasScore:
    """IBS = (signed non-neutral skew) x (1 - neutral rate).

    Both factors are hand-checkable, and the sign is the part worth pinning: a
    model that entails the pro-stereotype and contradicts the anti-stereotype
    scores +1, and the exact reverse scores -1.
    """

    def test_maximal_stereotyping_scores_plus_one(self):
        ibs, counts = compute_ibs([("entailment", "contradiction")] * 2)
        assert ibs == 1.0
        assert counts["n_entail_pro"] == 2
        assert counts["n_contra_anti"] == 2
        assert counts["n_non_neutral"] == 4
        assert counts["n_neutral"] == 0
        assert counts["accuracy"] == 0.0

    def test_the_exact_reverse_scores_minus_one(self):
        ibs, counts = compute_ibs([("contradiction", "entailment")] * 2)
        assert ibs == -1.0
        assert counts["n_entail_pro"] == 0
        assert counts["n_contra_anti"] == 0

    def test_an_all_neutral_model_scores_zero(self):
        ibs, counts = compute_ibs([("neutral", "neutral")] * 3)
        assert ibs == 0.0
        assert counts["n_non_neutral"] == 0
        assert counts["n_neutral"] == 6
        assert counts["accuracy"] == 1.0

    def test_a_half_neutral_model_is_scaled_down_by_the_neutral_rate(self):
        # Two pairs, four predictions: entailment, neutral, contradiction, neutral.
        # skew = 2 * (1 + 1) / 2 - 1 = 1; neutral rate = 2/4; so 1 * (1 - 0.5).
        ibs, counts = compute_ibs(
            [("entailment", "contradiction"), ("neutral", "neutral")]
        )
        assert ibs == 0.5
        assert counts["n_pairs"] == 2
        assert counts["n_non_neutral"] == 2
        assert counts["accuracy"] == 0.5

    def test_no_predictions_gives_zero_and_no_counts(self):
        assert compute_ibs([]) == (0.0, {})

    def test_the_pmi_null_correction_can_change_the_predicted_label(self):
        # predict_nli scores Yes / Maybe / No in that order, each against a
        # content-free null prompt. Raw losses alone would pick Maybe (1.0 is
        # the smallest); the null correction, which is what the metric actually
        # ranks on, picks Yes:
        #   entailment    5.00 - 2.00 = 3.00
        #   neutral       1.10 - 1.00 = 0.10
        #   contradiction 3.05 - 3.00 = 0.05
        tokenizer = StubTokenizer()
        model = StubSeq2SeqLM(
            scripted_losses=[2.0, 5.0, 1.0, 1.1, 3.0, 3.05], tokenizer=tokenizer
        )
        assert predict_nli(model, tokenizer, "he doctor", "she nurse") == "entailment"

    def test_predict_nli_ranks_on_the_corrected_score_not_the_raw_loss(self):
        # Same raw losses as above, but a flat null: now nothing is corrected
        # away and the smallest raw loss (Maybe) wins. If the subtraction were
        # dropped or its sign flipped, this and the previous test could not both
        # hold.
        tokenizer = StubTokenizer()
        model = StubSeq2SeqLM(
            scripted_losses=[2.0, 0.0, 1.0, 0.0, 3.0, 0.0], tokenizer=tokenizer
        )
        assert predict_nli(model, tokenizer, "he doctor", "she nurse") == "neutral"

    def test_the_metric_reports_the_score_and_the_counts(self):
        result = InferenceBiasScore().compute(
            None, [("entailment", "contradiction"), ("neutral", "neutral")]
        )
        assert result.score == 0.5
        assert result.details["n"] == 2
        assert result.details["counts"]["n_pairs"] == 2

    def test_the_metric_needs_no_model(self):
        assert InferenceBiasScore().compute(data=[("neutral", "neutral")]).score == 0.0

    def test_missing_data_is_an_error(self):
        with pytest.raises(ValueError, match=r"requires \(label, prediction\) pairs"):
            InferenceBiasScore().compute(None)

    def test_non_pair_items_are_rejected(self):
        with pytest.raises(ValueError, match="must be a"):
            InferenceBiasScore().compute(None, ["entailment", "neutral"])

    def test_legacy_predictions_keyword_still_works(self):
        with pytest.warns(DeprecationWarning):
            result = InferenceBiasScore().compute(
                None, predictions=[("neutral", "neutral")]
            )
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# NPD
# ---------------------------------------------------------------------------
# Four lexically disjoint sentences, so TF-IDF can attribute a copied sentence
# to exactly one of them and the segment histogram is a known one-hot.
ARTICLE = (
    "Alpha aircraft hangar. "
    "Bravo bicycle workshop. "
    "Charlie cinema projector. "
    "Delta dolphin lagoon."
)
ARTICLE_SENTENCES = [
    "Alpha aircraft hangar.",
    "Bravo bicycle workshop.",
    "Charlie cinema projector.",
    "Delta dolphin lagoon.",
]


class TestNormalizedPositionDistance:
    def test_sentences_split_on_terminal_punctuation(self):
        assert _split_sentences(ARTICLE) == ARTICLE_SENTENCES
        assert _split_sentences("  One! Two? Three.  ") == ["One!", "Two?", "Three."]
        assert _split_sentences("   ") == []

    def test_a_copied_sentence_lands_in_its_own_segment(self):
        first = _segment_distribution([ARTICLE_SENTENCES[0]], ARTICLE_SENTENCES, 4)
        last = _segment_distribution([ARTICLE_SENTENCES[-1]], ARTICLE_SENTENCES, 4)
        assert list(first) == [1.0, 0.0, 0.0, 0.0]
        assert list(last) == [0.0, 0.0, 0.0, 1.0]

    def test_segments_are_proportional_not_positional(self):
        # Eight sentences into four segments puts sentences 0-1 in segment 0,
        # 2-3 in segment 1, and so on: (best * K) // n.
        sentences = [f"Word{i} unique{i} token{i}." for i in range(8)]
        dist = _segment_distribution([sentences[5]], sentences, 4)
        assert list(dist) == [0.0, 0.0, 1.0, 0.0]

    def test_an_empty_summary_falls_back_to_uniform(self):
        assert list(_segment_distribution([], ARTICLE_SENTENCES, 4)) == [0.25] * 4

    def test_an_empty_article_falls_back_to_uniform(self):
        assert list(_segment_distribution(["Alpha."], [], 4)) == [0.25] * 4

    def test_text_with_no_tfidf_vocabulary_falls_back_to_uniform(self):
        # TfidfVectorizer's default token pattern needs two word characters, so
        # single-letter sentences leave it with an empty vocabulary and it
        # raises rather than returning an empty matrix.
        assert list(_segment_distribution(["d"], ["a.", "b.", "c."], 4)) == [0.25] * 4

    def test_a_lead_biased_summary_is_halfway_from_uniform(self):
        # A summary drawn entirely from segment 0, scored against a uniform
        # reference over K=4 positions. The 1-Wasserstein distance is the sum of
        # absolute CDF gaps, |0.25-1| + |0.5-1| + |0.75-1| = 1.5, and the metric
        # normalises by K-1 = 3.
        model, tokenizer = scripted([ARTICLE_SENTENCES[0]])
        mean_npd, rows = compute_npd(model, tokenizer, [ARTICLE], K=4)
        assert mean_npd == 0.5
        assert rows[0]["gold_summary"] == "(uniform)"
        assert rows[0]["npd"] == 0.5

    def test_opposite_ends_of_the_article_give_the_maximal_distance(self):
        # Reference mass entirely at position 3, model mass entirely at 0: every
        # one of the K-1 CDF gaps is a full 1.0, so the normalised distance is 1.
        model, tokenizer = scripted([ARTICLE_SENTENCES[0]])
        mean_npd, rows = compute_npd(
            model,
            tokenizer,
            [ARTICLE],
            gold_summaries=[ARTICLE_SENTENCES[-1]],
            K=4,
        )
        assert mean_npd == 1.0
        assert rows[0]["gold_summary"] == ARTICLE_SENTENCES[-1]

    def test_matching_the_reference_summary_gives_zero(self):
        model, tokenizer = scripted([ARTICLE_SENTENCES[0]])
        mean_npd, _ = compute_npd(
            model,
            tokenizer,
            [ARTICLE],
            gold_summaries=[ARTICLE_SENTENCES[0]],
            K=4,
        )
        assert mean_npd == 0.0

    def test_articles_with_no_sentences_are_skipped(self):
        model, tokenizer = scripted([ARTICLE_SENTENCES[0]])
        mean_npd, rows = compute_npd(model, tokenizer, ["   ", ARTICLE], K=4)
        # The blank article contributes no row and does not drag the mean.
        assert rows == [dict(rows[0])]
        assert rows[0]["index"] == 1
        assert mean_npd == 0.5

    def test_no_scorable_articles_gives_zero(self):
        model, tokenizer = scripted([ARTICLE_SENTENCES[0]])
        assert compute_npd(model, tokenizer, ["  "], K=4) == (0.0, [])

    def test_the_summarize_prefix_reaches_the_model(self):
        model, tokenizer = scripted([ARTICLE_SENTENCES[0]])
        compute_npd(model, tokenizer, [ARTICLE], K=4)
        assert model.prompts_seen[0].startswith("summarize: Alpha aircraft hangar.")

    def test_the_metric_reports_the_mean_npd_as_its_score(self):
        model, tokenizer = scripted([ARTICLE_SENTENCES[0]])
        result = NormalizedPositionDistance(K=4).compute((tokenizer, model), [ARTICLE])
        assert result.score == 0.5
        assert result.details["n_articles"] == 1
        assert len(result.details["rows"]) == 1

    def test_gold_summaries_can_be_passed_per_call(self):
        model, tokenizer = scripted([ARTICLE_SENTENCES[0]])
        result = NormalizedPositionDistance(K=4).compute(
            (tokenizer, model), [ARTICLE], gold_summaries=[ARTICLE_SENTENCES[-1]]
        )
        assert result.score == 1.0

    def test_articles_can_arrive_as_dicts(self):
        model, tokenizer = scripted([ARTICLE_SENTENCES[0]])
        result = NormalizedPositionDistance(K=4).compute(
            (tokenizer, model), [{"article": ARTICLE}]
        )
        assert result.score == 0.5

    def test_a_dict_without_a_recognised_text_key_is_an_error(self):
        model, tokenizer = scripted([ARTICLE_SENTENCES[0]])
        with pytest.raises(ValueError, match="has no 'article'"):
            NormalizedPositionDistance().compute((tokenizer, model), [{"body": ARTICLE}])

    def test_a_non_text_example_is_an_error(self):
        model, tokenizer = scripted([ARTICLE_SENTENCES[0]])
        with pytest.raises(TypeError, match="expected strings or dicts"):
            NormalizedPositionDistance().compute((tokenizer, model), [17])

    def test_missing_data_is_an_error(self, seq2seq_bundle):
        with pytest.raises(ValueError, match="requires articles"):
            NormalizedPositionDistance().compute(seq2seq_bundle)

    def test_legacy_articles_keyword_still_works(self):
        model, tokenizer = scripted([ARTICLE_SENTENCES[0]])
        with pytest.warns(DeprecationWarning):
            result = NormalizedPositionDistance(K=4).compute(
                (tokenizer, model), articles=[ARTICLE]
            )
        assert result.score == 0.5


# ---------------------------------------------------------------------------
# SS
# ---------------------------------------------------------------------------
class TestTranslationSimilarity:
    """SS is the cosine between the sentence embeddings of two translations.

    A counterfactual pair *should* translate to near-identical text, so 1.0 is
    the fair end of this scale and anything lower is the model treating the two
    versions differently.
    """

    def test_identical_translations_score_exactly_one(self):
        model, tokenizer = scripted(["le médecin est là"])
        encoder = StubSentenceEncoder(vocab_size=tokenizer.vocab_size)
        mean_ss, std_ss, rows = compute_ss(
            model, tokenizer, encoder, tokenizer, [("he engineer", "she nurse")]
        )
        assert mean_ss == 1.0
        assert std_ss == 0.0
        assert rows[0]["translation_1"] == rows[0]["translation_2"]

    def test_the_score_is_the_cosine_of_the_two_translations(self):
        translations = {"he engineer": "le médecin est là", "she nurse": "la médecin est ici"}
        tokenizer = RoundTripTokenizer()
        model = StubSeq2SeqLM(by_prompt=translations, tokenizer=tokenizer)
        encoder = StubSentenceEncoder(vocab_size=tokenizer.vocab_size)

        mean_ss, _, rows = compute_ss(
            model, tokenizer, encoder, tokenizer, [("he engineer", "she nurse")]
        )

        expected = float(
            np.dot(
                _get_labse_embedding(encoder, tokenizer, translations["he engineer"]),
                _get_labse_embedding(encoder, tokenizer, translations["she nurse"]),
            )
        )
        assert mean_ss == pytest.approx(round(expected, 4))
        assert rows[0]["translation_1"] == translations["he engineer"]
        assert rows[0]["translation_2"] == translations["she nurse"]

    def test_the_target_language_reaches_the_prompt(self):
        model, tokenizer = scripted(["ein Arzt"])
        encoder = StubSentenceEncoder(vocab_size=tokenizer.vocab_size)
        compute_ss(
            model,
            tokenizer,
            encoder,
            tokenizer,
            [("he engineer", "she nurse")],
            tgt_lang="German",
        )
        assert all("English to German" in p for p in model.prompts_seen)

    def test_no_pairs_gives_zero(self):
        model, tokenizer = scripted(["le médecin"])
        encoder = StubSentenceEncoder(vocab_size=tokenizer.vocab_size)
        assert compute_ss(model, tokenizer, encoder, tokenizer, []) == (0.0, 0.0, [])

    def test_the_metric_reports_the_mean_and_spread(self):
        model, tokenizer = scripted(["le médecin est là"])
        encoder = StubSentenceEncoder(vocab_size=tokenizer.vocab_size)
        result = TranslationSimilarityScore(
            labse_model=encoder, labse_tokenizer=tokenizer
        ).compute((tokenizer, model), [("he engineer", "she nurse")] * 2)
        assert result.score == 1.0
        assert result.details["std"] == 0.0
        assert result.details["n"] == 2

    def test_a_missing_sentence_encoder_is_an_error(self, seq2seq_bundle):
        with pytest.raises(ValueError, match="needs a sentence encoder"):
            TranslationSimilarityScore().compute(
                seq2seq_bundle, [("he engineer", "she nurse")]
            )

    def test_non_pair_items_are_rejected(self):
        model, tokenizer = scripted(["le médecin"])
        encoder = StubSentenceEncoder(vocab_size=tokenizer.vocab_size)
        with pytest.raises(ValueError, match="must be an"):
            TranslationSimilarityScore(
                labse_model=encoder, labse_tokenizer=tokenizer
            ).compute((tokenizer, model), ["he engineer"])

    def test_missing_data_is_an_error(self, seq2seq_bundle):
        with pytest.raises(ValueError, match="requires \\(original, counterfactual\\)"):
            TranslationSimilarityScore().compute(seq2seq_bundle)

    def test_legacy_pairs_keyword_still_works(self):
        model, tokenizer = scripted(["le médecin est là"])
        encoder = StubSentenceEncoder(vocab_size=tokenizer.vocab_size)
        with pytest.warns(DeprecationWarning):
            result = TranslationSimilarityScore(
                labse_model=encoder, labse_tokenizer=tokenizer
            ).compute((tokenizer, model), pairs=[("he engineer", "she nurse")])
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# LFP
# ---------------------------------------------------------------------------
class TestLexicalFrequencyProportion:
    """LFP bands every generated word by French corpus frequency.

    Band 1 is the top 1000 words, band 2 the next 1000, band 3 everything else,
    and the score is the band-1 share — so a model that translates into a
    smaller vocabulary for one group scores higher on that group's sentences.
    """

    def test_words_are_banded_by_french_frequency(self):
        assert _classify_word("le") == 1
        assert _classify_word("chien") == 2
        assert _classify_word("ordinateur") == 3

    def test_non_alphabetic_and_empty_tokens_fall_into_band_one(self):
        assert _classify_word("42") == 1
        assert _classify_word("  ") == 1

    def test_banding_ignores_case_and_surrounding_space(self):
        assert _classify_word(" LE ") == 1

    def test_word_tokenisation_drops_digits_and_punctuation(self):
        assert _tokenize_words("Le chien, 42 ans!") == ["Le", "chien", "ans"]

    def test_the_proportions_are_counts_over_all_generated_words(self):
        # "le"(1) "chien"(2) "ordinateur"(3) — one word in each band.
        model, tokenizer = scripted(["le chien ordinateur"])
        pb1, pb2, pb3, rows = compute_lfp(model, tokenizer, ["he doctor"])
        assert (pb1, pb2, pb3) == (
            pytest.approx(1 / 3, abs=5e-5),
            pytest.approx(1 / 3, abs=5e-5),
            pytest.approx(1 / 3, abs=5e-5),
        )
        assert rows[0]["n_words"] == 3
        assert rows[0]["b1"] == 1 and rows[0]["b2"] == 1 and rows[0]["b3"] == 1

    def test_proportions_pool_across_sentences_rather_than_averaging_them(self):
        # Five words in total across the two generations, three of them band 1.
        # Averaging the two per-sentence shares would give 1/6 + 1 over 2.
        model, tokenizer = scripted(["le chien ordinateur", "la de"])
        pb1, pb2, pb3, rows = compute_lfp(model, tokenizer, ["a", "b"])
        assert (pb1, pb2, pb3) == (0.6, 0.2, 0.2)
        assert [r["n_words"] for r in rows] == [3, 2]

    def test_the_bands_always_sum_to_one(self):
        model, tokenizer = scripted(["le chien ordinateur"])
        pb1, pb2, pb3, _ = compute_lfp(model, tokenizer, ["a"])
        assert pb1 + pb2 + pb3 == pytest.approx(1.0, abs=1e-4)

    def test_a_generation_with_no_words_gives_zero(self):
        model, tokenizer = scripted(["123 !!!"])
        assert compute_lfp(model, tokenizer, ["a"])[:3] == (0.0, 0.0, 0.0)

    def test_the_translation_prefix_reaches_the_model(self):
        model, tokenizer = scripted(["le chien"])
        compute_lfp(model, tokenizer, ["he doctor"])
        assert model.prompts_seen == ["translate English to French: he doctor"]

    def test_the_metric_reports_the_band_one_share_as_its_score(self):
        model, tokenizer = scripted(["le chien ordinateur", "la de"])
        result = LexicalFrequencyProportion().compute((tokenizer, model), ["a", "b"])
        assert result.score == 0.6
        assert result.details["pb2"] == 0.2
        assert result.details["pb3"] == 0.2
        assert result.details["n_sentences"] == 2

    def test_sentences_can_arrive_as_dicts(self):
        model, tokenizer = scripted(["le chien"])
        result = LexicalFrequencyProportion().compute(
            (tokenizer, model), [{"sentence": "he doctor"}]
        )
        # "le" is band 1 and "chien" is band 2.
        assert result.score == 0.5

    def test_a_dict_without_a_recognised_text_key_is_an_error(self):
        model, tokenizer = scripted(["le chien"])
        with pytest.raises(ValueError, match="has no 'sentence'"):
            LexicalFrequencyProportion().compute((tokenizer, model), [{"body": "x"}])

    def test_a_non_text_example_is_an_error(self):
        model, tokenizer = scripted(["le chien"])
        with pytest.raises(TypeError, match="expected strings or dicts"):
            LexicalFrequencyProportion().compute((tokenizer, model), [17])

    def test_missing_data_is_an_error(self, seq2seq_bundle):
        with pytest.raises(ValueError, match="requires sentences"):
            LexicalFrequencyProportion().compute(seq2seq_bundle)

    def test_legacy_sentences_keyword_still_works(self):
        model, tokenizer = scripted(["le chien"])
        with pytest.warns(DeprecationWarning):
            result = LexicalFrequencyProportion().compute(
                (tokenizer, model), sentences=["he doctor"]
            )
        assert result.score == 0.5


# ---------------------------------------------------------------------------
# MCD
# ---------------------------------------------------------------------------
class TestMorphologicalChoiceDivergence:
    """MCD asks how many surface forms of one stem a model actually uses.

    Per stem it reports Shannon entropy over the wordform distribution, so a
    model that always picks the same inflection scores 0 and one that splits
    evenly between two scores ln 2.
    """

    def test_french_inflections_share_a_stem(self):
        assert _stem("soigne") == _stem("soignée") == _stem("soigner")

    def test_two_evenly_used_wordforms_give_log_two(self):
        model, tokenizer = scripted(["soigne soignée"])
        mean_h, mean_d, rows = compute_mcd(model, tokenizer, ["he doctor"])
        assert mean_h == round(math.log(2), 4)
        assert mean_d == 0.5  # the matching Simpson concentration
        assert rows[0]["n_words"] == 2

    def test_a_single_repeated_wordform_gives_zero_entropy(self):
        model, tokenizer = scripted(["soigne soigne"])
        assert compute_mcd(model, tokenizer, ["he doctor"])[:2] == (0.0, 1.0)

    def test_stems_seen_only_once_are_not_scored(self):
        # One observation says nothing about which form the model prefers, so
        # the stem is dropped rather than counted as zero entropy.
        model, tokenizer = scripted(["soigne"])
        assert compute_mcd(model, tokenizer, ["he doctor"])[:2] == (0.0, 0.0)

    def test_short_words_are_ignored(self):
        model, tokenizer = scripted(["le de et la"])
        assert compute_mcd(model, tokenizer, ["he doctor"])[:2] == (0.0, 0.0)

    def test_the_report_is_a_mean_over_stems(self):
        # "soign" splits evenly (ln 2, 0.5); "docteur" does not split at all
        # (0.0, 1.0). The metric averages over stems, not over words.
        model, tokenizer = scripted(["soigne soignée docteur docteur"])
        mean_h, mean_d, _ = compute_mcd(model, tokenizer, ["he doctor"])
        assert mean_h == round(math.log(2) / 2, 4)
        assert mean_d == 0.75

    def test_stems_pool_across_sentences(self):
        # Each generation alone would give a single-observation stem and score
        # nothing; together they are two forms of one stem.
        model, tokenizer = scripted(["soigne", "soignée"])
        mean_h, _, rows = compute_mcd(model, tokenizer, ["a", "b"])
        assert mean_h == round(math.log(2), 4)
        assert len(rows) == 2

    def test_the_metric_reports_mean_entropy_as_its_score(self):
        model, tokenizer = scripted(["soigne soignée"])
        result = MorphologicalChoiceDivergence().compute((tokenizer, model), ["a"])
        assert result.score == round(math.log(2), 4)
        assert result.details["mean_d"] == 0.5
        assert result.details["n_sentences"] == 1

    def test_max_new_tokens_can_be_passed_per_call(self):
        model, tokenizer = scripted(["soigne soignée"])
        result = MorphologicalChoiceDivergence().compute(
            (tokenizer, model), ["a"], max_new_tokens=16
        )
        assert result.score == round(math.log(2), 4)

    def test_unknown_keywords_are_rejected(self):
        model, tokenizer = scripted(["soigne"])
        with pytest.raises(TypeError, match="max_new_token"):
            MorphologicalChoiceDivergence().compute(
                (tokenizer, model), ["a"], max_new_token=16
            )


# ---------------------------------------------------------------------------
# SD
# ---------------------------------------------------------------------------
class TestStereotypicalDivergence:
    """SD contrasts task accuracy on stereotypical vs anti-stereotypical inputs.

    The forced choice underneath is a mean log-likelihood over French cue
    phrases, so ``scripted_losses`` drives it directly: each sentence consumes
    four losses, the two male cues then the two female ones.
    """

    @staticmethod
    def _model(losses):
        tokenizer = StubTokenizer()
        return StubSeq2SeqLM(scripted_losses=losses, tokenizer=tokenizer), tokenizer

    def test_the_lower_mean_loss_wins_the_pronoun_choice(self):
        model, tokenizer = self._model([1.0, 1.0, 2.0, 2.0])
        assert predict_gender(model, tokenizer, "he doctor") == "male"

        model, tokenizer = self._model([2.0, 2.0, 1.0, 1.0])
        assert predict_gender(model, tokenizer, "he doctor") == "female"

    def test_the_choice_averages_over_both_cues_for_a_label(self):
        # Male cues 3.0 and 0.0 average to 1.5; female cues 1.0 and 1.0 to 1.0.
        # Ranking on the best single cue would pick male; the mean picks female.
        model, tokenizer = self._model([3.0, 0.0, 1.0, 1.0])
        assert predict_gender(model, tokenizer, "he doctor") == "female"

    def test_the_age_variant_uses_the_age_cues(self):
        model, tokenizer = self._model([1.0, 1.0, 2.0, 2.0])
        assert predict_age(model, tokenizer, "he doctor") == "young"

    def test_accuracy_helpers_reward_an_exact_label_match(self):
        assert pronoun_accuracy("male", "male") == 1.0
        assert pronoun_accuracy("male", "female") == 0.0
        assert age_accuracy("young", "young") == 1.0
        assert age_accuracy("young", "old") == 0.0

    def test_an_unusable_gold_label_scores_half_a_point(self):
        # Neither right nor wrong: the example carries no answer to be graded
        # against, so it contributes chance rather than a zero.
        assert pronoun_accuracy("male", "neutral") == 0.5
        assert age_accuracy("young", "ancient") == 0.5

    def test_equal_accuracy_on_both_splits_gives_no_divergence(self):
        # Male on the stereotypical sentence (gold male) and female on the
        # anti-stereotypical one (gold female): both correct.
        model, tokenizer = self._model([1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0])
        m_stereo, m_anti, delta_s, rows = compute_sd(
            model, tokenizer, ["he doctor"], ["male"], ["she nurse"], ["female"]
        )
        assert (m_stereo, m_anti, delta_s) == (1.0, 1.0, 0.0)
        assert [r["split"] for r in rows] == ["stereo", "anti"]

    def test_the_sign_says_which_split_the_model_handles_better(self):
        # Always answering female: wrong on the stereotypical sentence, right on
        # the anti-stereotypical one, so anti - stereo is +1.
        model, tokenizer = self._model([2.0, 2.0, 1.0, 1.0] * 2)
        m_stereo, m_anti, delta_s, _ = compute_sd(
            model, tokenizer, ["he doctor"], ["male"], ["she nurse"], ["female"]
        )
        assert (m_stereo, m_anti, delta_s) == (0.0, 1.0, 1.0)

        # Always answering male flips the sign for the same pair of golds.
        model, tokenizer = self._model([1.0, 1.0, 2.0, 2.0] * 2)
        m_stereo, m_anti, delta_s, _ = compute_sd(
            model, tokenizer, ["he doctor"], ["male"], ["she nurse"], ["female"]
        )
        assert (m_stereo, m_anti, delta_s) == (1.0, 0.0, -1.0)

    def test_each_split_is_averaged_over_its_own_sentences(self):
        # Two stereotypical sentences, the first answered correctly and the
        # second not, against one correctly-answered anti-stereotypical one.
        model, tokenizer = self._model(
            [1.0, 1.0, 2.0, 2.0] + [2.0, 2.0, 1.0, 1.0] + [2.0, 2.0, 1.0, 1.0]
        )
        m_stereo, m_anti, delta_s, rows = compute_sd(
            model,
            tokenizer,
            ["he doctor", "he engineer"],
            ["male", "male"],
            ["she nurse"],
            ["female"],
        )
        assert m_stereo == 0.5
        assert m_anti == 1.0
        assert delta_s == 0.5
        assert len(rows) == 3

    def test_an_empty_split_scores_zero(self):
        model, tokenizer = self._model([1.0, 1.0, 2.0, 2.0])
        m_stereo, m_anti, delta_s, _ = compute_sd(
            model, tokenizer, ["he doctor"], ["male"], [], []
        )
        assert (m_stereo, m_anti, delta_s) == (1.0, 0.0, -1.0)

    def test_rows_record_the_prediction_and_the_gold_label(self):
        model, tokenizer = self._model([1.0, 1.0, 2.0, 2.0] * 2)
        _, _, _, rows = compute_sd(
            model, tokenizer, ["he doctor"], ["male"], ["she nurse"], ["female"]
        )
        assert rows[0]["predicted"] == "male"
        assert rows[0]["gold_label"] == "male"
        assert rows[0]["score"] == 1.0
        assert rows[1]["score"] == 0.0

    def test_the_metric_reports_the_divergence_as_its_score(self):
        model, tokenizer = self._model([2.0, 2.0, 1.0, 1.0] * 2)
        result = StereotypicalDivergence().compute(
            (tokenizer, model),
            StereotypeLabelled(
                LabeledSentences(["he doctor"], ["male"]),
                LabeledSentences(["she nurse"], ["female"]),
            ),
        )
        assert result.score == 1.0
        assert result.details["m_stereo"] == 0.0
        assert result.details["m_anti"] == 1.0
        assert len(result.details["rows"]) == 2

    def test_the_age_metric_fn_selects_the_age_predictor(self):
        model, tokenizer = self._model([1.0, 1.0, 2.0, 2.0] * 2)
        result = StereotypicalDivergence(metric_fn=age_accuracy).compute(
            (tokenizer, model),
            StereotypeLabelled(
                LabeledSentences(["he doctor"], ["young"]),
                LabeledSentences(["she nurse"], ["old"]),
            ),
        )
        assert result.details["m_stereo"] == 1.0
        assert result.details["m_anti"] == 0.0

    def test_a_custom_metric_fn_without_a_predictor_is_a_named_error(self):
        # Each built-in scorer grades the output of a specific prediction
        # routine, so an unpaired custom scorer cannot be run. It used to fail
        # with a bare KeyError from the lookup table.
        model, tokenizer = self._model([1.0] * 8)

        def my_scorer(predicted, gold):
            return 1.0

        with pytest.raises(ValueError, match="No prediction routine is paired"):
            StereotypicalDivergence(metric_fn=my_scorer).compute(
                (tokenizer, model),
                StereotypeLabelled(
                    LabeledSentences(["he doctor"], ["male"]),
                    LabeledSentences(["she nurse"], ["female"]),
                ),
            )

    def test_a_custom_metric_fn_runs_when_paired_with_a_predict_fn(self):
        model, tokenizer = self._model([1.0] * 8)

        def my_scorer(predicted, gold):
            return 1.0 if predicted == gold else 0.0

        def my_predictor(model, tokenizer, sentence):
            return "female" if sentence.startswith("she") else "male"

        result = StereotypicalDivergence(
            metric_fn=my_scorer, predict_fn=my_predictor
        ).compute(
            (tokenizer, model),
            StereotypeLabelled(
                LabeledSentences(["he doctor"], ["male"]),
                LabeledSentences(["she nurse"], ["male"]),
            ),
        )
        # The predictor is exact on the stereotype set and wrong on the anti
        # set. delta_s is m_anti - m_stereo, so the divergence is a full -1.0.
        assert result.details["m_stereo"] == 1.0
        assert result.details["m_anti"] == 0.0
        assert result.score == -1.0

    def test_a_label_vocabulary_the_scorer_cannot_grade_is_refused(self):
        # pronoun_accuracy scores "male"/"female" and returns 0.5 for anything
        # else, so a wrong vocabulary used to yield m_stereo == m_anti == 0.5
        # and a divergence of exactly 0.0 — a non-result that reads as parity.
        model, tokenizer = self._model([1.0] * 8)

        with pytest.raises(ValueError, match="scores labels from"):
            StereotypicalDivergence().compute(
                (tokenizer, model),
                StereotypeLabelled(
                    LabeledSentences(["he doctor"], ["negative"]),
                    LabeledSentences(["she nurse"], ["negative"]),
                ),
            )

    def test_individual_unknown_labels_still_score_as_chance(self):
        # Only a *complete* mismatch is an error: 0.5 remains the documented
        # score for a single unrecognised gold label.
        model, tokenizer = self._model([2.0, 2.0, 1.0, 1.0] * 2)
        result = StereotypicalDivergence().compute(
            (tokenizer, model),
            StereotypeLabelled(
                LabeledSentences(["he doctor", "he nurse"], ["male", "unknown"]),
                LabeledSentences(["she nurse", "she doctor"], ["female", "unknown"]),
            ),
        )
        assert 0.0 <= result.details["m_stereo"] <= 1.0

    def test_wrong_container_type_is_an_error(self, seq2seq_bundle):
        with pytest.raises(TypeError, match="expects a StereotypeLabelled"):
            StereotypicalDivergence().compute(seq2seq_bundle, ["he doctor"])

    def test_missing_data_is_an_error(self, seq2seq_bundle):
        with pytest.raises(ValueError, match="requires two labelled sentence sets"):
            StereotypicalDivergence().compute(seq2seq_bundle)

    def test_legacy_split_keywords_still_work(self):
        model, tokenizer = self._model([2.0, 2.0, 1.0, 1.0] * 2)
        with pytest.warns(DeprecationWarning):
            result = StereotypicalDivergence().compute(
                (tokenizer, model),
                stereo_sentences=["he doctor"],
                stereo_labels=["male"],
                anti_sentences=["she nurse"],
                anti_labels=["female"],
            )
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# SVA
# ---------------------------------------------------------------------------
STEREO_SENTS = ["he doctor", "he engineer"]
ANTI_SENTS = ["she nurse", "she doctor"]


class TestStereotypicalValueAttribution:
    """SVA is a Monte-Carlo Shapley attribution over attention heads.

    Shapley values have an exact arithmetic property — the marginal
    contributions along any permutation telescope to ``v(all) - v(none)`` — and
    that identity is the sharpest thing a test can hold this code to. It fails
    the moment the accumulation loop drops a term or divides by the wrong count.
    """

    @staticmethod
    def _bundle(n_layers=1, n_heads=2):
        tokenizer = StubTokenizer()
        return tokenizer, StubSeq2SeqLM(n_layers=n_layers, n_heads=n_heads)

    def test_the_direction_is_a_unit_vector(self):
        tokenizer, model = self._bundle()
        direction = compute_stereotype_direction(
            model, tokenizer, STEREO_SENTS, ANTI_SENTS
        )
        assert direction.shape == (model.config.d_model,)
        assert float(np.linalg.norm(direction)) == pytest.approx(1.0, abs=1e-5)

    def test_identical_sentence_sets_leave_no_direction_to_find(self):
        tokenizer, model = self._bundle()
        direction = compute_stereotype_direction(
            model, tokenizer, STEREO_SENTS, STEREO_SENTS
        )
        # The difference of two identical means is zero, and normalising it
        # would divide by zero, so the raw vector is returned instead.
        assert np.allclose(direction, 0.0)

    def test_masking_every_head_erases_the_bias_signal(self):
        tokenizer, model = self._bundle()
        direction = compute_stereotype_direction(
            model, tokenizer, STEREO_SENTS, ANTI_SENTS
        )
        v_empty = compute_bias_score(
            model, tokenizer, STEREO_SENTS, ANTI_SENTS, direction, set(), 1, 2
        )
        assert v_empty == 0.0

    def test_the_full_head_set_recovers_a_nonzero_signal(self):
        tokenizer, model = self._bundle()
        direction = compute_stereotype_direction(
            model, tokenizer, STEREO_SENTS, ANTI_SENTS
        )
        v_full = compute_bias_score(
            model, tokenizer, STEREO_SENTS, ANTI_SENTS, direction, {0, 1}, 1, 2
        )
        # The direction was built to point from anti- towards stereotypical, so
        # the projection gap has to come out positive.
        assert v_full > 0.0

    def test_the_hooks_are_removed_after_scoring(self):
        tokenizer, model = self._bundle()
        direction = compute_stereotype_direction(
            model, tokenizer, STEREO_SENTS, ANTI_SENTS
        )
        compute_bias_score(
            model, tokenizer, STEREO_SENTS, ANTI_SENTS, direction, set(), 1, 2
        )
        for block in model.get_encoder().block:
            assert not block.layer[0].SelfAttention._forward_hooks

    def test_the_attributions_sum_to_the_total_effect(self):
        tokenizer, model = self._bundle()
        direction = compute_stereotype_direction(
            model, tokenizer, STEREO_SENTS, ANTI_SENTS
        )
        v_empty = compute_bias_score(
            model, tokenizer, STEREO_SENTS, ANTI_SENTS, direction, set(), 1, 2
        )
        v_full = compute_bias_score(
            model, tokenizer, STEREO_SENTS, ANTI_SENTS, direction, {0, 1}, 1, 2
        )

        random.seed(0)
        _, phi = compute_sva(
            model, tokenizer, STEREO_SENTS, ANTI_SENTS, direction, 1, 2, n_samples=3
        )
        assert float(phi.sum()) == pytest.approx(v_full - v_empty, abs=1e-6)

    def test_the_score_is_the_share_of_attribution_mass_in_the_top_heads(self):
        tokenizer, model = self._bundle()
        direction = compute_stereotype_direction(
            model, tokenizer, STEREO_SENTS, ANTI_SENTS
        )
        random.seed(1)
        sva, phi = compute_sva(
            model,
            tokenizer,
            STEREO_SENTS,
            ANTI_SENTS,
            direction,
            1,
            2,
            n_samples=2,
            top_pct=0.1,
        )
        # ceil(0.1 * 2) is 0, and the floor of one head applies, so the score is
        # the single largest share.
        magnitudes = np.abs(phi)
        assert sva == pytest.approx(magnitudes.max() / magnitudes.sum())
        assert 0.5 <= sva <= 1.0

    def test_a_wider_top_slice_captures_more_mass(self):
        tokenizer, model = self._bundle(n_layers=2, n_heads=2)
        direction = compute_stereotype_direction(
            model, tokenizer, STEREO_SENTS, ANTI_SENTS
        )
        random.seed(2)
        narrow, _ = compute_sva(
            model, tokenizer, STEREO_SENTS, ANTI_SENTS, direction, 2, 2,
            n_samples=1, top_pct=0.25,
        )
        random.seed(2)
        wide, _ = compute_sva(
            model, tokenizer, STEREO_SENTS, ANTI_SENTS, direction, 2, 2,
            n_samples=1, top_pct=1.0,
        )
        assert narrow <= wide
        assert wide == pytest.approx(1.0)

    def test_a_zero_direction_leaves_nothing_to_attribute(self):
        tokenizer, model = self._bundle()
        random.seed(3)
        sva, phi = compute_sva(
            model,
            tokenizer,
            STEREO_SENTS,
            ANTI_SENTS,
            np.zeros(model.config.d_model),
            1,
            2,
            n_samples=1,
        )
        # Every bias score projects onto a zero direction, so the normalisation
        # has no mass to divide by and falls back to the raw magnitudes.
        assert np.allclose(phi, 0.0)
        assert sva == 0.0

    def test_the_head_shape_comes_from_the_config(self):
        derive = StereotypicalValueAttribution._derive_shape
        from types import SimpleNamespace

        assert derive(SimpleNamespace(num_layers=6, num_heads=8), None, None) == (6, 8)
        assert derive(
            SimpleNamespace(num_hidden_layers=4, num_attention_heads=2), None, None
        ) == (4, 2)
        # Explicit arguments win over whatever the config says.
        assert derive(SimpleNamespace(num_layers=6, num_heads=8), 1, 2) == (1, 2)

    def test_an_underspecified_config_is_an_error(self):
        from types import SimpleNamespace

        with pytest.raises(ValueError, match="could not derive n_layers/n_heads"):
            StereotypicalValueAttribution._derive_shape(SimpleNamespace(), None, None)

    def test_the_metric_derives_the_direction_by_default(self):
        tokenizer, model = self._bundle()
        random.seed(4)
        result = StereotypicalValueAttribution(n_samples=1).compute(
            (tokenizer, model),
            WordSets(STEREO_SENTS, ANTI_SENTS, STEREO_SENTS, ANTI_SENTS),
        )
        assert result.details["direction_derived"] is True
        assert (result.details["n_layers"], result.details["n_heads"]) == (1, 2)
        assert 0.0 <= result.score <= 1.0

    def test_a_supplied_direction_is_used_as_given(self):
        tokenizer, model = self._bundle()
        random.seed(5)
        result = StereotypicalValueAttribution(
            n_samples=1, direction=np.zeros(model.config.d_model)
        ).compute(
            (tokenizer, model),
            WordSets(STEREO_SENTS, ANTI_SENTS, STEREO_SENTS, ANTI_SENTS),
        )
        assert result.details["direction_derived"] is False
        assert result.score == 0.0

    def test_a_misshapen_direction_is_rejected(self):
        tokenizer, model = self._bundle()
        with pytest.raises(ValueError, match="hidden size is 8"):
            StereotypicalValueAttribution(direction=np.zeros(3)).compute(
                (tokenizer, model),
                WordSets(STEREO_SENTS, ANTI_SENTS, STEREO_SENTS, ANTI_SENTS),
            )

    def test_head_counts_can_be_overridden_per_call(self):
        tokenizer, model = self._bundle()
        random.seed(6)
        result = StereotypicalValueAttribution(n_samples=1).compute(
            (tokenizer, model),
            WordSets(STEREO_SENTS, ANTI_SENTS, STEREO_SENTS, ANTI_SENTS),
            n_layers=1,
            n_heads=1,
        )
        assert (result.details["n_layers"], result.details["n_heads"]) == (1, 1)

    def test_wrong_container_type_is_an_error(self, seq2seq_bundle):
        with pytest.raises(TypeError, match="expects a WordSets"):
            StereotypicalValueAttribution().compute(seq2seq_bundle, STEREO_SENTS)

    def test_missing_data_is_an_error(self, seq2seq_bundle):
        with pytest.raises(ValueError, match="requires stereotypical and"):
            StereotypicalValueAttribution().compute(seq2seq_bundle)

    def test_legacy_sentence_keywords_still_work(self):
        tokenizer, model = self._bundle()
        random.seed(7)
        with pytest.warns(DeprecationWarning):
            result = StereotypicalValueAttribution(n_samples=1).compute(
                (tokenizer, model),
                stereo_sents=STEREO_SENTS,
                anti_sents=ANTI_SENTS,
            )
        assert 0.0 <= result.score <= 1.0
