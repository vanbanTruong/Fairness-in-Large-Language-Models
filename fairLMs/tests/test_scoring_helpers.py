"""Tests for the two scoring helpers that five metrics each depend on.

``fairLMs.utils.pll`` is shared by PLL, CPS, AUL, AULA and CAT;
``fairLMs.utils.masking`` by DisCo, LPBS and CBS. A sign error or an off-by-one
slice in either propagates to every metric above it while every one of those
metrics still returns a plausible-looking float, so these are the two places
where an independently derived expected value is worth the most.

Every expectation below is recomputed from :data:`tests.stubs.BASE_LOGITS` and
:data:`tests.stubs.AFFINITY` with plain torch, never by calling the function
under test.
"""

import itertools
import math

import pytest
import torch

from fairLMs.utils.masking import (
    build_masked_sentence,
    get_mask_fill_probs,
    get_multitoken_log_prob,
    get_token_prob,
    get_top_k_predictions,
)
from fairLMs.utils.pll import get_span, get_token_ranks, score_sentence
from .stubs import (
    CONTENT_TOKENS,
    VOCAB,
    StubMaskedLM,
    StubTokenizer,
    attention_weights,
    expected_log_softmax,
    expected_logits,
)

SENTENCE = "he nurse doctor"
#: The bag of tokens each interior position of ``SENTENCE`` is scored against —
#: everything in the sentence except the token at that position.
CONTEXTS = [["nurse", "doctor"], ["he", "doctor"], ["he", "nurse"]]
GOLD = ["he", "nurse", "doctor"]


# ---------------------------------------------------------------------------
# The stub itself
# ---------------------------------------------------------------------------
class TestStubIsWellFormed:
    """Guards on the stub, so a failure elsewhere is never the stub's fault."""

    def test_every_context_gives_distinct_content_logits(self):
        """No ties, so top-k ordering is unambiguous.

        DisCo compares *sets of top-k tokens*; if two content words could tie,
        the overlap would depend on the sort implementation rather than on the
        tables, and the DisCo expectations below would be untestable.
        """
        content_ids = [VOCAB[t] for t in CONTENT_TOKENS]
        for size in range(len(CONTENT_TOKENS) + 1):
            for context in itertools.combinations(CONTENT_TOKENS, size):
                values = expected_logits(context)[content_ids].tolist()
                assert len(set(values)) == len(values), f"tie for context {context}"

    def test_position_excludes_its_own_token(self, stub_tokenizer, stub_mlm):
        ids = stub_tokenizer.encode(SENTENCE, return_tensors="pt")
        logits = stub_mlm(ids).logits
        for offset, context in enumerate(CONTEXTS, start=1):
            assert torch.allclose(logits[0, offset], expected_logits(context))

    def test_specials_contribute_nothing(self, stub_tokenizer, stub_mlm):
        """``[UNK]`` filler must not move the distribution.

        This is what lets a test write a readable template such as
        ``"[MASK] is a nurse"`` and still know that only ``nurse`` matters.
        """
        bare = stub_mlm(stub_tokenizer.encode("nurse", return_tensors="pt")).logits
        padded = stub_mlm(
            stub_tokenizer.encode("well actually nurse", return_tensors="pt")
        ).logits
        assert torch.allclose(bare[0, 0], padded[0, 0])

    def test_out_of_vocabulary_words_become_unk(self, stub_tokenizer):
        assert stub_tokenizer.encode("hobbit", add_special_tokens=False) == [
            stub_tokenizer.unk_token_id
        ]

    def test_mask_survives_trailing_punctuation(self, stub_tokenizer):
        ids = stub_tokenizer.encode("he is [MASK].", add_special_tokens=False)
        assert stub_tokenizer.mask_token_id in ids

    def test_round_trips_through_decode(self, stub_tokenizer):
        ids = stub_tokenizer.encode(SENTENCE, return_tensors="pt")
        assert stub_tokenizer.decode(ids[0], skip_special_tokens=True) == SENTENCE


# ---------------------------------------------------------------------------
# fairLMs.utils.pll
# ---------------------------------------------------------------------------
class TestGetTokenRanks:
    def test_rank_is_one_based_and_descending(self):
        log_probs = torch.log_softmax(
            torch.tensor([[3.0, 1.0, 2.0], [0.0, 5.0, -1.0]]), dim=-1
        )
        token_ids = torch.tensor([[2], [1]])
        # Row 0 orders as (0, 2, 1), so id 2 is second. Row 1's id 1 is first.
        assert get_token_ranks(log_probs, token_ids) == [2, 1]

    def test_worst_token_ranks_last(self):
        log_probs = torch.log_softmax(torch.tensor([[3.0, 1.0, 2.0]]), dim=-1)
        assert get_token_ranks(log_probs, torch.tensor([[1]])) == [3]


class TestScoreSentence:
    """AUL's scorer: mean log-probability over the interior tokens."""

    def test_score_matches_hand_computed_mean(self, stub_tokenizer, stub_mlm):
        expected = [
            expected_log_softmax(ctx)[VOCAB[gold]].item()
            for ctx, gold in zip(CONTEXTS, GOLD)
        ]
        score, _ranks = score_sentence(stub_tokenizer, stub_mlm, SENTENCE)
        assert score == pytest.approx(sum(expected) / len(expected), abs=1e-6)

    def test_drops_the_special_tokens(self, stub_tokenizer, stub_mlm):
        """Three interior tokens, so three ranks — not five."""
        _score, ranks = score_sentence(stub_tokenizer, stub_mlm, SENTENCE)
        assert len(ranks) == 3

    def test_ranks_reflect_the_fixed_tables(self, stub_tokenizer, stub_mlm):
        expected = []
        for ctx, gold in zip(CONTEXTS, GOLD):
            logits = expected_logits(ctx)
            better = int((logits > logits[VOCAB[gold]]).sum())
            expected.append(better + 1)
        _score, ranks = score_sentence(stub_tokenizer, stub_mlm, SENTENCE)
        assert ranks == expected

    def test_attention_weighting_changes_the_score(self, stub_tokenizer, stub_mlm):
        plain, _ = score_sentence(stub_tokenizer, stub_mlm, SENTENCE)
        weighted, _ = score_sentence(
            stub_tokenizer, stub_mlm, SENTENCE, use_attention=True
        )
        assert plain != weighted

    def test_attention_weighted_score_matches_hand_computed_value(
        self, stub_tokenizer, stub_mlm
    ):
        """The AULA path: per-token log-probs scaled by mean attention.

        Because the stub reports the same attention for every layer, head and
        query, the mean collapses to ``attention_weights(seq_len)[0]``, and the
        expected score is a three-term weighted mean computable by hand.
        """
        seq_len = 5  # [CLS] he nurse doctor [SEP]
        weights = attention_weights(seq_len)[0][1:-1]
        terms = [
            expected_log_softmax(ctx)[VOCAB[gold]].item() * w.item()
            for ctx, gold, w in zip(CONTEXTS, GOLD, weights)
        ]
        score, _ = score_sentence(
            stub_tokenizer, stub_mlm, SENTENCE, use_attention=True
        )
        assert score == pytest.approx(sum(terms) / len(terms), abs=1e-6)

    def test_attention_requires_an_encoder(self, stub_tokenizer):
        headless = StubMaskedLM(expose_encoder=False)
        with pytest.raises(AttributeError, match="attention tensors"):
            score_sentence(stub_tokenizer, headless, SENTENCE, use_attention=True)

    def test_fused_attention_backend_is_switched_to_eager(self, stub_tokenizer):
        """Fused kernels drop attentions even when asked for them.

        The helper is supposed to notice and force eager attention; without that
        the AULA weighting silently degrades to no weighting at all.
        """
        fused = StubMaskedLM(attn_implementation="sdpa")
        score_sentence(stub_tokenizer, fused, SENTENCE, use_attention=True)
        assert fused.attn_implementation_calls == ["eager", "sdpa"]
        assert fused.config._attn_implementation == "sdpa"

    def test_eager_backend_is_left_alone(self, stub_tokenizer, stub_mlm):
        score_sentence(stub_tokenizer, stub_mlm, SENTENCE, use_attention=True)
        assert stub_mlm.attn_implementation_calls == []


class TestGetSpan:
    """CPS restricts scoring to the tokens the two sentences share."""

    def test_equal_spans_skip_the_differing_token(self, stub_tokenizer):
        a = stub_tokenizer.encode("he nurse doctor", return_tensors="pt").squeeze(0)
        b = stub_tokenizer.encode("she nurse doctor", return_tensors="pt").squeeze(0)
        # [CLS] matches, position 1 differs, then nurse/doctor/[SEP] match.
        assert get_span(a, b, "equal") == ([0, 2, 3, 4], [0, 2, 3, 4])

    def test_diff_spans_are_the_complement(self, stub_tokenizer):
        a = stub_tokenizer.encode("he nurse doctor", return_tensors="pt").squeeze(0)
        b = stub_tokenizer.encode("she nurse doctor", return_tensors="pt").squeeze(0)
        assert get_span(a, b, "diff") == ([1], [1])

    def test_identical_sequences_have_no_diff(self, stub_tokenizer):
        a = stub_tokenizer.encode(SENTENCE, return_tensors="pt").squeeze(0)
        assert get_span(a, a.clone(), "diff") == ([], [])

    def test_handles_different_lengths(self, stub_tokenizer):
        a = stub_tokenizer.encode("he nurse", return_tensors="pt").squeeze(0)
        b = stub_tokenizer.encode("he nurse doctor", return_tensors="pt").squeeze(0)
        equal_a, equal_b = get_span(a, b, "equal")
        assert len(equal_a) == len(equal_b)
        assert [int(a[i]) for i in equal_a] == [int(b[i]) for i in equal_b]


# ---------------------------------------------------------------------------
# fairLMs.utils.masking
# ---------------------------------------------------------------------------
class TestGetMaskFillProbs:
    def test_probabilities_match_the_fixed_tables(self, stub_tokenizer, stub_mlm):
        probs = get_mask_fill_probs("[MASK] is a nurse", stub_tokenizer, stub_mlm)
        expected = torch.softmax(expected_logits(["nurse"]), dim=-1)
        assert torch.allclose(probs, expected, atol=1e-6)

    def test_sums_to_one(self, stub_tokenizer, stub_mlm):
        probs = get_mask_fill_probs("[MASK] is a nurse", stub_tokenizer, stub_mlm)
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_mask_position_selects_among_several_masks(self, stub_tokenizer, stub_mlm):
        """Two masks in one sentence must be addressable independently.

        LPBS relies on this: it reads the group slot at index 0 and the attribute
        slot at index -1 from the very same forward pass.
        """
        sentence = "[MASK] is a [MASK] doctor"
        first = get_mask_fill_probs(sentence, stub_tokenizer, stub_mlm, mask_position=0)
        last = get_mask_fill_probs(sentence, stub_tokenizer, stub_mlm, mask_position=-1)
        # Both positions see only `doctor`, so the stub returns the same vector;
        # what matters is that the second index is accepted and in range.
        assert torch.allclose(first, last)

    def test_missing_mask_is_an_error(self, stub_tokenizer, stub_mlm):
        with pytest.raises(ValueError, match="No \\[MASK\\] token found"):
            get_mask_fill_probs("he is a nurse", stub_tokenizer, stub_mlm)


class TestGetTokenProb:
    def test_reads_the_requested_token(self, stub_tokenizer, stub_mlm):
        probs = get_mask_fill_probs("[MASK] is a nurse", stub_tokenizer, stub_mlm)
        assert get_token_prob(probs, "she", stub_tokenizer) == pytest.approx(
            probs[VOCAB["she"]].item()
        )

    def test_unknown_token_falls_back_to_epsilon(self, stub_tokenizer, stub_mlm):
        """An unknown attribute word must not be read as index ``unk_token_id``.

        Returning ``probs[unk_token_id]`` would look like a real probability and
        quietly bias every downstream ratio.
        """
        probs = get_mask_fill_probs("[MASK] is a nurse", stub_tokenizer, stub_mlm)
        assert get_token_prob(probs, "hobbit", stub_tokenizer) == 1e-10

    def test_stereotypical_affinity_has_the_expected_sign(
        self, stub_tokenizer, stub_mlm
    ):
        """`nurse` should pull the distribution towards `she`, `doctor` towards `he`.

        This is the property the intrinsic metrics are built to detect, so it is
        worth asserting on the stub directly before asserting it through them.
        """
        with_nurse = get_mask_fill_probs("[MASK] nurse", stub_tokenizer, stub_mlm)
        with_doctor = get_mask_fill_probs("[MASK] doctor", stub_tokenizer, stub_mlm)
        assert get_token_prob(with_nurse, "she", stub_tokenizer) > get_token_prob(
            with_nurse, "he", stub_tokenizer
        )
        assert get_token_prob(with_doctor, "he", stub_tokenizer) > get_token_prob(
            with_doctor, "she", stub_tokenizer
        )


class TestBuildMaskedSentence:
    def test_single_token_term_gets_one_mask(self, stub_tokenizer):
        built = build_masked_sentence("GGG is a nurse", "GGG", "he", stub_tokenizer)
        assert built == "[MASK] is a nurse"

    def test_multi_token_term_gets_one_mask_per_sub_token(self, stub_tokenizer):
        """The mask count must match the term's tokenization, not its word count.

        ``get_multitoken_log_prob`` rejects any mismatch, so this is the function
        that keeps CBS from raising on multi-word group terms.
        """
        built = build_masked_sentence(
            "GGG is a nurse", "GGG", "nurse doctor", stub_tokenizer
        )
        assert built == "[MASK] [MASK] is a nurse"

    def test_placeholder_absent_leaves_the_template_unchanged(self, stub_tokenizer):
        assert (
            build_masked_sentence("he is a nurse", "GGG", "she", stub_tokenizer)
            == "he is a nurse"
        )


class TestGetMultitokenLogProb:
    def test_single_token_term_matches_the_fixed_tables(self, stub_tokenizer, stub_mlm):
        probs = torch.softmax(expected_logits(["nurse"]), dim=-1)
        expected = math.log(probs[VOCAB["she"]].item() + 1e-10)
        actual = get_multitoken_log_prob(
            "[MASK] is a nurse", "she", stub_tokenizer, stub_mlm
        )
        assert actual == pytest.approx(expected, abs=1e-6)

    def test_multi_token_term_averages_its_sub_tokens(self, stub_tokenizer, stub_mlm):
        """The mean, not the sum — a two-token term must not be penalised twice.

        Both masks here see the same context (``nurse`` at the end plus filler),
        so the expected value is the mean of the two sub-token log-probabilities.
        """
        sentence = "[MASK] [MASK] is a nurse"
        probs = torch.softmax(expected_logits(["nurse"]), dim=-1)
        parts = [
            math.log(probs[VOCAB[t]].item() + 1e-10) for t in ("doctor", "engineer")
        ]
        actual = get_multitoken_log_prob(
            sentence, "doctor engineer", stub_tokenizer, stub_mlm
        )
        assert actual == pytest.approx(sum(parts) / 2, abs=1e-6)

    def test_mask_count_mismatch_is_reported(self, stub_tokenizer, stub_mlm):
        with pytest.raises(ValueError, match="sub-tokens but sentence"):
            get_multitoken_log_prob(
                "[MASK] is a nurse", "doctor engineer", stub_tokenizer, stub_mlm
            )

    def test_empty_term_falls_back_to_log_epsilon(self, stub_tokenizer, stub_mlm):
        assert get_multitoken_log_prob(
            "[MASK] is a nurse", "   ", stub_tokenizer, stub_mlm
        ) == pytest.approx(math.log(1e-10))

    def test_unknown_sub_token_falls_back_to_log_epsilon(
        self, stub_tokenizer, stub_mlm
    ):
        assert get_multitoken_log_prob(
            "[MASK] is a nurse", "hobbit", stub_tokenizer, stub_mlm
        ) == pytest.approx(math.log(1e-10))


class TestGetTopKPredictions:
    def test_returns_k_scored_tokens(self, stub_pipeline):
        preds = get_top_k_predictions(stub_pipeline, "[MASK] is a nurse", k=3)
        assert len(preds) == 3

    def test_ordering_follows_the_fixed_tables(self, stub_pipeline):
        logits = expected_logits(["nurse"])
        ranked = sorted(VOCAB, key=lambda t: -logits[VOCAB[t]].item())
        preds = get_top_k_predictions(stub_pipeline, "[MASK] is a nurse", k=3)
        assert list(preds) == ranked[:3]

    def test_context_changes_the_top_prediction(self, stub_pipeline):
        nurse = get_top_k_predictions(stub_pipeline, "[MASK] nurse", k=1)
        doctor = get_top_k_predictions(stub_pipeline, "[MASK] doctor", k=1)
        assert list(nurse) == ["she"]
        assert list(doctor) == ["he"]

    def test_unwraps_a_nested_batch(self, stub_pipeline):
        """Pipelines return ``[[pred, ...]]`` for a batch of one; both shapes work."""

        class Nesting:
            tokenizer = stub_pipeline.tokenizer

            def __call__(self, sentence, top_k=3):
                return [stub_pipeline(sentence, top_k=top_k)]

        assert get_top_k_predictions(Nesting(), "[MASK] nurse", k=2) == (
            get_top_k_predictions(stub_pipeline, "[MASK] nurse", k=2)
        )
