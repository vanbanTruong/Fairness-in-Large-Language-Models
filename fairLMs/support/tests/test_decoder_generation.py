"""Tests for the decoder metrics driven by a local model: SLL, DNP, DRD, CA.

Two of these read a next-token distribution (SLL, DNP) and two count words in
generated text (DRD, CA). The distributions come from the stub's fixed tables;
the text comes from :class:`tests.stubs.ScriptedCausalLM`, which replays a list
of continuations instead of sampling. Both halves are therefore exactly
predictable, which matters because every one of these four metrics normalises its
counts — and a normalisation is precisely the kind of thing that can be wrong by
a factor while still landing in a plausible range.
"""

import math

import numpy as np
import pytest
import torch

from fairLMs.definitions import (
    ConceptSpec,
    CooccurrenceAssociation,
    DemographicNextTokenProportion,
    DemographicPrompts,
    DemographicRepresentationDivergence,
    OccupationTriples,
    StereotypicalLogLikelihood,
)
from .stubs import VOCAB, ScriptedCausalLM, StubTokenizer, expected_log_softmax


def _next_token_log_prob(prompt_content, word):
    """The stub's ``log P(word)`` at the final position of a prompt.

    The last position is ``[SEP]``, which contributes nothing itself, so the
    distribution there is conditioned on the prompt's content words alone.
    """
    return expected_log_softmax(prompt_content)[VOCAB[word]].item()


@pytest.fixture
def scripted_bundle(scripted_decoder):
    return (StubTokenizer(), scripted_decoder)


def _bundle(*continuations):
    return (StubTokenizer(), ScriptedCausalLM(list(continuations)))


# ---------------------------------------------------------------------------
# SLL
# ---------------------------------------------------------------------------
class TestStereotypicalLogLikelihood:
    def test_gap_matches_the_hand_computed_difference(self, scripted_bundle):
        """The gap is ``log P(stereotype word) - log P(counter word)``.

        The three template variants (neutral / competent / incompetent) coincide
        under this stub, because the two adjectives are outside its five-word
        vocabulary and so contribute nothing to the distribution. The gap itself
        is still the quantity under test.
        """
        expected = _next_token_log_prob(["nurse"], "she") - _next_token_log_prob(
            ["nurse"], "he"
        )
        result = StereotypicalLogLikelihood().compute(
            scripted_bundle, OccupationTriples([("nurse", "she", "he")])
        )
        for variant in ("NV", "CV", "IV"):
            assert result.details[variant] == pytest.approx(expected, abs=1e-6)

    def test_sign_follows_the_stereotypical_association(self, scripted_bundle):
        """`nurse` favours `she`; `doctor` favours `he`. The gaps must disagree."""
        nurse = StereotypicalLogLikelihood().compute(
            scripted_bundle, OccupationTriples([("nurse", "she", "he")])
        )
        doctor = StereotypicalLogLikelihood().compute(
            scripted_bundle, OccupationTriples([("doctor", "she", "he")])
        )
        assert nurse.details["NV"] > 0 > doctor.details["NV"]

    def test_opposing_occupations_cancel_in_the_per_variant_mean(self, scripted_bundle):
        """Each variant averages over occupations *before* the absolute value.

        ``nurse`` and ``doctor`` pull in opposite directions, so the per-variant
        means nearly cancel and the headline score is small. A metric that took
        the absolute gap per occupation and then averaged would report a large
        number here — the same inputs, a different claim.
        """
        gaps = [
            _next_token_log_prob([occ], "she") - _next_token_log_prob([occ], "he")
            for occ in ("nurse", "doctor")
        ]
        result = StereotypicalLogLikelihood().compute(
            scripted_bundle,
            OccupationTriples([("nurse", "she", "he"), ("doctor", "she", "he")]),
        )
        assert result.details["NV"] == pytest.approx(float(np.mean(gaps)), abs=1e-6)
        assert result.score == pytest.approx(abs(float(np.mean(gaps))), abs=1e-6)
        assert result.score < max(abs(g) for g in gaps)

    def test_score_is_the_mean_absolute_variant_gap(self, scripted_bundle):
        result = StereotypicalLogLikelihood().compute(
            scripted_bundle, OccupationTriples([("nurse", "she", "he")])
        )
        variants = [result.details[v] for v in ("NV", "CV", "IV")]
        assert result.score == pytest.approx(float(np.mean([abs(v) for v in variants])))
        assert result.details["n_occupations"] == 1

    def test_triples_must_have_three_fields(self):
        with pytest.raises(ValueError, match="must be a 3-tuple"):
            OccupationTriples([("nurse", "she")])

    def test_missing_data_is_reported(self, scripted_bundle):
        with pytest.raises(ValueError, match="requires occupation triples"):
            StereotypicalLogLikelihood().compute(scripted_bundle, None)

    def test_legacy_occupation_pairs_keyword_warns(self, scripted_bundle):
        with pytest.warns(DeprecationWarning, match="OccupationTriples"):
            StereotypicalLogLikelihood().compute(
                scripted_bundle, occupation_pairs=[("nurse", "she", "he")]
            )

    def test_bare_sequence_is_coerced(self, scripted_bundle):
        result = StereotypicalLogLikelihood().compute(
            scripted_bundle, [("nurse", "she", "he")]
        )
        assert result.details["n_occupations"] == 1


# ---------------------------------------------------------------------------
# DNP
# ---------------------------------------------------------------------------
def _normalised_shares(prompt_content, stereo, counter, neutral):
    """Re-derive DNP's three normalised proportions from the tables."""
    mass = lambda words: sum(
        math.exp(_next_token_log_prob(prompt_content, w)) for w in words
    )
    p_s, p_sp, p_d = mass(stereo), mass(counter), mass(neutral)
    total = p_s + p_sp + p_d
    return p_s / total, p_sp / total, p_d / total


class TestDemographicNextTokenProportion:
    def test_shares_match_the_hand_computed_normalisation(self, scripted_bundle):
        """The three groups are renormalised to sum to one, discarding the rest.

        DNP is a share *among the listed words*, not a raw probability, so the
        denominator has to be the sum of the three masses and nothing else.
        """
        stereo, counter, neutral = ["she"], ["he"], ["engineer"]
        want_s, want_sp, want_d = _normalised_shares(
            ["nurse"], stereo, counter, neutral
        )
        result = DemographicNextTokenProportion().compute(
            scripted_bundle,
            DemographicPrompts(["the nurse"], stereo, counter, neutral),
        )
        assert result.details["mean_ps"] == pytest.approx(round(want_s, 4))
        assert result.details["mean_psp"] == pytest.approx(round(want_sp, 4))
        assert result.details["mean_pd"] == pytest.approx(round(want_d, 4))

    def test_shares_sum_to_one(self, scripted_bundle):
        result = DemographicNextTokenProportion().compute(
            scripted_bundle,
            DemographicPrompts(["the nurse"], ["she"], ["he"], ["engineer"]),
        )
        total = (
            result.details["mean_ps"]
            + result.details["mean_psp"]
            + result.details["mean_pd"]
        )
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_score_is_the_neutral_share(self, scripted_bundle):
        """The reported score is ``mean_pd`` — the *neutral* mass, not the gap.

        Easy to misread, so worth pinning: a caller looking for a disparity has
        to reach into ``details``.
        """
        result = DemographicNextTokenProportion().compute(
            scripted_bundle,
            DemographicPrompts(["the nurse"], ["she"], ["he"], ["engineer"]),
        )
        assert result.score == pytest.approx(result.details["mean_pd"])

    def test_context_shifts_the_shares_in_the_expected_direction(self, scripted_bundle):
        """A `nurse` prompt must give `she` more of the mass than a `doctor` prompt."""
        data = lambda occupation: DemographicPrompts(
            [f"the {occupation}"], ["she"], ["he"], ["engineer"]
        )
        nurse = DemographicNextTokenProportion().compute(scripted_bundle, data("nurse"))
        doctor = DemographicNextTokenProportion().compute(
            scripted_bundle, data("doctor")
        )
        assert nurse.details["mean_ps"] > doctor.details["mean_ps"]

    def test_several_prompts_are_averaged(self, scripted_bundle):
        result = DemographicNextTokenProportion().compute(
            scripted_bundle,
            DemographicPrompts(
                ["the nurse", "the doctor"], ["she"], ["he"], ["engineer"]
            ),
        )
        per_prompt = [
            _normalised_shares([occ], ["she"], ["he"], ["engineer"])[0]
            for occ in ("nurse", "doctor")
        ]
        assert result.details["mean_ps"] == pytest.approx(
            round(float(np.mean(per_prompt)), 4)
        )
        assert result.details["n_prompts"] == 2

    def test_neutral_words_are_required(self, scripted_bundle):
        """Without a neutral baseline there is nothing to normalise against."""
        with pytest.raises(ValueError, match="normalises against a neutral baseline"):
            DemographicNextTokenProportion().compute(
                scripted_bundle, DemographicPrompts(["the nurse"], ["she"], ["he"])
            )

    def test_missing_word_lists_are_reported(self, scripted_bundle):
        with pytest.raises(ValueError, match="requires prompts plus demographic"):
            DemographicNextTokenProportion().compute(scripted_bundle, ["the nurse"])

    def test_legacy_keywords_warn(self, scripted_bundle):
        with pytest.warns(DeprecationWarning, match="DemographicPrompts"):
            DemographicNextTokenProportion().compute(
                scripted_bundle,
                prompts=["the nurse"],
                stereo_words=["she"],
                counter_words=["he"],
                neutral_words=["engineer"],
            )


# ---------------------------------------------------------------------------
# DRD
# ---------------------------------------------------------------------------
class TestDemographicRepresentationDivergence:
    def test_imbalanced_mentions_produce_the_expected_divergence(self):
        """Three `he` to one `she` gives shares of 0.75/0.25.

        ``drd = 0.5*|p_s - 0.5| + 0.5*|p_sp - 0.5|`` = 0.5*0.25 + 0.5*0.25 = 0.25.
        """
        result = DemographicRepresentationDivergence().compute(
            _bundle("he he he", "she"),
            DemographicPrompts(["a", "b"], ["he"], ["she"]),
        )
        assert result.details["n_stereotype_total"] == 3
        assert result.details["n_counter_total"] == 1
        assert result.score == pytest.approx(0.25)

    def test_balanced_mentions_produce_zero(self):
        result = DemographicRepresentationDivergence().compute(
            _bundle("he", "she"), DemographicPrompts(["a", "b"], ["he"], ["she"])
        )
        assert result.score == pytest.approx(0.0)

    def test_a_one_sided_corpus_saturates(self):
        result = DemographicRepresentationDivergence().compute(
            _bundle("he he"), DemographicPrompts(["a"], ["he"], ["she"])
        )
        assert result.score == pytest.approx(0.5)

    def test_no_mentions_are_undefined_not_balanced(self):
        result = DemographicRepresentationDivergence().compute(
            _bundle("doctor"), DemographicPrompts(["a"], ["he"], ["she"])
        )
        assert math.isnan(result.score)
        assert result.status == "insufficient_evidence"
        assert result.details["mention_coverage"] == 0
        assert result.to_dict()["score"] is None

    def test_counts_whole_words_only(self):
        """`she` must not be counted as containing `he`.

        A substring match would make every `she` also a `he`, quietly halving
        every measured disparity.
        """
        result = DemographicRepresentationDivergence().compute(
            _bundle("she she"), DemographicPrompts(["a"], ["he"], ["she"])
        )
        assert result.details["n_stereotype_total"] == 0
        assert result.details["n_counter_total"] == 2

    def test_rows_record_the_generated_text(self):
        result = DemographicRepresentationDivergence().compute(
            _bundle("he doctor"), DemographicPrompts(["a"], ["he"], ["she"])
        )
        assert result.details["rows"][0]["generated"] == "he doctor"
        assert result.details["rows"][0]["n_s"] == 1

    def test_max_new_tokens_is_forwarded(self):
        """A budget of one token must truncate the continuation."""
        result = DemographicRepresentationDivergence(max_new_tokens=1).compute(
            _bundle("he she"), DemographicPrompts(["a"], ["he"], ["she"])
        )
        assert result.details["rows"][0]["generated"] == "he"

    def test_prompts_can_come_from_dict_examples(self):
        with pytest.raises(ValueError, match="requires prompts plus demographic"):
            DemographicRepresentationDivergence().compute(
                _bundle("he"), [{"prompt": "a"}]
            )


# ---------------------------------------------------------------------------
# CA
# ---------------------------------------------------------------------------
CA_TEMPLATE = "People interested in {concept} are usually"


class TestCooccurrenceAssociation:
    def test_one_sided_generations_give_the_maximal_distance(self):
        """All mentions in one group means total-variation distance from uniform is 0.5.

        With two groups the reference is (0.5, 0.5); an observed (1, 0) gives
        ``0.5 * (0.5 + 0.5) = 0.5``, the maximum for k=2.
        """
        result = CooccurrenceAssociation(n_samples=2).compute(
            _bundle("he he he", "he"),
            ConceptSpec(["career"], CA_TEMPLATE, {"men": ["he"], "women": ["she"]}),
        )
        assert result.score == pytest.approx(0.5)
        assert result.details["n_valid"] == 1
        assert result.details["n_skipped"] == 0

    def test_balanced_generations_give_zero(self):
        result = CooccurrenceAssociation(n_samples=2).compute(
            _bundle("he", "she"),
            ConceptSpec(["career"], CA_TEMPLATE, {"men": ["he"], "women": ["she"]}),
        )
        assert result.score == pytest.approx(0.0)

    def test_concepts_with_no_group_terms_are_skipped_not_scored_as_zero(self):
        """No mention of any group term means no observation, not a balanced one.

        Counting it as 0 would make a model that simply never mentions anyone look
        maximally fair.
        """
        result = CooccurrenceAssociation(n_samples=2).compute(
            _bundle("doctor", "engineer"),
            ConceptSpec(["career"], CA_TEMPLATE, {"men": ["he"], "women": ["she"]}),
        )
        assert math.isnan(result.score)
        assert result.details["n_valid"] == 0
        assert result.details["n_skipped"] == 1

    def test_partially_skipped_concepts_average_over_the_rest(self):
        result = CooccurrenceAssociation(n_samples=1).compute(
            _bundle("he", "doctor"),
            ConceptSpec(
                ["career", "family"], CA_TEMPLATE, {"men": ["he"], "women": ["she"]}
            ),
        )
        assert result.details["n_valid"] == 1
        assert result.details["n_skipped"] == 1
        assert result.score == pytest.approx(0.5)

    def test_n_samples_controls_how_many_generations_are_drawn(self):
        _tok, model = bundle = _bundle("he", "she", "he", "she")
        CooccurrenceAssociation(n_samples=3).compute(
            bundle,
            ConceptSpec(["career"], CA_TEMPLATE, {"men": ["he"], "women": ["she"]}),
        )
        assert len(model.generate_calls) == 3

    def test_template_needs_a_concept_placeholder(self):
        with pytest.raises(ValueError, match=r"must contain a '\{concept\}'"):
            ConceptSpec(["career"], "People are usually", {"men": ["he"]})

    def test_group_terms_must_be_a_mapping(self):
        with pytest.raises(TypeError, match="must be a mapping"):
            ConceptSpec(["career"], CA_TEMPLATE, ["he", "she"])

    def test_wrong_container_is_reported(self):
        with pytest.raises(TypeError, match="expects a ConceptSpec"):
            CooccurrenceAssociation().compute(_bundle("he"), ["career"])

    def test_missing_data_is_reported(self):
        with pytest.raises(ValueError, match="requires concepts"):
            CooccurrenceAssociation().compute(_bundle("he"), None)

    def test_legacy_keywords_warn(self):
        with pytest.warns(DeprecationWarning, match="ConceptSpec"):
            CooccurrenceAssociation(n_samples=1).compute(
                _bundle("he"),
                concepts=["career"],
                prompt_template=CA_TEMPLATE,
                group_terms={"men": ["he"], "women": ["she"]},
            )


class TestScriptedDecoderContract:
    """Guards on the scripted stub, so these expectations rest on something firm."""

    def test_continuations_are_replayed_in_order(self):
        tokenizer, model = _bundle("he", "she doctor")
        inputs = tokenizer("a", return_tensors="pt")
        first = model.generate(**inputs, max_new_tokens=10)
        second = model.generate(**inputs, max_new_tokens=10)
        prompt_len = inputs["input_ids"].shape[1]
        assert tokenizer.decode(first[0, prompt_len:], skip_special_tokens=True) == "he"
        assert (
            tokenizer.decode(second[0, prompt_len:], skip_special_tokens=True)
            == "she doctor"
        )

    def test_the_script_cycles_when_exhausted(self):
        tokenizer, model = _bundle("he")
        inputs = tokenizer("a", return_tensors="pt")
        for _ in range(3):
            model.generate(**inputs, max_new_tokens=10)
        assert model.generate_calls == ["he", "he", "he"]

    def test_num_return_sequences_draws_that_many(self):
        tokenizer, model = _bundle("he", "she")
        inputs = tokenizer("a", return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=10, num_return_sequences=2)
        assert out.shape[0] == 2

    def test_next_token_logits_come_from_the_fixed_tables(self):
        tokenizer, model = _bundle("he")
        ids = tokenizer.encode("the nurse", return_tensors="pt")
        logits = model(ids).logits[0, -1]
        assert torch.allclose(
            torch.log_softmax(logits, dim=-1),
            expected_log_softmax(["nurse"]),
            atol=1e-6,
        )
