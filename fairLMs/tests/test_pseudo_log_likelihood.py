"""Tests for the pseudo-log-likelihood family: PLL, CPS, AUL, AULA, CAT.

All five ask a masked LM which of two sentences it prefers, and all five report
that preference as a percentage. That makes them easy to get subtly wrong and
hard to notice: a flipped comparison, a sum where a mean belongs, or a
per-category denominator taken from the wrong counter all still yield a number
between 0 and 100.

The stub masked LM makes the *right* number knowable. ``_pll`` and ``_aul`` below
re-derive each sentence's score straight from the stub's tables, so the expected
win rate is arithmetic rather than a recording of previous behaviour.
"""

import pytest

from fairLMs.definitions import (
    AllUnmaskedLikelihoodAttentionScore,
    AllUnmaskedLikelihoodScore,
    ContextAssociationTestScore,
    CrowSPairsScore,
    PseudoLogLikelihoodScore,
    SentenceTriples,
)
from .stubs import (
    VOCAB,
    StubMaskedLM,
    attention_weights,
    expected_log_softmax,
    expected_logits,
)


# ---------------------------------------------------------------------------
# Independent re-derivation of the stub's sentence scores
# ---------------------------------------------------------------------------
def _token_log_probs(sentence):
    """``log P(token | every other token)`` for each word, from the tables.

    The stub scores a position against the bag of *other* tokens in the row,
    which is exactly what masking one position at a time produces. So the same
    list serves both the masked scorers (PLL, CPS, CAT) and the unmasked ones
    (AUL, AULA).
    """
    tokens = sentence.split()
    return [
        expected_log_softmax(tokens[:i] + tokens[i + 1 :])[VOCAB[gold]].item()
        for i, gold in enumerate(tokens)
    ]


def _pll(sentence):
    """PLL and CAT score a sentence with the **sum** over its tokens."""
    return sum(_token_log_probs(sentence))


def _aul(sentence):
    """AUL uses the **mean**, which is what makes it length-independent."""
    values = _token_log_probs(sentence)
    return sum(values) / len(values)


def _aula(sentence):
    """AULA weights each token by mean attention before averaging."""
    values = _token_log_probs(sentence)
    weights = attention_weights(len(values) + 2)[0][1:-1].tolist()
    scaled = [v * w for v, w in zip(values, weights)]
    return sum(scaled) / len(scaled)


def _ranks(sentence):
    tokens = sentence.split()
    out = []
    for i, gold in enumerate(tokens):
        logits = expected_logits(tokens[:i] + tokens[i + 1 :])
        out.append(int((logits > logits[VOCAB[gold]]).sum()) + 1)
    return out


def _win_rate(pairs, score_fn):
    wins = sum(
        score_fn(p["stereotype"]) > score_fn(p["anti_stereotype"]) for p in pairs
    )
    return wins / len(pairs) * 100


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
#: One pair the stub prefers stereotypically and one it prefers the other way,
#: so a metric that inverted its comparison would report 50 as 50 — hence the
#: per-category breakdown, which pins down *which* pair went which way.
PAIRS = [
    {
        "stereotype": "he doctor",
        "anti_stereotype": "she doctor",
        "bias_type": "gender",
    },
    {
        "stereotype": "he nurse",
        "anti_stereotype": "she nurse",
        "bias_type": "profession",
    },
]


@pytest.fixture
def pairs():
    return [dict(p) for p in PAIRS]


class TestPseudoLogLikelihoodScore:
    def test_score_is_the_stereotypical_win_rate(self, stub_bundle, pairs):
        result = PseudoLogLikelihoodScore().compute(stub_bundle, pairs)
        assert result.score == pytest.approx(_win_rate(pairs, _pll))
        assert result.score == pytest.approx(50.0)

    def test_per_category_breakdown_identifies_which_pair_won(self, stub_bundle, pairs):
        """A 50 % total can hide an inverted comparison; this cannot.

        The stub associates `doctor` with `he` and `nurse` with `she`, so the
        gender pair must score 100 and the profession pair 0. Swap the sides and
        both flip while the headline score stays put.
        """
        result = PseudoLogLikelihoodScore().compute(stub_bundle, pairs)
        assert result.by_category == {"gender": 100.0, "profession": 0.0}

    def test_accuracy_is_the_share_of_rank_one_tokens(self, stub_bundle, pairs):
        ranks = [
            r
            for p in pairs
            for key in ("stereotype", "anti_stereotype")
            for r in _ranks(p[key])
        ]
        expected = sum(r == 1 for r in ranks) / len(ranks) * 100
        result = PseudoLogLikelihoodScore().compute(stub_bundle, pairs)
        assert result.details["accuracy"] == pytest.approx(expected)
        assert result.details["n_pairs"] == 2

    def test_all_stereotypical_pairs_score_one_hundred(self, stub_bundle):
        favouring_stereotype = [
            {
                "stereotype": "he doctor",
                "anti_stereotype": "she doctor",
                "bias_type": "g",
            },
            {
                "stereotype": "she nurse",
                "anti_stereotype": "he nurse",
                "bias_type": "g",
            },
        ]
        result = PseudoLogLikelihoodScore().compute(stub_bundle, favouring_stereotype)
        assert result.score == pytest.approx(100.0)

    def test_reversing_every_pair_reverses_the_score(self, stub_bundle, pairs):
        flipped = [
            {
                "stereotype": p["anti_stereotype"],
                "anti_stereotype": p["stereotype"],
                "bias_type": p["bias_type"],
            }
            for p in pairs
        ]
        result = PseudoLogLikelihoodScore().compute(stub_bundle, flipped)
        assert result.by_category == {"gender": 0.0, "profession": 100.0}

    def test_empty_sentences_are_refused(self, stub_bundle):
        with pytest.raises(ValueError, match="non-empty"):
            PseudoLogLikelihoodScore().compute(
                stub_bundle,
                [{"stereotype": "", "anti_stereotype": "", "bias_type": "g"}],
            )

    def test_one_word_sentences_are_scored_against_an_empty_context(self, stub_bundle):
        """A single word still has one maskable position, scored against ``BASE`` alone.

        ``he`` outranks ``she`` unconditionally, so exactly one of the two
        sentences puts its gold token first.
        """
        result = PseudoLogLikelihoodScore().compute(
            stub_bundle,
            [{"stereotype": "he", "anti_stereotype": "she", "bias_type": "g"}],
        )
        assert result.details["accuracy"] == pytest.approx(50.0)
        assert result.score == pytest.approx(100.0)


class TestCrowSPairsScore:
    """CPS scores only the tokens the two sentences have in common."""

    def test_score_reflects_the_shared_span_only(self, stub_bundle, pairs):
        """Each pair differs in its first word, so only the profession is scored.

        The comparison therefore reduces to ``log P(doctor | he)`` against
        ``log P(doctor | she)`` — a single number per side.
        """
        result = CrowSPairsScore().compute(stub_bundle, pairs)
        expected = (
            (
                _shared_span_wins("doctor", "he", "she")
                + _shared_span_wins("nurse", "he", "she")
            )
            / 2
            * 100
        )
        assert result.score == pytest.approx(expected)
        assert result.by_category == {"gender": 100.0, "profession": 0.0}

    def test_ignores_the_differing_token(self, stub_bundle):
        """Changing only the *unshared* word must not move the score.

        This is the whole point of CPS over PLL: the modified token is excluded,
        so its own likelihood cannot dominate the comparison.
        """
        base = CrowSPairsScore().compute(stub_bundle, [dict(PAIRS[0])])
        other = CrowSPairsScore().compute(
            stub_bundle,
            [
                {
                    "stereotype": "engineer doctor",
                    "anti_stereotype": "she doctor",
                    "bias_type": "gender",
                }
            ],
        )
        assert base.score == other.score

    def test_identical_sentences_leave_nothing_unshared(self, stub_bundle):
        result = CrowSPairsScore().compute(
            stub_bundle,
            [
                {
                    "stereotype": "he doctor",
                    "anti_stereotype": "he doctor",
                    "bias_type": "g",
                }
            ],
        )
        assert result.score == 0.0


def _shared_span_wins(gold, stereo_context, anti_context):
    stereo = expected_log_softmax([stereo_context])[VOCAB[gold]].item()
    anti = expected_log_softmax([anti_context])[VOCAB[gold]].item()
    return int(stereo > anti)


class TestAllUnmaskedLikelihoodScore:
    def test_score_matches_the_mean_based_win_rate(self, stub_bundle, pairs):
        result = AllUnmaskedLikelihoodScore().compute(stub_bundle, pairs)
        assert result.score == pytest.approx(_win_rate(pairs, _aul))

    def test_mean_and_sum_disagree_on_a_length_imbalanced_pair(self, stub_bundle):
        """AUL normalises for length; PLL does not. Here that changes the winner.

        ``she nurse engineer`` has the *lower* total log-probability than
        ``he nurse`` but the *higher* per-token average, so PLL calls the pair
        anti-stereotypical and AUL calls it stereotypical. A metric that summed
        where it should average — or vice versa — would fail exactly one of these
        two assertions.
        """
        pair = [
            {
                "stereotype": "she nurse engineer",
                "anti_stereotype": "he nurse",
                "bias_type": "gender",
            }
        ]
        assert _pll(pair[0]["stereotype"]) < _pll(pair[0]["anti_stereotype"])
        assert _aul(pair[0]["stereotype"]) > _aul(pair[0]["anti_stereotype"])

        assert PseudoLogLikelihoodScore().compute(stub_bundle, pair).score == 0.0
        assert AllUnmaskedLikelihoodScore().compute(stub_bundle, pair).score == 100.0

    def test_use_attention_is_forwarded_from_the_constructor(self, stub_bundle, pairs):
        """``use_attention=True`` turns AUL into AULA, so it must reach the scorer."""
        _tok, model = stub_bundle
        AllUnmaskedLikelihoodScore(use_attention=True).compute(stub_bundle, pairs)
        assert model.attention_was_requested()

    def test_use_attention_defaults_to_off(self, stub_bundle, pairs):
        _tok, model = stub_bundle
        AllUnmaskedLikelihoodScore().compute(stub_bundle, pairs)
        assert not model.attention_was_requested()

    def test_get_params_round_trips(self):
        metric = AllUnmaskedLikelihoodScore(use_attention=True)
        assert metric.get_params() == {"use_attention": True}
        assert type(metric)(**metric.get_params()).use_attention is True


class TestAllUnmaskedLikelihoodAttentionScore:
    def test_score_matches_the_attention_weighted_win_rate(self, stub_bundle, pairs):
        result = AllUnmaskedLikelihoodAttentionScore().compute(stub_bundle, pairs)
        assert result.score == pytest.approx(_win_rate(pairs, _aula))

    def test_attention_is_on_by_default(self, stub_bundle, pairs):
        """AULA *is* the attention-weighted variant; it cannot default to off."""
        _tok, model = stub_bundle
        AllUnmaskedLikelihoodAttentionScore().compute(stub_bundle, pairs)
        assert model.attention_was_requested()

    def test_weighting_changes_the_underlying_scores(self):
        """Guard on the fixture: unweighted and weighted scores must differ.

        If they happened to coincide, the previous test would pass even with the
        attention weighting removed entirely.
        """
        assert _aul("he doctor") != _aula("he doctor")


class TestContextAssociationTestScore:
    """StereoSet's SS / LMS / iCAT over stereotype-anti-unrelated triples."""

    def test_balanced_triples_give_a_perfect_icat(self, stub_bundle):
        """iCAT rewards a model that is fluent *and* unbiased.

        One triple the stub prefers stereotypically and one it does not gives
        ``ss = 50``; both prefer a meaningful sentence over the unrelated one, so
        ``lms = 100``. Then ``icat = 100 * min(50, 50) / 50 = 100``.
        """
        triples = SentenceTriples(
            stereotype=["he doctor", "he nurse"],
            anti_stereotype=["she doctor", "she nurse"],
            unrelated=["engineer engineer", "engineer engineer"],
        )
        result = ContextAssociationTestScore().compute(stub_bundle, triples)
        assert result.details["ss"] == pytest.approx(50.0)
        assert result.details["lms"] == pytest.approx(100.0)
        assert result.score == pytest.approx(100.0)

    def test_uniformly_stereotypical_triples_collapse_icat_to_zero(self, stub_bundle):
        """``min(ss, 100 - ss)`` is what makes iCAT punish bias in either direction.

        Both triples here go the stub's stereotypical way, so ``ss = 100`` and
        iCAT must be 0 despite ``lms`` still being a perfect 100. Drop the
        ``min`` and this returns 200.
        """
        triples = SentenceTriples(
            stereotype=["he doctor", "she nurse"],
            anti_stereotype=["she doctor", "he nurse"],
            unrelated=["engineer engineer", "engineer engineer"],
        )
        result = ContextAssociationTestScore().compute(stub_bundle, triples)
        assert result.details["ss"] == pytest.approx(100.0)
        assert result.details["lms"] == pytest.approx(100.0)
        assert result.score == pytest.approx(0.0)

    def test_lms_falls_when_the_unrelated_sentence_wins(self, stub_bundle):
        """`he doctor` beats `engineer engineer`, so reversing the roles halves lms."""
        triples = SentenceTriples(
            stereotype=["engineer engineer", "he doctor"],
            anti_stereotype=["engineer engineer", "she doctor"],
            unrelated=["he doctor", "engineer engineer"],
        )
        result = ContextAssociationTestScore().compute(stub_bundle, triples)
        assert result.details["lms"] == pytest.approx(50.0)

    def test_rows_record_every_log_probability(self, stub_bundle):
        triples = SentenceTriples(["he doctor"], ["she doctor"], ["engineer engineer"])
        rows = (
            ContextAssociationTestScore().compute(stub_bundle, triples).details["rows"]
        )
        assert rows[0]["lp_stereo"] == pytest.approx(_pll("he doctor"), abs=1e-5)
        assert rows[0]["lp_anti"] == pytest.approx(_pll("she doctor"), abs=1e-5)
        assert rows[0]["lp_related"] == pytest.approx(
            _pll("engineer engineer"), abs=1e-5
        )

    def test_accepts_a_sequence_of_dicts(self, stub_bundle):
        """Datasets yield dicts; the wrapper coerces them to ``SentenceTriples``."""
        examples = [
            {
                "stereotype": "he doctor",
                "anti_stereotype": "she doctor",
                "unrelated": "engineer engineer",
            }
        ]
        result = ContextAssociationTestScore().compute(stub_bundle, examples)
        assert result.details["n"] == 1
        assert result.details["ss"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Argument handling shared by all five
# ---------------------------------------------------------------------------
class TestPairMetricArgumentHandling:
    @pytest.mark.parametrize(
        "metric",
        [PseudoLogLikelihoodScore, CrowSPairsScore, AllUnmaskedLikelihoodScore],
    )
    def test_missing_keys_are_named(self, stub_bundle, metric):
        with pytest.raises(ValueError, match="missing key\\(s\\) anti_stereotype"):
            metric().compute(stub_bundle, [{"stereotype": "he doctor"}])

    @pytest.mark.parametrize(
        "metric",
        [PseudoLogLikelihoodScore, CrowSPairsScore, AllUnmaskedLikelihoodScore],
    )
    def test_unknown_kwarg_raises(self, stub_bundle, metric, pairs):
        with pytest.raises(TypeError, match="use_attentino"):
            metric().compute(stub_bundle, pairs, use_attentino=True)

    def test_empty_data_is_reported(self, stub_bundle):
        with pytest.raises(ValueError, match="is empty"):
            PseudoLogLikelihoodScore().compute(stub_bundle, [])

    def test_missing_data_is_reported(self, stub_bundle):
        with pytest.raises(ValueError, match="requires"):
            PseudoLogLikelihoodScore().compute(stub_bundle, None)

    def test_legacy_sentence_pairs_keyword_warns(self, stub_bundle, pairs):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            result = PseudoLogLikelihoodScore().compute(
                stub_bundle, sentence_pairs=pairs
            )
        assert result.score == pytest.approx(50.0)

    def test_dataset_like_object_is_loaded(self, stub_bundle, pairs):
        class Dataset:
            def load(self):
                return pairs

        result = PseudoLogLikelihoodScore().compute(stub_bundle, Dataset())
        assert result.details["n_pairs"] == 2

    def test_model_is_required(self, pairs):
        with pytest.raises(TypeError, match="model is required"):
            PseudoLogLikelihoodScore().compute(None, pairs)

    def test_accepts_tokenizer_keyword_beside_a_bare_model(self, stub_tokenizer, pairs):
        result = PseudoLogLikelihoodScore().compute(
            StubMaskedLM(), pairs, tokenizer=stub_tokenizer
        )
        assert result.score == pytest.approx(50.0)

    def test_context_association_requires_triples(self, stub_bundle):
        with pytest.raises(ValueError, match="requires sentence triples"):
            ContextAssociationTestScore().compute(stub_bundle, None)

    def test_metrics_declare_their_architecture(self):
        for metric in (
            PseudoLogLikelihoodScore,
            CrowSPairsScore,
            AllUnmaskedLikelihoodScore,
            AllUnmaskedLikelihoodAttentionScore,
            ContextAssociationTestScore,
        ):
            assert metric.architectures == ("encoder_only",)
            assert metric.bias_type == "intrinsic"
