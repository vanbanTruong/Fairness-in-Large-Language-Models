"""Tests for the attention-head disparity metrics: NIE and GBE.

Both work by reaching *inside* a decoder — NIE captures the input to every
attention output projection and patches it with a counterfactual activation, GBE
masks the Value heads and differentiates the loss with respect to the mask. That
plumbing is the part most likely to break silently: if a hook stops firing or the
mask falls off the autograd graph, both metrics keep returning a plausible
fraction between 0 and 1.

So this file runs them against :class:`tests.stubs.TinyCausalLM`, a real
two-layer two-head module with GPT-2's names and deterministic weights, and
separately pins the pure aggregation functions with hand-made matrices.
"""

import numpy as np
import pytest
import torch

from fairLMs.definition.decoder_only.intrinsic_bias.attention_head_based_disparity.gbe.gbe import (
    compute_gbe,
    compute_gbe_mass,
    gbe_permutation_null,
    null_summary,
    random_partition,
)
from fairLMs.definition.decoder_only.intrinsic_bias.attention_head_based_disparity.nie.nie import (
    compute_nie,
    compute_nie_matrix,
)
from fairLMs.metrics import GradientBasedBiasEstimation, NaturalIndirectEffect, ProbeSet, WordSets
from fairLMs.metrics.attention_head import _derive_head_shape
from .stubs import VOCAB, StubTokenizer, TinyCausalLM

PROBES = [
    {
        "prompt": "he doctor",
        "cf_text": "she nurse",
        "stereo_token_id": VOCAB["he"],
        "anti_token_id": VOCAB["she"],
    }
]

WORD_SETS = WordSets(
    target_1=["he", "doctor"],
    target_2=["she", "nurse"],
    attribute_1=["doctor", "engineer"],
    attribute_2=["nurse", "he"],
)


@pytest.fixture
def decoder_bundle(tiny_decoder):
    return (StubTokenizer(), tiny_decoder)


# ---------------------------------------------------------------------------
# Pure aggregation
# ---------------------------------------------------------------------------
class TestComputeNie:
    def test_score_is_the_share_of_heads_above_threshold(self):
        """NIE aggregates to "how many heads mediate", using |value|, not value.

        A head with a large *negative* indirect effect mediates just as much as a
        positive one, so dropping the absolute value would halve the score.
        """
        matrix = np.array([[0.01, 0.001], [0.0, -0.5]])
        assert compute_nie(matrix, threshold=0.003) == 0.5

    def test_threshold_moves_the_score(self):
        matrix = np.array([[0.01, 0.001], [0.0, -0.5]])
        assert compute_nie(matrix, threshold=0.4) == 0.25
        assert compute_nie(matrix, threshold=1.0) == 0.0

    def test_default_threshold_is_applied(self):
        assert compute_nie(np.array([[0.004]])) == 1.0
        assert compute_nie(np.array([[0.002]])) == 0.0


class TestComputeGbe:
    def test_score_is_the_share_of_positive_gradients(self):
        matrix = np.array([[1.0, -1.0], [0.0, 2.0]])
        assert compute_gbe(matrix) == 0.5

    def test_zero_does_not_count_as_positive(self):
        assert compute_gbe(np.zeros((2, 2))) == 0.0

    def test_mass_weights_by_magnitude_not_by_count(self):
        """``compute_gbe_mass`` answers a different question from ``compute_gbe``.

        Two heads out of four are positive, so the *count* is 0.5, but they carry
        three quarters of the total absolute gradient.
        """
        matrix = np.array([[1.0, -1.0], [0.0, 2.0]])
        assert compute_gbe(matrix) == 0.5
        assert compute_gbe_mass(matrix) == pytest.approx(0.75)

    def test_mass_of_an_all_zero_matrix_is_undefined(self):
        assert np.isnan(compute_gbe_mass(np.zeros((2, 2))))

    def test_mass_of_a_non_finite_matrix_is_undefined(self):
        assert np.isnan(compute_gbe_mass(np.array([[np.inf, 1.0]])))


class TestNullCalibration:
    def test_partition_preserves_the_sizes_and_the_pool(self):
        rng = np.random.default_rng(0)
        first, second = random_partition(["a", "b"], ["c", "d", "e"], rng)
        assert (len(first), len(second)) == (2, 3)
        assert sorted(first + second) == ["a", "b", "c", "d", "e"]

    def test_summary_counts_deviations_in_either_direction(self):
        """The null is two-sided around ``chance``, so 0.1 and 0.9 are equally extreme."""
        summary = null_summary(0.9, np.array([0.4, 0.5, 0.6]), chance=0.5)
        assert summary["p_value"] == pytest.approx(0.25)
        assert summary["null_mean"] == pytest.approx(0.5)
        assert summary["significant"] is False

    def test_summary_flags_significance_when_the_null_never_reaches_the_observation(self):
        """A null pinned at chance cannot match an observation far from it.

        Note the null is scored by *distance from chance*, so an all-zero null
        would be as extreme as an all-one one; sitting it exactly on chance is
        what makes the observation stand out.
        """
        summary = null_summary(1.0, np.full(100, 0.5), chance=0.5)
        assert summary["p_value"] < 0.05
        assert summary["significant"] is True

    def test_summary_handles_an_empty_null(self):
        summary = null_summary(0.9, np.array([]), chance=0.5)
        assert np.isnan(summary["p_value"])
        assert summary["significant"] is False

    def test_summary_handles_a_non_finite_observation(self):
        summary = null_summary(float("nan"), np.array([0.4, 0.6]))
        assert np.isnan(summary["null_mean"])

    def test_non_finite_nulls_are_dropped(self):
        summary = null_summary(0.9, np.array([0.5, np.nan, np.inf]), chance=0.5)
        assert summary["null_mean"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# NIE against a real (tiny) decoder
# ---------------------------------------------------------------------------
class TestNaturalIndirectEffect:
    def test_matrix_has_one_entry_per_head(self, decoder_bundle, tiny_decoder):
        result = NaturalIndirectEffect().compute(decoder_bundle, ProbeSet(PROBES))
        assert result.details["shape"] == (
            tiny_decoder.config.n_layer,
            tiny_decoder.config.n_head,
        )
        assert result.details["matrix_computed"] is True

    def test_patching_actually_changes_the_prediction(self, decoder_bundle):
        """A non-zero matrix is the only evidence that the hooks fired.

        If the forward pre-hook stopped mutating the activation, every head would
        record an indirect effect of exactly zero and NIE would report 0.0 — a
        legitimate-looking "no mediation" result rather than a failure.
        """
        result = NaturalIndirectEffect().compute(decoder_bundle, ProbeSet(PROBES))
        assert np.any(result.details["nie"] != 0.0)

    def test_shape_is_derived_from_the_model_config(self, decoder_bundle):
        """``n_layers``/``n_heads``/``head_dim`` used to be required arguments."""
        result = NaturalIndirectEffect().compute(decoder_bundle, ProbeSet(PROBES))
        assert result.details["shape"] == (2, 2)

    def test_explicit_shape_overrides_the_config(self, decoder_bundle):
        result = NaturalIndirectEffect(n_layers=1, n_heads=2, head_dim=4).compute(
            decoder_bundle, ProbeSet(PROBES)
        )
        assert result.details["shape"] == (1, 2)

    def test_legacy_uppercase_shape_keywords_are_accepted(self, decoder_bundle):
        result = NaturalIndirectEffect().compute(
            decoder_bundle, ProbeSet(PROBES), N_LAYERS=1, N_HEADS=1, HEAD_DIM=4
        )
        assert result.details["shape"] == (1, 1)

    def test_threshold_is_forwarded_and_reported(self, decoder_bundle):
        strict = NaturalIndirectEffect(threshold=10.0).compute(
            decoder_bundle, ProbeSet(PROBES)
        )
        assert strict.details["threshold"] == 10.0
        assert strict.score == 0.0

    def test_precomputed_matrix_skips_the_model(self):
        """``nie=`` lets a caller score a matrix they already have.

        No model is touched, which is also how an expensive sweep gets reused
        across thresholds.
        """
        result = NaturalIndirectEffect(nie=[[0.01, 0.0]]).compute(None, None)
        assert result.score == 0.5
        assert result.details["matrix_computed"] is False
        assert result.details["shape"] == (1, 2)

    def test_is_deterministic(self, decoder_bundle):
        first = NaturalIndirectEffect().compute(decoder_bundle, ProbeSet(PROBES))
        second = NaturalIndirectEffect().compute(decoder_bundle, ProbeSet(PROBES))
        assert np.allclose(first.details["nie"], second.details["nie"])

    def test_no_probes_gives_a_zero_matrix(self, decoder_bundle, tiny_decoder):
        matrix = compute_nie_matrix(
            tiny_decoder, StubTokenizer(), torch.device("cpu"), [], 2, 2, 4
        )
        assert matrix.shape == (2, 2)
        assert not matrix.any()

    def test_blank_probes_are_skipped_not_counted(self, tiny_decoder):
        """A probe with an empty prompt cannot be patched, so it must not dilute.

        Averaging over ``max(1, n_valid)`` means a skipped probe would otherwise
        pull every head's effect towards zero.
        """
        blank = [
            {
                "prompt": "   ",
                "cf_text": "she nurse",
                "stereo_token_id": VOCAB["he"],
                "anti_token_id": VOCAB["she"],
            }
        ]
        matrix = compute_nie_matrix(
            tiny_decoder, StubTokenizer(), torch.device("cpu"), blank, 2, 2, 4
        )
        assert not matrix.any()

    def test_probes_need_the_documented_keys(self):
        with pytest.raises(ValueError, match="is missing cf_text"):
            ProbeSet([{"prompt": "he doctor", "stereo_token_id": 5, "anti_token_id": 6}])

    def test_probe_must_be_a_mapping(self):
        with pytest.raises(TypeError, match="must be a mapping"):
            ProbeSet(["he doctor"])

    def test_missing_data_is_reported(self, decoder_bundle):
        with pytest.raises(ValueError, match="requires probes"):
            NaturalIndirectEffect().compute(decoder_bundle, None)

    def test_legacy_probes_keyword_warns(self, decoder_bundle):
        with pytest.warns(DeprecationWarning, match="ProbeSet"):
            NaturalIndirectEffect().compute(decoder_bundle, probes=PROBES)

    def test_unknown_kwarg_raises(self, decoder_bundle):
        with pytest.raises(TypeError, match="threshhold"):
            NaturalIndirectEffect().compute(
                decoder_bundle, ProbeSet(PROBES), threshhold=0.1
            )


class TestDeriveHeadShape:
    """The shape derivation has to cope with two different naming conventions."""

    def test_reads_gpt2_style_names(self):
        config = type("C", (), {"n_layer": 3, "n_head": 4, "n_embd": 32})()
        assert _derive_head_shape(config, None, None, None) == (3, 4, 8)

    def test_reads_bert_style_names(self):
        config = type(
            "C",
            (),
            {"num_hidden_layers": 6, "num_attention_heads": 2, "hidden_size": 16},
        )()
        assert _derive_head_shape(config, None, None, None) == (6, 2, 8)

    def test_explicit_values_win(self):
        config = type("C", (), {"n_layer": 3, "n_head": 4, "n_embd": 32})()
        assert _derive_head_shape(config, 1, 2, 4) == (1, 2, 4)

    def test_an_uninformative_config_is_reported(self):
        with pytest.raises(ValueError, match="pass them to the constructor explicitly"):
            _derive_head_shape(type("C", (), {})(), None, None, None)


# ---------------------------------------------------------------------------
# GBE against a real (tiny) decoder
# ---------------------------------------------------------------------------
class TestGradientBasedBiasEstimation:
    def test_matrix_has_one_gradient_per_head(self, decoder_bundle):
        result = GradientBasedBiasEstimation().compute(decoder_bundle, WORD_SETS)
        assert result.details["shape"] == (2, 2)
        assert result.details["matrix_computed"] is True

    def test_the_value_mask_reaches_the_loss(self, decoder_bundle):
        """A non-zero gradient is the evidence that the mask is on the graph.

        ``compute_gbe_matrix`` raises if the hooks never fire or if ``masks.grad``
        comes back ``None``, but a mask that is *connected yet ineffective* would
        produce an all-zero matrix and a GBE of 0.0 with no complaint.
        """
        result = GradientBasedBiasEstimation().compute(decoder_bundle, WORD_SETS)
        assert np.any(result.details["gbe_matrix"] != 0.0)
        assert np.all(np.isfinite(result.details["gbe_matrix"]))

    def test_score_is_the_positive_fraction_of_that_matrix(self, decoder_bundle):
        result = GradientBasedBiasEstimation().compute(decoder_bundle, WORD_SETS)
        assert result.score == pytest.approx(compute_gbe(result.details["gbe_matrix"]))

    def test_loss_scale_scales_the_gradient_linearly(self, decoder_bundle):
        """The loss is ``|d| * loss_scale``, so the gradient must scale with it.

        ``loss_scale`` used to be accepted and ignored; a fixed multiple is the
        cheapest way to show it is not. The tolerance is relative because the
        backward pass runs in float32 through a softmax and two layer norms.
        """
        base = GradientBasedBiasEstimation(loss_scale=1.0).compute(
            decoder_bundle, WORD_SETS
        )
        scaled = GradientBasedBiasEstimation(loss_scale=3.0).compute(
            decoder_bundle, WORD_SETS
        )
        assert np.allclose(
            scaled.details["gbe_matrix"], 3.0 * base.details["gbe_matrix"], rtol=1e-3
        )

    def test_restores_the_original_requires_grad_flags(self, tiny_decoder):
        """The metric switches every parameter to ``requires_grad=False`` and back.

        Leaving them off would silently break any training the caller does next.
        """
        before = {n: p.requires_grad for n, p in tiny_decoder.named_parameters()}
        GradientBasedBiasEstimation().compute((StubTokenizer(), tiny_decoder), WORD_SETS)
        after = {n: p.requires_grad for n, p in tiny_decoder.named_parameters()}
        assert before == after
        assert all(before.values())

    def test_removes_its_hooks(self, tiny_decoder):
        """Left-behind hooks would corrupt every later forward pass."""
        GradientBasedBiasEstimation().compute((StubTokenizer(), tiny_decoder), WORD_SETS)
        for block in tiny_decoder.transformer.h:
            assert not block.attn.c_attn._forward_hooks
            assert not block.attn.c_proj._forward_pre_hooks

    def test_is_deterministic(self, decoder_bundle):
        first = GradientBasedBiasEstimation().compute(decoder_bundle, WORD_SETS)
        second = GradientBasedBiasEstimation().compute(decoder_bundle, WORD_SETS)
        assert np.allclose(first.details["gbe_matrix"], second.details["gbe_matrix"])

    def test_precomputed_matrix_skips_the_model(self):
        result = GradientBasedBiasEstimation(gbe_matrix=[[1.0, -1.0]]).compute(None, None)
        assert result.score == 0.5
        assert result.details["matrix_computed"] is False

    def test_missing_data_is_reported(self, decoder_bundle):
        with pytest.raises(ValueError, match="requires four word sets"):
            GradientBasedBiasEstimation().compute(decoder_bundle, None)

    def test_wrong_container_is_reported(self, decoder_bundle):
        with pytest.raises(TypeError, match="expects a WordSets"):
            GradientBasedBiasEstimation().compute(decoder_bundle, ["he", "she"])

    def test_legacy_xyab_keywords_warn(self, decoder_bundle):
        with pytest.warns(DeprecationWarning, match="WordSets"):
            GradientBasedBiasEstimation().compute(
                decoder_bundle,
                X=["he", "doctor"],
                Y=["she", "nurse"],
                A=["doctor", "engineer"],
                B=["nurse", "he"],
            )

    def test_permutation_null_returns_one_value_per_draw(self, tiny_decoder):
        """The null answers "would a random regrouping look this biased?".

        Two draws is enough to pin the shape and that both statistics come back
        finite; the calibration itself is ``null_summary``'s job.
        """
        proportions, masses = gbe_permutation_null(
            tiny_decoder,
            StubTokenizer(),
            torch.device("cpu"),
            ["he", "doctor"],
            ["she", "nurse"],
            ["doctor", "engineer"],
            ["nurse", "he"],
            n_perm=2,
            seed=0,
            progress=False,
        )
        assert proportions.shape == masses.shape == (2,)
        assert np.all(np.isfinite(proportions))

    @pytest.mark.parametrize("permute", ["targets", "attributes", "both"])
    def test_every_permute_mode_runs(self, tiny_decoder, permute):
        proportions, _masses = gbe_permutation_null(
            tiny_decoder,
            StubTokenizer(),
            torch.device("cpu"),
            ["he", "doctor"],
            ["she", "nurse"],
            ["doctor", "engineer"],
            ["nurse", "he"],
            n_perm=1,
            seed=0,
            permute=permute,
            progress=False,
        )
        assert proportions.shape == (1,)

    def test_unknown_permute_mode_is_reported(self, tiny_decoder):
        with pytest.raises(ValueError, match="unknown permute mode"):
            gbe_permutation_null(
                tiny_decoder,
                StubTokenizer(),
                torch.device("cpu"),
                ["he"],
                ["she"],
                ["doctor"],
                ["nurse"],
                n_perm=1,
                permute="sideways",
                progress=False,
            )
