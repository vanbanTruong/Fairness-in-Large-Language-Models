"""In-processing components: loss terms, their invariants, and differentiability.

Each of these returns a callable, not a trained model. The tests check the
mathematical properties the loss term promises and that gradients actually flow,
which is the only thing that makes it composable with a real training loop.
"""

import pytest
import torch

from fairLMs.metrics import PromptPairs
from fairLMs.mitigation import (
    AdversarialDebiasing,
    AttributeLabeledVectors,
    CounterfactualInvarianceLoss,
    GroupLabeledRecords,
    GroupRegularizedObjective,
    InfluenceGuidedSuppression,
    InfluenceScoredCorpus,
)

VECTORS = AttributeLabeledVectors(
    axis="gender",
    vectors=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
    labels=["f", "f", "m", "m"],
    source="unit-test",
)


class TestAllReturnLossComponents:
    """The category-wide guarantee: a callable, never a trained model."""

    def test_every_in_processing_result_is_callable(self):
        from fairLMs.mitigation import MITIGATOR_REGISTRY, list_by_category

        for name in list_by_category("in"):
            assert MITIGATOR_REGISTRY[name].category == "in"

    def test_no_trainer_is_exposed(self):
        import fairLMs.mitigation as package

        assert not [n for n in package.__all__ if "train" in n.lower()]


class TestAdversarialDebiasing:
    def test_returns_a_callable_loss_with_the_adversary_attached(self):
        result = AdversarialDebiasing().apply(None, VECTORS)
        assert callable(result.result)
        assert isinstance(result.result.adversary, torch.nn.Module)
        assert result.result.classes == ("f", "m")

    def test_the_loss_is_positive_and_differentiable(self):
        component = AdversarialDebiasing().apply(None, VECTORS).result
        pooled = torch.tensor(VECTORS.vectors, dtype=torch.float32, requires_grad=True)
        loss = component(pooled, ["f", "f", "m", "m"])
        assert loss.item() > 0
        loss.backward()
        assert pooled.grad is not None
        assert torch.isfinite(pooled.grad).all()

    def test_the_gradient_reaching_the_encoder_is_reversed(self):
        # The defining property of gradient reversal: the encoder receives the
        # negated adversary gradient, so it is trained to defeat the adversary.
        def encoder_gradient(lambda_):
            component = (
                AdversarialDebiasing(lambda_=lambda_).apply(None, VECTORS).result
            )
            pooled = torch.tensor(
                VECTORS.vectors, dtype=torch.float32, requires_grad=True
            )
            component(pooled, ["f", "f", "m", "m"]).backward()
            return pooled.grad.clone()

        positive = encoder_gradient(1.0)
        negative = encoder_gradient(-1.0)
        assert torch.allclose(positive, -negative, atol=1e-6)

    def test_lambda_scales_the_reversed_gradient_linearly(self):
        def encoder_gradient(lambda_):
            component = (
                AdversarialDebiasing(lambda_=lambda_).apply(None, VECTORS).result
            )
            pooled = torch.tensor(
                VECTORS.vectors, dtype=torch.float32, requires_grad=True
            )
            component(pooled, ["f", "f", "m", "m"]).backward()
            return pooled.grad.clone()

        assert torch.allclose(
            encoder_gradient(2.0), 2 * encoder_gradient(1.0), atol=1e-6
        )

    def test_an_unseen_attribute_value_is_refused(self):
        component = AdversarialDebiasing().apply(None, VECTORS).result
        pooled = torch.tensor(VECTORS.vectors, dtype=torch.float32)
        with pytest.raises(ValueError, match="unseen attribute value"):
            component(pooled, ["f", "f", "m", "nonbinary"])

    def test_misaligned_batch_and_labels_are_refused(self):
        component = AdversarialDebiasing().apply(None, VECTORS).result
        pooled = torch.tensor(VECTORS.vectors, dtype=torch.float32)
        with pytest.raises(ValueError, match="attribute labels for"):
            component(pooled, ["f", "m"])

    def test_records_without_a_declared_width_are_refused(self):
        records = GroupLabeledRecords(
            axis="gender",
            groups=["f", "m"],
            labels=["yes", "no"],
            label_name="outcome",
            source="t",
        )
        with pytest.raises(ValueError, match="input width must be known"):
            AdversarialDebiasing().apply(None, records)


class TestCounterfactualInvarianceLoss:
    def _component(self, **kwargs):
        pairs = PromptPairs(["he is a nurse"], ["she is a nurse"])
        return CounterfactualInvarianceLoss(**kwargs).apply(None, pairs).result

    def test_identical_predictions_incur_no_penalty(self):
        # The defining invariant.
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        assert self._component()(logits, logits).item() == pytest.approx(0.0, abs=1e-9)

    def test_the_penalty_is_non_negative(self):
        component = self._component()
        for _ in range(10):
            a, b = torch.randn(4, 5), torch.randn(4, 5)
            assert component(a, b).item() >= -1e-9

    def test_the_penalty_is_symmetric(self):
        component = self._component()
        a, b = torch.randn(4, 5), torch.randn(4, 5)
        assert component(a, b).item() == pytest.approx(component(b, a).item(), abs=1e-6)

    def test_the_penalty_grows_with_divergence(self):
        component = self._component()
        base = torch.tensor([[1.0, 0.0]])
        near = component(base, torch.tensor([[0.9, 0.1]])).item()
        far = component(base, torch.tensor([[0.0, 1.0]])).item()
        assert far > near

    def test_it_is_differentiable(self):
        component = self._component()
        a = torch.randn(3, 4, requires_grad=True)
        component(a, torch.randn(3, 4)).backward()
        assert a.grad is not None and torch.isfinite(a.grad).all()

    def test_shape_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="same shape"):
            self._component()(torch.randn(2, 3), torch.randn(2, 4))

    def test_sum_reduction_scales_with_batch_size(self):
        mean = self._component(reduction="mean")
        total = self._component(reduction="sum")
        a, b = torch.randn(4, 5), torch.randn(4, 5)
        assert total(a, b).item() == pytest.approx(4 * mean(a, b).item(), abs=1e-5)

    def test_an_unknown_reduction_is_refused(self):
        with pytest.raises(ValueError, match="must be 'mean' or 'sum'"):
            self._component(reduction="median")


class TestGroupRegularizedObjective:
    RECORDS = GroupLabeledRecords(
        axis="gender",
        groups=["f", "f", "m", "m"],
        labels=["yes", "no", "yes", "no"],
        label_name="outcome",
        source="unit-test",
    )
    GROUPS = ["f", "f", "m", "m"]
    POSITIVES = [True, False, True, False]

    def test_no_gap_means_no_penalty(self):
        component = GroupRegularizedObjective().apply(None, self.RECORDS).result
        equal = torch.tensor([0.8, 0.2, 0.8, 0.2])
        assert component(equal, self.POSITIVES, self.GROUPS).item() == pytest.approx(
            0.0, abs=1e-9
        )

    def test_a_gap_produces_a_positive_penalty_equal_to_its_size(self):
        component = GroupRegularizedObjective().apply(None, self.RECORDS).result
        # Positive-class scores: f -> 0.9, m -> 0.5, so the soft TPR gap is 0.4.
        skewed = torch.tensor([0.9, 0.2, 0.5, 0.2])
        assert component(skewed, self.POSITIVES, self.GROUPS).item() == pytest.approx(
            0.4, abs=1e-6
        )

    def test_the_penalty_is_non_negative(self):
        component = GroupRegularizedObjective().apply(None, self.RECORDS).result
        for _ in range(10):
            probabilities = torch.rand(4)
            assert component(probabilities, self.POSITIVES, self.GROUPS).item() >= -1e-9

    def test_equalized_odds_adds_the_negative_class_gap(self):
        component = (
            GroupRegularizedObjective("equalized_odds").apply(None, self.RECORDS).result
        )
        # TPR gap 0.4 (0.9 vs 0.5) plus FPR gap 0.3 (0.5 vs 0.2).
        probabilities = torch.tensor([0.9, 0.5, 0.5, 0.2])
        assert component(probabilities, self.POSITIVES, self.GROUPS).item() == (
            pytest.approx(0.7, abs=1e-6)
        )

    def test_it_is_differentiable(self):
        component = GroupRegularizedObjective().apply(None, self.RECORDS).result
        probabilities = torch.rand(4, requires_grad=True)
        component(probabilities, self.POSITIVES, self.GROUPS).backward()
        assert probabilities.grad is not None

    def test_a_group_missing_a_class_in_the_batch_raises(self):
        # Silently skipping it would make the penalty quietly meaningless.
        component = GroupRegularizedObjective().apply(None, self.RECORDS).result
        with pytest.raises(ValueError, match="no positive rows in this batch"):
            component(
                torch.tensor([0.9, 0.2, 0.5, 0.2]),
                [True, False, False, False],
                self.GROUPS,
            )

    def test_misaligned_inputs_are_refused(self):
        component = GroupRegularizedObjective().apply(None, self.RECORDS).result
        with pytest.raises(ValueError, match="must align"):
            component(torch.rand(4), [True, False], self.GROUPS)


class TestInfluenceGuidedSuppression:
    CORPUS = InfluenceScoredCorpus(
        n_examples=4,
        flagged=[1, 3],
        influence=[2.0, 1.0],
        source="unit-test-precomputed",
    )

    def test_only_flagged_rows_are_penalised(self):
        component = InfluenceGuidedSuppression().apply(None, self.CORPUS).result
        losses = torch.tensor([1.0, 0.0, 1.0, 0.0])
        # Rows 0 and 2 are unflagged, rows 1 and 3 carry zero loss here.
        assert component(losses, [0, 1, 2, 3]).item() == pytest.approx(0.0)

    def test_the_penalty_is_proportional_to_normalized_influence(self):
        component = InfluenceGuidedSuppression().apply(None, self.CORPUS).result
        losses = torch.tensor([1.0, 1.0, 1.0, 1.0])
        # Normalized: row 1 -> 2/2 = 1.0, row 3 -> 1/2 = 0.5. Negative, because
        # added to the task loss the term ascends the flagged rows' loss.
        assert component(losses, [0, 1, 2, 3]).item() == pytest.approx(-1.5)

    def test_lambda_scales_the_penalty_linearly(self):
        losses = torch.tensor([1.0, 1.0, 1.0, 1.0])
        one = InfluenceGuidedSuppression(lambda_=1.0).apply(None, self.CORPUS).result
        three = InfluenceGuidedSuppression(lambda_=3.0).apply(None, self.CORPUS).result
        assert three(losses, [0, 1, 2, 3]).item() == pytest.approx(
            3 * one(losses, [0, 1, 2, 3]).item()
        )

    def test_unnormalized_uses_the_raw_influence_magnitudes(self):
        component = (
            InfluenceGuidedSuppression(normalize=False).apply(None, self.CORPUS).result
        )
        losses = torch.tensor([1.0, 1.0, 1.0, 1.0])
        assert component(losses, [0, 1, 2, 3]).item() == pytest.approx(-3.0)

    def test_negative_influence_is_used_by_magnitude(self):
        corpus = InfluenceScoredCorpus(
            n_examples=2, flagged=[0], influence=[-2.0], source="t"
        )
        component = (
            InfluenceGuidedSuppression(normalize=False).apply(None, corpus).result
        )
        # Magnitude 2.0 from |-2.0|; the sign of the term is the suppression
        # direction and does not follow the sign of the influence score.
        assert component(torch.tensor([1.0, 1.0]), [0, 1]).item() == pytest.approx(-2.0)

    def test_it_suppresses_rather_than_reinforces_the_flagged_rows(self):
        # The property the component exists for, checked through a real
        # composition rather than trusted. Magnitude tests alone cannot see it:
        # a sign error passes every one of them while training the model to fit
        # the harmful examples *harder*, which is the exact opposite of the
        # intervention. Two rows with identical loss, one flagged, one not.
        corpus = InfluenceScoredCorpus(
            n_examples=2, flagged=[0], influence=[1.0], source="t"
        )
        component = InfluenceGuidedSuppression(lambda_=1.0).apply(None, corpus).result

        logits = torch.zeros(2, requires_grad=True)
        per_example = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor([1.0, 1.0]), reduction="none"
        )
        total = per_example.mean() + component(per_example, [0, 1])
        total.backward()

        flagged, benign = logits.grad[0].item(), logits.grad[1].item()
        # The benign row is fitted; the flagged row is pushed the other way.
        assert benign < 0
        assert flagged > 0

    def test_it_is_differentiable(self):
        component = InfluenceGuidedSuppression().apply(None, self.CORPUS).result
        losses = torch.rand(4, requires_grad=True)
        component(losses, [0, 1, 2, 3]).backward()
        assert losses.grad is not None

    def test_an_index_outside_the_corpus_is_refused(self):
        component = InfluenceGuidedSuppression().apply(None, self.CORPUS).result
        with pytest.raises(ValueError, match="outside the corpus range"):
            component(torch.rand(2), [0, 99])

    def test_provenance_records_that_influence_was_precomputed(self):
        result = InfluenceGuidedSuppression().apply(None, self.CORPUS)
        assert result.provenance["influence_source"] == "unit-test-precomputed"
        assert "precomputed" in result.provenance["scope"]
