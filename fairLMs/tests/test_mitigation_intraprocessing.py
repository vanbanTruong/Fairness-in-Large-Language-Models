"""Intra-processing components, and the guarantee that makes them useful.

The load-bearing claim: an intra-processing result *is* a ``ModelAdapter``, so
any metric in ``METRIC_REGISTRY`` re-evaluates the mitigated model with no
modification. That claim is tested here end to end against a real metric.
"""

import numpy as np
import pytest
import torch

from fairLMs.applicability import TASK_PROFILES
from fairLMs.metrics import METRIC_REGISTRY, WEAT, WordSets
from fairLMs.mitigation import (
    AttributeLabeledVectors,
    IterativeNullspaceProjection,
    ProjectedModelAdapter,
    PromptSpec,
    SelfDebiasedModelAdapter,
    SelfDebiasing,
    SubspaceProjection,
)
from fairLMs.mitigation.intraprocessing import _nullspace_projection
from fairLMs.models.base import LoadedModel, ModelAdapter
from .stubs import StubSentenceEncoder, StubTokenizer


class StubEncoderAdapter(ModelAdapter):
    """A real ``ModelAdapter`` over an offline encoder, so no download is needed."""

    name = "stub-encoder"
    task = "encoder"

    def __init__(self):
        self.model = StubSentenceEncoder(d_model=8)
        # The projection hook attaches here; real HF models expose the same API.
        self.model.get_input_embeddings = lambda: self.model.encoder.embedding
        self.tokenizer = StubTokenizer()

    def load(self) -> LoadedModel:
        return LoadedModel(
            name=self.name,
            tokenizer=self.tokenizer,
            model=self.model,
            device=torch.device("cpu"),
            task=self.task,
        )


def _vectors(**overrides):
    kwargs = dict(
        axis="gender",
        vectors=[[2.0, 0.1], [1.8, -0.1], [-2.0, 0.1], [-1.9, -0.2]],
        labels=["f", "f", "m", "m"],
        source="unit-test",
        representation_layer="input_embeddings",
        pooling="token",
        pair_ids=["p1", "p2", "p1", "p2"],
    )
    kwargs.update(overrides)
    if len(kwargs["labels"]) != 4:
        kwargs["pair_ids"] = None
    return AttributeLabeledVectors(**kwargs)


class TestProjectionMath:
    def test_the_projection_is_idempotent(self):
        # P_perp is a projection: applying it twice changes nothing.
        basis = np.array([[1.0], [0.0], [0.0]])
        projection = _nullspace_projection(basis)
        assert np.allclose(projection @ projection, projection)

    def test_the_projection_is_symmetric(self):
        basis = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        projection = _nullspace_projection(basis)
        assert np.allclose(projection, projection.T)

    def test_the_basis_directions_are_annihilated(self):
        basis = np.array([[1.0], [0.0], [0.0]])
        projection = _nullspace_projection(basis)
        assert np.allclose(projection @ basis, 0.0)

    def test_orthogonal_directions_survive_untouched(self):
        basis = np.array([[1.0], [0.0], [0.0]])
        projection = _nullspace_projection(basis)
        orthogonal = np.array([0.0, 3.0, 4.0])
        assert np.allclose(projection @ orthogonal, orthogonal)

    def test_a_rank_deficient_basis_does_not_raise(self):
        # Two collinear columns: the Gram matrix is singular, and the
        # pseudo-inverse is what keeps this a projection onto the real span.
        basis = np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
        projection = _nullspace_projection(basis)
        assert np.allclose(projection @ projection, projection)


class TestSubspaceProjection:
    def test_returns_a_model_adapter(self):
        result = SubspaceProjection().apply(StubEncoderAdapter(), _vectors())
        assert result.category == "intra"
        assert isinstance(result.result, ModelAdapter)

    def test_the_estimated_subspace_removes_the_pair_direction(self):
        # Pairs differ only along axis 0, so that axis must be projected out.
        evidence = _vectors(
            vectors=[[1.0, 5.0], [2.0, 7.0], [0.0, 5.0], [1.0, 7.0]],
            labels=["f", "f", "m", "m"],
        )
        result = SubspaceProjection().apply(StubEncoderAdapter(), evidence)
        projection = result.result.projection
        assert np.allclose(projection @ np.array([1.0, 0.0]), 0.0, atol=1e-9)
        assert np.allclose(projection @ np.array([0.0, 1.0]), [0.0, 1.0], atol=1e-9)

    def test_more_components_than_pairs_is_refused(self):
        with pytest.raises(ValueError, match="cannot support"):
            SubspaceProjection(n_components=5).apply(StubEncoderAdapter(), _vectors())

    def test_a_zero_component_subspace_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            SubspaceProjection(n_components=0).apply(StubEncoderAdapter(), _vectors())

    def test_unequal_group_sizes_are_refused_rather_than_truncated(self):
        evidence = _vectors(
            vectors=[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            labels=["f", "f", "m"],
        )
        with pytest.raises(ValueError, match="explicit pair_ids"):
            SubspaceProjection().apply(StubEncoderAdapter(), evidence)

    def test_textual_pairs_without_an_encoder_are_refused(self):
        from fairLMs.metrics import PromptPairs

        with pytest.raises(ValueError, match="needs an encoder"):
            SubspaceProjection(layer="input_embeddings").apply(
                StubEncoderAdapter(), PromptPairs(["he"], ["she"])
            )

    def test_provenance_reports_the_probe_family_not_absolute_removal(self):
        result = SubspaceProjection().apply(StubEncoderAdapter(), _vectors())
        assert "linear" in result.provenance["probe_family"]
        claim = result.provenance["removal_claim"]
        assert "Gonen" in claim and "hide rather than remove" in claim

    def test_an_api_model_is_refused_for_lacking_hidden_states(self):
        from fairLMs.models.openai import OpenAIModel

        with pytest.raises(TypeError, match="hidden_states"):
            SubspaceProjection().apply(OpenAIModel(), _vectors())

    def test_a_decoder_only_model_is_refused(self):
        with pytest.raises(TypeError, match="encoder_only"):
            SubspaceProjection().apply(TASK_PROFILES["causal"], _vectors())


class TestIterativeNullspaceProjection:
    def test_returns_a_model_adapter(self):
        result = IterativeNullspaceProjection().apply(StubEncoderAdapter(), _vectors())
        assert isinstance(result.result, ModelAdapter)

    def test_the_probe_can_no_longer_read_the_attribute_after_projection(self):
        # The property INLP exists to produce, checked directly rather than
        # trusted: refit a probe on the projected representations.
        from fairLMs.mitigation.intraprocessing import (
            _fit_linear_probe,
            _probe_accuracy,
        )

        evidence = _vectors()
        result = IterativeNullspaceProjection(n_iterations=8).apply(
            StubEncoderAdapter(), evidence
        )
        projected = np.array(evidence.vectors) @ result.result.projection.T
        targets = np.array([1.0 if x == "m" else 0.0 for x in evidence.labels])
        weights = _fit_linear_probe(projected, targets, seed=0)
        assert _probe_accuracy(projected, targets, weights) <= 0.75

    def test_the_probe_accuracy_history_is_reported(self):
        result = IterativeNullspaceProjection().apply(StubEncoderAdapter(), _vectors())
        history = result.provenance["probe_accuracy_history"]
        assert history and all(0.0 <= a <= 1.0 for a in history)
        assert 0 <= result.provenance["final_probe_accuracy"] <= 1
        assert "held-out" in result.provenance["probe_evaluation"]
        assert result.provenance["n_iterations_run"] <= len(history)

    def test_the_composed_projection_is_idempotent(self):
        result = IterativeNullspaceProjection().apply(StubEncoderAdapter(), _vectors())
        projection = result.result.projection
        assert np.allclose(projection @ projection, projection, atol=1e-8)

    def test_a_non_binary_attribute_is_refused(self):
        evidence = _vectors(
            vectors=[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            labels=["f", "m", "nonbinary"],
        )
        with pytest.raises(ValueError, match="binary attribute probe"):
            IterativeNullspaceProjection().apply(StubEncoderAdapter(), evidence)

    def test_zero_iterations_are_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            IterativeNullspaceProjection(n_iterations=0).apply(
                StubEncoderAdapter(), _vectors()
            )

    def test_provenance_avoids_an_absolute_removal_claim(self):
        result = IterativeNullspaceProjection().apply(StubEncoderAdapter(), _vectors())
        assert "Gonen" in result.provenance["removal_claim"]

    def test_the_motivating_refusal_names_hidden_states(self):
        from fairLMs.models.openai import OpenAIModel

        with pytest.raises(TypeError) as exc:
            IterativeNullspaceProjection().apply(OpenAIModel(), _vectors())
        message = str(exc.value)
        assert "hidden_states" in message and "free_generation" in message


class TestSelfDebiasing:
    def _stub(self):
        class Decoder(ModelAdapter):
            name = "stub-decoder"
            task = "causal"

            def load(self):
                raise NotImplementedError

        return Decoder()

    def test_returns_a_model_adapter(self):
        spec = PromptSpec(templates=["Biased: {query}"], attribute="gender")
        result = SelfDebiasing().apply(self._stub(), spec)
        assert isinstance(result.result, ModelAdapter)
        assert isinstance(result.result, SelfDebiasedModelAdapter)

    def test_the_wrapper_profiles_as_the_model_it_wraps(self):
        spec = PromptSpec(templates=["Biased: {query}"])
        adapter = SelfDebiasing().apply(self._stub(), spec).result
        assert adapter.task == "causal"

    def test_a_negative_decay_is_refused(self):
        spec = PromptSpec(templates=["Biased: {query}"])
        with pytest.raises(ValueError, match="non-negative"):
            SelfDebiasing(decay=-1.0).apply(self._stub(), spec)

    def test_a_masked_lm_is_refused_for_lacking_the_scores_it_reads(self):
        # An MLM exposes neither token_logprobs nor free_generation, so the
        # capability check speaks first - and names both.
        spec = PromptSpec(templates=["Biased: {query}"])
        with pytest.raises(TypeError) as exc:
            SelfDebiasing().apply(TASK_PROFILES["mlm"], spec)
        assert "token_logprobs" in str(exc.value)
        assert "free_generation" in str(exc.value)

    def test_an_encoder_decoder_is_refused_on_architecture(self):
        # seq2seq satisfies both capabilities, so the architecture check is
        # what refuses it: self-debiasing is declared decoder-only.
        spec = PromptSpec(templates=["Biased: {query}"])
        with pytest.raises(TypeError, match="decoder_only"):
            SelfDebiasing().apply(TASK_PROFILES["seq2seq"], spec)


class TestTheAdapterGuarantee:
    """ "An intra-processing result satisfies the adapter interface, so any
    metric re-evaluates the mitigated model without modification."
    """

    def test_the_result_is_accepted_by_an_unmodified_registry_metric(self):
        base = StubEncoderAdapter()
        evidence = AttributeLabeledVectors(
            axis="gender",
            vectors=np.eye(8)[:4] * 2.0,
            labels=["f", "f", "m", "m"],
            source="unit-test",
            representation_layer="input_embeddings",
            pooling="token",
            pair_ids=["p1", "p2", "p1", "p2"],
        )
        mitigated = SubspaceProjection().apply(base, evidence).result

        words = WordSets(
            target_1=["he", "doctor"],
            target_2=["she", "nurse"],
            attribute_1=["engineer", "doctor"],
            attribute_2=["nurse", "she"],
        )
        # WEAT is taken straight from the registry and is not adapted in any way.
        metric = METRIC_REGISTRY["weat"]()
        result = metric.compute(mitigated, words)
        assert isinstance(float(result), float)

    def test_the_mitigated_model_scores_differently_from_the_base(self):
        # The projection must actually reach the forward pass, not merely be
        # recorded on an adapter that behaves identically.
        base = StubEncoderAdapter()
        evidence = AttributeLabeledVectors(
            axis="gender",
            vectors=np.eye(8)[:4] * 2.0,
            labels=["f", "f", "m", "m"],
            source="unit-test",
            representation_layer="input_embeddings",
            pooling="token",
            pair_ids=["p1", "p2", "p1", "p2"],
        )
        mitigated = SubspaceProjection().apply(StubEncoderAdapter(), evidence).result

        words = WordSets(
            target_1=["he", "doctor"],
            target_2=["she", "nurse"],
            attribute_1=["engineer", "doctor"],
            attribute_2=["nurse", "she"],
        )
        before = float(WEAT().compute(base, words))
        after = float(WEAT().compute(mitigated, words))
        assert before != pytest.approx(after)

    def test_the_wrapped_adapter_profiles_like_the_original(self):
        base = StubEncoderAdapter()
        mitigated = SubspaceProjection().apply(base, _vectors()).result
        from fairLMs.applicability import describe_model

        assert describe_model(mitigated).architecture == "encoder_only"
        assert "hidden_states" in describe_model(mitigated).capabilities

    def test_the_original_adapter_is_left_untouched(self):
        base = StubEncoderAdapter()
        mitigated = SubspaceProjection().apply(base, _vectors()).result
        assert mitigated.base is base
        assert not isinstance(base, ProjectedModelAdapter)

    def test_removing_the_hook_restores_the_base_behaviour(self):
        base = StubEncoderAdapter()
        # Several words per set: WEAT's effect size divides by a standard
        # deviation, which is undefined for a single term.
        words = WordSets(
            target_1=["he", "doctor"],
            target_2=["she", "nurse"],
            attribute_1=["engineer", "doctor"],
            attribute_2=["nurse", "she"],
        )
        expected = float(WEAT().compute(base, words))
        assert not np.isnan(expected)

        shared = StubEncoderAdapter()
        mitigated = (
            SubspaceProjection()
            .apply(
                shared,
                _vectors(
                    vectors=np.pad(np.array(_vectors().vectors), ((0, 0), (0, 6)))
                ),
            )
            .result
        )
        mitigated.load()
        mitigated.remove()
        assert float(WEAT().compute(shared, words)) == pytest.approx(expected)
