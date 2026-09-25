"""Behavioral regressions for the September 2026 code audit (no paid API calls)."""

from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from fairLMs.definitions.core.applicability import check_applicability
from fairLMs.datasets import StereoSet, XNLIReligionPairs
from fairLMs.definitions import (
    CrowSPairsScore,
    PseudoLogLikelihoodScore,
    CounterfactualRobustness,
    CounterfactualFairnessScore,
    BiasAmplifierScore,
    PromptPairs,
    DemographicRepresentationDivergence,
    DemographicPrompts,
    DemographicNextTokenProportion,
    EqualOpportunityGap,
    GroupPredictions,
    SentenceTriples,
    ContextAssociationTestScore,
    WEAT,
    VectorSets,
)
from fairLMs.definitions.models import HuggingFaceModel
from fairLMs.definitions.models.base import ModelAdapter, LoadedModel
from fairLMs.definitions.models.openai import OpenAILoadedModel
from fairLMs.mitigation import (
    SubspaceProjection,
    IterativeNullspaceProjection,
    AttributeLabeledVectors,
    ProjectedModelAdapter,
    SelfDebiasing,
    PromptSpec,
    MetricEvaluation,
    compare_before_after,
)
from fairLMs.mitigation._generation import damped_probabilities
from .stubs import StubOpenAIClient, StubTokenizer, StubMaskedLM, VOCAB, VOCAB_SIZE


def stereo_row():
    # Official ClassLabel: 0 anti-stereotype, 1 stereotype, 2 unrelated.
    return dict(
        id="fixture-1",
        target="doctor",
        bias_type="gender",
        context="doctor",
        sentences=dict(sentence=["she", "he", "engineer"], gold_label=[0, 1, 2]),
    )


def test_stereoset_preserves_official_roles_and_context(monkeypatch):
    monkeypatch.setattr(StereoSet, "_load_hf", lambda self: [stereo_row()])
    row = StereoSet(as_triples=True).load()[0]
    assert row["stereotype"] == "he" and row["anti_stereotype"] == "she"
    assert row["context"] == row["scoring_context"] == "doctor"
    assert row["id"] == "fixture-1" and row["target"] == "doctor"
    assert SentenceTriples.from_examples([row]).contexts == ("doctor",)


def test_stereoset_uses_declared_classlabel_names(monkeypatch):
    class Rows(list):
        features = {
            "sentences": SimpleNamespace(
                feature={
                    "gold_label": SimpleNamespace(
                        names=["stereotype", "anti-stereotype", "unrelated"]
                    )
                }
            )
        }

    monkeypatch.setattr(StereoSet, "_load_hf", lambda self: Rows([stereo_row()]))
    assert StereoSet().load()[0]["stereotype"] == "she"


@pytest.mark.parametrize("bad", ["missing_context", "bad_label", "bad_length"])
def test_stereoset_refuses_uninterpretable_records(monkeypatch, bad):
    row = stereo_row()
    if bad == "missing_context":
        row.pop("context")
    elif bad == "bad_label":
        row["sentences"]["gold_label"][0] = 123
    else:
        row["sentences"]["gold_label"].pop()
    monkeypatch.setattr(StereoSet, "_load_hf", lambda self: [row])
    with pytest.raises(ValueError):
        StereoSet().load()


class RecordingMaskedLM(StubMaskedLM):
    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, input_ids=None, **kwargs):
        self.batches.append(input_ids.detach().clone())
        assert input_ids.device == next(self.parameters()).device
        return super().forward(input_ids=input_ids, **kwargs)


def test_context_is_visible_and_not_masked_in_pll():
    model, tokenizer = RecordingMaskedLM(), StubTokenizer()
    result = PseudoLogLikelihoodScore(batch_size=1).compute(
        (tokenizer, model),
        [dict(stereotype="he", anti_stereotype="she", scoring_context="doctor")],
    )
    assert result.details["n_pairs"] == 1
    assert len(model.batches) == 2
    for ids in model.batches:
        assert ids[0, 1] == VOCAB["doctor"]
        assert ids[0, 2] == tokenizer.mask_token_id


def test_cat_context_survives_typed_conversion(monkeypatch):
    monkeypatch.setattr(StereoSet, "_load_hf", lambda self: [stereo_row()])
    model = RecordingMaskedLM()
    ContextAssociationTestScore().compute(
        (StubTokenizer(), model), StereoSet(as_triples=True)
    )
    assert len(model.batches) == 3
    assert all(batch[0, 1].item() == VOCAB["doctor"] for batch in model.batches)


def test_mask_batches_are_bounded_and_numerically_equivalent():
    pairs = [
        dict(
            stereotype="he doctor nurse engineer",
            anti_stereotype="she doctor nurse engineer",
        )
    ]
    bounded = RecordingMaskedLM()
    small = PseudoLogLikelihoodScore(batch_size=1).compute(
        (StubTokenizer(), bounded), pairs
    )
    large = PseudoLogLikelihoodScore(batch_size=16).compute(
        (StubTokenizer(), StubMaskedLM()), pairs
    )
    assert max(batch.shape[0] for batch in bounded.batches) == 1
    assert small.score == large.score
    assert small.details["accuracy"] == large.details["accuracy"]


def test_optional_bias_type_and_bad_later_row_are_handled_before_loading():
    result = CrowSPairsScore().compute(
        (StubTokenizer(), StubMaskedLM()),
        [dict(stereotype="he doctor", anti_stereotype="she doctor")],
    )
    assert "unspecified" in result.by_category

    class NoLoad(ModelAdapter):
        task = "mlm"

        def load(self):
            raise AssertionError("must validate first")

    with pytest.raises(ValueError, match="example 1"):
        CrowSPairsScore().compute(
            NoLoad(), [dict(stereotype="a", anti_stereotype="b"), {"stereotype": "c"}]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cps_and_pll_move_tokenizer_inputs_to_cuda():
    model = RecordingMaskedLM().to("cuda")
    pairs = [dict(stereotype="he doctor", anti_stereotype="she doctor")]
    for metric in [CrowSPairsScore(), PseudoLogLikelihoodScore()]:
        metric.compute((StubTokenizer(), model), pairs)
    assert model.batches and all(x.is_cuda for x in model.batches)


def test_xnli_is_unique_counterfactual_evidence(monkeypatch):
    monkeypatch.setattr(XNLIReligionPairs, "_load_hf", lambda self: [])
    rows = XNLIReligionPairs().load()
    assert len(rows) == 150
    keys = [(r["factual"], r["counterfactual"]) for r in rows]
    assert not any((b, a) in keys for a, b in keys)
    assert all("stereotype" not in r for r in rows)
    assert len(PromptPairs.from_examples(rows)) == 150


def test_religion_swaps_do_not_modify_substrings(monkeypatch):
    monkeypatch.setattr(
        XNLIReligionPairs,
        "_load_hf",
        lambda self: [
            dict(premise="Christianity is discussed."),
            dict(premise="A Christian is here."),
        ],
    )
    rows = XNLIReligionPairs(
        include_templates=False, religion_swaps=[("christian", "muslim")]
    ).load()
    assert len(rows) == 1 and rows[0]["counterfactual"] == "A Muslim is here."


@pytest.mark.parametrize(
    "metric", [CounterfactualRobustness(), CounterfactualFairnessScore()]
)
def test_api_bundle_model_is_authoritative(metric):
    client = StubOpenAIClient(
        top_tokens={"nurse": "she", "doctor": "he"},
        distributions={"nurse": {"she": 1.0}, "doctor": {"he": 1.0}},
    )
    bundle = OpenAILoadedModel(name="requested", model="requested", client=client)
    result = metric.compute(bundle, PromptPairs(["nurse"], ["doctor"]))
    assert len(client.calls) == 2
    assert {call["model"] for call in client.calls} == {"requested"}
    assert result.provenance["model"]["effective_model"] == "requested"


def test_api_explicit_override_and_transport_refusal():
    client = StubOpenAIClient(top_tokens={"nurse": "she", "doctor": "he"})
    bundle = OpenAILoadedModel(name="requested", model="requested", client=client)
    CounterfactualRobustness(completion_model="override").compute(
        bundle, PromptPairs(["nurse"], ["doctor"])
    )
    assert {c["model"] for c in client.calls} == {"override"}
    local = HuggingFaceModel("must-not-download", task="causal")
    for metric in [
        CounterfactualRobustness(),
        CounterfactualFairnessScore(),
        BiasAmplifierScore(),
    ]:
        with pytest.raises(TypeError, match="completions_api"):
            check_applicability(metric, local)


def test_cr_retries_transient_errors_and_exits_on_success(monkeypatch):
    from fairLMs.definitions.decoder_only.extrinsic_bias.counterfactual_fairness.cr import (
        cr,
    )

    monkeypatch.setattr(cr.time, "sleep", lambda delay: None)
    client = StubOpenAIClient(top_tokens={"nurse": "she"})
    original = client.completions.create
    attempts = []

    class Transient(Exception):
        status_code = 429

    def create(**kw):
        attempts.append(kw)
        if len(attempts) == 1:
            raise Transient()
        return original(**kw)

    client.completions.create = create
    assert cr._top1_token(client, "nurse") == "she"
    assert len(attempts) == 2
    client.completions.create = lambda **kw: (_ for _ in ()).throw(
        ValueError("permanent")
    )
    with pytest.raises(ValueError, match="permanent"):
        cr._top1_token(client, "nurse")


def test_probability_damping_matches_reference_and_logit_shift_invariance():
    regular, biased = torch.tensor([2.0, 1.0]), torch.tensor([6.0, -6.0])
    p, q = regular.softmax(0), biased.softmax(0)
    expected = p * torch.exp(-50 * (q - p).clamp_min(0)).clamp_min(0.01)
    expected /= expected.sum()
    actual = damped_probabilities(regular, [biased], decay=50)
    assert torch.allclose(actual, expected)
    assert torch.allclose(
        actual, damped_probabilities(regular + 100, [biased - 100], decay=50)
    )
    assert actual.argmax() == 1
    assert torch.allclose(damped_probabilities(regular, [biased], decay=0), p)


class ControlledDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(max_position_embeddings=128)
        self.calls = []
        self.base_generations = 0

    @property
    def device(self):
        return self.anchor.device

    def forward(self, input_ids=None, **kwargs):
        self.calls.append(input_ids.clone())
        logits = torch.full((*input_ids.shape, VOCAB_SIZE), -100.0, device=self.device)
        diagnosis = (input_ids == VOCAB["nurse"]).any().item()
        logits[..., VOCAB["he"]] = 6.0 if diagnosis else 2.0
        logits[..., VOCAB["she"]] = -6.0 if diagnosis else 1.0
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids=None, **kwargs):
        self.base_generations += 1
        return torch.cat(
            (input_ids, torch.tensor([[VOCAB["he"]]], device=self.device)), dim=1
        )


class DecoderAdapter(ModelAdapter):
    name, task = "controlled", "causal"

    def __init__(self):
        self.model, self.tokenizer = ControlledDecoder(), StubTokenizer()

    def load(self):
        return LoadedModel(
            name=self.name,
            task=self.task,
            device=self.model.device,
            model=self.model,
            tokenizer=self.tokenizer,
        )


def test_public_generation_metric_observes_self_debiasing():
    base = DecoderAdapter()
    edited = (
        SelfDebiasing()
        .apply(base, PromptSpec(templates=["doctor {query}", "nurse {query}"]))
        .result
    )
    data = DemographicPrompts(["doctor"], ["he"], ["she"])
    metric = DemographicRepresentationDivergence(max_new_tokens=1)
    before = metric.compute(base, data)
    after = metric.compute(edited, data)
    assert before.details["n_stereotype_total"] == 1
    assert after.details["n_counter_total"] == 1
    assert (
        base.model.base_generations == 1
    )  # only the baseline called original.generate
    assert len(base.model.calls) == 3  # plain + both prefixes
    loaded = edited.load()
    with pytest.raises(TypeError, match="token_logprobs"):
        DemographicNextTokenProportion().compute(
            loaded, DemographicPrompts(["doctor"], ["he"], ["she"], ["engineer"])
        )
    with pytest.raises(TypeError, match="edited free_generation"):
        loaded.model(input_ids=torch.tensor([[1]]))


def test_self_debiasing_sampling_is_reproducible_and_stays_in_token_space():
    base = DecoderAdapter()
    edited = (
        SelfDebiasing(seed=4)
        .apply(base, PromptSpec(templates=["nurse {query}"]))
        .result.load()
    )
    inputs = base.tokenizer("doctor", return_tensors="pt")
    first = edited.model.generate(
        **inputs, max_new_tokens=3, do_sample=True, top_p=0.9, num_return_sequences=2
    )
    second = edited.model.generate(
        **inputs, max_new_tokens=3, do_sample=True, top_p=0.9, num_return_sequences=2
    )
    assert torch.equal(first, second)
    assert first.shape[0] == 2 and first.shape[1] == inputs["input_ids"].shape[1] + 3


def paired_vectors(**kwargs):
    base = dict(
        axis="g",
        source="controlled",
        vectors=[[0.0, 1.0], [0.0, 2.0], [0.0, 0.0], [0.0, 1.0]],
        labels=["a", "a", "b", "b"],
        pair_ids=["one", "two", "one", "two"],
        representation_layer="input_embeddings",
        pooling="token",
    )
    base.update(kwargs)
    return AttributeLabeledVectors(**base)


class DeclaredEncoder(ModelAdapter):
    name, task = "unused", "encoder"

    def load(self):
        raise AssertionError("projection estimation needs no model load")


def test_constant_pair_differences_remove_the_correct_direction():
    projection = (
        SubspaceProjection()
        .apply(DeclaredEncoder(), paired_vectors())
        .result.projection
    )
    assert np.allclose(np.array([0.0, 1.0]) @ projection, 0)
    assert np.allclose(np.array([1.0, 0.0]) @ projection, [1.0, 0.0])
    with pytest.raises(ValueError, match="rank"):
        SubspaceProjection().apply(
            DeclaredEncoder(), paired_vectors(vectors=np.ones((4, 2)))
        )


def test_projection_requires_layer_and_pairing_evidence():
    for changes, message in [
        ({"representation_layer": None}, "representation layer"),
        ({"pair_ids": None}, "pair_ids"),
    ]:
        with pytest.raises(ValueError, match=message):
            SubspaceProjection().apply(DeclaredEncoder(), paired_vectors(**changes))
    with pytest.raises(ValueError, match="differs"):
        SubspaceProjection(layer="another.layer").apply(
            DeclaredEncoder(), paired_vectors()
        )


def test_pair_id_alignment_survives_row_reordering():
    e = paired_vectors()
    order = [3, 0, 2, 1]
    shuffled = paired_vectors(
        vectors=e.vectors[order],
        labels=[e.labels[i] for i in order],
        pair_ids=[e.pair_ids[i] for i in order],
    )
    first = SubspaceProjection().apply(DeclaredEncoder(), e).result.projection
    assert np.allclose(
        first, SubspaceProjection().apply(DeclaredEncoder(), shuffled).result.projection
    )


def test_projection_uses_the_named_layer_and_refuses_width_mismatch():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(2, 2)
            self.rotate = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.embedding.weight.copy_(torch.eye(2))
                self.rotate.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, ids):
            return self.rotate(self.embedding(ids))

    class Adapter(ModelAdapter):
        name, task = "layer-test", "encoder"

        def __init__(self):
            self.net = Net()

        def load(self):
            return LoadedModel(
                name=self.name,
                model=self.net,
                tokenizer=None,
                device=torch.device("cpu"),
                task=self.task,
            )

    base = Adapter()
    before = base.net(torch.tensor([0, 1])).detach().clone()
    edited = ProjectedModelAdapter(
        base,
        np.diag([0.0, 1.0]),
        method="test",
        axis="g",
        probe_family="test",
        layer="rotate",
    )
    after = edited.load().model(torch.tensor([0, 1]))
    assert torch.allclose(after[:, 0], torch.zeros(2)) and after[0, 1] == 1
    edited.remove()
    assert torch.equal(base.net(torch.tensor([0, 1])), before)
    wrong = ProjectedModelAdapter(
        base,
        np.eye(3),
        method="test",
        axis="g",
        probe_family="test",
        layer="input_embeddings",
    )
    with pytest.raises(ValueError, match="dimension"):
        wrong.load()


def test_inlp_reports_a_probe_on_final_heldout_vectors():
    from fairLMs.mitigation.intraprocessing import _fit_linear_probe, _probe_accuracy
    from sklearn.model_selection import train_test_split

    e = paired_vectors(vectors=[[2.0, 0.0], [3.0, 0.1], [-2.0, 0.0], [-3.0, -0.1]])
    result = IterativeNullspaceProjection(n_iterations=1, seed=0).apply(
        DeclaredEncoder(), e
    )
    train, validation = train_test_split(
        np.arange(4), test_size=2, stratify=[0, 0, 1, 1], random_state=0
    )
    features = e.vectors @ result.result.projection
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    weights = _fit_linear_probe(features[train], labels[train], seed=0)
    assert result.provenance["final_probe_accuracy"] == _probe_accuracy(
        features[validation], labels[validation], weights
    )
    assert (
        result.provenance["n_probe_train"]
        == result.provenance["n_probe_validation"]
        == 2
    )


def test_comparison_uses_distinct_predictions_and_keeps_configuration():
    metric = EqualOpportunityGap(g1="B", g2="A", positive_label="yes")
    before = GroupPredictions(["yes", "yes"], ["yes", "no"], ["A", "B"])
    after = GroupPredictions(["yes", "yes"], ["yes", "yes"], ["A", "B"])
    report = compare_before_after(
        None,
        None,
        metrics={"gap": MetricEvaluation(metric=metric, data=before, after_data=after)},
    )
    delta = report.fairness[0]
    assert delta.before == -1 and delta.after == 0 and delta.delta == 1
    assert delta.provenance["before"]["configuration"]["positive_label"] == "yes"
    metric.set_params(positive_label="no")
    assert delta.provenance["before"]["configuration"]["positive_label"] == "yes"
    assert json.loads(report.to_json())["fairness"]["extrinsic"][0]["delta"] == 1


def test_comparison_refuses_reusing_precomputed_predictions_implicitly():
    data = GroupPredictions([1, 1], [1, 0], ["a", "b"])
    report = compare_before_after(None, None, metrics={"equal_opportunity_gap": data})
    assert report.fairness[0].delta is None
    assert "after_data" in report.fairness[0].error


def test_comparison_factory_and_nonfinite_scores_serialize():
    calls = []

    def predictions(model):
        calls.append(model)
        return GroupPredictions([0, 1], [0, 1], ["a", "b"])

    report = compare_before_after(
        "base",
        "candidate",
        metrics={
            "gap": MetricEvaluation(
                metric=EqualOpportunityGap(), evidence_factory=predictions
            )
        },
        utility={"nan": lambda m: float("nan")},
    )
    assert calls == ["base", "candidate"]
    assert report.fairness[0].before is None and report.utility[0].after is None
    assert json.loads(report.to_json())["fairness"]["extrinsic"][0]["delta"] is None


def test_metric_provenance_records_seed_and_input_hash():
    rng = np.random.default_rng(0)
    data = VectorSets(*(rng.normal(size=(4, 3)) for _ in range(4)))
    result = WEAT(seed=12, n_samples=3).compute(None, data)
    assert result.provenance["configuration"]["seed"] == 12
    assert len(result.provenance["evidence"]["sha256"]) == 64
    assert json.loads(result.to_json())["provenance"]["metric"] == "weat"


def test_cat_counts_both_meaningful_alternatives(monkeypatch):
    from fairLMs.definitions.encoder_only.intrinsic_bias.probability_based.pseudo_log_likelihood_metrics.cat import (
        cat,
    )

    scores = {"stereo": -1.0, "unrelated": -2.0, "anti": -3.0}
    monkeypatch.setattr(
        cat,
        "sentence_pll",
        lambda model, tokenizer, sentence, **kwargs: scores[sentence],
    )
    ss, lms, icat, rows = cat.compute_ss(
        None, None, ["stereo"], ["anti"], ["unrelated"]
    )
    assert ss == 100 and lms == 50 and icat == 0
    assert rows[0]["meaningful_wins"] == 1


@pytest.mark.parametrize("placement", ["dispatched", "4bit", "8bit"])
def test_hf_adapter_preserves_backend_placement(monkeypatch, placement):
    import transformers

    class PlacedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Embedding(10, 2)
            if placement == "dispatched":
                self.hf_device_map = {"": "cpu"}
            elif placement == "4bit":
                self.is_loaded_in_4bit = True
            else:
                self.is_loaded_in_8bit = True

        def get_input_embeddings(self):
            return self.embeddings

        def to(self, *args, **kwargs):
            raise AssertionError("must preserve backend placement")

    model = PlacedModel()
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: StubTokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModelForMaskedLM, "from_pretrained", lambda *a, **k: model
    )
    result = HuggingFaceModel("offline", task="mlm", device="cpu").load()
    assert result.model is model and result.device == model.embeddings.weight.device
