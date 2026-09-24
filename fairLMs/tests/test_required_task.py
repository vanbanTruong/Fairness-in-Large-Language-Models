"""The ``required_task`` declaration and the check that enforces it.

A metric reads a specific quantity — vocabulary logits, a bare hidden state, a
label distribution — and which of those a checkpoint exposes is decided by the
``task`` it was loaded with. The pairing used to be unchecked: the ``task`` was
recorded on the ``LoadedModel`` and then discarded by ``get_tokenizer_model``,
so a mismatch surfaced either as an ``AttributeError`` deep inside a metric or,
where a head was randomly initialized, as plausible numbers from noise.
"""

import pytest
import torch

from fairLMs.definitions import METRIC_REGISTRY, CrowSPairsScore, WEAT
from fairLMs.definitions.resolve import check_task, get_tokenizer_model
from fairLMs.definitions.models.base import LoadedModel


def _loaded(task):
    """A LoadedModel carrying nothing but the task label under test."""
    return LoadedModel(
        name="stub",
        tokenizer=object(),
        model=object(),
        device=torch.device("cpu"),
        task=task,
    )


class TestDeclarations:
    def test_every_registered_metric_declares_the_attribute(self):
        for name, cls in METRIC_REGISTRY.items():
            assert hasattr(cls, "required_task"), name

    def test_a_declared_task_is_one_the_loader_understands(self):
        known = {"mlm", "encoder", "sequence_classification", "seq2seq", "causal"}
        for name, cls in METRIC_REGISTRY.items():
            if cls.required_task is not None:
                assert cls.required_task in known, (name, cls.required_task)

    def test_the_declaration_agrees_with_the_architecture_family(self):
        family = {
            "mlm": "encoder_only",
            "encoder": "encoder_only",
            "sequence_classification": "encoder_only",
            "causal": "decoder_only",
            "seq2seq": "encoder_decoder",
        }
        for name, cls in METRIC_REGISTRY.items():
            if cls.required_task is None:
                continue
            assert family[cls.required_task] in cls.architectures, name

    def test_pll_wants_a_head_and_weat_wants_the_trunk(self):
        # The two encoder-only families disagree, which is the whole reason the
        # task cannot be inferred from the checkpoint.
        assert CrowSPairsScore.required_task == "mlm"
        assert WEAT.required_task == "encoder"


class TestCheck:
    def test_a_mismatch_names_both_sides_and_the_fix(self):
        with pytest.raises(TypeError) as exc:
            check_task(_loaded("encoder"), CrowSPairsScore())
        message = str(exc.value)
        assert "CrowSPairsScore" in message
        assert "'mlm'" in message and "'encoder'" in message
        assert "HuggingFaceModel" in message

    def test_a_match_is_silent(self):
        assert check_task(_loaded("mlm"), CrowSPairsScore()) is None

    def test_a_metric_with_no_requirement_accepts_any_task(self):
        class AnyHead:
            required_task = None

        for task in ("mlm", "encoder", "causal", "seq2seq"):
            assert check_task(_loaded(task), AnyHead()) is None

    def test_an_unknown_task_is_not_second_guessed(self):
        # A LoadedModel built by hand may carry no task at all. Refusing on
        # missing information would break the (tokenizer, model) escape hatch.
        assert check_task(_loaded(None), CrowSPairsScore()) is None

    def test_no_metric_means_no_check(self):
        assert check_task(_loaded("encoder"), None) is None


class TestResolveIntegration:
    def test_resolve_refuses_a_mismatched_loaded_model(self):
        with pytest.raises(TypeError, match="requires a model loaded with"):
            get_tokenizer_model(_loaded("encoder"), metric=CrowSPairsScore())

    def test_resolve_passes_a_matching_loaded_model_through(self):
        loaded = _loaded("mlm")
        tok, model, device = get_tokenizer_model(loaded, metric=CrowSPairsScore())
        assert (tok, model) == (loaded.tokenizer, loaded.model)
        assert device == loaded.device

    def test_a_raw_tuple_still_bypasses_the_check(self):
        # Tuples carry no task, so this path is unchanged. Tests across the
        # suite depend on it, and so does anyone holding a bare HF model.
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

        tokenizer, model = object(), Tiny()
        tok, resolved, device = get_tokenizer_model(
            (tokenizer, model), metric=CrowSPairsScore()
        )
        assert tok is tokenizer and resolved is model
        assert device == torch.device("cpu")

    def test_the_default_call_is_unchanged(self):
        # metric= is keyword-only and optional: existing callers keep working.
        loaded = _loaded("encoder")
        assert get_tokenizer_model(loaded)[0] is loaded.tokenizer
