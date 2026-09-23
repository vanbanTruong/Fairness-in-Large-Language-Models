"""Tests for the argument-normalising dispatch layer.

``fairLMs.metrics.resolve`` sits between every metric and its model. It accepts
five different spellings of "here is a model" and three of "here is a dataset",
and picks a device when the caller did not. A bug here misroutes silently: the
metric still runs, still returns a float, and nobody learns that the tokenizer
came from one place and the weights from another.

The branches are cheap to reach and there are a lot of them, so they are all
covered here rather than incidentally through whichever metric happens to use
them.
"""

import pytest
import torch

from fairLMs.metrics.resolve import (
    get_examples,
    get_openai_bundle,
    get_tokenizer_model,
    require_kwargs,
)
from fairLMs.models import HuggingFaceModel
from fairLMs.models.base import LoadedModel
from fairLMs.models.openai import OpenAILoadedModel, OpenAIModel
from .stubs import StubMaskedLM, StubOpenAIClient, StubTokenizer


class TestGetExamples:
    def test_none_passes_through(self):
        assert get_examples(None) is None

    def test_dataset_is_loaded(self):
        class Dataset:
            def load(self):
                return iter([{"a": 1}, {"a": 2}])

        assert get_examples(Dataset()) == [{"a": 1}, {"a": 2}]

    def test_list_is_copied_not_aliased(self):
        original = [1, 2, 3]
        result = get_examples(original)
        assert result == original
        assert result is not original

    def test_tuple_becomes_a_list(self):
        assert get_examples((1, 2)) == [1, 2]

    def test_arbitrary_iterable_is_materialised(self):
        """Hugging Face datasets and generators arrive as plain iterables."""
        assert get_examples(iter([1, 2])) == [1, 2]

    def test_non_iterable_is_reported(self):
        with pytest.raises(TypeError, match="must be a FairnessDataset or sequence"):
            get_examples(42)


class TestGetTokenizerModel:
    def test_missing_model_is_reported(self):
        with pytest.raises(TypeError, match="model is required for this metric"):
            get_tokenizer_model(None)

    def test_loaded_model_is_unpacked(self):
        loaded = LoadedModel(
            name="stub",
            tokenizer=StubTokenizer(),
            model=StubMaskedLM(),
            device=torch.device("cpu"),
            task="mlm",
        )
        tok, model, device = get_tokenizer_model(loaded)
        assert tok is loaded.tokenizer
        assert model is loaded.model
        assert device == torch.device("cpu")

    def test_adapter_is_loaded_once(self):
        """``ModelAdapter.load()`` is called, and adapters are expected to cache."""
        loaded = LoadedModel(
            name="stub",
            tokenizer=StubTokenizer(),
            model=StubMaskedLM(),
            device=torch.device("cpu"),
            task="mlm",
        )
        adapter = HuggingFaceModel("stub")
        adapter._loaded = loaded  # stand in for a completed download

        tok, model, device = get_tokenizer_model(adapter)
        assert (tok, model) == (loaded.tokenizer, loaded.model)
        assert get_tokenizer_model(adapter)[1] is model

    def test_pair_infers_the_device_from_the_model(self):
        tokenizer, model = StubTokenizer(), StubMaskedLM()
        tok, mod, device = get_tokenizer_model((tokenizer, model))
        assert (tok, mod) == (tokenizer, model)
        assert device == next(model.parameters()).device

    def test_triple_honours_an_explicit_device_string(self):
        """A string device must be coerced, not passed through as ``str``.

        Downstream code calls ``tensor.to(device)``, which accepts both — but it
        also compares devices, and ``"cpu" != torch.device("cpu")``.
        """
        _tok, _mod, device = get_tokenizer_model(
            (StubTokenizer(), StubMaskedLM(), "cpu")
        )
        assert isinstance(device, torch.device)
        assert device == torch.device("cpu")

    def test_triple_accepts_a_device_object(self):
        wanted = torch.device("cpu")
        _tok, _mod, device = get_tokenizer_model(
            (StubTokenizer(), StubMaskedLM(), wanted)
        )
        assert device is wanted

    def test_triple_with_a_none_device_falls_back_to_the_model(self):
        model = StubMaskedLM()
        _tok, _mod, device = get_tokenizer_model((StubTokenizer(), model, None))
        assert device == next(model.parameters()).device

    def test_list_is_accepted_like_a_tuple(self):
        tokenizer, model = StubTokenizer(), StubMaskedLM()
        assert get_tokenizer_model([tokenizer, model])[:2] == (tokenizer, model)

    def test_bare_model_plus_tokenizer_keyword(self):
        tokenizer, model = StubTokenizer(), StubMaskedLM()
        tok, mod, device = get_tokenizer_model(model, tokenizer)
        assert (tok, mod) == (tokenizer, model)
        assert device == next(model.parameters()).device

    def test_bare_model_without_a_tokenizer_is_reported(self):
        """A model alone is not enough, and the error must say what to pass."""
        with pytest.raises(TypeError, match="Pass a HuggingFaceModel / LoadedModel"):
            get_tokenizer_model(StubMaskedLM())

    def test_one_element_sequence_is_not_a_pair(self):
        with pytest.raises(TypeError, match="Pass a HuggingFaceModel / LoadedModel"):
            get_tokenizer_model([StubMaskedLM()])


class TestGetOpenAIBundle:
    def test_loaded_bundle_passes_through(self):
        bundle = OpenAILoadedModel(
            name="x", client=StubOpenAIClient(), model="davinci-002"
        )
        assert get_openai_bundle(bundle) is bundle

    def test_adapter_is_loaded(self):
        adapter = OpenAIModel(model_name="davinci-002")
        adapter._loaded = OpenAILoadedModel(
            name="davinci-002", client=StubOpenAIClient(), model="davinci-002"
        )
        assert get_openai_bundle(adapter) is adapter._loaded

    def test_raw_client_is_wrapped(self):
        """A bare ``OpenAI()`` client is recognised by its ``.completions``."""
        client = StubOpenAIClient()
        bundle = get_openai_bundle(client)
        assert bundle.client is client
        assert bundle.model == "davinci-002"

    def test_chat_only_client_is_refused_before_request(self):
        class ChatClient:
            chat = object()

        with pytest.raises(TypeError, match="OpenAI"):
            get_openai_bundle(ChatClient())

    def test_unrelated_object_is_reported(self):
        with pytest.raises(TypeError, match="Pass an OpenAIModel"):
            get_openai_bundle(StubMaskedLM())

    def test_none_builds_a_default_adapter_from_the_environment(self, monkeypatch):
        """``None`` means "construct a client from ``OPENAI_API_KEY``".

        Pinned with a dummy key so the branch is exercised without depending on
        the developer's ambient environment. No request is made.
        """
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
        bundle = get_openai_bundle(None)
        assert bundle.model == "davinci-002"
        assert bundle.client is not None


class TestRequireKwargs:
    def test_present_keys_pass(self):
        require_kwargs({"a": 1, "b": 2}, "a", "b")

    def test_missing_keys_are_all_named(self):
        with pytest.raises(TypeError, match="a, c"):
            require_kwargs({"b": 2}, "a", "b", "c")

    def test_explicit_none_counts_as_missing(self):
        """A key present but ``None`` is the common shape of a forgotten argument."""
        with pytest.raises(TypeError, match="a"):
            require_kwargs({"a": None}, "a")
