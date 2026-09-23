"""Shared fixtures.

Two kinds of model fixture live here. The ``encoder_model`` fixture downloads
real weights and skips when it cannot, so it only ever asserts things that are
true of BERT in particular. The ``stub_*`` fixtures are offline stand-ins with
hand-authored distributions (see :mod:`tests.stubs`); they never skip, and the
values they produce are known in advance, so tests built on them can assert
exact numbers instead of just types.
"""

import pytest

from .stubs import (
    RoundTripTokenizer,
    ScriptedCausalLM,
    StubFillMaskPipeline,
    StubMaskedLM,
    StubSeq2SeqLM,
    StubSentenceEncoder,
    StubTokenizer,
    TinyCausalLM,
)


def _try_load(task):
    from fairLMs.models import HuggingFaceModel

    try:
        adapter = HuggingFaceModel("bert-base-uncased", task=task)
        adapter.load()
        return adapter
    except Exception as exc:  # offline, no cache, gated, …
        pytest.skip(f"bert-base-uncased ({task}) unavailable: {type(exc).__name__}")


@pytest.fixture(scope="session")
def encoder_model():
    """A small cached encoder. Skips the test if it cannot be loaded."""
    return _try_load("encoder")


@pytest.fixture
def word_sets():
    from fairLMs.metrics import WordSets

    return WordSets(
        target_1=["Adam", "Chip", "Harry", "Josh"],
        target_2=["Alonzo", "Jamel", "Lerone", "Percell"],
        attribute_1=["caress", "freedom", "health", "love"],
        attribute_2=["abuse", "crash", "filth", "murder"],
    )


@pytest.fixture
def context_sets():
    from fairLMs.metrics import ContextSets

    return ContextSets(
        target_1={"Adam": ["Adam went home.", "Adam is here."]},
        target_2={"Alonzo": ["Alonzo went home.", "Alonzo is here."]},
        attribute_1={"freedom": ["Freedom matters.", "We value freedom."]},
        attribute_2={"abuse": ["Abuse is harmful.", "They reported abuse."]},
    )


@pytest.fixture
def vector_sets():
    import numpy as np

    from fairLMs.metrics import VectorSets

    rng = np.random.default_rng(0)
    return VectorSets(*(rng.normal(size=(4, 16)) for _ in range(4)))


# ---------------------------------------------------------------------------
# Offline stubs
# ---------------------------------------------------------------------------
@pytest.fixture
def stub_tokenizer():
    return StubTokenizer()


@pytest.fixture
def stub_mlm():
    return StubMaskedLM()


@pytest.fixture
def stub_bundle(stub_tokenizer, stub_mlm):
    """The ``(tokenizer, model)`` tuple form every metric wrapper accepts."""
    return (stub_tokenizer, stub_mlm)


@pytest.fixture
def stub_pipeline(stub_mlm, stub_tokenizer):
    return StubFillMaskPipeline(stub_mlm, stub_tokenizer)


@pytest.fixture
def tiny_decoder():
    return TinyCausalLM()


@pytest.fixture
def scripted_decoder():
    return ScriptedCausalLM()


@pytest.fixture
def stub_seq2seq():
    return StubSeq2SeqLM()


@pytest.fixture
def stub_sentence_encoder():
    return StubSentenceEncoder()


@pytest.fixture
def round_trip_tokenizer():
    """A tokenizer that decodes back to exactly what was encoded.

    Needed by the metrics that score the *words* of a generation rather than a
    distribution, since the five-word stub vocabulary would turn any scripted
    text into a row of ``[UNK]``.
    """
    return RoundTripTokenizer()
