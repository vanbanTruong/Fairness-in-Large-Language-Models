"""Reference backends for the backend-dependent construction slots.

The embedding backend is exercised with a tiny randomly initialized encoder
and a vocabulary written to a temporary file, so nothing is downloaded. The
grammar and parser backends depend on optional extras; their live paths are
skipped when the dependency or pipeline is absent, and their pure-Python
surface (validation, revision strings, tree depth) is tested unconditionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from fairLMs.datasets.diagnostics import (
    DependencyParserBackend,
    EmbeddingBackend,
    GrammarCheckerBackend,
)
from fairLMs.datasets.diagnostics.backends import (
    HuggingFaceEmbeddingBackend,
    LanguageToolGrammarBackend,
    SpacyDependencyBackend,
)

# --------------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------------


def test_reference_backends_satisfy_their_protocols_structurally():
    assert isinstance(HuggingFaceEmbeddingBackend(), EmbeddingBackend)
    assert isinstance(LanguageToolGrammarBackend(), GrammarCheckerBackend)
    assert isinstance(SpacyDependencyBackend(), DependencyParserBackend)


def test_backend_constructors_validate_their_configuration():
    with pytest.raises(ValueError):
        HuggingFaceEmbeddingBackend(model_name="  ")
    with pytest.raises(ValueError):
        HuggingFaceEmbeddingBackend(pooling="max")
    with pytest.raises(ValueError):
        HuggingFaceEmbeddingBackend(batch_size=0)
    with pytest.raises(ValueError):
        LanguageToolGrammarBackend(language="")
    with pytest.raises(ValueError):
        SpacyDependencyBackend(model="")


def test_backends_refuse_a_bare_string_as_the_text_sequence():
    with pytest.raises(TypeError):
        HuggingFaceEmbeddingBackend().encode("one string")
    with pytest.raises(TypeError):
        LanguageToolGrammarBackend().count_errors("one string")
    with pytest.raises(TypeError):
        SpacyDependencyBackend().depths("one string")


def test_empty_input_returns_empty_output_without_loading_anything():
    assert HuggingFaceEmbeddingBackend(model_name="never-loaded").encode([]) == []
    assert LanguageToolGrammarBackend().count_errors([]) == []
    assert SpacyDependencyBackend(model="never-loaded").depths([]) == []


def test_revision_strings_name_the_model_and_the_configuration():
    embed = HuggingFaceEmbeddingBackend(model_name="org/model", model_revision="abc123")
    assert embed.revision.startswith("org/model@abc123;pooling=mean;transformers=")
    assert LanguageToolGrammarBackend(language="en-GB").revision.endswith("language=en-GB")
    assert "pipeline=en_core_web_sm@unloaded" in SpacyDependencyBackend().revision


# --------------------------------------------------------------------------
# Embedding backend on a tiny random encoder
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_encoder(tmp_path_factory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "the",
        "nurse",
        "engineer",
        "is",
        "a",
        "an",
        "brilliant",
        "tidy",
        "desk",
    ]
    vocab_file = tmp_path_factory.mktemp("vocab") / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizer(str(vocab_file), do_lower_case=True)

    torch.manual_seed(0)
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    model = transformers.BertModel(config)
    model.eval()
    return tokenizer, model


def test_embedding_backend_encodes_in_order_with_one_vector_per_text(tiny_encoder):
    import math

    tokenizer, model = tiny_encoder
    backend = HuggingFaceEmbeddingBackend.from_components(
        tokenizer, model, revision="tiny-random-bert@test"
    )
    texts = ["the nurse is tidy", "an engineer is brilliant", "the nurse is tidy"]
    vectors = backend.encode(texts)

    assert len(vectors) == 3
    assert {len(vector) for vector in vectors} == {16}
    assert all(math.isfinite(component) for vector in vectors for component in vector)
    assert all(isinstance(component, float) for vector in vectors for component in vector)
    # Identical texts get identical vectors regardless of their position.
    assert vectors[0] == pytest.approx(vectors[2])
    assert vectors[0] != pytest.approx(vectors[1])
    assert backend.revision.startswith("tiny-random-bert@test@default;pooling=mean")


def test_embedding_backend_batching_does_not_change_the_vectors(tiny_encoder):
    tokenizer, model = tiny_encoder
    texts = ["the nurse is tidy", "an engineer", "a brilliant engineer is tidy"]
    one_by_one = HuggingFaceEmbeddingBackend.from_components(
        tokenizer, model, revision="r", batch_size=1
    ).encode(texts)
    batched = HuggingFaceEmbeddingBackend.from_components(
        tokenizer, model, revision="r", batch_size=8
    ).encode(texts)
    for left, right in zip(one_by_one, batched):
        assert left == pytest.approx(right, abs=1e-5)


def test_embedding_backend_pooling_rule_is_part_of_the_declaration(tiny_encoder):
    tokenizer, model = tiny_encoder
    mean = HuggingFaceEmbeddingBackend.from_components(
        tokenizer, model, revision="r", pooling="mean"
    )
    cls = HuggingFaceEmbeddingBackend.from_components(
        tokenizer, model, revision="r", pooling="cls"
    )
    assert mean.encode(["the nurse"])[0] != pytest.approx(cls.encode(["the nurse"])[0])
    assert "pooling=mean" in mean.revision and "pooling=cls" in cls.revision


def test_embedding_backend_from_components_requires_a_revision_label(tiny_encoder):
    tokenizer, model = tiny_encoder
    with pytest.raises(ValueError):
        HuggingFaceEmbeddingBackend.from_components(tokenizer, model, revision=" ")


# --------------------------------------------------------------------------
# Dependency depth: the tree walk is pure Python
# --------------------------------------------------------------------------


@dataclass
class _FakeToken:
    children: list = field(default_factory=list)


def test_tree_depth_counts_tokens_on_the_longest_root_to_leaf_path():
    leaf = _FakeToken()
    mid = _FakeToken(children=[leaf])
    root = _FakeToken(children=[_FakeToken(), mid])
    assert SpacyDependencyBackend._tree_depth(_FakeToken()) == 1
    assert SpacyDependencyBackend._tree_depth(mid) == 2
    assert SpacyDependencyBackend._tree_depth(root) == 3


def test_spacy_backend_depths_on_a_real_pipeline_when_available():
    spacy = pytest.importorskip("spacy")
    backend = SpacyDependencyBackend()
    try:
        spacy.load(backend.model)
    except OSError:
        pytest.skip(f"spaCy pipeline {backend.model!r} is not installed")

    depths = backend.depths(["The quick brown fox jumps over the lazy dog.", "Hi.", ""])
    assert len(depths) == 3
    assert depths[0] >= 2
    assert depths[1] >= 1
    assert depths[2] == 0
    assert "@unloaded" not in backend.revision


def test_language_tool_backend_counts_when_available():
    pytest.importorskip("language_tool_python")
    backend = LanguageToolGrammarBackend()
    try:
        counts = backend.count_errors(["She were here.", "She was here."])
    except Exception as exc:  # no Java, no network, no server
        pytest.skip(f"LanguageTool unavailable: {type(exc).__name__}")
    finally:
        backend.close()
    assert len(counts) == 2
    assert all(isinstance(count, int) and count >= 0 for count in counts)
