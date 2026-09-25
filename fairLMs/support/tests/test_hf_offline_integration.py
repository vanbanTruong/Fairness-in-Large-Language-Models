"""Real transformers classes and local checkpoints; no Hub or paid requests.

Random weights establish API/device compatibility, not fairness effectiveness.
Failures in these tests must not be converted into network-related skips.
"""

import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DistilBertConfig,
    DistilBertForMaskedLM,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
)

from fairLMs.definitions import (
    AllUnmaskedLikelihoodAttentionScore,
    CrowSPairsScore,
    PseudoLogLikelihoodScore,
    DemographicPrompts,
    DemographicRepresentationDivergence,
)
from fairLMs.mitigation import ProjectedModelAdapter, SelfDebiasing, PromptSpec
from fairLMs.definitions.models import HuggingFaceModel
from fairLMs.definitions.models.base import LoadedModel, ModelAdapter

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "he",
    "she",
    "doctor",
    "nurse",
    "is",
    "a",
    ".",
]
PAIRS = [{"stereotype": "he is a doctor .", "anti_stereotype": "she is a doctor ."}]


def bert_tokenizer():
    """Build the WordPiece backend explicitly, the way ``word_tokenizer`` does.

    ``BertTokenizerFast(vocab_file=...)`` silently stopped honouring the file in
    transformers 5.x: the vocabulary came back holding only the five special
    tokens, every content word became ``[UNK]``, and because ``[UNK]`` counts as
    special there were no scoreable positions left. Passing a fully constructed
    ``tokenizer_object`` is the one route both 4.x and 5.x read the same way.

    The normalizer is not optional. ``BertTokenizerFast.__init__`` on 4.x reads
    ``backend_tokenizer.normalizer`` and raises a bare ``TypeError`` when it is
    ``None``.
    """
    engine = Tokenizer(
        models.WordPiece({t: i for i, t in enumerate(VOCAB)}, unk_token="[UNK]")
    )
    engine.normalizer = normalizers.BertNormalizer(lowercase=True)
    engine.pre_tokenizer = pre_tokenizers.Whitespace()
    engine.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", VOCAB.index("[CLS]")),
            ("[SEP]", VOCAB.index("[SEP]")),
        ],
    )
    return BertTokenizerFast(
        tokenizer_object=engine,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


@pytest.fixture
def bert_checkpoint(tmp_path):
    bert_tokenizer().save_pretrained(tmp_path)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = BertForMaskedLM(
            BertConfig(
                vocab_size=len(VOCAB),
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=24,
                max_position_embeddings=64,
            )
        )
    model.save_pretrained(tmp_path)
    return tmp_path


def test_local_bert_checkpoint_loads_and_scores(bert_checkpoint):
    adapter = HuggingFaceModel(str(bert_checkpoint), task="mlm", device="cpu")
    for metric in [
        CrowSPairsScore(batch_size=2),
        PseudoLogLikelihoodScore(batch_size=2),
        AllUnmaskedLikelihoodAttentionScore(),
    ]:
        result = metric.compute(adapter, PAIRS)
        assert np.isfinite(result.score)
        json.loads(result.to_json())
    contextual = [dict(PAIRS[0], scoring_context="a nurse .")]
    assert np.isfinite(PseudoLogLikelihoodScore().compute(adapter, contextual).score)


def test_distilbert_attention_without_bert_or_roberta_attribute(bert_checkpoint):
    tokenizer = BertTokenizerFast.from_pretrained(bert_checkpoint)
    model = DistilBertForMaskedLM(
        DistilBertConfig(
            vocab_size=len(VOCAB),
            dim=16,
            hidden_dim=24,
            n_layers=1,
            n_heads=2,
            max_position_embeddings=64,
        )
    ).eval()
    assert not hasattr(model, "bert") and not hasattr(model, "roberta")
    original_backend = model.config._attn_implementation
    result = AllUnmaskedLikelihoodAttentionScore().compute((tokenizer, model), PAIRS)
    assert np.isfinite(result.score)
    assert model.config._attn_implementation == original_backend


def test_named_bert_layer_projection_and_cleanup(bert_checkpoint):
    adapter = HuggingFaceModel(str(bert_checkpoint), task="mlm", device="cpu")
    loaded = adapter.load()
    inputs = loaded.tokenizer("he is a doctor", return_tensors="pt")
    with torch.no_grad():
        before = (
            loaded.model(**inputs, output_hidden_states=True).hidden_states[-1].clone()
        )
        edited = ProjectedModelAdapter(
            adapter,
            np.diag([0.0] + [1.0] * 15),
            method="integration",
            axis="gender",
            probe_family="linear",
            layer="bert.encoder.layer.0",
        )
        after = (
            edited.load().model(**inputs, output_hidden_states=True).hidden_states[-1]
        )
        assert torch.equal(after[..., 0], torch.zeros_like(after[..., 0]))
        assert torch.allclose(before[..., 1:], after[..., 1:])
        edited.remove()
        assert torch.equal(
            before, loaded.model(**inputs, output_hidden_states=True).hidden_states[-1]
        )


def word_tokenizer():
    engine = Tokenizer(
        models.WordLevel({t: i for i, t in enumerate(VOCAB)}, unk_token="[UNK]")
    )
    engine.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=engine,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[SEP]",
        model_input_names=["input_ids", "attention_mask"],
    )


def test_local_gpt2_self_debiasing_through_public_metric(tmp_path):
    tokenizer = word_tokenizer()
    tokenizer.save_pretrained(tmp_path)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(VOCAB),
            n_positions=64,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=0,
        )
    )
    model.save_pretrained(tmp_path)
    adapter = HuggingFaceModel(str(tmp_path), task="causal", device="cpu")
    edited = (
        SelfDebiasing(seed=0)
        .apply(adapter, PromptSpec(templates=["nurse {query}", "doctor {query}"]))
        .result
    )
    metric = DemographicRepresentationDivergence(max_new_tokens=2)
    result = metric.compute(
        edited, DemographicPrompts(["he is a doctor"], ["he"], ["she"])
    )
    assert result.status in ("ready", "insufficient_evidence")
    json.loads(result.to_json())
    ids = edited.load().tokenizer("he", return_tensors="pt")
    with pytest.raises(TypeError, match="num_beams"):
        edited.load().model.generate(**ids, num_beams=2)


def test_t5_encoder_modeloutput_projection():
    model = T5ForConditionalGeneration(
        T5Config(
            vocab_size=len(VOCAB),
            d_model=16,
            d_ff=24,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            decoder_start_token_id=0,
            eos_token_id=3,
            pad_token_id=0,
        )
    ).eval()

    class Adapter(ModelAdapter):
        name, task = "random-t5", "seq2seq"

        def load(self):
            return LoadedModel(
                name=self.name,
                task=self.task,
                model=model,
                tokenizer=word_tokenizer(),
                device=torch.device("cpu"),
            )

    edited = ProjectedModelAdapter(
        Adapter(),
        np.diag([0.0] + [1.0] * 15),
        method="integration",
        axis="gender",
        probe_family="linear",
        layer="encoder",
    )
    with torch.no_grad():
        try:
            output = edited.load().model.get_encoder()(
                input_ids=torch.tensor([[5, 7]]), output_hidden_states=True
            )
            assert torch.equal(output.last_hidden_state[..., 0], torch.zeros(1, 2))
            assert torch.equal(output.hidden_states[-1], output.last_hidden_state)
        finally:
            edited.remove()
