"""Offline stand-ins for the model backends the metrics expect.

The similarity-based family (WEAT / SEAT / CEAT) is already testable without a
checkpoint, because ``VectorSets`` lets a caller hand the metric its embeddings
directly. Nothing else in the library has that escape hatch: every other metric
reaches through to a tokenizer and a model, so its arithmetic could only be
exercised against whatever a downloaded checkpoint happened to say — which in CI
means skipping.

These stubs close the gap one layer down. The masked LM here returns a *fixed*
distribution over a five-word vocabulary, defined by two small tables
(:data:`BASE_LOGITS` and :data:`AFFINITY`) that a test can re-apply for itself.
That is the point: the expected value of a metric becomes something a test
derives independently, rather than a number recorded from a previous run. A
recorded number pins behaviour; a re-derived one catches a flipped sign or a
transposed denominator.

The scoring rule
----------------
For a row of token ids and a position ``p``::

    logits[p] = BASE + sum(AFFINITY[t] for t at every position except p)

So the distribution at a position depends on the *bag* of other tokens in the
sequence and not on their order or on ``p`` itself. Two consequences are worth
knowing when reading the tests:

* Special tokens (``[CLS]``, ``[SEP]``, ``[MASK]``, ``[PAD]``, ``[UNK]``)
  contribute nothing, so only the five content words shape a distribution.
* Excluding position ``p`` is what makes the rule meaningful for a masked LM:
  the token being predicted never gets to vote for itself, whether or not the
  caller actually replaced it with ``[MASK]``.
"""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

__all__ = [
    "CONTENT_TOKENS",
    "SPECIAL_TOKENS",
    "VOCAB",
    "BASE_LOGITS",
    "AFFINITY",
    "SPECIAL_LOGIT",
    "base_logits",
    "affinity_table",
    "expected_logits",
    "expected_log_softmax",
    "attention_weights",
    "StubTokenizer",
    "StubEncoding",
    "StubMaskedLM",
    "StubFillMaskPipeline",
    "ScriptedCausalLM",
    "TinyCausalLM",
    "RoundTripTokenizer",
    "StubSeq2SeqLM",
    "StubSentenceEncoder",
    "StubOpenAIClient",
]

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
CONTENT_TOKENS = ("he", "she", "nurse", "doctor", "engineer")

VOCAB: Dict[str, int] = {
    tok: i for i, tok in enumerate(SPECIAL_TOKENS + CONTENT_TOKENS)
}
ID_TO_TOKEN: Dict[int, str] = {i: tok for tok, i in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)

# Special tokens are pushed far enough down that they never enter a top-k, but
# not to -inf: log_softmax must stay finite for the pseudo-log-likelihood paths.
SPECIAL_LOGIT = -20.0

#: Unconditional preference for each content word.
BASE_LOGITS: Dict[str, float] = {
    "he": 1.00,
    "she": 0.20,
    "nurse": 0.60,
    "doctor": -0.40,
    "engineer": -1.10,
}

#: What the presence of each content word adds to the logits of the others.
#: Deliberately asymmetric and gendered, so a metric that is supposed to detect
#: a stereotypical association has something to detect, with a known sign.
AFFINITY: Dict[str, Dict[str, float]] = {
    "he": {"nurse": -0.55, "doctor": 1.05, "engineer": 0.80},
    "she": {"nurse": 1.10, "doctor": -0.45, "engineer": -0.30},
    "nurse": {"he": -0.70, "she": 1.30},
    "doctor": {"he": 1.15, "she": -0.65},
    "engineer": {"he": 0.45, "she": -0.35},
}


def base_logits() -> torch.Tensor:
    """``BASE`` as a dense vector over the whole vocabulary."""
    vec = torch.full((VOCAB_SIZE,), SPECIAL_LOGIT, dtype=torch.float32)
    for token, value in BASE_LOGITS.items():
        vec[VOCAB[token]] = value
    return vec


def affinity_table() -> torch.Tensor:
    """``(vocab, vocab)`` table whose row ``t`` is ``AFFINITY[t]``."""
    table = torch.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=torch.float32)
    for context, effects in AFFINITY.items():
        for target, value in effects.items():
            table[VOCAB[context], VOCAB[target]] = value
    return table


def expected_logits(context_tokens: Iterable[str]) -> torch.Tensor:
    """Re-derive the stub's logits from the tables, for use in assertions.

    ``context_tokens`` is the bag of tokens visible *around* the position being
    scored — i.e. the caller has already dropped the token at that position.
    Written independently of :class:`StubMaskedLM` on purpose: a test that
    computed its expectation by calling the model under test would only be
    checking that the model is deterministic.
    """
    vec = base_logits()
    table = affinity_table()
    for token in context_tokens:
        vec = vec + table[VOCAB.get(token, VOCAB["[UNK]"])]
    return vec


def expected_log_softmax(context_tokens: Iterable[str]) -> torch.Tensor:
    return torch.log_softmax(expected_logits(context_tokens), dim=-1)


def attention_weights(seq_len: int) -> torch.Tensor:
    """The fixed attention pattern every stub layer and head reports.

    ``A[i, j] = (j + 1) / sum(1..seq_len)`` — identical for every query row, so
    averaging over layers, heads and queries (which is what the AULA path does)
    returns that same vector and stays trivially checkable by hand.
    """
    keys = torch.arange(1, seq_len + 1, dtype=torch.float32)
    row = keys / keys.sum()
    return row.expand(seq_len, seq_len).contiguous()


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
_STRIP = ".,!?;:\"'()"


def _normalize(chunk: str) -> str:
    chunk = chunk.strip(_STRIP)
    if chunk in VOCAB:  # a special token, or a content word already lowercase
        return chunk
    return chunk.lower()


class StubEncoding(dict):
    """Minimal stand-in for ``transformers.BatchEncoding``.

    Supports the three access styles the library uses interchangeably:
    ``enc["input_ids"]``, ``enc.input_ids`` and ``model(**enc)``.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to(self, device: Any) -> "StubEncoding":
        for key, value in list(self.items()):
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)
        return self


class StubTokenizer:
    """Whitespace tokenizer over the five-word stub vocabulary.

    Every out-of-vocabulary word maps to ``[UNK]``, which contributes nothing to
    the stub's distributions. That is deliberate: it lets a test write a
    natural-looking template such as ``"[MASK] is a nurse"`` and still know that
    only ``nurse`` moves the numbers.
    """

    mask_token = "[MASK]"
    unk_token = "[UNK]"
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    pad_token = "[PAD]"
    eos_token = "[SEP]"

    mask_token_id = VOCAB["[MASK]"]
    unk_token_id = VOCAB["[UNK]"]
    cls_token_id = VOCAB["[CLS]"]
    sep_token_id = VOCAB["[SEP]"]
    pad_token_id = VOCAB["[PAD]"]
    eos_token_id = VOCAB["[SEP]"]

    vocab_size = VOCAB_SIZE
    model_max_length = 512

    def get_vocab(self) -> Dict[str, int]:
        return dict(VOCAB)

    def tokenize(self, text: str) -> List[str]:
        return [t for t in (_normalize(c) for c in str(text).split()) if t]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return VOCAB.get(tokens, self.unk_token_id)
        return [VOCAB.get(t, self.unk_token_id) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return ID_TO_TOKEN.get(ids, self.unk_token)
        return [ID_TO_TOKEN.get(int(i), self.unk_token) for i in ids]

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None,
        **_: Any,
    ):
        ids = self.convert_tokens_to_ids(self.tokenize(text))
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def __call__(
        self,
        text: Any = None,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None,
        padding: Any = False,
        text_target: Any = None,
        **_: Any,
    ) -> StubEncoding:
        if text is None:
            text = text_target
        if text is None:
            raise ValueError("StubTokenizer needs text or text_target.")
        texts = [text] if isinstance(text, str) else list(text)
        rows = [
            self.encode(
                t,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
            )
            for t in texts
        ]
        width = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[i, : len(row)] = 1
        if return_tensors != "pt":
            input_ids = input_ids.tolist()
            attention_mask = attention_mask.tolist()
        return StubEncoding(input_ids=input_ids, attention_mask=attention_mask)

    def decode(self, ids, skip_special_tokens: bool = False, **_: Any) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()
        tokens = [ID_TO_TOKEN.get(int(i), self.unk_token) for i in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return " ".join(tokens)

    def batch_decode(self, sequences, skip_special_tokens: bool = False, **_: Any):
        return [
            self.decode(seq, skip_special_tokens=skip_special_tokens)
            for seq in sequences
        ]


class RoundTripTokenizer:
    """Tokenizer whose ``decode(encode(text))`` returns ``text`` unchanged.

    :class:`StubTokenizer` maps everything outside its five content words to
    ``[UNK]``, which is exactly right for the probability-based metrics and
    exactly wrong for LFP, MCD and NPD: those score the *words* of a generated
    string against French frequency bands, French stems, and the article the
    string was supposedly summarised from. A test can only hand-check them if
    the generation it scripts survives the decode intact.

    So this one grows its vocabulary on demand, one id per whitespace-separated
    chunk, and joins back on single spaces. Punctuation stays glued to its chunk,
    which is harmless: both LFP/MCD (word regex) and NPD (sentence split) do
    their own re-tokenisation of the decoded string.
    """

    mask_token = "[MASK]"
    unk_token = "[UNK]"
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    pad_token = "[PAD]"
    eos_token = "[SEP]"

    mask_token_id = VOCAB["[MASK]"]
    unk_token_id = VOCAB["[UNK]"]
    cls_token_id = VOCAB["[CLS]"]
    sep_token_id = VOCAB["[SEP]"]
    pad_token_id = VOCAB["[PAD]"]
    eos_token_id = VOCAB["[SEP]"]

    model_max_length = 512

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self.vocab_size = capacity
        self._vocab: Dict[str, int] = dict(VOCAB)
        self._inverse: Dict[int, str] = dict(ID_TO_TOKEN)

    def _id_for(self, chunk: str) -> int:
        if chunk not in self._vocab:
            if len(self._vocab) >= self.capacity:
                raise ValueError(
                    f"RoundTripTokenizer is full at {self.capacity} tokens; raise "
                    f"capacity= if a test really needs a larger vocabulary."
                )
            new_id = len(self._vocab)
            self._vocab[chunk] = new_id
            self._inverse[new_id] = chunk
        return self._vocab[chunk]

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    def tokenize(self, text: str) -> List[str]:
        return str(text).split()

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._id_for(tokens)
        return [self._id_for(t) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._inverse.get(ids, self.unk_token)
        return [self._inverse.get(int(i), self.unk_token) for i in ids]

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None,
        **_: Any,
    ):
        ids = self.convert_tokens_to_ids(self.tokenize(text))
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def __call__(
        self,
        text: Any = None,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None,
        padding: Any = False,
        text_target: Any = None,
        **_: Any,
    ) -> StubEncoding:
        if text is None:
            text = text_target
        if text is None:
            raise ValueError("RoundTripTokenizer needs text or text_target.")
        texts = [text] if isinstance(text, str) else list(text)
        rows = [
            self.encode(
                t,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
            )
            for t in texts
        ]
        width = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[i, : len(row)] = 1
        if return_tensors != "pt":
            input_ids = input_ids.tolist()
            attention_mask = attention_mask.tolist()
        return StubEncoding(input_ids=input_ids, attention_mask=attention_mask)

    def decode(self, ids, skip_special_tokens: bool = False, **_: Any) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()
        tokens = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return " ".join(tokens)

    def batch_decode(self, sequences, skip_special_tokens: bool = False, **_: Any):
        return [
            self.decode(seq, skip_special_tokens=skip_special_tokens)
            for seq in sequences
        ]


# ---------------------------------------------------------------------------
# Fixed-distribution models
# ---------------------------------------------------------------------------
class _FixedLogitModel(nn.Module):
    """Shared engine: ``BASE + affinity of every other token in the row``."""

    def __init__(self) -> None:
        super().__init__()
        # A single parameter so `next(model.parameters()).device` works, which is
        # how several metrics discover where to put their tensors.
        self.anchor = nn.Parameter(torch.zeros(1))
        self.register_buffer("base", base_logits())
        self.register_buffer("affinity", affinity_table())

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def fixed_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``(batch, seq, vocab)`` logits under the rule documented above."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        per_position = self.affinity[input_ids]  # (B, L, V)
        row_total = per_position.sum(dim=1, keepdim=True)  # (B, 1, V)
        return self.base + row_total - per_position


class _StubAttentionEncoder(nn.Module):
    """Sub-encoder that reports :func:`attention_weights` for every layer/head."""

    def __init__(self, n_layers: int, n_heads: int, hidden_size: int) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.attention_requests: List[bool] = []

    def forward(self, input_ids=None, output_attentions: bool = False, **_: Any):
        self.attention_requests.append(bool(output_attentions))
        seq_len = input_ids.shape[-1]
        hidden = torch.zeros(input_ids.shape[0], seq_len, self.hidden_size)
        attentions = None
        if output_attentions:
            single = attention_weights(seq_len)
            attentions = tuple(
                single.expand(
                    input_ids.shape[0], self.n_heads, seq_len, seq_len
                ).contiguous()
                for _ in range(self.n_layers)
            )
        return SimpleNamespace(last_hidden_state=hidden, attentions=attentions)


class StubMaskedLM(_FixedLogitModel):
    """Masked LM with a hand-authored distribution and fixed attention.

    Parameters
    ----------
    expose_encoder:
        Whether to publish a ``.bert`` sub-encoder. Set ``False`` to reach the
        branch that refuses ``use_attention=True`` on a model that cannot report
        attentions.
    attn_implementation:
        Value for ``config._attn_implementation``. Anything other than
        ``"eager"`` makes the scoring helpers ask for eager attention first,
        because fused kernels silently drop the weights.
    """

    def __init__(
        self,
        n_layers: int = 2,
        n_heads: int = 2,
        hidden_size: int = 8,
        expose_encoder: bool = True,
        attn_implementation: str = "eager",
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=VOCAB_SIZE,
            hidden_size=hidden_size,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            _attn_implementation=attn_implementation,
        )
        self.attn_implementation_calls: List[str] = []
        if expose_encoder:
            self.bert = _StubAttentionEncoder(n_layers, n_heads, hidden_size)

    def set_attn_implementation(self, name: str) -> None:
        self.attn_implementation_calls.append(name)
        self.config._attn_implementation = name

    def attention_was_requested(self) -> bool:
        """Whether any forward pass so far asked the encoder for attentions.

        Lets a test distinguish AUL from AULA by observation rather than by
        trusting that a ``use_attention`` flag was plumbed through.
        """
        encoder = getattr(self, "bert", None)
        return bool(encoder is not None and any(encoder.attention_requests))

    def forward(
        self, input_ids=None, attention_mask=None, output_attentions=False, **_: Any
    ):
        attentions = None
        if output_attentions and hasattr(self, "bert"):
            attentions = self.bert(
                input_ids=input_ids, output_attentions=True
            ).attentions
        return SimpleNamespace(
            logits=self.fixed_logits(input_ids), attentions=attentions
        )


class StubFillMaskPipeline:
    """Fill-mask pipeline over :class:`StubMaskedLM`, for DisCo.

    Callable and carrying a ``.tokenizer``, which is how the DisCo wrapper
    recognises an already-built pipeline and skips constructing a real one.
    """

    task = "fill-mask"

    def __init__(
        self,
        model: Optional[StubMaskedLM] = None,
        tokenizer: Optional[StubTokenizer] = None,
    ) -> None:
        self.model = model if model is not None else StubMaskedLM()
        self.tokenizer = tokenizer if tokenizer is not None else StubTokenizer()

    def __call__(self, sentence: str, top_k: int = 5) -> List[Dict[str, Any]]:
        input_ids = self.tokenizer.encode(sentence, return_tensors="pt")
        positions = (input_ids == self.tokenizer.mask_token_id).nonzero()
        if positions.numel() == 0:
            raise ValueError(f"No [MASK] token in {sentence!r}")
        pos = int(positions[0, 1])
        with torch.no_grad():
            probs = self.model(input_ids).logits[0, pos].softmax(dim=-1)
        k = min(top_k, probs.numel())
        scores, ids = torch.topk(probs, k)
        return [
            {
                "token": int(i),
                "token_str": ID_TO_TOKEN[int(i)],
                "score": float(s),
                "sequence": sentence.replace("[MASK]", ID_TO_TOKEN[int(i)], 1),
            }
            for s, i in zip(scores, ids)
        ]


class ScriptedCausalLM(_FixedLogitModel):
    """Causal LM whose next-token logits are fixed and whose text is scripted.

    The metrics that consume *generated strings* (DRD, CA) do not care how the
    text was produced, only what it says, so ``generate`` replays a list of
    continuations instead of sampling. Everything about those metrics — the
    mention counts, the total-variation distance — then has an exact expected
    value. ``do_sample``, ``temperature`` and ``top_p`` are accepted and ignored
    for the same reason.
    """

    def __init__(self, continuations: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=VOCAB_SIZE,
            n_layer=2,
            n_head=2,
            n_embd=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            hidden_size=8,
            is_encoder_decoder=False,
        )
        self.continuations = list(continuations or ["he doctor", "she nurse"])
        self._cycle = itertools.cycle(self.continuations)
        self.tokenizer = StubTokenizer()
        self.generate_calls: List[str] = []

    def forward(self, input_ids=None, attention_mask=None, **_: Any):
        logits = self.fixed_logits(input_ids)
        return SimpleNamespace(logits=logits, hidden_states=None, past_key_values=None)

    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_new_tokens: int = 20,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        **_: Any,
    ) -> torch.Tensor:
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        prompt = input_ids[0]
        pad = self.tokenizer.pad_token_id if pad_token_id is None else pad_token_id

        continuations = []
        for _i in range(num_return_sequences):
            text = next(self._cycle)
            self.generate_calls.append(text)
            ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
            continuations.append(ids[:max_new_tokens])

        # Real `generate` returns one rectangular tensor, padding the shorter
        # sequences in the batch. Callers slice it at the prompt length and then
        # decode with `skip_special_tokens=True`, which drops the padding again —
        # so the padding has to be there for that slice to line up.
        width = max((len(c) for c in continuations), default=0)
        rows = [
            torch.cat(
                [
                    prompt,
                    torch.tensor(c + [pad] * (width - len(c)), dtype=torch.long),
                ]
            )
            for c in continuations
        ]
        return torch.stack(rows)


# ---------------------------------------------------------------------------
# A real (tiny) GPT-2-shaped decoder, for the metrics that install hooks
# ---------------------------------------------------------------------------
class _TinyAttention(nn.Module):
    """Causal self-attention laid out like GPT-2's, because the hooks say so.

    NIE captures and overwrites the *input* to ``attn.c_proj``; GBE rewrites the
    *output* of ``attn.c_attn`` to mask Value heads and then differentiates
    through it. Both therefore need the real module names, the packed
    ``(q, k, v)`` projection, and an unbroken autograd path from the mask to the
    logits — which a scripted stub cannot provide.
    """

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        q, k, v = self.c_attn(x).split(dim, dim=2)
        shape = (batch, seq, self.n_head, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
        weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        merged = (weights @ v).transpose(1, 2).reshape(batch, seq, dim)
        return self.c_proj(merged)


class _TinyBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = _TinyAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd), nn.GELU(), nn.Linear(2 * n_embd, n_embd)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class _TinyTransformer(nn.Module):
    def __init__(
        self, vocab_size: int, n_positions: int, n_embd: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.h = nn.ModuleList(_TinyBlock(n_embd, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, input_ids: torch.Tensor):
        seq = input_ids.shape[1]
        positions = torch.arange(seq, device=input_ids.device)
        hidden = self.wte(input_ids) + self.wpe(positions)
        states = [hidden]
        for block in self.h:
            hidden = block(hidden)
            states.append(hidden)
        hidden = self.ln_f(hidden)
        states[-1] = hidden
        return hidden, tuple(states)


class TinyCausalLM(nn.Module):
    """Two-layer, two-head GPT-2-shaped decoder with deterministic weights.

    Small enough to run a full NIE head sweep in milliseconds, and real enough
    that the forward hooks, the in-place activation patch and the backward pass
    through the Value mask all have to actually work.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        n_layer: int = 2,
        n_head: int = 2,
        head_dim: int = 4,
        n_positions: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        n_embd = n_head * head_dim
        self.transformer = _TinyTransformer(
            vocab_size, n_positions, n_embd, n_head, n_layer
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_positions=n_positions,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            hidden_size=n_embd,
            _attn_implementation="eager",
            is_encoder_decoder=False,
        )
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for param in self.parameters():
                param.copy_(
                    torch.randn(param.shape, generator=generator, dtype=param.dtype)
                    * 0.3
                )

    @property
    def device(self) -> torch.device:
        return self.lm_head.weight.device

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        output_hidden_states: bool = False,
        **_: Any,
    ):
        hidden, states = self.transformer(input_ids)
        return SimpleNamespace(
            logits=self.lm_head(hidden),
            hidden_states=states if output_hidden_states else None,
        )


# ---------------------------------------------------------------------------
# Encoder-decoder stubs
# ---------------------------------------------------------------------------
class _StubSelfAttention(nn.Module):
    """Stands in for ``MT5Attention``: the module SVA hangs its head mask on."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, hidden_states, **_: Any):
        return (hidden_states * self.scale,)


class _StubBlockLayer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.SelfAttention = _StubSelfAttention(d_model)

    def forward(self, hidden_states, **_: Any):
        return self.SelfAttention(hidden_states)


class _StubBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.layer = nn.ModuleList([_StubBlockLayer(d_model)])

    def forward(self, hidden_states, **_: Any):
        return self.layer[0](hidden_states)[0]


class _StubEncoder(nn.Module):
    """Encoder whose sentence embedding is a deterministic function of its ids.

    Sentences that share a token bag get the same embedding, so a test can
    reason about which pairs a downstream classifier or cosine similarity is
    able to separate.
    """

    def __init__(
        self, d_model: int, n_layers: int, vocab_size: int = VOCAB_SIZE
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.block = nn.ModuleList(_StubBlock(d_model) for _ in range(n_layers))
        self.embedding = nn.Embedding(vocab_size, d_model)
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.randn(self.embedding.weight.shape, generator=generator) * 0.5
            )

    def forward(self, input_ids=None, attention_mask=None, **_: Any):
        hidden = self.embedding(input_ids)
        for block in self.block:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=None)


class StubSeq2SeqLM(nn.Module):
    """Seq2seq stub: scripted generations, deterministic encoder, fixed loss.

    ``generate`` replays a caller-supplied script (optionally keyed by a
    substring of the prompt) because every metric built on it — LFP, MCD, NPD,
    the translation similarity score — scores the *text*, not the sampling.
    ``forward(labels=...)`` reports a loss derived from the candidate's token
    bag, which is what SD's forced-choice comparison needs. When the candidates
    are words outside the stub vocabulary — SD's French pronoun cues, IBS's
    ``Yes``/``No``/``Maybe`` — they all collapse to ``[UNK]`` and the token bag
    can no longer tell them apart, so ``scripted_losses`` overrides it with a
    loss per ``forward`` call in the order the metric makes them.
    """

    def __init__(
        self,
        generations: Optional[Sequence[str]] = None,
        by_prompt: Optional[Dict[str, str]] = None,
        candidate_losses: Optional[Dict[str, float]] = None,
        scripted_losses: Optional[Sequence[float]] = None,
        tokenizer: Any = None,
        d_model: int = 8,
        n_layers: int = 2,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer if tokenizer is not None else StubTokenizer()
        vocab_size = int(getattr(self.tokenizer, "vocab_size", VOCAB_SIZE))
        self.encoder = _StubEncoder(d_model, n_layers, vocab_size=vocab_size)
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            d_model=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            num_hidden_layers=n_layers,
            num_heads=n_heads,
            num_attention_heads=n_heads,
            vocab_size=vocab_size,
            is_encoder_decoder=True,
        )
        self.generations = list(generations or ["he doctor"])
        self.by_prompt = dict(by_prompt or {})
        self.candidate_losses = dict(candidate_losses or {})
        self._cycle = itertools.cycle(self.generations)
        self._loss_cycle = (
            itertools.cycle([float(v) for v in scripted_losses])
            if scripted_losses
            else None
        )
        self.prompts_seen: List[str] = []
        self.labels_seen: List[List[int]] = []

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def get_encoder(self):
        return self.encoder

    def forward(self, input_ids=None, attention_mask=None, labels=None, **_: Any):
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        loss = None
        if labels is not None:
            ids = labels.reshape(-1).tolist()
            self.labels_seen.append(ids)
            tokens = [ID_TO_TOKEN.get(int(i), "[UNK]") for i in ids]
            # Cross-entropy is a positive quantity, so a *preferred* candidate
            # must come out with a *lower* loss. Deriving it from the candidate's
            # own base logits keeps that direction honest, and keeps SD's
            # `-loss` comparison meaningful rather than arbitrary.
            content = [t for t in tokens if t in BASE_LOGITS]
            if self._loss_cycle is not None:
                loss = torch.tensor(next(self._loss_cycle), dtype=torch.float32)
            elif self.candidate_losses:
                key = " ".join(content) or " ".join(tokens)
                loss = torch.tensor(self.candidate_losses.get(key, 1.0))
            else:
                penalty = sum(-BASE_LOGITS[t] for t in content) / max(1, len(content))
                loss = torch.tensor(2.0 + penalty, dtype=torch.float32)
        return SimpleNamespace(
            loss=loss,
            logits=None,
            encoder_last_hidden_state=encoder_out.last_hidden_state,
        )

    def generate(self, input_ids=None, attention_mask=None, **_: Any) -> torch.Tensor:
        prompt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        self.prompts_seen.append(prompt)
        text = next((v for k, v in self.by_prompt.items() if k in prompt), None)
        if text is None:
            text = next(self._cycle)
        ids = self.tokenizer.encode(text, return_tensors="pt")
        return ids


class StubSentenceEncoder(nn.Module):
    """LaBSE-shaped sentence encoder: ``model(**inputs).last_hidden_state``."""

    def __init__(self, d_model: int = 8, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.encoder = _StubEncoder(d_model, n_layers=1, vocab_size=vocab_size)
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(hidden_size=d_model)

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def forward(self, input_ids=None, attention_mask=None, **_: Any):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)


# ---------------------------------------------------------------------------
# OpenAI Completions stub
# ---------------------------------------------------------------------------
class _StubCompletions:
    def __init__(self, owner: "StubOpenAIClient") -> None:
        self._owner = owner

    def create(self, *, model: str, prompt: str, **kwargs: Any):
        return self._owner._create(model=model, prompt=prompt, **kwargs)


class StubOpenAIClient:
    """Scripted ``client.completions.create`` for the OpenAI-backed metrics.

    Three metrics read three different corners of the response, so all three are
    scripted independently:

    ``token_scores``
        Per-word log-probabilities, echoed back with character offsets. This is
        what ``continuation_logprob`` averages over, and hence what drives
        BiasAsker's forced choice.
    ``token_score_fn``
        ``(word, prompt) -> score or None``, consulted before ``token_scores``.
        Needed for anything order-sensitive: BiasAsker asks the same question
        with the two groups swapped, so a client that scored a word the same way
        regardless of prompt could never produce a disagreement between the two
        orderings.
    ``top_tokens``
        Substring → greedy first token, for CR's top-1 comparison.
    ``distributions``
        Substring → ``{token: probability}``, for CTF's total-variation distance.
    """

    def __init__(
        self,
        token_scores: Optional[Dict[str, float]] = None,
        default_token_score: float = -5.0,
        top_tokens: Optional[Dict[str, str]] = None,
        distributions: Optional[Dict[str, Dict[str, float]]] = None,
        token_score_fn: Optional[Any] = None,
    ) -> None:
        self.completions = _StubCompletions(self)
        self.token_scores = dict(token_scores or {})
        self.default_token_score = default_token_score
        self.top_tokens = dict(top_tokens or {})
        self.distributions = dict(distributions or {})
        self.token_score_fn = token_score_fn
        self.calls: List[Dict[str, Any]] = []

    def _match(self, table: Dict[str, Any], prompt: str):
        for key, value in table.items():
            if key in prompt:
                return value
        return None

    def _word_score(self, word: str, prompt: str) -> float:
        if self.token_score_fn is not None:
            score = self.token_score_fn(word, prompt)
            if score is not None:
                return score
        return self.token_scores.get(word, self.default_token_score)

    def _echo_logprobs(self, prompt: str):
        """Echo the prompt back word by word, with character offsets.

        ``continuation_logprob`` keeps only the tokens at or past
        ``len(prompt_without_continuation)``, so word-level offsets are enough to
        make the continuation's score exactly the mean over its own words.
        """
        tokens, offsets, logprobs = [], [], []
        cursor = 0
        for word in prompt.split(" "):
            if word:
                tokens.append(word)
                offsets.append(cursor)
                logprobs.append(self._word_score(word.strip(".?,!"), prompt))
            cursor += len(word) + 1
        return SimpleNamespace(
            tokens=tokens,
            text_offset=offsets,
            token_logprobs=logprobs,
            top_logprobs=None,
        )

    def _create(self, *, model: str, prompt: str, **kwargs: Any):
        self.calls.append({"model": model, "prompt": prompt, **kwargs})

        if kwargs.get("echo"):
            return SimpleNamespace(
                choices=[SimpleNamespace(text="", logprobs=self._echo_logprobs(prompt))]
            )

        dist = self._match(self.distributions, prompt)
        if dist is not None:
            top = {tok: math.log(p) for tok, p in dist.items()}
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        text=max(dist, key=dist.get),
                        logprobs=SimpleNamespace(
                            tokens=[max(dist, key=dist.get)],
                            token_logprobs=[max(top.values())],
                            text_offset=[len(prompt)],
                            top_logprobs=[top],
                        ),
                    )
                ]
            )

        token = self._match(self.top_tokens, prompt) or ""
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    text=token,
                    logprobs=SimpleNamespace(
                        tokens=[token],
                        token_logprobs=[self.default_token_score],
                        text_offset=[len(prompt)],
                        top_logprobs=None,
                    ),
                )
            ]
        )
