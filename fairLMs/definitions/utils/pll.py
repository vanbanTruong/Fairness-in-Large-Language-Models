"""Pseudo-log-likelihood scoring helpers (AUL, AULA, CAT, CPS, PLL)."""

from __future__ import annotations

import difflib

import torch


def _input_device(model):
    embedding = (
        model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    )
    device = (
        embedding.weight.device
        if embedding is not None
        else next(model.parameters()).device
    )
    if device.type == "meta":
        raise ValueError("Scoring does not support meta-offloaded input embeddings.")
    return device


def get_token_ranks(log_probs, token_ids):
    ranks = []
    for i in range(log_probs.shape[0]):
        sorted_indices = torch.argsort(log_probs[i], descending=True)
        gold_id = token_ids[i].item()
        rank = (sorted_indices == gold_id).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    return ranks


def score_sentence(tokenizer, model, sentence, use_attention=False, *, context=None):
    input_ids, positions = scoring_input(tokenizer, model, sentence, context=context)
    previous_backend = getattr(
        getattr(model, "config", None), "_attn_implementation", None
    )
    switch_backend = (
        use_attention
        and previous_backend not in (None, "eager")
        and hasattr(model, "set_attn_implementation")
    )
    if switch_backend:
        model.set_attn_implementation("eager")
    try:
        with torch.no_grad():
            output = (
                model(input_ids, output_attentions=True)
                if use_attention
                else model(input_ids)
            )
    finally:
        if switch_backend:
            model.set_attn_implementation(previous_backend)
    with torch.no_grad():
        logits = output.logits.squeeze(0)
        log_probs = torch.log_softmax(logits, dim=-1)
        token_ids = input_ids.view(-1, 1).detach().to(log_probs.device)
        token_log_probs = log_probs.gather(1, token_ids)[positions].squeeze(1)

        if use_attention:
            attentions = getattr(output, "attentions", None)
            if not attentions or any(a is None for a in attentions):
                raise AttributeError(
                    "use_attention=True requires model(..., output_attentions=True) "
                    "to return attention tensors; load a supported model with eager attention."
                )
            all_attentions = torch.stack([a.squeeze(0) for a in attentions], dim=0)
            mean_attention = all_attentions.mean(dim=(0, 1))
            token_attention = mean_attention.mean(dim=0)
            token_log_probs = token_log_probs * token_attention[positions]

    score = torch.mean(token_log_probs).item()
    ranks = get_token_ranks(log_probs[positions], token_ids[positions])
    return score, ranks


def get_span(tokens1, tokens2, operation):
    tokens1 = [str(x) for x in tokens1.tolist()]
    tokens2 = [str(x) for x in tokens2.tolist()]

    matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        if (operation == "equal" and op[0] == "equal") or (
            operation == "diff" and op[0] != "equal"
        ):
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2


def scoring_input(tokenizer, model, sentence, *, context=None):
    """Tokenize once; return device-correct IDs and scoreable token positions.

    A supplied context stays visible but is never masked/scored. Fast tokenizers
    use character offsets; slow tokenizers must have a stable prefix boundary.
    Overlong evidence is refused instead of silently losing the comparison.
    """
    prefix = context.rstrip() + " " if context is not None else ""
    text = prefix + sentence
    device = _input_device(model)
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(limit, int) and ids.shape[1] > limit:
        raise ValueError(
            f"Evidence has {ids.shape[1]} tokens; model limit is {limit}. Shorten the evidence explicitly."
        )
    tokens = ids[0].tolist()
    if hasattr(tokenizer, "get_special_tokens_mask"):
        special = tokenizer.get_special_tokens_mask(
            tokens, already_has_special_tokens=True
        )
        positions = [i for i, flag in enumerate(special) if not flag]
    else:
        positions = list(range(1, len(tokens) - 1))
    if context is not None:
        if getattr(tokenizer, "is_fast", False):
            offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
            if len(offsets) != len(tokens):
                raise ValueError(
                    "Tokenizer offset mapping disagrees with encoded tokens."
                )
            if any(start < len(prefix.rstrip()) < end for start, end in offsets):
                raise ValueError("A token crosses the context/candidate boundary.")
            positions = [i for i in positions if offsets[i][1] > len(prefix)]
        else:
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            if hasattr(prefix_ids, "tolist"):
                prefix_ids = prefix_ids.tolist()
            content = [tokens[i] for i in positions]
            if content[: len(prefix_ids)] != list(prefix_ids):
                raise ValueError(
                    "Context token boundary is unstable; use a fast tokenizer with offsets."
                )
            positions = positions[len(prefix_ids) :]
    if not positions:
        raise ValueError("No scoreable candidate tokens remain.")
    return ids, positions


def masked_position_score(tokenizer, model, input_ids, positions, *, batch_size=16):
    """Sum masked-token log probabilities using bounded mask batches."""
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")
    if not positions:
        return 0.0, [-1]
    values, ranks = [], []
    for start in range(0, len(positions), batch_size):
        spans = torch.tensor(
            positions[start : start + batch_size], device=input_ids.device
        )
        batch = input_ids.repeat(len(spans), 1)
        rows = torch.arange(len(spans), device=input_ids.device)
        batch[rows, spans] = tokenizer.mask_token_id
        with torch.no_grad():
            output = model(batch).logits
            logits = output[rows.to(output.device), spans.to(output.device), :]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
        gold = input_ids[0, spans].to(log_probs.device)
        values.extend(log_probs[rows.to(log_probs.device), gold].tolist())
        ranks.extend(get_token_ranks(log_probs, gold.view(-1, 1)))
    return sum(values), ranks
