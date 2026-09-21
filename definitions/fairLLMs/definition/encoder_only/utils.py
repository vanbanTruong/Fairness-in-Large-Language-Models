import numpy as np
import scipy.special
import itertools
import torch
import math
import logging
import difflib

log = logging.getLogger(__name__)

#-------------------------------------------------------------------------------
# WEAT, SEAT, CEAT Helpers
#-------------------------------------------------------------------------------
def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        raise ValueError("Cannot compute cosine similarity with a zero vector")
    return np.dot(a, b) / denom

def association(word_vec, A_vecs, B_vecs):
    """s(w, A, B)"""
    s_A = np.mean([cosine_similarity(word_vec, a) for a in A_vecs])
    s_B = np.mean([cosine_similarity(word_vec, b) for b in B_vecs])
    return s_A - s_B

def association_vectorized(word_vec, A_vecs, B_vecs):
    """Vectorized s(w, A, B)"""
    A = np.array(A_vecs)  # shape (|A|, d)
    B = np.array(B_vecs)  # shape (|B|, d)
    w = word_vec / np.linalg.norm(word_vec)
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return (w @ A_norm.T).mean() - (w @ B_norm.T).mean()

def cohens_d(s_T1, s_T2):
    all_s = list(s_T1) + list(s_T2)
    return (np.mean(s_T1) - np.mean(s_T2)) / np.std(all_s, ddof=1)

def permutation_pval(s_T1, s_T2, n_samples=10_000):
    s_T1 = np.array(s_T1, dtype=np.float64)
    s_T2 = np.array(s_T2, dtype=np.float64)
    n = len(s_T1)
    assert len(s_T1) == len(s_T2), "Target sets must be the same size"

    combined = np.concatenate([s_T1, s_T2])
    observed = s_T1.sum()  # reduced test statistic

    num_partitions = int(scipy.special.binom(2 * n, n))

    if num_partitions <= n_samples:
        # Exact test over all C(2n, n) even partitions
        count = 0
        total = 0
        for Xi_idx in itertools.combinations(range(2 * n), n):
            si = combined[list(Xi_idx)].sum()
            if si >= observed:
                count += 1
            total += 1
        return count / total
    else:
        count = 1
        total = 1
        rng_combined = combined.copy()
        for _ in range(n_samples - 1):
            np.random.shuffle(rng_combined)
            si = rng_combined[:n].sum()
            if si >= observed:
                count += 1
            total += 1
        return count / total

def encode_sentence(model, tokenizer, text, pooling="mean", device="cpu"):
    """Encode a sentence (or batch) to a single vector using a transformer encoder."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state
    if pooling == "cls":
        return hidden[:, 0, :].squeeze().cpu().numpy()
    elif pooling == "mean":
        mask = inputs["attention_mask"].unsqueeze(-1).type_as(hidden)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return (summed / counts).squeeze().cpu().numpy()
    else:
        raise ValueError(f"unknown pooling '{pooling}'")
    
def encode_in_context(model, tokenizer, word, contexts, pooling = "cls", device = "cpu"):
    vecs = []
    for sentence in contexts:
        vec = encode_sentence(model, tokenizer, sentence, pooling)
        vecs.append(vec)
    return np.array(vecs)  # shape (n_contexts, d)

def ceat_effect_size(effect_sizes, sample_sizes):
    weights = np.array(sample_sizes)
    return np.average(effect_sizes, weights=weights)

#-------------------------------------------------------------------------------
# DisCo, LPBS, CBS Helpers
#-------------------------------------------------------------------------------

def get_top_k_predictions(pipe, sentence, k=3):
        preds = pipe(sentence, top_k=k)
        if preds and isinstance(preds[0], list):
            preds = preds[0]
        return {p["token_str"]: p["score"] for p in preds}

def get_mask_fill_probs(sentence, tokenizer, model, mask_position=0):
    
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    mask_positions = torch.where(input_ids == tokenizer.mask_token_id)[1]

    if len(mask_positions) == 0:
        raise ValueError(f"No [MASK] token found in sentence: {sentence!r}")

    pos = mask_positions[mask_position].item()

    with torch.no_grad():
        logits = model(input_ids, return_dict=True).logits

    return logits[0, pos, :].softmax(dim=0)


def get_token_prob(probs, token, tokenizer):
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid == tokenizer.unk_token_id:
        log.warning("Token '%s' is unknown to the tokenizer; using epsilon.", token)
        return 1e-10
    return probs[tid].item()

def get_multitoken_log_prob(sentence, term, tokenizer, model):
    
    sub_tokens = tokenizer.tokenize(term)
    if not sub_tokens:
        log.warning("Term '%s' tokenizes to nothing; returning log(epsilon).", term)
        return math.log(1e-10)

    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    mask_positions = torch.where(input_ids == tokenizer.mask_token_id)[1]

    if len(mask_positions) != len(sub_tokens):
        raise ValueError(
            f"Term '{term}' has {len(sub_tokens)} sub-tokens but sentence "
            f"has {len(mask_positions)} [MASK] tokens. Use build_masked_sentence "
            f"to construct the sentence correctly."
        )

    with torch.no_grad():
        logits = model(input_ids, return_dict=True).logits

    log_probs = []
    for pos, sub_token in zip(mask_positions, sub_tokens):
        tid = tokenizer.convert_tokens_to_ids(sub_token)
        if tid == tokenizer.unk_token_id:
            log.warning(
                "Sub-token '%s' of term '%s' is unknown; using epsilon.", sub_token, term
            )
            log_probs.append(math.log(1e-10))
        else:
            prob = logits[0, pos.item(), :].softmax(dim=0)[tid].item()
            log_probs.append(math.log(prob + 1e-10))

    return float(np.mean(log_probs))


def build_masked_sentence(template, term_placeholder, term, tokenizer):
    
    n_tokens = len(tokenizer.tokenize(term))
    mask_str = " ".join([tokenizer.mask_token] * n_tokens)
    return template.replace(term_placeholder, mask_str)

#-------------------------------------------------------------------------------
# AUL, AULA, ICAT, CAT, CPS, PLL Helpers
#-------------------------------------------------------------------------------

def get_token_ranks(log_probs, token_ids):
    
    ranks = []
    for i in range(log_probs.shape[0]):
        # Sort tokens by descending log-probability
        sorted_indices = torch.argsort(log_probs[i], descending=True)
        gold_id = token_ids[i].item()
        rank = (sorted_indices == gold_id).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    return ranks


def score_sentence(tokenizer, model, sentence, use_attention=False):
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits.squeeze(0)
        log_probs = torch.log_softmax(logits, dim=-1)
        token_ids = input_ids.view(-1, 1).detach()
        token_log_probs = log_probs.gather(1, token_ids)[1:-1].squeeze(1)

        if use_attention:
            bert_output = model.bert(input_ids=input_ids, output_attentions=True)
            all_attentions = torch.stack([a.squeeze(0) for a in bert_output.attentions], dim=0)
            mean_attention = all_attentions.mean(dim=(0, 1))
            token_attention = mean_attention.mean(dim=0)
            token_log_probs = token_log_probs * token_attention[1:-1]

    score = torch.mean(token_log_probs).item()
    ranks = get_token_ranks(log_probs[1:-1], token_ids[1:-1])
    return score, ranks

def get_span(tokens1, tokens2, operation):
    
    tokens1 = [str(x) for x in tokens1.tolist()]
    tokens2 = [str(x) for x in tokens2.tolist()]

    matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        if (operation == 'equal' and op[0] == 'equal') \
                or (operation == 'diff' and op[0] != 'equal'):
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2
