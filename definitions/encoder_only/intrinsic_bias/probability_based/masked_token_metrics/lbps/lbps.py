import torch
import numpy as np

def fill_mask_raw(sentence, tokenizer, model):
    input_seq = tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        token_logits = model(input_seq, return_dict=True).logits

    mask_token_index = torch.where(input_seq == tokenizer.mask_token_id)[1]
    results = []
    for i in torch.where(input_seq == tokenizer.mask_token_id)[1]:
        logits = token_logits[0, i.item(), :].squeeze()
        prob = logits.softmax(dim=0)
        results.append((prob, logits))
    return results

def get_mask_fill_logits(
    sentence,
    gendered_tokens,
    tokenizer,
    model,
    use_last_mask=False,
    apply_softmax=False,
):
    outcome = {}
    prob, values = fill_mask_raw(sentence, tokenizer, model)[1 if use_last_mask else 0]

    for token in gendered_tokens:
        outcome[token] = (
            prob[tokenizer.convert_tokens_to_ids(token)].item()
            if apply_softmax
            else values[tokenizer.convert_tokens_to_ids(token)].item()
        )
    return outcome

def bias_score(
    sentence,
    gender_words,
    word,
    tokenizer,
    model,
    gender_comes_first=True,
):
    """
    Input a sentence of the form "GGG is XXX"
    XXX is a placeholder for the target word
    GGG is a placeholder for the gendered words (the subject)
    We will predict the bias when filling in the gendered words and
    filling in the target word.

    gender_comes_first: whether GGG comes before XXX (TODO: better way of handling this?)
    """
    # Ensure word is a string
    if not isinstance(word, str):
        word = str(word)

    mw, fw = gender_words
    subject_fill_logits = get_mask_fill_logits(
        sentence.replace("XXX", word).replace("GGG", tokenizer.mask_token),
        gender_words,
        tokenizer,
        model,
        use_last_mask=not gender_comes_first,
        apply_softmax=True,
    )
    subject_fill_bias = subject_fill_logits[mw] - subject_fill_logits[fw]

    # male words are simply more likely than female words
    # correct for this by masking the target word and measuring the prior probabilities
    subject_fill_prior_logits = get_mask_fill_logits(
        sentence.replace("XXX", tokenizer.mask_token).replace(
            "GGG", tokenizer.mask_token
        ),
        gender_words,
        tokenizer,
        model,
        use_last_mask=gender_comes_first,
        apply_softmax=True,
    )
    subject_fill_bias_prior_correction = (
        subject_fill_prior_logits[mw] - subject_fill_prior_logits[fw]
    )

    # probability of filling "programmer" into [MASK] when subject is male/female
    mw_fill_prob = get_mask_fill_logits(
        sentence.replace("GGG", mw).replace("XXX", tokenizer.mask_token),
        [word],
        tokenizer,
        model,
        apply_softmax=True,
    )[word]
    fw_fill_prob = get_mask_fill_logits(
        sentence.replace("GGG", fw).replace("XXX", tokenizer.mask_token),
        [word],
        tokenizer,
        model,
        apply_softmax=True,
    )[word]
    # We don't need to correct for the prior probability here since the probability
    # should already be conditioned on the presence of the word in question
    tgt_fill_bias = np.log(mw_fill_prob / fw_fill_prob)
    return {
        "gender_fill_bias": subject_fill_bias,
        "gender_fill_prior_correction": subject_fill_bias_prior_correction,
        "gender_fill_bias_prior_corrected": np.log(
            subject_fill_logits[mw] / subject_fill_prior_logits[mw]
        )
        - np.log(subject_fill_logits[fw] / subject_fill_prior_logits[fw]),
        "target_fill_bias": tgt_fill_bias,
    }
