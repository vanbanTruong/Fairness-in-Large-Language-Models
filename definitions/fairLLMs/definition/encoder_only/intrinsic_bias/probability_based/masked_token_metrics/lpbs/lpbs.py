import math
import numpy as np
import torch
from fairLLMs.definition.encoder_only.utils import get_mask_fill_probs, get_token_prob, build_masked_sentence

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
        return 1e-10
    return probs[tid].item()


def build_masked_sentence(template, term_placeholder, term, tokenizer):
    
    n_tokens = len(tokenizer.tokenize(term))
    mask_str = " ".join([tokenizer.mask_token] * n_tokens)
    return template.replace(term_placeholder, mask_str)

DEFAULT_TEMPLATE = "{Group} is a {Attribute}"

def _attribute_bias_score(attribute, gender_words, template, tokenizer, model,
                          gender_comes_first=True):

    mw, fw = gender_words

    if not isinstance(attribute, str):
        attribute = str(attribute)

    sentence_attr = build_masked_sentence(
    template.replace("XXX", attribute),
    "GGG", mw, tokenizer
)
    
    probs_attr = get_mask_fill_probs(
        sentence_attr, tokenizer, model,
        mask_position=0 if gender_comes_first else -1,
    )
    p_mw_attr = get_token_prob(probs_attr, mw, tokenizer)
    p_fw_attr = get_token_prob(probs_attr, fw, tokenizer)

    gender_fill_bias = p_mw_attr - p_fw_attr
    
    temp_template = template.replace("XXX", tokenizer.mask_token)
    sentence_prior = build_masked_sentence(temp_template, "GGG", mw, tokenizer)
    probs_prior = get_mask_fill_probs(
    sentence_prior, tokenizer, model,
    mask_position=0 if gender_comes_first else -1, 
)
    p_mw_prior = get_token_prob(probs_prior, mw, tokenizer)
    p_fw_prior = get_token_prob(probs_prior, fw, tokenizer)

    gender_fill_prior_correction = p_mw_prior - p_fw_prior

    lpbs = (
        math.log((p_mw_attr + 1e-10) / (p_mw_prior + 1e-10))
        - math.log((p_fw_attr + 1e-10) / (p_fw_prior + 1e-10))
    )

    sentence_mw = template.replace("GGG", mw).replace("XXX", tokenizer.mask_token)
    sentence_fw = template.replace("GGG", fw).replace("XXX", tokenizer.mask_token)

    probs_mw = get_mask_fill_probs(sentence_mw, tokenizer, model, mask_position=0)
    probs_fw = get_mask_fill_probs(sentence_fw, tokenizer, model, mask_position=0)

    p_attr_mw = get_token_prob(probs_mw, attribute, tokenizer)
    p_attr_fw = get_token_prob(probs_fw, attribute, tokenizer)

    target_fill_bias = math.log((p_attr_mw + 1e-10) / (p_attr_fw + 1e-10))

    return {
        "gender_fill_bias": gender_fill_bias,
        "gender_fill_prior_correction": gender_fill_prior_correction,
        "gender_fill_bias_prior_corrected": lpbs,
        "target_fill_bias": target_fill_bias,
    }


def compute_lpbs(tokenizer, model, gender_words, attribute_words,
                 template=DEFAULT_TEMPLATE, gender_comes_first=True):
    
    if len(gender_words) != 2:
        raise ValueError(
            f"gender_words must contain exactly two tokens, got {len(gender_words)}."
        )
    if list(gender_words)[0] == list(gender_words)[1]:
        raise ValueError("gender_words must be two distinct tokens.")

    outcomes = []

    for attribute in attribute_words:
        scores = _attribute_bias_score(
            attribute, gender_words, template, tokenizer, model, gender_comes_first
        )
        scores["attribute"] = attribute
        outcomes.append(scores)

    lpbs_scores = [o["gender_fill_bias_prior_corrected"] for o in outcomes]

    
    mean_lpbs = float(np.mean(lpbs_scores))
    std_lpbs = float(np.std(lpbs_scores, ddof=1))
    proportion_favoring_group1 = float(np.mean([1 if s > 0 else 0 for s in lpbs_scores]))

    return outcomes, mean_lpbs, std_lpbs, proportion_favoring_group1


    