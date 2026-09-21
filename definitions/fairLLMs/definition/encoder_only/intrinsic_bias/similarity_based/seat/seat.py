import numpy as np
import torch

from fairLLMs.definition.encoder_only.utils import association_vectorized, cohens_d, permutation_pval, encode_sentence

def apply_templates(terms, templates = None):
    
    if templates is None:
        templates = [
    "This is {}.",
    "That is {}.",
    "It is {}.",
    "There is {}.",
    "Here is {}.",
]
    sentences = []
    for term in terms:
        for tmpl in templates:
            sentences.append(tmpl.format(term))
    return sentences


def compute_seat(model, tokenizer, T1_terms, T2_terms, A1_terms, A2_terms,
    templates= None, pooling = "mean", n_samples = 10_000, device = None):
    if len(T1_terms) != len(T2_terms):
        raise ValueError(
            f"T1 and T2 must have the same number of terms, "
            f"got {len(T1_terms)} and {len(T2_terms)}."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if templates is None:
        templates = [
    "This is {}.",
    "This is a {}.",
    "{} is here.",
    "This will {}.",
    "{} are things."
]

    n_templates = len(templates)
    
    T1_sentences = apply_templates(T1_terms, templates)
    T1_vecs = encode_sentence(model, tokenizer, T1_sentences, pooling = "mean")
    T1_vecs = T1_vecs.reshape(len(T1_terms), n_templates, -1).mean(axis=1)

    T2_sentences = apply_templates(T2_terms, templates)
    T2_vecs = encode_sentence(model, tokenizer, T2_sentences, pooling = "mean")
    T2_vecs = T2_vecs.reshape(len(T2_terms), n_templates, -1).mean(axis=1)

    A1_sentences = apply_templates(A1_terms, templates)
    A1_vecs = encode_sentence(model, tokenizer, A1_sentences, pooling = "mean")
    A1_vecs = A1_vecs.reshape(len(A1_terms), n_templates, -1).mean(axis=1)

    A2_sentences = apply_templates(A2_terms, templates)
    A2_vecs = encode_sentence(model, tokenizer, A2_sentences, pooling = "mean")
    A2_vecs = A2_vecs.reshape(len(A2_terms), n_templates, -1).mean(axis=1)

    s_T1 = np.array([association_vectorized(t, A1_vecs, A2_vecs) for t in T1_vecs])
    s_T2 = np.array([association_vectorized(t, A1_vecs, A2_vecs) for t in T2_vecs])

    effect_size = cohens_d(s_T1, s_T2)
    p_value = permutation_pval(s_T1, s_T2, n_samples)

    return effect_size, p_value