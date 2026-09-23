import numpy as np
import torch

from fairLMs.definition.encoder_only.utils import association_vectorized, cohens_d, permutation_pval, encode_sentence

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
    templates= None, pooling = "mean", n_samples = 10_000, device = None,
    seed = None):
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

    def _encode_terms(terms):
        # pooling and device are honoured here; hardcoding them silently ignored
        # the caller's choice and pinned inputs to CPU (breaking CUDA runs).
        sentences = apply_templates(terms, templates)
        vecs = encode_sentence(
            model, tokenizer, sentences, pooling=pooling, device=device
        )
        return vecs.reshape(len(terms), n_templates, -1).mean(axis=1)

    T1_vecs = _encode_terms(T1_terms)
    T2_vecs = _encode_terms(T2_terms)
    A1_vecs = _encode_terms(A1_terms)
    A2_vecs = _encode_terms(A2_terms)

    s_T1 = np.array([association_vectorized(t, A1_vecs, A2_vecs) for t in T1_vecs])
    s_T2 = np.array([association_vectorized(t, A1_vecs, A2_vecs) for t in T2_vecs])

    effect_size = cohens_d(s_T1, s_T2)
    p_value = permutation_pval(s_T1, s_T2, n_samples, seed=seed)

    return effect_size, p_value