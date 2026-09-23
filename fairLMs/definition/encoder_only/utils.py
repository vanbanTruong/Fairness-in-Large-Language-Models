"""Backward-compatible re-exports of shared encoder-only utilities.

Prefer importing from ``fairLMs.utils`` in new code.
"""

from fairLMs.utils import (  # noqa: F401
    association,
    association_vectorized,
    build_masked_sentence,
    ceat_effect_size,
    cohens_d,
    cosine_similarity,
    encode_in_context,
    encode_sentence,
    get_mask_fill_probs,
    get_multitoken_log_prob,
    get_span,
    get_token_prob,
    get_token_ranks,
    get_top_k_predictions,
    permutation_pval,
    score_sentence,
)
