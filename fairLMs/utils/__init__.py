"""Shared utilities extracted from metric leaf scripts."""

from fairLMs.utils.association import (
    association,
    association_vectorized,
    ceat_effect_size,
    cohens_d,
    cosine_similarity,
    permutation_pval,
)
from fairLMs.utils.embeddings import (
    embed_encoder_hidden,
    encode_in_context,
    encode_sentence,
)
from fairLMs.utils.io import results_to_csv
from fairLMs.utils.masking import (
    build_masked_sentence,
    get_mask_fill_probs,
    get_multitoken_log_prob,
    get_token_prob,
    get_top_k_predictions,
)
from fairLMs.utils.paths import (
    artifacts_root,
    bbq_category_files,
    data_root,
    package_root,
    resolve_bbq_dir,
    resolve_crows_pairs_csv,
)
from fairLMs.utils.pll import get_span, get_token_ranks, score_sentence
from fairLMs.utils.stats import mean_confidence_interval

__all__ = [
    "association",
    "association_vectorized",
    "artifacts_root",
    "bbq_category_files",
    "build_masked_sentence",
    "ceat_effect_size",
    "cohens_d",
    "cosine_similarity",
    "data_root",
    "embed_encoder_hidden",
    "encode_in_context",
    "encode_sentence",
    "get_mask_fill_probs",
    "get_multitoken_log_prob",
    "get_span",
    "get_token_prob",
    "get_token_ranks",
    "get_top_k_predictions",
    "mean_confidence_interval",
    "package_root",
    "permutation_pval",
    "resolve_bbq_dir",
    "resolve_crows_pairs_csv",
    "results_to_csv",
    "score_sentence",
]
