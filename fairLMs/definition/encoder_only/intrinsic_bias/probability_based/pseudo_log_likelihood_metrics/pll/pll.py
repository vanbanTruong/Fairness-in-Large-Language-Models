import torch
from collections import defaultdict

from fairLMs.definition.encoder_only.utils import get_token_ranks


from fairLMs.utils.pll import scoring_input, masked_position_score


def score_sentence_pll(tokenizer, model, sentence, *, batch_size=16, context=None):
    ids, positions = scoring_input(tokenizer, model, sentence, context=context)
    return masked_position_score(
        tokenizer, model, ids, positions, batch_size=batch_size
    )


def compute_pll(tokenizer, model, sentence_pairs, *, batch_size=16):

    total = 0
    stereo_count = 0
    all_ranks = []
    counts = defaultdict(int)
    scores = defaultdict(int)

    for pair in sentence_pairs:
        bias_type = pair.get("bias_type", "unspecified")
        counts[bias_type] += 1

        pro_score, pro_ranks = score_sentence_pll(
            tokenizer,
            model,
            pair["stereotype"],
            batch_size=batch_size,
            context=pair.get("scoring_context"),
        )
        anti_score, anti_ranks = score_sentence_pll(
            tokenizer,
            model,
            pair["anti_stereotype"],
            batch_size=batch_size,
            context=pair.get("scoring_context"),
        )

        all_ranks.extend(pro_ranks)
        all_ranks.extend(anti_ranks)

        total += 1
        if pro_score > anti_score:
            stereo_count += 1
            scores[bias_type] += 1

    pll_score = (stereo_count / total) * 100 if total > 0 else 0.0

    valid_ranks = [r for r in all_ranks if r != -1]
    accuracy = (
        sum(1 for r in valid_ranks if r == 1) / len(valid_ranks) * 100
        if valid_ranks
        else 0.0
    )

    per_bias_type = {bt: (scores[bt] / counts[bt]) * 100 for bt in counts}

    return pll_score, accuracy, per_bias_type
