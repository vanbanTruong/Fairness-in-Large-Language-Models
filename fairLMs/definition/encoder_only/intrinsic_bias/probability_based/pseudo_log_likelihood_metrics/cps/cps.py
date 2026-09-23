import torch
from collections import defaultdict

from fairLMs.definition.encoder_only.utils import get_span, get_token_ranks


from fairLMs.utils.pll import scoring_input, masked_position_score


def score_sentence_cps(
    tokenizer, model, sentence, spans, *, batch_size=16, context=None
):
    input_ids, allowed = scoring_input(tokenizer, model, sentence, context=context)
    return masked_position_score(
        tokenizer,
        model,
        input_ids,
        [i for i in spans if i in allowed],
        batch_size=batch_size,
    )


def compute_cps(tokenizer, model, sentence_pairs, *, batch_size=16):

    total = 0
    stereo_count = 0
    all_ranks = []
    counts = defaultdict(int)
    scores = defaultdict(int)

    for pair in sentence_pairs:
        bias_type = pair.get("bias_type", "unspecified")
        counts[bias_type] += 1

        context = pair.get("scoring_context")
        pro_input, pro_allowed = scoring_input(
            tokenizer, model, pair["stereotype"], context=context
        )
        anti_input, anti_allowed = scoring_input(
            tokenizer, model, pair["anti_stereotype"], context=context
        )
        pro_ids, anti_ids = pro_input[0], anti_input[0]

        pro_spans, anti_spans = get_span(pro_ids, anti_ids, operation="equal")
        pro_spans = [s for s in pro_spans if s in pro_allowed]
        anti_spans = [s for s in anti_spans if s in anti_allowed]

        pro_score, pro_ranks = score_sentence_cps(
            tokenizer,
            model,
            pair["stereotype"],
            pro_spans,
            batch_size=batch_size,
            context=context,
        )
        anti_score, anti_ranks = score_sentence_cps(
            tokenizer,
            model,
            pair["anti_stereotype"],
            anti_spans,
            batch_size=batch_size,
            context=context,
        )

        pro_score = round(pro_score, 3)
        anti_score = round(anti_score, 3)

        all_ranks.extend(pro_ranks)
        all_ranks.extend(anti_ranks)

        total += 1
        if pro_score > anti_score:
            stereo_count += 1
            scores[bias_type] += 1

    cps_score = (stereo_count / total) * 100 if total > 0 else 0.0

    valid_ranks = [r for r in all_ranks if r != -1]
    accuracy = (
        sum(1 for r in valid_ranks if r == 1) / len(valid_ranks) * 100
        if valid_ranks
        else 0.0
    )

    per_bias_type = {bt: (scores[bt] / counts[bt]) * 100 for bt in counts}

    return cps_score, accuracy, per_bias_type
