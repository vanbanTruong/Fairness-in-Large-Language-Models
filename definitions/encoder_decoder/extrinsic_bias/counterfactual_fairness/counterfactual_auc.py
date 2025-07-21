def compute_auc(embeddings):
    positive_scores = [e["score"] for e in embeddings if e["label"] == 1]
    negative_scores = [e["score"] for e in embeddings if e["label"] == 0]
    P = len(positive_scores)
    N = len(negative_scores)

    if P == 0 or N == 0:
        return 0.5

    comparisons = 0
    for s_i in positive_scores:
        for s_j in negative_scores:
            if s_i > s_j:
                comparisons += 1

    auc = comparisons / (P * N)
    return round(auc, 2)
