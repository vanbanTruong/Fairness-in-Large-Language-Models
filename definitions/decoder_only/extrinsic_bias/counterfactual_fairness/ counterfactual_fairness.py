def compute_cr(pairs, predictions):
    changed = 0
    for pair in pairs:
        original_output = predictions.get(pair["original"], "")
        cf_output = predictions.get(pair["cf"], "")
        if original_output != cf_output:
            changed += 1
    return round(changed / len(pairs), 2)

def compute_ctf(token_scores):
    diffs = []
    seen = set()
    for text, score in token_scores.items():
        if text in seen:
            continue
        for comp_text, comp_score in token_scores.items():
            if text != comp_text and text.split()[0] != comp_text.split()[0]:
                diffs.append(abs(score - comp_score))
        seen.add(text)
    return round(sum(diffs), 2)
