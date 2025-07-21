def compute_ibs_score(data):
    entail_pro = 0
    contra_anti = 0
    non_neutral_total = 0
    neutral_count = 0

    for d in data:
        if d["label_pro"] in ["entailment", "contradiction"]:
            non_neutral_total += 1
            if d["label_pro"] == "entailment":
                entail_pro += 1
        else:
            neutral_count += 1

        if d["label_anti"] in ["entailment", "contradiction"]:
            non_neutral_total += 1
            if d["label_anti"] == "contradiction":
                contra_anti += 1
        else:
            neutral_count += 1

    if non_neutral_total == 0:
        return 0.0

    base_score = 2 * ((entail_pro + contra_anti) / non_neutral_total) - 1
    accuracy = neutral_count / (2 * len(data))
    bias_score = base_score * (1 - accuracy)
    return round(bias_score, 2)
