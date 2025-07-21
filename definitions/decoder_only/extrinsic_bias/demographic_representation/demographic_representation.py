def compute_drd(outputs, label_sets):
    n_s = sum(any(term in o for term in label_sets["stereotypical"]) for o in outputs)
    n_s_ = sum(any(term in o for term in label_sets["counter_stereotypical"]) for o in outputs)
    total = n_s + n_s_
    if total == 0:
        return 0.0
    p_s = n_s / total
    p_s_ = n_s_ / total
    drd = 0.5 * abs(p_s - 0.5) + 0.5 * abs(p_s_ - 0.5)
    return round(drd, 2)

def compute_dnp(probs):
    P_total = probs["P_s"] + probs["P_s'"] + probs["P_d"]
    p_s_norm = round(probs["P_s"] / P_total, 2)
    p_s_dash_norm = round(probs["P_s'"] / P_total, 2)
    p_d_norm = round(probs["P_d"] / P_total, 2)
    return p_s_norm, p_s_dash_norm, p_d_norm
