import numpy as np
from encoder_only.extrinsic.equal_opportunity.data import extract_instances_by_group

def compute_tpr(grouped_counts):
    tprs = {}
    for (group, label), stats in grouped_counts.items():
        tp = stats["TP"]
        total = stats["Total"]
        tpr = tp / total if total > 0 else 0
        tprs[(group, label)] = tpr
    return tprs

def compute_gap(tprs, groups=("male", "female")):
    results = []
    for label in set(y for _, y in tprs.keys()):
        tpr_g1 = tprs.get((groups[0], label), 0)
        tpr_g2 = tprs.get((groups[1], label), 0)
        gap = abs(tpr_g1 - tpr_g2)
        results.append({
            "Label": label,
            "Group 1": groups[0],
            "Group 2": groups[1],
            "TPR Group 1": tpr_g1,
            "TPR Group 2": tpr_g2,
            "Gap_g,y": gap
        })
    return results
