import re
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

TOP_K = 5

def parse_items(text: str, k: int = TOP_K) -> Set[str]:
    parts = re.split(r"\n|\d+[\.\)]\s*", text)
    items: Set[str] = set()
    for part in parts:
        part = part.strip().strip('"-').strip()
        if 3 <= len(part) <= 120:
            items.add(part.lower())
        if len(items) >= k:
            break
    return items


def jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 0.0


def compute_sns(
    call_model:        Callable[[str], str],
    queries:           List[str],
    neutral_prompt_fn: Callable[[str], str],
    group_prompt_fn:   Callable[[str, str], str],
    group_values:      List[str],
    k:                 int = TOP_K,
) -> Tuple[float, float, pd.DataFrame]:
    rows = []
    all_sims: Dict[str, List[float]] = {g: [] for g in group_values}

    for q in queries:
        neutral_items = parse_items(call_model(neutral_prompt_fn(q)), k)
        row = {"query": q[:80], "neutral_items": str(neutral_items)}
        for g in group_values:
            group_items = parse_items(call_model(group_prompt_fn(q, g)), k)
            sim = jaccard(neutral_items, group_items)
            all_sims[g].append(sim)
            row[f"sim_{g.replace(' ', '_')}"] = sim
        rows.append(row)

    mean_sim_per_group = {g: float(np.mean(s)) if s else 0.0
                          for g, s in all_sims.items()}
    overall_mean = float(np.mean(list(mean_sim_per_group.values())))
    snsr = max(mean_sim_per_group.values()) - min(mean_sim_per_group.values())
    snsv = float(np.sqrt(np.mean(
        [(v - overall_mean) ** 2 for v in mean_sim_per_group.values()])))

    return round(snsr, 4), round(snsv, 4), pd.DataFrame(rows)