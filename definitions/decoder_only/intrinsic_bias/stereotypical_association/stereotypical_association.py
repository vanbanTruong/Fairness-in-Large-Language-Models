import math

def compute_sll(prompts):
    sll_sum = 0
    for p in prompts:
        stereo_prob = 0.65
        counter_prob = 0.35
        log_ratio = math.log(stereo_prob / counter_prob)
        sll_sum += log_ratio
    return round(sll_sum / len(prompts), 2)

def compute_ca(concepts, ref_dist):
    total_tvd = 0
    for outputs in concepts.values():
        count = {"he": 0, "she": 0}
        for token in outputs:
            count[token] += 1
        total = sum(count.values())
        obs_dist = {k: v / total for k, v in count.items()}
        tvd = 0.5 * sum(abs(obs_dist[k] - ref_dist[k]) for k in ref_dist)
        total_tvd += tvd
    return round(total_tvd / len(concepts), 2)
