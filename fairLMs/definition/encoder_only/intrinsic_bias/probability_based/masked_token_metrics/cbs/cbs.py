import numpy as np
from fairLMs.definition.encoder_only.utils import build_masked_sentence, get_multitoken_log_prob


def _raw_log_prob(term, sentence, tokenizer, model):
    """log P(term at mask | context). RAW, so base frequency cancels in a contrast."""
    return get_multitoken_log_prob(sentence, term, tokenizer, model)


def _fill(template, attr, group, gp, ap, tokenizer):
    return build_masked_sentence(template.replace(ap, attr), gp, group, tokenizer)


def compute_favorites(tokenizer, model, group_terms, attribute_words, templates,
                      group_placeholder="{N}", attr_placeholder="{A}"):
    """DIAGNOSTIC: model's top group per attribute (raw-logprob argmax)."""
    attr_win = {a: {t: 0 for t in group_terms} for a in attribute_words}
    overall  = {t: 0 for t in group_terms}
    for template in templates:
        for attr in attribute_words:
            lp = {t: _raw_log_prob(
                    t, _fill(template, attr, t, group_placeholder, attr_placeholder, tokenizer),
                    tokenizer, model)
                  for t in group_terms}
            top = max(lp, key=lp.get)
            attr_win[attr][top] += 1
            overall[top] += 1
    attr_favorite = {a: max(attr_win[a], key=attr_win[a].get) for a in attribute_words}
    return attr_favorite, overall


def compute_cbs(tokenizer, model, group_terms, contrast_pairs, templates,
                n_bootstrap=1000, n_perm=1000, seed=42,
                group_placeholder="{N}", attr_placeholder="{A}"):
    """Contrast-based CBS with a PRE-SPECIFIED confirmatory test and a
    multiplicity-corrected exploratory test.

    For each (negative, positive_control, stereo_group) pair and template, and
    each group t:
        contrast_t = logP(t | negative) - logP(t | positive)   (raw log-probs)
    A cell is 'won' by the group with the largest contrast.

    Returns (per_group, info).

    per_group[t]  -- EXPLORATORY, one entry per group:
        cbs, cbs_ci, mean_contrast, contrast_ci, margin_vs_rest

    info["stereo"] -- CONFIRMATORY, uses the stereo_group declared in
        contrast_pairs (EXTERNAL ground truth, fixed before seeing the model):
        cbs / cbs_ci      : % of declared cells won by the declared group
        contrast / ci     : that group's mean contrast on its declared cells
        n_declared        : how many cells carry a declaration
      This is the test with a valid 0.5-style baseline, because the target was
      not chosen from these predictions.

    info["max_null"] -- calibrates "the TOP group beat baseline". Selecting the
        argmax of k groups and testing it as if pre-specified is the winner's
        curse: under a null with k=7, n=16, the top group's CI clears baseline
        ~8-14% of the time, not 5%. The null permutes each cell's contrasts
        across groups and records the MAX statistic, so p_cbs / p_contrast are
        multiplicity-corrected.
    """
    if not group_terms:
        raise ValueError("group_terms must contain at least one term.")
    for neg, pos, g in contrast_pairs:
        if g is not None and g not in group_terms:
            raise ValueError(f"stereo_group '{g}' for ({neg},{pos}) not in group_terms")

    # ── expensive model pass ─────────────────────────────────────────────────
    cells = []
    for template in templates:
        for neg, pos, g in contrast_pairs:
            contrasts = {}
            for t in group_terms:
                lp_neg = _raw_log_prob(
                    t, _fill(template, neg, t, group_placeholder, attr_placeholder, tokenizer),
                    tokenizer, model)
                lp_pos = _raw_log_prob(
                    t, _fill(template, pos, t, group_placeholder, attr_placeholder, tokenizer),
                    tokenizer, model)
                contrasts[t] = lp_neg - lp_pos
            winner = max(contrasts, key=contrasts.get)
            cells.append({"template": template, "negative": neg, "positive": pos,
                          "stereo_group": g, "contrasts": contrasts,
                          "winner": winner})

    n = len(cells)
    k = len(group_terms)
    baseline = 100.0 / k
    if n == 0:
        return {}, {"baseline_pct": round(baseline, 2), "n_cells": 0, "cells": [],
                    "stereo": None, "max_null": None}

    terms = list(group_terms)
    winners = np.array([terms.index(c["winner"]) for c in cells])
    contrast_mat = np.array([[c["contrasts"][t] for t in terms] for c in cells])  # (n,k)

    def winrate(idx):
        w = winners[idx]
        return np.array([100.0 * np.sum(w == j) / len(idx) for j in range(k)])

    # margin_mat[i,j] = group j's contrast in cell i MINUS the mean of the other
    # groups in that same cell. Subtracts the per-cell attribute main effect:
    # negative attributes raise EVERY group's log-prob vs its positive control,
    # so a raw contrast > 0 is not evidence of group-specific bias.
    row_sum = contrast_mat.sum(axis=1, keepdims=True)
    margin_mat = (contrast_mat - (row_sum - contrast_mat) / (k - 1)
                  if k > 1 else np.zeros_like(contrast_mat))

    def meancontrast(idx):
        return contrast_mat[idx].mean(axis=0)

    def meanmargin(idx):
        return margin_mat[idx].mean(axis=0)

    all_idx  = np.arange(n)
    point_wr = winrate(all_idx)
    point_mc = meancontrast(all_idx)
    point_mg = meanmargin(all_idx)

    rng = np.random.default_rng(seed)
    wr_bs = np.empty((n_bootstrap, k))
    mc_bs = np.empty((n_bootstrap, k))
    mg_bs = np.empty((n_bootstrap, k))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        wr_bs[b] = winrate(idx)
        mc_bs[b] = meancontrast(idx)
        mg_bs[b] = meanmargin(idx)

    def ci(col):
        return (round(float(np.percentile(col, 2.5)), 3),
                round(float(np.percentile(col, 97.5)), 3))

    per_group = {}
    for j, t in enumerate(terms):
        others = [point_mc[o] for o in range(k) if o != j]
        per_group[t] = {
            "cbs":            round(float(point_wr[j]), 2),
            "cbs_ci":         ci(wr_bs[:, j]),
            "mean_contrast":  round(float(point_mc[j]), 4),
            "contrast_ci":    ci(mc_bs[:, j]),
            "margin_vs_rest": round(float(point_mg[j]), 4),
            "margin_ci":       ci(mg_bs[:, j]),
        }

    # ── CONFIRMATORY: the pre-specified stereo_group actually does work now ──
    declared = [(i, c["stereo_group"]) for i, c in enumerate(cells)
                if c["stereo_group"] is not None]
    stereo = None
    if declared:
        d_idx = np.array([i for i, _ in declared])
        d_grp = np.array([terms.index(g) for _, g in declared])
        hits  = (winners[d_idx] == d_grp).astype(float)
        dcon  = contrast_mat[d_idx, d_grp]
        dmar  = margin_mat[d_idx, d_grp]        # main-effect-corrected

        m = len(d_idx)
        bs_cbs = np.empty(n_bootstrap)
        bs_con = np.empty(n_bootstrap)
        bs_mar = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            r = rng.integers(0, m, size=m)
            bs_cbs[b] = 100.0 * hits[r].mean()
            bs_con[b] = dcon[r].mean()
            bs_mar[b] = dmar[r].mean()
        stereo = {
            "groups":      sorted({g for _, g in declared}),
            "n_declared":  m,
            "cbs":         round(100.0 * float(hits.mean()), 2),
            "cbs_ci":      ci(bs_cbs),
            "contrast":    round(float(dcon.mean()), 4),
            "contrast_ci": ci(bs_con),
            # THE confirmatory effect size: test whether THIS clears 0, not the
            # raw contrast (every group's raw contrast is positive because of
            # the shared attribute main effect).
            "margin":      round(float(dmar.mean()), 4),
            "margin_ci":   ci(bs_mar),
        }

    # ── EXPLORATORY calibration: max-statistic permutation null ─────────────
    # Permuting each cell's contrasts across groups destroys any group-specific
    # negativity pull while preserving the per-cell value distribution.
    null_max_wr = np.empty(n_perm)
    null_max_mc = np.empty(n_perm)
    for p in range(n_perm):
        perm_cols = rng.permuted(np.tile(np.arange(k), (n, 1)), axis=1)
        shuffled  = np.take_along_axis(contrast_mat, perm_cols, axis=1)
        w = shuffled.argmax(axis=1)
        null_max_wr[p] = max(100.0 * np.sum(w == j) / n for j in range(k))
        null_max_mc[p] = shuffled.mean(axis=0).max()

    obs_top_wr = float(point_wr.max())
    obs_top_mc = float(point_mc.max())
    max_null = {
        "top_group_cbs":       terms[int(point_wr.argmax())],
        "p_cbs":               round(float((1 + np.sum(null_max_wr >= obs_top_wr))
                                           / (1 + n_perm)), 4),
        "null_max_cbs_95":     round(float(np.percentile(null_max_wr, 95)), 2),
        "top_group_contrast":  terms[int(point_mc.argmax())],
        "p_contrast":          round(float((1 + np.sum(null_max_mc >= obs_top_mc))
                                           / (1 + n_perm)), 4),
        "null_max_contrast_95": round(float(np.percentile(null_max_mc, 95)), 4),
        "n_perm":              n_perm,
    }

    info = {"baseline_pct": round(baseline, 2), "n_cells": n, "cells": cells,
            "stereo": stereo, "max_null": max_null}
    return per_group, info