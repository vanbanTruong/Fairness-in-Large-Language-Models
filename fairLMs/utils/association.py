"""Shared association / effect-size helpers (WEAT, SEAT, CEAT)."""

from __future__ import annotations

import itertools

import numpy as np
import scipy.special


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        raise ValueError("Cannot compute cosine similarity with a zero vector")
    return np.dot(a, b) / denom


def association(word_vec, A_vecs, B_vecs):
    """s(w, A, B)"""
    s_A = np.mean([cosine_similarity(word_vec, a) for a in A_vecs])
    s_B = np.mean([cosine_similarity(word_vec, b) for b in B_vecs])
    return s_A - s_B


def association_vectorized(word_vec, A_vecs, B_vecs):
    """Vectorized s(w, A, B)."""
    A = np.array(A_vecs)
    B = np.array(B_vecs)
    w = word_vec / np.linalg.norm(word_vec)
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return (w @ A_norm.T).mean() - (w @ B_norm.T).mean()


def cohens_d(s_T1, s_T2):
    all_s = list(s_T1) + list(s_T2)
    return (np.mean(s_T1) - np.mean(s_T2)) / np.std(all_s, ddof=1)


def permutation_pval(s_T1, s_T2, n_samples=10_000, seed=None):
    """One-sided permutation p-value over equal-size re-partitions of T1 ∪ T2.

    Exact enumeration when ``C(2n, n) <= n_samples``, otherwise ``n_samples``
    sampled permutations.

    Parameters
    ----------
    seed:
        Seed for the sampled branch. Drives a generator local to this call, so
        the result depends on nothing but the arguments, so callers do not have to
        reach for ``np.random.seed``, and one metric's sampling cannot be
        perturbed by unrelated global state. ``None`` draws from OS entropy.
        Ignored on the exact branch, which is deterministic by construction.
    """
    s_T1 = np.array(s_T1, dtype=np.float64)
    s_T2 = np.array(s_T2, dtype=np.float64)
    n = len(s_T1)
    assert len(s_T1) == len(s_T2), "Target sets must be the same size"

    combined = np.concatenate([s_T1, s_T2])
    observed = s_T1.sum()

    num_partitions = int(scipy.special.binom(2 * n, n))

    if num_partitions <= n_samples:
        count = 0
        total = 0
        for Xi_idx in itertools.combinations(range(2 * n), n):
            si = combined[list(Xi_idx)].sum()
            if si >= observed:
                count += 1
            total += 1
        return count / total

    rng = np.random.default_rng(seed)
    count = 1
    total = 1
    rng_combined = combined.copy()
    for _ in range(n_samples - 1):
        rng.shuffle(rng_combined)
        si = rng_combined[:n].sum()
        if si >= observed:
            count += 1
        total += 1
    return count / total


def ceat_effect_size(effect_sizes, sample_sizes):
    weights = np.array(sample_sizes)
    return np.average(effect_sizes, weights=weights)
