import numpy as np
import torch
from scipy import stats
from fairLMs.definition.encoder_only.utils import association_vectorized, cohens_d, encode_in_context


def dersimonian_laird(effects, variances):
    d = np.asarray(effects, float)
    v = np.asarray(variances, float)
    ok = np.isfinite(d) & np.isfinite(v) & (v > 0)
    d, v = d[ok], v[ok]
    k = len(d)
    if k < 2:
        return {"CES": float(d.mean()) if k else float("nan"),
                "tau2": 0.0, "Q": float("nan"), "Q_p": float("nan"),
                "se": float("nan"), "p": float("nan")}
    w = 1.0 / v
    d_fixed = np.sum(w * d) / np.sum(w)
    Q = float(np.sum(w * (d - d_fixed) ** 2))
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    w_star = 1.0 / (v + tau2)
    ces = float(np.sum(w_star * d) / np.sum(w_star))
    se = float(np.sqrt(1.0 / np.sum(w_star)))
    z = ces / se if se > 0 else float("nan")
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    return {"CES": ces, "tau2": tau2, "Q": Q,
            "Q_p": float(stats.chi2.sf(Q, k - 1)),
            "se": se, "p": p}


def compute_ceat(model, tokenizer, T1_contexts, T2_contexts, A1_contexts, A2_contexts,
                 pooling="cls", sample_size=10, n_trials=100, seed=None, device=None):
    """CEAT with random-effects (DerSimonian-Laird) pooling.
    Returns the DL dict: CES, tau2, Q, Q_p, se, p."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T1_context_vecs = {t: encode_in_context(model, tokenizer, t, s, pooling, device)
                       for t, s in T1_contexts.items()}
    T2_context_vecs = {t: encode_in_context(model, tokenizer, t, s, pooling, device)
                       for t, s in T2_contexts.items()}
    A1_context_vecs = {t: encode_in_context(model, tokenizer, t, s, pooling, device)
                       for t, s in A1_contexts.items()}
    A2_context_vecs = {t: encode_in_context(model, tokenizer, t, s, pooling, device)
                       for t, s in A2_contexts.items()}

    all_pools = {**T1_context_vecs, **T2_context_vecs,
                 **A1_context_vecs, **A2_context_vecs}
    for term, vecs in all_pools.items():
        if len(vecs) < sample_size:
            raise ValueError(
                f"Term '{term}' has only {len(vecs)} context embeddings "
                f"but sample_size={sample_size}. Provide more contexts or "
                f"reduce sample_size."
            )

    rng = np.random.default_rng(seed)
    trial_ds = []
    trial_vs = []

    def _sample_mean(term_vecs):
        idx = rng.integers(0, len(term_vecs), size=sample_size)
        return term_vecs[idx].mean(axis=0)

    for _ in range(n_trials):
        T1_sampled = np.array([_sample_mean(v) for v in T1_context_vecs.values()])
        T2_sampled = np.array([_sample_mean(v) for v in T2_context_vecs.values()])
        A1_sampled = np.array([_sample_mean(v) for v in A1_context_vecs.values()])
        A2_sampled = np.array([_sample_mean(v) for v in A2_context_vecs.values()])

        s_T1 = np.array([association_vectorized(t, A1_sampled, A2_sampled) for t in T1_sampled])
        s_T2 = np.array([association_vectorized(t, A1_sampled, A2_sampled) for t in T2_sampled])

        trial_ds.append(cohens_d(s_T1, s_T2))
        combined = np.concatenate([s_T1, s_T2])
        trial_vs.append(float(np.var(combined, ddof=1)) + 1e-10)

    return dersimonian_laird(np.array(trial_ds), np.array(trial_vs))