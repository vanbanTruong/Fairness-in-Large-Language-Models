import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from encoder_only.utils import association_vectorized, cohens_d
from encoder_only.intrinsic_bias.similarity_based.ceat.generate_embeddings import GUO_CANDIDATES, filter_and_equalize

_DIR = os.path.dirname(os.path.abspath(__file__))
N_TRIALS    = 10000
SAMPLE_SIZE = 1
SEEDS = [42, 137, 256, 391, 512, 631, 748, 859, 973, 1024,
         1138, 1247, 1356, 1465, 1574, 1683, 1792, 1901, 2010, 2119]
TEST_LABEL = {3: "C1 – Race", 6: "C2 – Gender", 9: "C3 – Disease", 10: "C4 – Age"}


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


def load_pools(num):
    path = os.path.join(_DIR, f"bert_weat{num}.pickle")
    with open(path, "rb") as f:
        d = pickle.load(f)
    return {w: np.asarray(v) for w, v in d.items()}


def compute_ceat_from_pools(T1, T2, A1, A2, sample_size=SAMPLE_SIZE,
                            n_trials=N_TRIALS, seed=None):
    rng = np.random.default_rng(seed)

    def _sample_mean(vecs):
        idx = rng.integers(0, len(vecs), size=sample_size)
        return vecs[idx].mean(axis=0)

    ds, vs = [], []
    for _ in range(n_trials):
        t1 = np.array([_sample_mean(v) for v in T1.values()])
        t2 = np.array([_sample_mean(v) for v in T2.values()])
        a1 = np.array([_sample_mean(v) for v in A1.values()])
        a2 = np.array([_sample_mean(v) for v in A2.values()])
        s1 = np.array([association_vectorized(t, a1, a2) for t in t1])
        s2 = np.array([association_vectorized(t, a1, a2) for t in t2])
        ds.append(cohens_d(s1, s2))
        combined = np.concatenate([s1, s2])
        vs.append(float(np.var(combined, ddof=1)) + 1e-10)

    return dersimonian_laird(np.array(ds), np.array(vs))


def run_multi_seed(num, groups):
    pools = load_pools(num)
    keys = set(pools)
    resolved = filter_and_equalize(groups, keys) 
    (_, t1w), (_, t2w), (_, a1w), (_, a2w) = resolved
    
    missing = [w for grp in (t1w, t2w, a1w, a2w) for w in grp if w not in pools]
    if missing:
        raise KeyError(f"WEAT{num}: resolved words absent from pool: {missing}")

    T1 = {w: pools[w] for w in t1w}
    T2 = {w: pools[w] for w in t2w}
    A1 = {w: pools[w] for w in a1w}
    A2 = {w: pools[w] for w in a2w}
    print(f"  |T1|={len(T1)} |T2|={len(T2)} |A1|={len(A1)} |A2|={len(A2)}")

    per_seed = []
    for i, seed in enumerate(SEEDS):
        print(f"    seed {seed} ({i+1}/{len(SEEDS)})", end="\r")
        per_seed.append(compute_ceat_from_pools(T1, T2, A1, A2, seed=seed))
    print()

    ces_arr  = np.array([r["CES"]  for r in per_seed])
    tau2_arr = np.array([r["tau2"] for r in per_seed])
    primary = per_seed[0]
    ces, se = primary["CES"], primary["se"]
    ci_low  = ces - 1.96 * se if np.isfinite(se) else float("nan")
    ci_high = ces + 1.96 * se if np.isfinite(se) else float("nan")

    return {
        "test":        TEST_LABEL[num],
        "ceat_score":         round(float(ces), 4),
        "se":          round(float(se), 4) if np.isfinite(se) else None,
    }


def main():
    print("=" * 72)
    print("  CEAT on bert-base-uncased  |  Guo Reddit context embeddings")
    print("  Random-effects (DerSimonian-Laird) pooling")
    print("=" * 72)
    results = []
    for num, groups in GUO_CANDIDATES.items():
        print(f"\nWEAT {num} ({TEST_LABEL[num]})")
        results.append(run_multi_seed(num, groups))

    print("\n" + "=" * 84)
    pd.DataFrame(results).to_csv(os.path.join(_DIR, "ceat_results.csv"), index=False)


if __name__ == "__main__":
    main()