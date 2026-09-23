import numpy as np
from encoder_only.utils import association_vectorized, cohens_d, permutation_pval

def compute_weat(T1_vecs, T2_vecs, A_vecs, B_vecs,):
    s_T1 = np.array([association_vectorized(t, A_vecs, B_vecs) for t in T1_vecs])
    s_T2 = np.array([association_vectorized(t, A_vecs, B_vecs) for t in T2_vecs])
    
    d     = cohens_d(s_T1, s_T2)
    p     = permutation_pval(s_T1, s_T2, n_samples = 10000)
    
    return d, p