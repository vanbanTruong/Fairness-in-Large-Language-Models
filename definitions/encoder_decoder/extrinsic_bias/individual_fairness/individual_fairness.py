from data import embed_sentence
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_similarity(trans_1, trans_2):
    vec1 = embed_sentence(trans_1).reshape(1, -1)
    vec2 = embed_sentence(trans_2).reshape(1, -1)
    sim_score = cosine_similarity(vec1, vec2)[0][0]
    return round(sim_score, 2)

def ss_score(pairs):
    similarities = []
    for i in range(0, len(pairs), 2):
        s1 = pairs[i][1]
        s2 = pairs[i + 1][1]
        sim = compute_similarity(s1, s2)
        similarities.append(sim)
    return round(sum(similarities) / len(similarities), 2)
