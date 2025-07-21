from scipy.stats import wasserstein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def map_summary_to_segments(summary, article, segments):
    tfidf = TfidfVectorizer().fit_transform(article + summary)
    sim_matrix = cosine_similarity(tfidf[-len(summary):], tfidf[:-len(summary)])
    positions = sim_matrix.argmax(axis=1)
    segment_counts = [0] * segments
    seg_size = len(article) // segments
    for p in positions:
        idx = min(p // seg_size, segments - 1)
        segment_counts[idx] += 1
    total = sum(segment_counts)
    return [count / total for count in segment_counts]

def compute_npd(data, segments=3):
    dists = []
    for example in data:
        article = example["article"]
        gold = example["gold_summary"]
        model = example["model_summary"]

        p_gold = map_summary_to_segments(gold, article, segments)
        p_model = map_summary_to_segments(model, article, segments)
        dist = wasserstein_distance(p_gold, p_model)
        dists.append(dist)
    
    return round(sum(dists) / len(dists), 2)
