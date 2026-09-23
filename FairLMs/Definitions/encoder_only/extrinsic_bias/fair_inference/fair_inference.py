import numpy as np

def compute_nn(predictions):
    return np.mean([p["neutral"] for p in predictions])

def compute_fn(predictions):
    return np.mean([
        p["neutral"] == max(p.values()) for p in predictions
    ])

def compute_threshold(predictions, tau):
    return np.mean([p["neutral"] > tau for p in predictions])

def evaluate_fair_inference(predictions):
    nn = compute_nn(predictions)
    fn = compute_fn(predictions)
    t05 = compute_threshold(predictions, 0.5)
    t07 = compute_threshold(predictions, 0.7)
    return nn, fn, t05, t07