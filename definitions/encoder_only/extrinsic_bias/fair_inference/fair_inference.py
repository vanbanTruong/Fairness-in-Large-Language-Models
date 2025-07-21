def compute_nn(predictions):
    return sum(p["neutral"] for p in predictions) / len(predictions)

def compute_fn(predictions):
    return sum(1 for p in predictions if p["neutral"] == max(p.values())) / len(predictions)

def compute_threshold(predictions, tau):
    return sum(1 for p in predictions if p["neutral"] > tau) / len(predictions)

def evaluate_fair_inference(predictions):
    nn = compute_nn(predictions)
    fn = compute_fn(predictions)
    t05 = compute_threshold(predictions, 0.5)
    t07 = compute_threshold(predictions, 0.7)
    return nn, fn, t05, t07