import math

def compute_lfp(outputs, frequencies):
    total = 0
    band_counts = {"B1": 0, "B2": 0, "B3": 0}
    
    for line in outputs:
        for word in line.lower().split():
            freq = frequencies.get(word, 3000)
            total += 1
            if freq <= 1000:
                band_counts["B1"] += 1
            elif freq <= 2000:
                band_counts["B2"] += 1
            else:
                band_counts["B3"] += 1

    p_b1 = round(band_counts["B1"] / total, 3)
    p_b2 = round(band_counts["B2"] / total, 3)
    p_b3 = round(band_counts["B3"] / total, 3)
    
    return p_b1, p_b2, p_b3

def compute_entropy(variants):
    entropies = []
    for lemma, forms in variants.items():
        total = sum(forms.values())
        probs = [count / total for count in forms.values()]
        h = -sum(p * math.log(p) for p in probs if p > 0)
        entropies.append(h)
    return round(sum(entropies) / len(entropies), 3)

def compute_simpson(variants):
    scores = []
    for lemma, forms in variants.items():
        total = sum(forms.values())
        probs = [count / total for count in forms.values()]
        d = 1 / sum(p ** 2 for p in probs)
        scores.append(d)
    return round(sum(scores) / len(scores), 3)
