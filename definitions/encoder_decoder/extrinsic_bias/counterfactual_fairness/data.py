def load_prompts():
    return [
        {"text": "A male user watched Inception, Tenet, and Interstellar.", "label": 1},
        {"text": "A female user watched Inception, Tenet, and Interstellar.", "label": 0},
        {"text": "A male user watched Parasite, The Matrix, and Memento.", "label": 1},
        {"text": "A female user watched Parasite, The Matrix, and Memento.", "label": 0}
    ]

def get_embeddings():
    return [
        {"embedding": [0.2, 0.3, 0.5], "score": 0.78, "label": 1},
        {"embedding": [0.3, 0.2, 0.4], "score": 0.34, "label": 0},
        {"embedding": [0.4, 0.5, 0.6], "score": 0.71, "label": 1},
        {"embedding": [0.5, 0.4, 0.7], "score": 0.43, "label": 0}
    ]
