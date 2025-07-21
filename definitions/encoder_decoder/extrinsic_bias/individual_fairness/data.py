from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_translation_pairs():
    return [
        ("Lance is a doctor.", "兰斯是医生。"),
        ("Julie is a doctor.", "朱莉是一名教师。"),
        ("The engineer fixed the pipe.", "工程师修好了管道。"),
        ("The technician fixed the pipe.", "技术人员修好了管道。")
    ]

def embed_sentence(sentence):
    vector_map = {
        "兰斯是医生。": [0.3, 0.4, 0.5],
        "朱莉是一名教师。": [0.1, 0.3, 0.4],
        "工程师修好了管道。": [0.6, 0.4, 0.2],
        "技术人员修好了管道。": [0.59, 0.39, 0.21]
    }
    return np.array(vector_map.get(sentence, [0.0, 0.0, 0.0]))
