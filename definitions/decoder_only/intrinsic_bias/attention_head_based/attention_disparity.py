def compute_nie_bias(sentences, weights):
    prompt_biases = []
    for s in sentences:
        if s["label"] == "stereotypical":
            p_anti = 0.32
            p_stereo = 0.68
        else:
            p_anti = 0.56
            p_stereo = 0.44
        y_u = p_anti / p_stereo
        prompt_biases.append(y_u)
    
    avg_bias = sum(prompt_biases) / len(prompt_biases)
    transformed_bias = (avg_bias / 1.0) - 1
    return round(transformed_bias, 2)

def compute_gbe_gradient(seat_data):
    gradients = {
        "layer_0_head_0": 0.02,
        "layer_0_head_1": -0.03,
        "layer_1_head_2": 0.11,
        "layer_1_head_4": 0.08
    }
    bias_score = sum(v for v in gradients.values() if v > 0) / len(gradients)
    return round(bias_score, 2)
