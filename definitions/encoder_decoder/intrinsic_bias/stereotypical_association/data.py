def load_translation_pairs():
    return [
        {"source": "The nurse helped the patient", "gender": "male", "translation": "La enfermera"},
        {"source": "The nurse helped the patient", "gender": "female", "translation": "La enfermera"},
        {"source": "The mechanic fixed the car", "gender": "female", "translation": "El mecánico"},
        {"source": "The mechanic fixed the car", "gender": "male", "translation": "El mecánico"},
    ]

def get_stereotypical_examples():
    return [
        {"src": "The nurse", "expected": "La enfermera", "gender": "female"},
        {"src": "The mechanic", "expected": "El mecánico", "gender": "male"}
    ]

def get_antistereotypical_examples():
    return [
        {"src": "The nurse", "expected": "El enfermero", "gender": "male"},
        {"src": "The mechanic", "expected": "La mecánica", "gender": "female"}
    ]

def get_attention_heads():
    return ["enc_0_1", "enc_1_3", "dec_0_2", "dec_2_4"]
