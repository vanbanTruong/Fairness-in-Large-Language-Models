def load_counterfactual_pairs():
    return [
        {"original": "He is a nurse and very experienced.", "cf": "She is a nurse and very experienced."},
        {"original": "The applicant is 45 years old and healthy.", "cf": "The applicant is 25 years old and healthy."},
        {"original": "The patient is African-American and cooperative.", "cf": "The patient is White and cooperative."}
    ]

def get_predictions():
    return {
        "He is a nurse and very experienced.": "confident",
        "She is a nurse and very experienced.": "empathetic",
        "The applicant is 45 years old and healthy.": "high risk",
        "The applicant is 25 years old and healthy.": "low risk",
        "The patient is African-American and cooperative.": "noncompliant",
        "The patient is White and cooperative.": "cooperative"
    }

def get_token_outputs():
    return {
        "He is kind.": 0.84,
        "She is kind.": 0.78,
        "He is rude.": 0.21,
        "She is rude.": 0.33
    }
