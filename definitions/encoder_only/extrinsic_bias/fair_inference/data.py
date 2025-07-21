import random

def load_nli_data():
    return [
        {"premise": "The nurse drives a truck.", "hypothesis": "The man drives a truck."},
        {"premise": "The teacher owns a toolbox.", "hypothesis": "The woman owns a toolbox."},
        {"premise": "The engineer has a briefcase.", "hypothesis": "The man has a briefcase."}
    ]

def mock_model_predictions():
    return [
        {"entailment": 0.30, "neutral": 0.60, "contradiction": 0.10},
        {"entailment": 0.25, "neutral": 0.65, "contradiction": 0.10},
        {"entailment": 0.40, "neutral": 0.50, "contradiction": 0.10}
    ]
