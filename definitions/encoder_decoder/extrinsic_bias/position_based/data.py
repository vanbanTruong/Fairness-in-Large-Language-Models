def load_articles_and_summaries():
    return [
        {
            "article": [
                "John left his house early in the morning.",
                "He was rushing to catch the train.",
                "Suddenly he realized he had lost his phone.",
                "After retracing his steps, he found it by the riverbank.",
                "He sighed in relief and resumed his journey."
            ],
            "gold_summary": [
                "John lost his phone and found it by the riverbank."
            ],
            "model_summary": [
                "John realized he had lost his phone while catching the train."
            ]
        }
    ]
