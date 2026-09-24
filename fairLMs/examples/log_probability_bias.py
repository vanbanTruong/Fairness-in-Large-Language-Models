"""Log Probability Bias Score via the public fairLMs API."""

from fairLMs.definitions import LogProbabilityBiasScore
from fairLMs.definitions.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="mlm")
    result = LogProbabilityBiasScore().compute(
        model=model,
        attribute_words=["nurse", "surgeon", "teacher", "engineer"],
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
