"""CrowS-Pairs Score via the public fairLMs API."""

from fairLMs.datasets import CrowSPairs
from fairLMs.metrics import CrowSPairsScore
from fairLMs.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="mlm")
    result = CrowSPairsScore().compute(
        model=model,
        dataset=CrowSPairs(n_max=32),
    )
    print(result)
    print(f"score={result.score}")
    if result.by_category:
        print("by_category:", result.by_category)


if __name__ == "__main__":
    main()
