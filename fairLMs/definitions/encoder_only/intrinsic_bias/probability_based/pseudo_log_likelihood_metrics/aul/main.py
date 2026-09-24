"""Short demo of the public ``AllUnmaskedLikelihoodScore`` metric."""

from fairLMs.datasets import CrowSPairs
from fairLMs.definitions import AllUnmaskedLikelihoodScore
from fairLMs.definitions.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="mlm")
    result = AllUnmaskedLikelihoodScore().compute(
        model=model,
        dataset=CrowSPairs(n_max=16),
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
