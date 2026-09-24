"""Short demo of the public ``PseudoLogLikelihoodScore`` metric."""

from fairLMs.datasets import CrowSPairs
from fairLMs.definitions import PseudoLogLikelihoodScore
from fairLMs.definitions.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="mlm")
    result = PseudoLogLikelihoodScore().compute(
        model=model,
        dataset=CrowSPairs(n_max=32),
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
