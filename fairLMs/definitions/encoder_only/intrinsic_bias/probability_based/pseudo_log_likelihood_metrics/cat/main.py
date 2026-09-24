"""Short demo of the public ``ContextAssociationTestScore`` metric."""

from fairLMs.datasets import StereoSet
from fairLMs.definitions import ContextAssociationTestScore
from fairLMs.definitions.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="mlm")
    dataset = StereoSet(config="intrasentence", as_triples=True, n_max=16)
    result = ContextAssociationTestScore().compute(model=model, dataset=dataset)
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
