"""Short demo of the public ``SEAT`` metric."""

from fairLMs.definition.encoder_only.intrinsic_bias.similarity_based.weat.data import (
    ALL_TESTS,
)
from fairLMs.metrics import SEAT
from fairLMs.models import HuggingFaceModel


def main():
    test = ALL_TESTS[1]  # C2 gender — smaller word lists
    model = HuggingFaceModel("bert-base-uncased", task="encoder")
    result = SEAT(n_samples=1000).compute(
        model=model,
        T1_terms=test["t1"],
        T2_terms=test["t2"],
        A1_terms=test["a1"],
        A2_terms=test["a2"],
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
