"""WEAT via the public fairLMs API (one Caliskan-style test).

Shows the preferred shape: configuration in the constructor, data as a
validated container passed positionally to ``compute``.
"""

from fairLMs.data import weat_c1
from fairLMs.metrics import WEAT
from fairLMs.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="encoder")

    result = WEAT(pooling="mean", n_samples=10_000, seed=0).compute(model, weat_c1)

    print(result)
    print(f"score={result.score}")
    print(f"p_value={result.details['p_value']}")


if __name__ == "__main__":
    main()
