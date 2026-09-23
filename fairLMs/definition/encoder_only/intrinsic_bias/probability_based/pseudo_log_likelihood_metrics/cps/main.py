"""Short demo of the public ``CrowSPairsScore`` (CPS) metric."""

from fairLMs.datasets import CrowSPairs, StereoSet
from fairLMs.metrics import CrowSPairsScore
from fairLMs.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="mlm")
    metric = CrowSPairsScore()

    crows = CrowSPairs(n_max=32)
    result = metric.compute(model=model, dataset=crows)
    print("CrowS-Pairs:", result)
    print(f"score={result.score}")

    try:
        stereo = StereoSet(config="intersentence", n_max=16)
        result2 = metric.compute(model=model, dataset=stereo)
        print("StereoSet:", result2)
        print(f"score={result2.score}")
    except Exception as exc:  # noqa: BLE001 — optional secondary demo
        print(f"[skip] StereoSet demo: {exc}")


if __name__ == "__main__":
    main()
