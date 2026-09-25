"""Accuracy Disparity via the public fairLMs API (synthetic scores)."""

from fairLMs.definitions import AccuracyDisparity


def main():
    result = AccuracyDisparity().compute(
        scores_s=[1.0, 0.0, 1.0, 1.0],
        scores_sp=[0.0, 0.0, 1.0, 0.0],
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
