"""Equal Opportunity gap via the public fairLMs API (synthetic labels)."""

from fairLMs.metrics import EqualOpportunityGap


def main():
    result = EqualOpportunityGap().compute(
        y_true=[1, 1, 1, 1, 0, 0, 1, 1],
        y_pred=[1, 0, 1, 1, 0, 1, 1, 0],
        groups=["A", "A", "A", "B", "A", "B", "B", "B"],
        g1="A",
        g2="B",
        y=1,
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
