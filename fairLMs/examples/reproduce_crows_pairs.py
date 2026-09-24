"""Reproduce the published CrowS-Pairs score for bert-base-uncased.

Nangia et al. (2020) report 60.5 overall, 61.1 on stereotype pairs and 56.9 on
anti-stereotype pairs for BERT. This script runs FairLMs' ``CrowSPairsScore`` on
the bundled CrowS-Pairs file (1,508 pairs) and prints the overall score, the two
subset scores and the per-category breakdown. About four minutes on a CPU.

    python examples/reproduce_crows_pairs.py
"""

from __future__ import annotations

import argparse

from fairLMs.datasets import CrowSPairs
from fairLMs.definitions import CrowSPairsScore
from fairLMs.definitions.models import HuggingFaceModel

PUBLISHED = {"overall": 60.5, "stereo": 61.1, "antistereo": 56.9}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="bert-base-uncased")
    args = parser.parse_args()

    model = HuggingFaceModel(args.model, task="mlm")
    rows = list(CrowSPairs().load())
    metric = CrowSPairsScore()
    overall = metric.compute(model, rows)
    stereo = metric.compute(model, [r for r in rows if r["stereo_antistereo"] == "stereo"])
    anti = metric.compute(model, [r for r in rows if r["stereo_antistereo"] == "antistereo"])

    print(f"{'subset':12s} {'published':>10s} {'fairLMs':>10s}")
    for name, result in (("overall", overall), ("stereo", stereo), ("antistereo", anti)):
        print(f"{name:12s} {PUBLISHED[name]:10.1f} {float(result.score):10.2f}")
    print("\nby bias category:")
    for category, value in sorted(overall.by_category.items()):
        print(f"  {category:22s} {value:6.2f}")


if __name__ == "__main__":
    main()
