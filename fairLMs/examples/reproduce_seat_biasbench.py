"""Reproduce the bias-bench SEAT effect sizes for bert-base-uncased.

Meade et al. (2022, Table 1) report six gender SEAT effect sizes for the
un-debiased ``bert-base-uncased`` model: 0.931, 0.090, -0.124, 0.937, 0.783,
0.858 (average absolute effect size 0.620). This script downloads the sentence
sets of May et al. (2019) from the bias-bench repository, embeds them with
FairLMs' Hugging Face embedding backend and scores them with FairLMs'
association-test kernel. With mean pooling every value matches to three
decimals; ``--pooling cls`` shows how much the pooling rule alone changes them.

    python examples/reproduce_seat_biasbench.py
    python examples/reproduce_seat_biasbench.py --pooling cls --model bert-base-uncased
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np
from transformers import AutoModel, AutoTokenizer

from fairLMs.datasets.diagnostics.backends import HuggingFaceEmbeddingBackend
from fairLMs.definitions import WEAT
from fairLMs.definitions.data import VectorSets

BASE = "https://raw.githubusercontent.com/McGill-NLP/bias-bench/main/data/seat/sent-weat{}.jsonl"
TESTS = ("6", "6b", "7", "7b", "8", "8b")
PUBLISHED = {"6": 0.931, "6b": 0.090, "7": -0.124, "7b": 0.937, "8": 0.783, "8b": 0.858}


def load_test(test: str) -> dict:
    with urllib.request.urlopen(BASE.format(test), timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--pooling", default="mean", choices=("mean", "cls"))
    parser.add_argument("--n-samples", type=int, default=1000, help="permutation samples for p")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    backend = HuggingFaceEmbeddingBackend.from_components(
        tokenizer, model, revision=f"{args.model};pooling={args.pooling}", pooling=args.pooling
    )

    print(f"{'test':8s} {'published':>10s} {'fairLMs':>10s}   backend={backend.revision}")
    effects = []
    for test in TESTS:
        data = load_test(test)
        vectors = {
            role: np.asarray(backend.encode(data[role]["examples"]))
            for role in ("targ1", "targ2", "attr1", "attr2")
        }
        result = WEAT(seed=0, n_samples=args.n_samples).compute(
            None, VectorSets(vectors["targ1"], vectors["targ2"], vectors["attr1"], vectors["attr2"])
        )
        effects.append(abs(float(result.score)))
        print(f"SEAT-{test:4s} {PUBLISHED[test]:10.3f} {float(result.score):10.3f}   p={result.details.get('p_value')}")
    print(f"{'avg |d|':8s} {0.620:10.3f} {np.mean(effects):10.3f}")


if __name__ == "__main__":
    main()
