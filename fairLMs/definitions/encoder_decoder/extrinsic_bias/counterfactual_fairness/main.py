"""Demo of ``CounterfactualAucScore`` on WinoBias sentences (small slice)."""

from fairLMs.definitions.encoder_decoder.extrinsic_bias.counterfactual_fairness.stimuli import (
    load_winobias_sentences,
)
from fairLMs.definitions import CounterfactualAucScore
from fairLMs.definitions.models import HuggingFaceModel


def main():
    sentences, labels = load_winobias_sentences(n_max=24)
    print(f"AUC inputs: n={len(sentences)} label_balance={sum(labels)}/{len(labels)}")
    model = HuggingFaceModel("t5-small", task="seq2seq")
    result = CounterfactualAucScore().compute(
        model=model,
        sentences=sentences,
        labels=labels,
        n_seeds=3,
        test_ratio=0.3,
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
