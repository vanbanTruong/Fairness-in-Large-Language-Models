"""Demo of ``NormalizedPositionDistance`` on a small XSum article slice."""

from fairLMs.definition.encoder_decoder.extrinsic_bias.position_based.stimuli import (
    load_xsum_articles,
)
from fairLMs.metrics import NormalizedPositionDistance
from fairLMs.models import HuggingFaceModel


def main():
    articles = load_xsum_articles(n_max=4)
    print(f"Articles for NPD: {len(articles)}")
    model = HuggingFaceModel("t5-small", task="seq2seq")
    result = NormalizedPositionDistance().compute(
        model=model,
        articles=articles,
        max_new_tokens=32,
        K=5,
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
