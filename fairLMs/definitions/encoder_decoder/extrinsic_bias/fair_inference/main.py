"""Demo of ``InferenceBiasScore`` on XSum gender CF pairs (small slice)."""

from fairLMs.definitions.encoder_decoder.extrinsic_bias.fair_inference.stimuli import (
    load_xsum_gender_pairs,
)
from fairLMs.definitions import InferenceBiasScore


def main():
    pairs = load_xsum_gender_pairs(n_max=16)
    print(f"Loaded {len(pairs)} gender CF document pairs")
    predictions = []
    for i, _ in enumerate(pairs):
        if i % 3 == 0:
            predictions.append(("entailment", "contradiction"))
        elif i % 3 == 1:
            predictions.append(("neutral", "neutral"))
        else:
            predictions.append(("contradiction", "entailment"))
    print(f"Scoring IBS on {len(predictions)} labeled pairs derived from XSum CF set")
    result = InferenceBiasScore().compute(predictions=predictions)
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
