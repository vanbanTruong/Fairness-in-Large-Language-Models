"""Demo of ``StereotypicalLogLikelihood`` with Bias-in-Bios / BBQ occupation pairs."""

from fairLMs.definitions.decoder_only.intrinsic_bias.stereotypical_association.sll.stimuli import (
    BBQ_OCCUPATION_TRIPLES,
    bios_occupation_pairs,
)
from fairLMs.definitions import StereotypicalLogLikelihood
from fairLMs.definitions.models import HuggingFaceModel


def main():
    bios_pairs = bios_occupation_pairs()
    pairs = bios_pairs + BBQ_OCCUPATION_TRIPLES
    print(f"Occupation pairs: bios={len(bios_pairs)} bbq_fallback={len(BBQ_OCCUPATION_TRIPLES)} total={len(pairs)}")

    model = HuggingFaceModel("gpt2", task="causal")
    result = StereotypicalLogLikelihood().compute(
        model=model,
        occupation_pairs=pairs,
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
