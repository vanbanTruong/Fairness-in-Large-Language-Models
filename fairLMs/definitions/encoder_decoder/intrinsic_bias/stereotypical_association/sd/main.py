"""Demo of ``StereotypicalDivergence`` on a small WinoMT slice."""

from fairLMs.definitions.encoder_decoder.intrinsic_bias.stereotypical_association.sd.stimuli import (
    load_winomt_stereo_anti,
)
from fairLMs.definitions import StereotypicalDivergence
from fairLMs.definitions.models import HuggingFaceModel


def main():
    stereo_s, stereo_l, anti_s, anti_l = load_winomt_stereo_anti(n_max=6)
    print(f"SD inputs: stereo={len(stereo_s)} anti={len(anti_s)}")
    model = HuggingFaceModel("t5-small", task="seq2seq")
    result = StereotypicalDivergence().compute(
        model=model,
        stereo_sentences=stereo_s,
        stereo_labels=stereo_l,
        anti_sentences=anti_s,
        anti_labels=anti_l,
        max_new_tokens=16,
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
