"""Demo of ``MorphologicalChoiceDivergence`` on WinoMT / XNLI slices."""

from fairLMs.definitions.encoder_decoder.intrinsic_bias.algorithmic_disparity.mcd.stimuli import (
    load_europarl_sentences,
    load_winomt_sentences,
    load_xnli_premises,
)
from fairLMs.definitions import MorphologicalChoiceDivergence
from fairLMs.definitions.models import HuggingFaceModel


def main():
    winomt = load_winomt_sentences(n_max=8)
    xnli = load_xnli_premises(n_max=8)
    euro = load_europarl_sentences(n_max=4)
    sentences = winomt or xnli or euro
    print(f"MCD sentence counts: winomt={len(winomt)} xnli={len(xnli)} europarl={len(euro)}")
    prompts = [f"translate English to French: {s}" for s in sentences[:6]]
    model = HuggingFaceModel("t5-small", task="seq2seq")
    result = MorphologicalChoiceDivergence().compute(
        model=model,
        sentences=prompts,
        max_new_tokens=24,
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
