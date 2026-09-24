"""Demo of ``TranslationSimilarityScore`` on gender CF pairs (small slice)."""

from fairLMs.definitions.encoder_decoder.extrinsic_bias.individual_fairness.stimuli import (
    load_translation_pairs,
)
from fairLMs.definitions import TranslationSimilarityScore
from fairLMs.definitions.models import HuggingFaceModel


def main():
    pairs = load_translation_pairs(n_max=6)
    print(f"Translation CF pairs: {len(pairs)}")
    translator = HuggingFaceModel("t5-small", task="seq2seq")
    labse = HuggingFaceModel("sentence-transformers/LaBSE", task="encoder")
    loaded_labse = labse.load()
    result = TranslationSimilarityScore().compute(
        model=translator,
        labse_model=loaded_labse.model,
        labse_tokenizer=loaded_labse.tokenizer,
        pairs=pairs,
        tgt_lang="French",
        max_new_tokens=32,
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
