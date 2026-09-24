"""Demo of ``DiscoveryOfCorrelationsScore`` with original DisCo stimuli."""

from transformers import pipeline

from fairLMs.definitions.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.disco.stimuli import (
    CHRISTIAN_WORDS,
    FEMALE_WORDS,
    MALE_WORDS,
    MUSLIM_WORDS,
    RELIGION_TEMPLATES,
    occupation_templates,
)
from fairLMs.definitions import DiscoveryOfCorrelationsScore
from fairLMs.definitions.models import HuggingFaceModel


def main():
    # fill-mask pipeline expected by DisCo
    pipe = pipeline("fill-mask", model="bert-base-uncased", device=-1)
    metric = DiscoveryOfCorrelationsScore(k=3, n_bootstrap=100, seed=0)

    gender_templates = occupation_templates(cap=20)
    print(f"Gender DisCo templates: {len(gender_templates)}")
    result = metric.compute(
        model=pipe,
        group1_words=MALE_WORDS,
        group2_words=FEMALE_WORDS,
        templates=gender_templates,
    )
    print(result)
    print(f"gender_disco={result.score}")

    print(f"Religion DisCo templates: {len(RELIGION_TEMPLATES)}")
    result_r = metric.compute(
        model=pipe,
        group1_words=CHRISTIAN_WORDS,
        group2_words=MUSLIM_WORDS,
        templates=RELIGION_TEMPLATES,
    )
    print(result_r)
    print(f"religion_disco={result_r.score}")


if __name__ == "__main__":
    main()
