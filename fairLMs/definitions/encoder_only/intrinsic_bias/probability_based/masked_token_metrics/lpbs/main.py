"""Demo of ``LogProbabilityBiasScore`` with original WinoBias / Bios / XNLI stimuli."""

from fairLMs.definitions.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.lpbs.stimuli import (
    BIOS_ATTRIBUTES,
    BIOS_TEMPLATES,
    GENDER_PAIR,
    RELIGION_PAIR,
    WINOBIAS_OCCUPATIONS,
    WINOBIAS_TEMPLATES,
    XNLI_RELIGION_ATTRS,
    XNLI_TEMPLATES,
)
from fairLMs.definitions import LogProbabilityBiasScore
from fairLMs.definitions.models import HuggingFaceModel


def _run(name, model, gender_words, attrs, templates):
    print(f"\n=== {name} | n_attrs={len(attrs)} n_templates={len(templates)} ===")
    scores = []
    for template in templates:
        result = LogProbabilityBiasScore(
            gender_words=gender_words,
            template=template,
        ).compute(model=model, attribute_words=attrs)
        scores.append(result.score)
        print(f"  template={template!r} score={result.score:.4f}")
    mean = sum(scores) / len(scores)
    print(f"  mean_lpbs={mean:.4f}")
    return mean


def main():
    model = HuggingFaceModel("bert-base-uncased", task="mlm")
    _run("WinoBias occupations", model, GENDER_PAIR, WINOBIAS_OCCUPATIONS, WINOBIAS_TEMPLATES)
    _run("Bias-in-Bios professions", model, GENDER_PAIR, BIOS_ATTRIBUTES, BIOS_TEMPLATES[:5])
    _run("XNLI religion attrs", model, RELIGION_PAIR, XNLI_RELIGION_ATTRS, XNLI_TEMPLATES)


if __name__ == "__main__":
    main()
