"""Demo of ``CooccurrenceAssociation`` with Bias-in-Bios professions + gender groups."""

from fairLMs.definitions.decoder_only.intrinsic_bias.stereotypical_association.ca.stimuli import (
    BIOS_PROFESSIONS,
    GENDER_GROUPS,
    PROMPT_TEMPLATE,
    flatten_groups,
)
from fairLMs.definitions import CooccurrenceAssociation
from fairLMs.definitions.models import HuggingFaceModel


def main():
    concepts = BIOS_PROFESSIONS[:12]
    group_terms = flatten_groups(GENDER_GROUPS)
    print(f"Concepts={len(concepts)} group_terms={len(group_terms)}")

    model = HuggingFaceModel("gpt2", task="causal")
    result = CooccurrenceAssociation().compute(
        model=model,
        concepts=concepts,
        prompt_template=PROMPT_TEMPLATE,
        group_terms=group_terms,
        n_samples=8,
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
