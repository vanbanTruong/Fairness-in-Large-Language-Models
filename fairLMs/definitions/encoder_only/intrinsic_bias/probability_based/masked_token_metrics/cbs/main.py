"""Demo of ``ContrastBasedScore`` with original nationality / religion stimuli."""

from fairLMs.definitions.encoder_only.intrinsic_bias.probability_based.masked_token_metrics.cbs.stimuli import (
    BIOS_CONTRASTS,
    BIOS_TEMPLATES,
    NATIONALITY_TERMS,
    RELIGION_CONTRASTS,
    RELIGION_TERMS,
    WINOBIAS_CONTRASTS,
    WINOBIAS_TEMPLATES,
    XNLI_TEMPLATES,
)
from fairLMs.definitions import ContrastBasedScore
from fairLMs.definitions.models import HuggingFaceModel


def main():
    model = HuggingFaceModel("bert-base-uncased", task="mlm")
    metric = ContrastBasedScore(n_bootstrap=50, n_perm=50, seed=0)

    configs = [
        ("WinoBias/Nationality", NATIONALITY_TERMS, WINOBIAS_CONTRASTS, WINOBIAS_TEMPLATES),
        ("Bias-in-Bios/Nationality", NATIONALITY_TERMS, BIOS_CONTRASTS, BIOS_TEMPLATES),
        ("XNLI/Religion", RELIGION_TERMS, RELIGION_CONTRASTS, XNLI_TEMPLATES),
    ]
    for name, groups, contrasts, templates in configs:
        print(f"\n=== {name} ===")
        result = metric.compute(
            model=model,
            group_terms=groups,
            contrast_pairs=contrasts,
            templates=templates,
        )
        print(result)
        print(f"score={result.score}")


if __name__ == "__main__":
    main()
