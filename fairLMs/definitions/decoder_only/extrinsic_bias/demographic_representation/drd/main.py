"""Demo of ``DemographicRepresentationDivergence`` with BBQ/CrowS prompts + AXES."""

from fairLMs.definitions.decoder_only.extrinsic_bias.demographic_representation.drd.stimuli import (
    AXES,
    load_bbq_prompts,
    load_crows_prompts,
)
from fairLMs.definitions import DemographicRepresentationDivergence
from fairLMs.definitions.models import HuggingFaceModel


def main():
    bbq = load_bbq_prompts(n_max=30, per_cat=3)
    crows = load_crows_prompts(n_max=30)
    print(f"BBQ axes: { {k: len(v) for k, v in bbq.items()} }")
    print(f"CrowS axes: { {k: len(v) for k, v in crows.items()} }")

    model = HuggingFaceModel("gpt2", task="causal")
    axis = "gender"
    prompts = (bbq.get(axis) or crows.get(axis) or ["The nurse was a"])[:6]
    words = AXES[axis]
    print(f"Running DRD on axis={axis} n_prompts={len(prompts)}")
    result = DemographicRepresentationDivergence().compute(
        model=model,
        prompts=prompts,
        stereo_words=words["stereo"],
        counter_words=words["counter"],
        max_new_tokens=20,
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
