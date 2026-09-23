"""Demo of ``DemographicNextTokenProportion`` with BBQ/CrowS prompts + AXES."""

from fairLMs.definition.decoder_only.extrinsic_bias.demographic_representation.dnp.stimuli import (
    AXES,
    load_bbq_prompts,
    load_crows_prompts,
)
from fairLMs.metrics import DemographicNextTokenProportion
from fairLMs.models import HuggingFaceModel


def main():
    bbq = load_bbq_prompts(n_max=40, per_cat=4)
    crows = load_crows_prompts(n_max=40)
    print(f"BBQ axes with prompts: { {k: len(v) for k, v in bbq.items()} }")
    print(f"CrowS axes with prompts: { {k: len(v) for k, v in crows.items()} }")

    model = HuggingFaceModel("gpt2", task="causal")
    metric = DemographicNextTokenProportion()

    # Primary demo: gender axis (BBQ preferred, CrowS fallback)
    axis = "gender"
    prompts = (bbq.get(axis) or crows.get(axis) or ["The nurse was a", "The doctor was a"])[:12]
    words = AXES[axis]
    print(f"Running DNP on axis={axis} n_prompts={len(prompts)}")
    result = metric.compute(
        model=model,
        prompts=prompts,
        stereo_words=words["stereo"],
        counter_words=words["counter"],
        neutral_words=words["neutral"],
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
