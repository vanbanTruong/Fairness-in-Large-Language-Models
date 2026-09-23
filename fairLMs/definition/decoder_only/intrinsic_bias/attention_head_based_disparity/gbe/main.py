"""Demo of ``GradientBasedBiasEstimation`` with StereoSet word lists + gpt2."""

from fairLMs.definition.decoder_only.intrinsic_bias.attention_head_based_disparity.gbe.stimuli import (
    STEREOSET_A,
    STEREOSET_B,
    STEREOSET_X,
    STEREOSET_Y,
)
from fairLMs.metrics import GradientBasedBiasEstimation
from fairLMs.models import HuggingFaceModel


def main():
    print(
        f"StereoSet lists: |X|={len(STEREOSET_X)} |Y|={len(STEREOSET_Y)} "
        f"|A|={len(STEREOSET_A)} |B|={len(STEREOSET_B)}"
    )
    model = HuggingFaceModel(
        "gpt2",
        task="causal",
        model_kwargs={"attn_implementation": "eager"},
    )
    result = GradientBasedBiasEstimation().compute(
        model=model,
        X=STEREOSET_X,
        Y=STEREOSET_Y,
        A=STEREOSET_A,
        B=STEREOSET_B,
        verbose=False,
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
