"""Demo of ``CounterfactualRobustness`` with StereoSet / OpenML CF pairs (OpenAI-gated)."""

import os

from fairLMs.definitions.decoder_only.extrinsic_bias.counterfactual_fairness.cr.stimuli import (
    load_german_credit,
    load_heart_disease,
    load_stereoset_gender_cf,
)
from fairLMs.definitions import CounterfactualRobustness
from fairLMs.definitions.models import OpenAIModel


def main():
    f_ss, cf_ss = load_stereoset_gender_cf(n_max=20)
    f_gc, cf_gc = load_german_credit(n_max=20)
    f_hd, cf_hd = load_heart_disease(n_max=20)
    print(
        "Loaded CF pairs: StereoSet=%d GermanCredit=%d HeartDisease=%d"
        % (len(f_ss), len(f_gc), len(f_hd))
    )

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — stimuli restored; skipping CR.compute()")
        if f_ss:
            print("  sample factual:", repr(f_ss[0]))
            print("  sample CF:     ", repr(cf_ss[0]))
        return

    model = OpenAIModel()
    factual, counterfactual = (f_ss, cf_ss) if f_ss else (f_gc, cf_gc)
    if not factual:
        factual, counterfactual = f_hd, cf_hd
    result = CounterfactualRobustness().compute(
        model=model,
        factual_prompts=factual[:12],
        counterfactual_prompts=counterfactual[:12],
    )
    print(result)
    print("score=%s" % result.score)


if __name__ == "__main__":
    main()
