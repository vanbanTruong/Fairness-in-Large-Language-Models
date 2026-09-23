"""Demo of ``BiasAmplifierScore`` with BiasAsker groups/props (OpenAI-gated)."""

import os
from pathlib import Path

from fairLMs.definition.decoder_only.extrinsic_bias.performance_disparity.ba.stimuli import (
    BIASASKER_AB,
    BIASASKER_RB,
    load_groups_and_props,
)
from fairLMs.metrics import BiasAmplifierScore
from fairLMs.models import OpenAIModel

_LEAF = Path(__file__).resolve().parent


def main():
    buckets, props = load_groups_and_props(_LEAF, n_props=12, n_groups_per=8)
    groups = ["old", "young"]
    # Use short property phrases (strip leading "are ")
    properties = []
    for p in props:
        s = p.strip()
        if s.lower().startswith("are "):
            s = s[4:].strip().strip('"')
        if s:
            properties.append(s)
    properties = properties[:8]
    print(f"BiasAsker: groups={groups} n_props={len(properties)} age_rows old/young={len(buckets['old'])}/{len(buckets['young'])}")

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — loaded BiasAsker stimuli but skipping BA.compute()")
        print(f"Sample props: {properties[:3]}")
        return

    model = OpenAIModel()
    result = BiasAmplifierScore().compute(
        model=model,
        groups=groups,
        properties=properties,
        ab_template=BIASASKER_AB,
        rb_template=BIASASKER_RB,
        max_new_tokens=20,
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
