"""Demo of ``NaturalIndirectEffect`` with occupation probes + gpt2."""

from fairLMs.definitions.decoder_only.intrinsic_bias.attention_head_based_disparity.nie.stimuli import (
    occupation_prompts,
    prompts_to_probes,
)
from fairLMs.definitions import NaturalIndirectEffect
from fairLMs.definitions.models import HuggingFaceModel


def main():
    prompts = occupation_prompts(n_max=12)
    print(f"Built {len(prompts)} occupation probes")

    model = HuggingFaceModel(
        "gpt2",
        task="causal",
        model_kwargs={"attn_implementation": "eager"},
    )
    loaded = model.load()
    probes = prompts_to_probes(prompts, loaded.tokenizer)
    cfg = loaded.model.config
    n_layers = cfg.n_layer
    n_heads = cfg.n_head
    head_dim = cfg.n_embd // cfg.n_head
    print(f"gpt2: layers={n_layers} heads={n_heads} head_dim={head_dim} probes={len(probes)}")

    result = NaturalIndirectEffect(threshold=0.003).compute(
        model=model,
        probes=probes,
        N_LAYERS=n_layers,
        N_HEADS=n_heads,
        HEAD_DIM=head_dim,
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
