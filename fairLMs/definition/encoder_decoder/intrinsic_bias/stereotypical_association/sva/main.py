"""Demo of ``StereotypicalValueAttribution`` on WinoBias pairs (tiny slice)."""

import numpy as np

from fairLMs.definition.encoder_decoder.intrinsic_bias.stereotypical_association.sva.stimuli import (
    load_winobias_pairs,
)
from fairLMs.metrics import StereotypicalValueAttribution
from fairLMs.models import HuggingFaceModel


def main():
    stereo, anti = load_winobias_pairs(n_max=4)
    print(f"SVA inputs: stereo={len(stereo)} anti={len(anti)}")
    model = HuggingFaceModel("t5-small", task="seq2seq")
    loaded = model.load()
    # t5-small encoder layers/heads
    cfg = loaded.model.config
    n_layers = getattr(cfg, "num_layers", None) or getattr(cfg, "num_hidden_layers", 6)
    n_heads = getattr(cfg, "num_heads", None) or getattr(cfg, "num_attention_heads", 8)
    direction = np.ones(loaded.model.config.d_model, dtype=np.float32)
    direction /= np.linalg.norm(direction) + 1e-8
    result = StereotypicalValueAttribution().compute(
        model=model,
        stereo_sents=stereo,
        anti_sents=anti,
        direction=direction,
        n_layers=min(int(n_layers), 2),
        n_heads=min(int(n_heads), 2),
        n_samples=4,
    )
    print(result)
    print(f"score={result.score}")


if __name__ == "__main__":
    main()
