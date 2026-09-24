# DNP — Demographic Normalized Probability

Extrinsic demographic-representation metric: for each prompt, collect next-token
probability mass on stereo / counter / neutral word lists, renormalize to three
buckets, and average across prompts:

```
P̂_s  = p_s  / (p_s + p_s' + p_d)
P̂_s' = p_s' / (p_s + p_s' + p_d)
P̂_d  = p_d  / (p_s + p_s' + p_d)
```

Fair when all three means ≈ **1/3**. Log-prob is taken on the first subtoken of
each word (leading-space form preferred when present in the vocab).

## Public API

```python
from fairLMs.definitions import DemographicNextTokenProportion
from fairLMs.definitions.models import HuggingFaceModel

# Prompts often come from fairLMs.datasets.BBQ / CrowSPairs.
result = DemographicNextTokenProportion().compute(
    model=HuggingFaceModel("gpt2", task="causal"),
    prompts=["The person who"],
    stereo_words=["he"],
    counter_words=["she"],
    neutral_words=["they"],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `DemographicNextTokenProportion`; writes results CSV for continuity |
| `dnp.py` | Core: `_token_logprobs`, `compute_dnp` |
| `data/` | Prefer `fairLMs.datasets.BBQ` / `CrowSPairs`; BBQ downloads on demand and CrowS-Pairs is bundled |
| `dnp_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `meta-llama/Llama-2-7b-hf` | `main.py` → `MODEL_NAME` |
| Sample cap | `N_MAX = 200`; `SEED = 42`; BBQ `PER_CAT_MAX = 10000` | `main.py` |
| Device | CUDA if available, else CPU | `main.py` |
| Datasets | Local BBQ ambiguous contexts; CrowS-Pairs prefixes (cut at first differing demographic token); Natural Questions roles as **baseline** (gender axis only) | `main.py` |

## How to run

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definitions.decoder_only.extrinsic_bias.demographic_representation.dnp.main
```

## Output \& Results

`dnp_results.csv`: `dataset`, `P_s`, `P_sp`, `P_d`.

NQ is labeled as a baseline (not a bias benchmark) in the runner.
