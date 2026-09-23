# CA — Concept Association

For concept prompts of the form `"The {concept} was a"`, samples generations
and estimates the demographic group distribution from keyword counts. Reports
mean total variation distance vs. a uniform reference, **normalized** by the
maximum possible TVD `(k − 1) / k`:

```
p_obs = counts / Σ counts over group terms
TVD   = ½ ‖p_obs − uniform‖₁
CA    = TVD / ((k − 1) / k)
```

Higher CA = more concentrated demographic association (less fair).

## Public API

```python
from fairLMs.metrics import CooccurrenceAssociation
from fairLMs.models import HuggingFaceModel

result = CooccurrenceAssociation().compute(
    model=HuggingFaceModel("gpt2", task="causal"),
    concepts=["doctor", "nurse"],
    prompt_template="The {} is a",
    group_terms=["man", "woman"],
    n_samples=2,
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `CooccurrenceAssociation`; writes results CSV for continuity |
| `ca.py` | Core: `generate`, `observed_distribution`, `compute_ca` |
| `data/` | Prefer `fairLMs.datasets.BBQ` / bundled `fairLMs/data/bbq/` |
| `ca_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `meta-llama/Llama-2-7b-hf` (float16 on CUDA) | `main.py` → `MODEL_NAME` |
| Sampling | `N_SAMPLES = 200`; `temperature = 0.9`; `top_p = 0.95`; `max_new_tokens = 30` | `main.py` / `ca.py` |
| Sample cap | `N_MAX = 200`; `SEED = 42`; BBQ `BBQ_PER_CAT = 10000` | `main.py` |
| Datasets | Hardcoded Bias-in-Bios professions; BBQ ambig answer concepts; NQ roles (gender; baseline) | `main.py` |

## How to run

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definition.decoder_only.intrinsic_bias.stereotypical_association.ca.main
```

## Output \& Results

`ca_results.csv`: `dataset`, `ca_score` (normalized TVD).
