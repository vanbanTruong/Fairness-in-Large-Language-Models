# SVA — Shapley Value Attribution of Bias to Attention Heads

Attributes the model's stereotypical association to individual **encoder
attention heads** via Monte-Carlo Shapley values. A "bias direction" is first
computed as the normalized difference between the mean encoder embeddings of
hand-written stereotypical vs. anti-stereotypical sentences. The bias score of
any coalition of heads is the mean projection gap onto this direction when
only those heads are active (all others zeroed by forward hooks). Shapley
values are estimated with random permutations: heads are activated one by one
and each head is credited with its marginal change in the bias score. The SVA
score is the share of total |attribution| carried by the **top 10% of heads**
— it asks *how concentrated* the bias is.

Calibration: mt5-base has 12 × 12 = 144 encoder heads, so the top 10 % is ~15
heads; under a **uniform** attribution the score would be 15/144 ≈ **0.104**.
The checked-in values (≈0.31–0.35) are well above this, i.e., the metric does
carry information (unlike sign-counting approaches whose chance baseline sits
at their observed value).

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point: loads mT5, builds sentence sets, runs MC-Shapley, writes `sva_results.csv` |
| `sva.py` | Core: head-masking hooks, bias direction, coalition score, MC Shapley (`compute_sva`) |
| `sva_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `google/mt5-base` (encoder only; heads zeroed via forward hooks on `SelfAttention`) | `sva.py` |
| Bias direction | mean(stereo embeddings) − mean(anti embeddings), L2-normalized; seed texts hard-coded in `sva.py` | `compute_stereotype_direction()` |
| MC Shapley | `n_samples = 50` permutations over 144 heads; encoding batch 16 | `compute_sva()` |
| Score | sum of normalized \|φ\| over top `top_pct = 0.10` heads | `compute_sva()` |
| Sample cap | `N_MAX = 100`; SEED = 42 (permutations use the seeded global `random`) | `main.py` |
| Datasets | WinoMT (GitHub URL), Europarl (HF), WinoBias (HF) | `main.py` |

## How to run

```bash
pip install torch transformers datasets pandas
cd <this directory>
python main.py
```

**Cost warning**: each permutation re-encodes every sentence 144 times →
50 × 144 ≈ 7,200 coalition evaluations. This is by far the slowest metric in
this group; plan for hours on CPU.

## Output \& Results

[sva_results.csv](https://github.com/michaellarionov/FairnessDefinitionsLLMs/blob/main/fairLLMs/definition/encoder_decoder/intrinsic_bias/stereotypical_association/sva/sva_results.csv).

## Known issues / caveats

1. No implementation errors found: the permutation/marginal-contribution
   structure is a correct MC-Shapley estimator, and hooks handle tuple outputs
   and cleanup properly.
2. The empty coalition (all heads zeroed) is a degenerate encoder state —
   standard for Shapley but worth remembering when interpreting small
   coalitions' scores.
3. The stereo/anti seed sentences are hand-written in `sva.py`; the "dataset"
   rows vary the evaluation sentences, not the direction-defining texts.
4. WinoMT is a network dependency (GitHub raw URL). `main.py` was reviewed for
   settings only.
