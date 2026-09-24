# LFP — Lexical Frequency Profile (Translation)

Measures the **lexical richness of the model's French translations**: every
word of the generated French is classified into a frequency band using the
`wordfreq` package — band 1 = among the 1,000 most frequent French words,
band 2 = rank 1,001–2,000, band 3 = rarer — and the metric reports the
proportion of each band (pb1/pb2/pb3). Comparing profiles across datasets
reveals whether the model produces systematically simpler (higher pb1)
language for some inputs — an algorithmic-disparity signal.

## Public API

```python
from fairLMs.definitions import LexicalFrequencyProportion
from fairLMs.definitions.models import HuggingFaceModel

result = LexicalFrequencyProportion().compute(
    model=HuggingFaceModel("t5-small", task="seq2seq"),
    sentences=["translate English to French: The doctor is busy."],
)
print(result.score)
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Short public-API demo using `LexicalFrequencyProportion`; writes results CSV for continuity |
| `lfp.py` | Core: translation, word tokenization, frequency-band classification (`compute_lfp`) |
| `lfp_results.csv` | Output of the last run |

## Parameters and settings

| Parameter | Value | Where |
|---|---|---|
| Model | `google-t5/t5-base`, prefix `"translate English to French: "` | `lfp.py` / `main.py` |
| Decoding | beam search, `num_beams=4`, `max_new_tokens=128` | `lfp.py` |
| Frequency bands | `wordfreq.top_n_list("fr", 1000/2000)`; non-alphabetic tokens fall into band 1 | `lfp.py` → `_classify_word()` |
| Tokenization | Unicode letter runs (`[^\W\d_]+`) | `lfp.py` |
| Sample cap | `N_MAX = 200`; `SEED = 42` | `main.py` |
| Datasets | WinoMT (GitHub URL), Europarl (HF `Helsinki-NLP/europarl` en-fr), XNLI (HF) | `main.py` |

Bootstrap CIs are computed for console output but **not** written to CSV.

## How to run

**Preferred:** use the public API above (and/or examples under `examples/` at the repo root).

**Optional legacy demo** from the repository root:

```bash
pip install -e .
python -m fairLMs.definitions.encoder_decoder.intrinsic_bias.algorithmic_disparity.lfp.main
```

## Output \& Results

`lfp_results.csv`: `dataset`, `pb1`, `pb2`, `pb3`.
