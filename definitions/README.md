# FairnessDefinitionsLLMs

A Python library of fairness definitions and bias metrics for large language models (`fairLLMs`). Metrics are organized by **architecture** (encoder-only, decoder-only, encoder–decoder) and by **bias type** (intrinsic vs. extrinsic), with a runnable implementation and local README for each metric.

## Taxonomy

```
fairLLMs/definition/
├── encoder_only/          # Masked LMs (e.g. BERT)
│   ├── intrinsic_bias/    # Embedding / probability associations inside the model
│   └── extrinsic_bias/    # Downstream task disparities
├── decoder_only/          # Autoregressive LMs (e.g. GPT-2, Llama-2)
│   ├── intrinsic_bias/
│   └── extrinsic_bias/
└── encoder_decoder/       # Seq2seq models (e.g. T5, mT5, mBART)
    ├── intrinsic_bias/
    └── extrinsic_bias/
```

- **Intrinsic bias** — associations measurable from representations, token probabilities, or attention without a task head.
- **Extrinsic bias** — disparities that show up on downstream behavior (QA, NLI, generation, translation, summarization, etc.).

Each leaf directory typically contains `main.py` (runner), a metric module, optional `data/`, results CSV, and a `README.md` with parameters, references, and how to run.

## Metrics

### Encoder-only

| Category | Metric | Abbrev. |
|---|---|---|
| Similarity-based | Word / Sentence / Contextualized Embedding Association Test | WEAT, SEAT, CEAT |
| Masked-token | Contrast-Based Score, Discovery of Correlations, Log Probability Bias Score | CBS, DisCo, LPBS |
| Pseudo log-likelihood | All Unmasked Likelihood, Attention-weighted AUL, CrowS-Pairs Score, Pseudo Log-Likelihood, StereoSet CAT / iCAT | AUL, AULA, CPS, PLL, CAT |
| Extrinsic | BBQ-style context disparity, Equal Opportunity, Fair Inference | S_DIS / S_AMB, EO, FI |

### Decoder-only

| Category | Metric | Abbrev. |
|---|---|---|
| Stereotypical association | Concept Association, Stereotypical Log-Likelihood | CA, SLL |
| Attention-head disparity | Gradient-based Bias Estimation, Natural Indirect Effect | GBE, NIE |
| Demographic representation | Demographic Normalized Probability, Demographic Representation Disparity | DNP, DRD |
| Counterfactual fairness | Change Rate, Counterfactual Token Fairness | CR, CTF |
| Performance disparity | Accuracy Disparity, BiasAsker, Sensitive-to-Neutral Similarity | AD, BA, SNS |

### Encoder–decoder

| Category | Metric | Abbrev. |
|---|---|---|
| Stereotypical association | Stereotype Disparity, Shapley Value Attribution | SD, SVA |
| Algorithmic disparity | Lexical Frequency Profile, Morphological Complexity Disparity | LFP, MCD |
| Extrinsic | Counterfactual Fairness, Idealized Bias Score, Semantic Similarity, Normalized Position Disparity | AUC, IBS, SS, NPD |

Full method details, datasets, and run settings live in each metric’s README under `fairLLMs/definition/`.

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/michaellarionov/FairnessDefinitionsLLMs.git
cd FairnessDefinitionsLLMs
pip install -e .
# or, for the fuller dependency set used by many runners:
pip install -r requirements.txt
```

Some decoder-only metrics use gated Hugging Face models (e.g. Llama-2). Authenticate first:

```bash
export HF_TOKEN=...   # or: huggingface-cli login
```

API-based runners (where applicable) need an `OPENAI_API_KEY`.

## Running a metric

From the repository root, after `pip install -e .`:

```bash
python -m fairLLMs.definition.encoder_only.intrinsic_bias.similarity_based.seat.main
```

Equivalently, from a metric directory:

```bash
cd fairLLMs/definition/encoder_only/intrinsic_bias/similarity_based/seat
python main.py
```

Results are written as a CSV next to the runner (e.g. `seat_results.csv`). See that metric’s README for model names, seeds, sample caps, and expected columns.

## Project layout

| Path | Role |
|---|---|
| `fairLLMs/` | Installable package |
| `fairLLMs/definition/` | Metric implementations by architecture and bias type |
| `pyproject.toml` | Package metadata and core dependencies |
| `requirements.txt` | Broader runtime deps (datasets, OpenAI, etc.) |

## License

This project is licensed under the [MIT License](LICENSE).
