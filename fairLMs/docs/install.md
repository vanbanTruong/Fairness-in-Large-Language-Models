# Installation

```bash
pip install fairLMs
```

Requires Python 3.10 or later.

## Extras

| Extra | Command | Adds |
|---|---|---|
| `openai` | `pip install "fairLMs[openai]"` | `openai`, for API-served decoder wrappers |
| `dev` | `pip install "fairLMs[dev]"` | `pytest`, `pytest-cov`, plus `openai` |
| `all` | `pip install "fairLMs[all]"` | `openai`, `dev`, `accelerate`, `sentencepiece`, `gender-guesser`, `tqdm` |
| `docs` | `pip install "fairLMs[docs]"` | `mkdocs-material`, `mkdocstrings[python]`, `mkdocs-jupyter` |

`torch`, `transformers`, `datasets`, `scikit-learn`, `wordfreq` and `nltk` are
required dependencies, not extras: `fairLMs.definitions` imports every metric family
eagerly, so `import fairLMs` needs them present.

## From source

```bash
git clone https://github.com/michaellarionov/FairLMs.git
cd FairLMs
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Prefer a tag over `main` when installing from source. `main` tracks unreleased
work, so any push can change behaviour under you.

## Verify the install

```bash
python -c "import fairLMs; print(fairLMs.__version__)"
```

## Dataset licenses

The library is MIT licensed; each benchmark retains its own license.

The small CrowS-Pairs, WinoBias and Winogender snapshots are bundled and work
offline. BBQ and nine more -- StereoSet, Bias in Bios, XNLI, BOLD, HONEST,
RealToxicityPrompts, HolisticBias, EEC and GAP -- are downloaded from the
Hugging Face Hub at first use. Bias-NLI, RedditBias, Grep-BiasIR and UnQover are
distributed only from their own project pages, so their loaders take a `root=`
pointing at your copy and download nothing; TrustGPT has no data release and
builds its prompts from bundled templates. See
[Loaders](registry/loaders.md) for the per-loader breakdown.

BBQ is downloaded from its Hugging Face mirror into the standard cache on first
use, or can read caller-supplied JSONL files through `data_dir=`. Gated or
rate-limited Hub downloads pick up `HF_TOKEN` or
`HUGGING_FACE_HUB_TOKEN` from the environment automatically.
