# Comparison with related tools

This page carries the feature-level comparison that the JMLR MLOSS paper refers to.
Every cell is checked against the version named in the last row; a dash means the tool
does not provide the capability or does not document it. Corrections are welcome
through the issue tracker.

| | FairLMs | FairPy | bias-bench | LangFair | WEFE |
|---|---|---|---|---|---|
| Model evidence accepted | internal representations, token probabilities, generations, precomputed predictions | HuggingFace causal and masked checkpoints | encoder checkpoints (BERT, ALBERT, RoBERTa) and GPT-2 | generations of any hosted LLM (via LangChain) | static word vectors |
| Bias metrics | 33 (19 intrinsic, 14 extrinsic) | 6 (Hellinger distance, WEAT/SEAT, StereoSet, HONEST, log-likelihood, F1) | 3 benchmarks (SEAT, StereoSet, CrowS-Pairs) | 5 metric families (toxicity, stereotype, counterfactual, classification, recommendation) | 7 |
| Mitigation methods | 14 in four categories (pre 4, in 4, intra 3, post 3) | 6 (DiffPruning, INLP, SentenceDebias, CDA, dropout, Self-Debias) | 5 (CDA, dropout, INLP, SentenceDebias, Self-Debias) | – | 5 |
| Requirements checked before execution | architecture, capabilities, access level, evidence type; refusal names the missing item | – | – | – | query template `(t, a)` only |
| Before/after comparison | `compare_before_after` on the same declared metric set and evidence; intrinsic, extrinsic and utility deltas kept separate | manual | experiment scripts, with GLUE utility (`run_glue.py`) | re-run `AutoEval` | manual |
| Diagnostics of evaluation data and scorers | 14 (10 dataset, 4 scoring instrument) | – | – | – | – |
| Result provenance (configuration, evidence hash, library version) | yes | – | – | – | – |
| Installation | `pip install fairLMs` | source (`git clone`) | source (`pip install -e .`) | `pip install langfair` | `pip`, `conda` |
| Version compared | 0.5.0 | arXiv:2302.05508v2 (April 2025) | McGill-NLP/bias-bench, main branch, September 2026 | JOSS 10:7570 (2025) and README, September 2026 | JMLR 26 (2025), paper 22-1133 |

## Where the other tools are stronger

- **bias-bench** ships GLUE utility scripts and released debiased checkpoints for a
  fixed model set, so a full before/after study on those models needs no extra code.
- **LangFair** automates output-based evaluation of any hosted LLM through `AutoEval`
  and covers toxicity and recommendation fairness, which FairLMs does not.
- **WEFE** established the measurement-plus-mitigation contract for static embeddings,
  with a large user base and replications of published studies.
- **FairPy** is the closest in scope and documents concrete cross-model failures
  (word-level probability metrics under some tokenizers, differing output-layer
  conventions) that motivated FairLMs' declarations.

## Where FairLMs differs

FairLMs checks each component's declared requirements against the model and the
evidence before execution, re-measures a mitigated system under the same declared
metric set, and audits the evaluation datasets and scoring instruments themselves.
