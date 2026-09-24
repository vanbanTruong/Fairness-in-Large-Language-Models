# Reproduction of published values

This page records runs in which FairLMs was pointed at a public checkpoint and a
published benchmark and compared with the number published for that pair. Each
row states the exact configuration, so the comparison can be repeated with
`pip install fairLMs` and the snippet below.

## CrowS-Pairs on `bert-base-uncased`

Published values are from Nangia et al. (2020), Table 2 and the bias-category
table. FairLMs values were produced with `CrowSPairsScore()` on the bundled
`crows_pairs_anonymized.csv` (1,508 pairs), `HuggingFaceModel("bert-base-uncased",
task="mlm")`, CPU, library version 0.5.0. The metric is the percentage of pairs
for which the masked language model assigns the higher pseudo-log-likelihood to
the more stereotyping sentence (`sent_more`), for stereo and anti-stereo rows
alike, exactly as the official `metric.py` counts it.

| Subset | Pairs | Published | FairLMs |
|---|---:|---:|---:|
| All pairs | 1,508 | 60.5 | 60.48 |
| Stereotype pairs | 1,290 | 61.1 | 61.09 |
| Anti-stereotype pairs | 218 | 56.9 | 56.88 |

| Bias category | Published | FairLMs |
|---|---:|---:|
| Race / color | 58.1 | 58.14 |
| Gender / gender identity | 58.0 | 58.02 |
| Socioeconomic status / occupation | 59.9 | 59.88 |
| Nationality | 62.9 | 62.89 |
| Religion | 71.4 | 71.43 |
| Age | 55.2 | 55.17 |
| Sexual orientation | 67.9 | 67.86 |
| Physical appearance | 63.5 | 63.49 |
| Disability | 61.7 | 61.67 |

```python
from fairLMs.datasets import CrowSPairs
from fairLMs.definitions import CrowSPairsScore
from fairLMs.definitions.models import HuggingFaceModel

result = CrowSPairsScore().compute(HuggingFaceModel("bert-base-uncased", task="mlm"), CrowSPairs())
print(result.score)          # 60.48
print(result.by_category)    # per bias_type
```

The run took about four minutes on a laptop CPU.

### What the reproduction caught

The first run reported 58.5 overall and 48.5 for gender. The difference was
traced to the loader, which had swapped the two sentences for the 218
anti-stereotype rows; the official metric counts a preference for `sent_more`
regardless of the `stereo_antistereo` label. The loader was corrected in 0.5.0
and the regression is pinned by `tests/test_crows_pairs_orientation.py`. Scores
obtained with earlier development versions on anti-stereotype rows were inverted.

## SEAT on `bert-base-uncased`

Published values are the baseline row of Meade et al. (2022, bias-bench, Table 1)
for the six gender SEAT tests of May et al. (2019). The sentence sets are the
`sent-weat*.jsonl` files distributed in the bias-bench repository; they are not
bundled with FairLMs and are downloaded by `examples/reproduce_seat_biasbench.py`.
Each sentence was embedded with `HuggingFaceEmbeddingBackend(pooling="mean")`
(attention-masked mean of the last hidden state of `AutoModel("bert-base-uncased")`),
and the effect size is Caliskan's *d* from `WEAT().compute(None, VectorSets(...))`.

| Test | Published | FairLMs, mean pooling | FairLMs, CLS pooling |
|---|---:|---:|---:|
| SEAT-6 | 0.931 | 0.931 | 1.044 |
| SEAT-6b | 0.090 | 0.090 | 0.108 |
| SEAT-7 | −0.124 | −0.124 | 0.176 |
| SEAT-7b | 0.937 | 0.937 | 0.725 |
| SEAT-8 | 0.783 | 0.783 | 0.813 |
| SEAT-8b | 0.858 | 0.858 | 1.004 |
| Average absolute effect size | 0.620 | 0.620 | 0.645 |

The mean-pooling column matches every published value to three decimals. The
CLS column does not, which is why every FairLMs result records its pooling rule
in provenance: an effect size is comparable only under the same declared pooling.
This run exercises the shared association-test kernel and the embedding backend
on May et al.'s exact sentences; `SEAT()` proper substitutes terms into
templates and is configured with `pooling=` in the same way.

```bash
python examples/reproduce_seat_biasbench.py            # downloads the six sentence files
python examples/reproduce_seat_biasbench.py --pooling cls
```

## Not yet reproduced

StereoSet is shipped as a loader, but `ContextAssociationTestScore` aggregates
micro over triples and is not the official StereoSet scorer, so its value is not
compared with the published StereoSet numbers here. Contributions of further
reproduction rows, with the exact configuration stated, are welcome.
