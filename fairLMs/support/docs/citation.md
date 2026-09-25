# Citing FairLMs

## The paper

```bibtex
@unpublished{larionov_fairLMs_paper,
  author = {Larionov, Michael and Zhang, Jiale and Yin, Zhipeng and
            Wang, Zichong and Zhang, Wenbin},
  title  = {{FairLMs}: A Turnkey Library for Fairness in Language Models},
  note   = {Manuscript},
  year   = {2026}
}
```

!!! note "Update on acceptance"
    Venue, volume and page details are not filled in above because they are not
    settled. Replace this entry with the published `@article` when it is.

## The software

Cite the release you actually ran. Version numbers matter here more than usual:
metric definitions and their reported units have changed between versions, so
"FairLMs" without a version does not identify a measurement.

```bibtex
@software{larionov_fairLMs,
  author  = {Larionov, Michael and Zhang, Jiale and Yin, Zhipeng and
             Wang, Zichong and Zhang, Wenbin},
  title   = {{FairLMs}: A Turnkey Library for Fairness in Language Models},
  url     = {https://github.com/vanbanTruong/Fairness-in-Large-Language-Models/tree/main/fairLMs},
  version = {0.4.0},
  license = {MIT},
  year    = {2026}
}
```

The repository also ships a [`CITATION.cff`](https://github.com/vanbanTruong/Fairness-in-Large-Language-Models/blob/main/fairLMs/support/metadata/CITATION.cff),
so GitHub's "Cite this repository" button and tools such as `cffconvert` will
produce this metadata for you in APA, BibTeX or other formats.

Get the version you ran from the package itself rather than from these docs:

```python
import fairLMs
fairLMs.__version__
```

## The metrics

**Citing FairLMs is not a substitute for citing the metric.** Every metric here
re-implements a published definition; the contribution of this library is the
uniform interface, not the measures. A paper reporting a CrowS-Pairs score
should cite Nangia et al. alongside FairLMs, and likewise for WEAT, StereoSet,
BBQ and the rest.

Each metric's docstring names its source, visible in the
[API reference](api/metrics.md) and via `help()`:

```python
from fairLMs.definitions import CrowSPairsScore
help(CrowSPairsScore)     # "Pseudo-log-likelihood CrowS-Pairs Score (Nangia et al., 2020)."
```

## The datasets

Benchmarks carry their own licenses and citation requirements, independent of
this library's MIT license. CrowS-Pairs and BBQ are redistributed with the
package; eleven others are downloaded from the Hugging Face Hub at first use;
Bias-NLI, RedditBias, Grep-BiasIR and UnQover you supply yourself. See
[Loaders](registry/loaders.md) for which is which, and cite the benchmark
authors for any dataset you evaluate on.

## What to record for reproducibility

A bias score is not reproducible from the metric name alone. Alongside the
citation, report:

- **The FairLMs version**, from `fairLMs.__version__`.
- **The full metric configuration**, which `get_params()` gives you verbatim:
  this is what `seed`, `n_samples`, `pooling` and the rest were set to.
- **The checkpoint and its task**, e.g. `bert-base-uncased` with `task="mlm"`.
  The same checkpoint under a different head is a different measurement.
- **The dataset, split and any subsetting**, including `n_max`.
- **Anything in `details`** that qualifies the headline number: sample counts,
  the `seed` behind a permutation p-value, the `unit` on a diagnostic.

```python
from fairLMs.definitions import WEAT

metric = WEAT(seed=0)
result = metric.compute(model, data)

print(fairLMs.__version__)   # 0.4.0
print(metric.get_params())   # {'n_samples': 10000, 'pooling': 'mean', 'seed': 0}
print(result.details)        # includes p_value, seed, embedded_with, n_targets
```

Diagnostics serialize all of this for you: `report.to_json()` carries the
schema version, the spec, every component's provenance and assumptions, and any
rule that was applied. That JSON is the artifact to archive.
