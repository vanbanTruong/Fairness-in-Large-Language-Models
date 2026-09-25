# Contributing

## Development setup

```bash
git clone https://github.com/vanbanTruong/Fairness-in-Large-Language-Models.git
cd Fairness-in-Large-Language-Models/fairLMs
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
pytest --cov=fairLMs --cov-report=term-missing
```

## Adding a metric

See [Writing a metric](guides/custom-metric.md) for the full contract. In short:

1. Subclass `FairnessMetric`, declare `name`, `bias_type` and `architectures`,
   and keep `__init__` to configuration stored verbatim.
2. Implement `compute(model, data, **legacy) -> MetricResult`, resolving the
   model through `fairLMs.definitions.resolve.get_tokenizer_model` rather than
   loading a checkpoint yourself.
3. Add the class to `METRIC_REGISTRY` in `fairLMs/definitions/__init__.py` and to
   `__all__`.
4. Run the suite. The contract tests iterate over the registry, so
   registration alone earns the checks.

```bash
pytest -k your_metric_name
```

The low-level implementation belongs under the
`fairLMs/definitions/{architecture}/{bias_type}/…` taxonomy, with a short
`main.py` demonstrating the public API. Its root-level wrapper is the stable
surface and can evolve independently of the nested calculation module.

## Adding a loader

Subclass `FairnessDataset`, implement `load()`, and export the class from
`fairLMs/datasets/__init__.py`. The export list is what the generated
[Loaders](registry/loaders.md) page reads, and the first sentence of the class
docstring is the description it prints, so make it say what `load()` returns.

Set `data_origin` to where the bytes come from; the table groups loaders by it.
Prefer Hub download over vendoring. The existing bundled snapshots are small,
license-checked and checksum-pinned; anything you add under
`fairLMs/datasets/resources/` must meet the same conditions and be declared in
`[tool.setuptools.package-data]` or it will not ship in the wheel. Fetch a
published data file with `fairLMs.datasets._sources.hub_file` rather than
`datasets.load_dataset`: several benchmark repositories still ship a loading
script, which `datasets>=3` will not run. For a benchmark with no Hub release,
take a `root=` and resolve it with `_sources.resolve_root`, whose error names
the project page and the layout you expected.

## Adding a diagnostic

See [Writing a diagnostic](guides/custom-diagnostic.md). Register in
`DIAGNOSTIC_REGISTRY`; every component must report one of `ready`, `blocked`,
`not_applicable`, `failed`, and non-ready components must carry `value=None`.

## Docs

```bash
python support/scripts/gen_registry_docs.py    # regenerate registry tables
mkdocs serve -f support/mkdocs.yml
```

Registry pages are generated from the registries themselves, so edit the code, not
the tables. CI runs `gen_registry_docs.py --check` and
`mkdocs build --strict -f support/mkdocs.yml`,
so a stale table or a broken link fails the build.

## Releasing

Bump `fairLMs/_version.py`, then tag; pushing a `v*` tag triggers the publish
workflow, which verifies that the tag matches the package version.

```bash
git tag -a "v$(python support/scripts/package_version.py)" -m "Release" && git push origin --tags
```
