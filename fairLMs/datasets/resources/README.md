# Bundled dataset resources

FairLMs ships three small, redistribution-safe evaluation snapshots:

- `crows_pairs/`: the canonical CrowS-Pairs CSV;
- `wino_bias/`: all four WinoBias configurations, with validation and test
  Parquet splits;
- `winogender/`: all 720 sentences, the 120 source templates and occupation
  statistics.

`checksums.json` pins every dataset file. `NOTICE.md` records its source,
revision and third-party license; the two MIT license texts are also stored
inside their dataset directories.

Large datasets, including BBQ, are downloaded by their loaders into the
standard Hugging Face cache on first use. Callers can pass a local path to the
loader when network access is unavailable.
