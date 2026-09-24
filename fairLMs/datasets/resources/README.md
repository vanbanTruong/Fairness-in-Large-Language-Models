# Bundled dataset resources

FairLMs ships only the small CrowS-Pairs CSV needed for offline metric runs.
Its checksum and third-party attribution are recorded alongside it.

Large datasets, including BBQ, are downloaded by their loaders into the
standard Hugging Face cache on first use. Callers can pass a local path to the
loader when network access is unavailable.
