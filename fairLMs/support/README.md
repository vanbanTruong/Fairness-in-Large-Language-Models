# Project support files

This directory contains material that supports all three FairLMs functional
areas (`datasets`, `definitions`, and `mitigation`) rather than belonging to
one of them individually.

- `docs/` contains the project documentation.
- `examples/` contains runnable, cross-module workflows.
- `tests/` contains unit, integration, packaging, and workflow tests.
- `scripts/` contains documentation, packaging, and release tooling.
- `metadata/` records citation, upstream, and conversion information.
- `requirements/` contains convenience and verification environment lists;
  `pyproject.toml` remains the authoritative package dependency declaration.

The package entry points and build files stay one level above this directory
because Python packaging tools expect them at the project root.
