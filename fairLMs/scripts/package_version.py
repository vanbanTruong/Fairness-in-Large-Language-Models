#!/usr/bin/env python3
"""Print the package version from fairLMs/_version.py.

Reads the assignment via AST so it works without importing fairLMs (which pulls
in torch). Used by the release workflow to check that a tag matches the version,
and useful locally:

    git tag -a "v$(python scripts/package_version.py)" -m "..."

Deliberately not a regex: _version.py carries a docstring, and naive
quote-matching returns a docstring quote instead of the version.
"""

from __future__ import annotations

import ast
import pathlib
import sys

VERSION_FILE = pathlib.Path(__file__).resolve().parent.parent / "_version.py"


def package_version(path: pathlib.Path = VERSION_FILE) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__version__":
                if not isinstance(node.value, ast.Constant) or not isinstance(
                    node.value.value, str
                ):
                    raise SystemExit(
                        f"{path}: __version__ must be a string literal"
                    )
                return node.value.value
    raise SystemExit(f"{path}: no __version__ assignment found")


if __name__ == "__main__":
    sys.stdout.write(package_version() + "\n")
