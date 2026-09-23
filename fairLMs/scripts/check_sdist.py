"""Install and validate an actual source archive away from the checkout.

First install .[dev,docs] and build. Then:
    python scripts/check_sdist.py dist/fairLMs-<version>.tar.gz --docs
The archive is installed without downloading runtime dependencies. This changes
the current Python environment; use a disposable virtual environment in CI.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--docs", action="store_true")
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)
    env = dict(
        os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="fairLMs-sdist-") as tmp:
        with tarfile.open(archive) as bundle:
            # Extract only regular files/directories inside the destination;
            # compatible with Python 3.10 (no reliance on tar filter='data').
            for member in bundle.getmembers():
                target = (Path(tmp) / member.name).resolve()
                if not target.is_relative_to(Path(tmp)) or not (
                    member.isfile() or member.isdir()
                ):
                    raise ValueError(
                        f"Unsupported source archive member: {member.name}"
                    )
            bundle.extractall(tmp)
        roots = list(Path(tmp).iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError("Source archive must contain a single project directory.")
        root = roots[0]
        required = [
            "tests/__init__.py",
            "tests/conftest.py",
            "tests/stubs.py",
            "scripts/package_version.py",
            "scripts/gen_registry_docs.py",
            "mkdocs.yml",
            "docs/notebooks/tour.ipynb",
            "fairLMs/data/NOTICE.md",
            "fairLMs/data/checksums.json",
        ]
        for name in required:
            if not (root / name).is_file():
                raise FileNotFoundError(f"sdist omits {name}")

        def run(*command):
            subprocess.run([sys.executable, *command], cwd=root, env=env, check=True)

        run(
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            str(archive),
        )
        run(
            "-c",
            "import fairLMs, pathlib; assert pathlib.Path(fairLMs.__file__).is_relative_to(pathlib.Path.cwd()); print('Testing extracted source:', fairLMs.__file__)",
        )
        run("-m", "pytest", "-q", "-ra")
        run("scripts/gen_registry_docs.py", "--check")
        if args.docs:
            run("-m", "mkdocs", "build", "--strict")
    print(
        "Source archive installation, tests and requested documentation checks passed."
    )


if __name__ == "__main__":
    main()
