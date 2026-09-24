"""Packaging guards.

`fairLMs.definitions` eagerly imports every metric family, so any third-party
module imported at module scope under `fairLMs/definitions/` becomes a hard
requirement of `import fairLMs`. A dependency that is only listed in an extra
therefore breaks a plain `pip install fairLMs` — which is exactly what happened
with `wordfreq`, `nltk` and `scikit-learn` before 0.2.0.

This test catches that class of bug from the source tree, without needing a
clean-environment install.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import shlex
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT
RUNTIME_TREES = (
    "datasets",
    "definitions",
    "mitigation",
)
RUNTIME_MODULES = (
    "__init__.py",
    "_version.py",
)

# import name -> distribution name, where they differ
DIST_NAME = {
    "sklearn": "scikit-learn",
    "gender_guesser": "gender-guesser",
    "yaml": "pyyaml",
    "PIL": "pillow",
}


def _declared_core_dependencies() -> set[str]:
    """Distribution names in [project.dependencies], normalized."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    out = set()
    for spec in cfg["project"]["dependencies"]:
        name = spec.split(";")[0]
        for sep in (">=", "<=", "==", "!=", "~=", ">", "<", "["):
            name = name.split(sep)[0]
        out.add(name.strip().lower().replace("_", "-"))
    return out


def _module_level_third_party_imports() -> dict[str, set[str]]:
    """Third-party modules imported at module scope, mapped to source files.

    Includes imports nested directly in a top-level ``try:`` block, since those
    still execute at import time.
    """
    found: dict[str, set[str]] = {}
    paths = [PACKAGE / name for name in RUNTIME_MODULES]
    paths.extend(PACKAGE / "datasets" / name for name in (
        "__init__.py",
        "_sources.py",
        "base.py",
        "bbq.py",
        "bias_in_bios.py",
        "bias_nli.py",
        "bold.py",
        "crows_pairs.py",
        "eec.py",
        "gap.py",
        "grep_biasir.py",
        "holistic_bias.py",
        "honest.py",
        "real_toxicity_prompts.py",
        "reddit_bias.py",
        "stereoset.py",
        "trustgpt.py",
        "unqover.py",
        "wino_bias.py",
        "winogender.py",
        "xnli.py",
    ))
    for tree in RUNTIME_TREES:
        paths.extend((PACKAGE / tree).rglob("*.py"))

    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue

        nodes = []
        for node in tree.body:
            nodes.append(node)
            if isinstance(node, ast.Try):
                nodes.extend(node.body)

        for node in nodes:
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name == "fairLMs" or name in sys.stdlib_module_names:
                    continue
                found.setdefault(name, set()).add(str(path.relative_to(REPO_ROOT)))
    return found


def test_module_level_imports_are_declared_core_dependencies():
    """Anything imported at module scope must be a core dependency, not an extra."""
    declared = _declared_core_dependencies()
    offenders = {}
    for module, files in _module_level_third_party_imports().items():
        dist = DIST_NAME.get(module, module).lower().replace("_", "-")
        if dist not in declared:
            offenders[dist] = sorted(files)

    assert not offenders, (
        "These are imported at module scope but are not core dependencies, so "
        "`import fairLMs` fails on a clean install:\n"
        + "\n".join(
            f"  {dist}\n" + "\n".join(f"    {f}" for f in files)
            for dist, files in sorted(offenders.items())
        )
        + "\n\nEither add them to [project.dependencies] in pyproject.toml, or "
        "move the import inside the function that needs it."
    )


def test_version_is_single_sourced():
    """pyproject must read the version from fairLMs/_version.py, not duplicate it."""
    tomllib = pytest.importorskip("tomllib")
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    project = cfg["project"]
    assert "version" not in project, (
        "version is hardcoded in pyproject.toml; it must stay dynamic so "
        "fairLMs/_version.py is the single source of truth"
    )
    assert "version" in project.get("dynamic", [])
    attr = cfg["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "fairLMs._version.__version__"


def test_runtime_version_matches_version_module():
    import fairLMs
    from fairLMs import _version

    assert fairLMs.__version__ == _version.__version__


def test_definitions_is_the_primary_metric_api():
    import fairLMs
    from fairLMs import definitions

    assert fairLMs.definitions is definitions
    assert len(definitions.list_metrics()) == 33


def test_definitions_exports_metric_model_and_resource_apis():
    from fairLMs.definitions import HuggingFaceModel, WEAT, weat_c1
    from fairLMs.definitions.models import HuggingFaceModel as NestedModel
    from fairLMs.definitions.resources import weat_c1 as nested_word_set

    assert HuggingFaceModel is NestedModel
    assert weat_c1 is nested_word_set
    assert WEAT.name == "weat"


def test_runtime_root_contains_only_the_two_package_modules():
    root_modules = {path.name for path in PACKAGE.glob("*.py")}
    assert root_modules == {"__init__.py", "_version.py"}


def test_large_dataset_snapshots_are_not_vendored():
    datasets = PACKAGE / "datasets"
    for legacy in (".hidden", "constrained_form", "open_ended"):
        assert not (datasets / legacy).exists()
    bundled = [path for path in (datasets / "resources").rglob("*") if path.is_file()]
    assert bundled
    assert max(path.stat().st_size for path in bundled) < 5 * 1024 * 1024


def test_bundled_dataset_manifest_covers_and_verifies_every_data_file():
    resources = PACKAGE / "datasets" / "resources"
    manifest = json.loads((resources / "checksums.json").read_text(encoding="utf-8"))
    entries = {item["path"]: item for item in manifest["files"]}
    data_files = {
        path.relative_to(resources).as_posix(): path
        for path in resources.rglob("*")
        if path.is_file() and path.suffix in {".csv", ".tsv", ".parquet"}
    }

    assert set(entries) == set(data_files)
    for name, path in data_files.items():
        payload = path.read_bytes()
        assert entries[name]["bytes"] == len(payload)
        assert entries[name]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_version_helper_script_agrees_with_package():
    """scripts/package_version.py is what the release workflow tags against.

    It must return the real version without importing fairLMs. A regression here
    means a release could be tagged with a version that isn't the package's.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "package_version.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    from fairLMs import _version

    assert result.stdout.strip() == _version.__version__


def test_license_file_exists():
    """pyproject declares MIT; the file it points at must actually be there."""
    tomllib = pytest.importorskip("tomllib")
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    assert cfg["project"]["license"] == "MIT"
    for rel in cfg["project"]["license-files"]:
        assert (REPO_ROOT / rel).is_file(), f"missing declared license file: {rel}"


def test_sdist_manifest_includes_diagnostic_reproducibility_material():
    """Golden fixtures, evidence guides, and examples ship in the source artifact."""
    golden_root = REPO_ROOT / "tests" / "data" / "golden"
    golden_json = sorted(golden_root.rglob("*.json"))
    assert golden_json, "expected at least one golden JSON fixture"

    manifest = REPO_ROOT / "MANIFEST.in"
    assert manifest.is_file(), "MANIFEST.in is required to declare sdist fixtures"
    rules = [
        shlex.split(line, comments=True)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert ["recursive-include", "tests", "*.py", "*.json"] in rules, (
        "MANIFEST.in must recursively include tests/data/golden/**/*.json so "
        "paper-parity fixtures ship in the source distribution"
    )
    assert ["recursive-include", "docs", "*.md", "*.ipynb"] in rules
    assert ["recursive-include", "examples", "*.md", "*.py"] in rules
    assert [
        "recursive-include",
        "datasets/resources",
        "*.md",
        "*.json",
        "*.csv",
        "*.tsv",
        "*.parquet",
        "LICENSE",
    ] in rules
    assert (REPO_ROOT / "docs" / "preparing_audit_evidence.md").is_file()
    assert (REPO_ROOT / "examples" / "scorer_rate_gap_diagnostic.py").is_file()
    assert (REPO_ROOT / "examples" / "scorer_distribution_gap_diagnostic.py").is_file()
    assert (
        REPO_ROOT / "examples" / "scorer_counterfactual_sensitivity_diagnostic.py"
    ).is_file()
    assert (
        golden_root
        / "diagnostics"
        / "score_counterfactual_sensitivity"
        / "bbq_age_toxicity_v1.json"
    ).is_file()


def test_registry_documentation_generates_from_current_source():
    """Catch obsolete registry fields and documentation drift in the sdist too."""
    import subprocess

    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/gen_registry_docs.py"), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    diagnostics = (REPO_ROOT / "docs/registry/diagnostics.md").read_text()
    assert "fairLMs[construction-backends]" not in diagnostics
    # The backend slots are implemented; the page names their reference backends
    # and no longer describes them as unshipped.
    assert "not_implemented_in_this_release" not in diagnostics
    for reference in (
        "HuggingFaceEmbeddingBackend",
        "LanguageToolGrammarBackend",
        "SpacyDependencyBackend",
    ):
        assert reference in diagnostics
