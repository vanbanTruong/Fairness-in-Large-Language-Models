"""Bundled evidence, ready to pass to a metric.

Two kinds of bundled data live under this directory.

**Association-test word sets** (this module). The Caliskan-style WEAT/SEAT
stimuli, exposed as validated :class:`~fairLMs.metrics.data.WordSets` objects so
they go straight into ``compute`` with no unpacking::

    from fairLMs.data import weat_c1
    from fairLMs.metrics import WEAT
    from fairLMs.models import HuggingFaceModel

    model = HuggingFaceModel("bert-base-uncased", task="encoder")
    WEAT().compute(model, weat_c1)

These are re-exports: the term lists under
``fairLMs/definition/encoder_only/intrinsic_bias/similarity_based/`` remain the
source of truth, and this module only shortens the import and wraps them in the
container the metrics already expect. ``weat_*`` uses the term lists from
Caliskan et al. (2017); ``seat_*`` uses the expanded name lists from May et al.
(2019). Either works with either metric; the container is the same type.

CEAT is deliberately absent: it consumes :class:`ContextSets` (terms keyed to
context sentences), not four flat lists, so there is nothing here to hand it.

**Corpus files** (``bbq/*.jsonl``, ``crows_pairs/*.csv``). Resolved from disk by
the loaders in :mod:`fairLMs.datasets` and :mod:`fairLMs.utils.paths`, not
imported from here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from fairLMs.definition.encoder_only.intrinsic_bias.similarity_based.seat import (
    data as _seat_data,
)
from fairLMs.definition.encoder_only.intrinsic_bias.similarity_based.weat import (
    data as _weat_data,
)
from fairLMs.metrics.data import WordSets

__all__ = [
    "WORD_SETS",
    "WORD_SET_LABELS",
    "get_word_set",
    "list_word_sets",
    "seat_c1",
    "seat_c2",
    "seat_c3",
    "seat_c4",
    "weat_c1",
    "weat_c2",
    "weat_c3",
    "weat_c4",
]


def _to_word_sets(spec: Mapping[str, Any]) -> WordSets:
    """Wrap a ``{"name", "t1", "t2", "a1", "a2"}`` stimulus dict in a WordSets.

    The container validates on construction, so a malformed stimulus file fails
    at import rather than deep inside a metric.
    """
    return WordSets(
        target_1=spec["t1"],
        target_2=spec["t2"],
        attribute_1=spec["a1"],
        attribute_2=spec["a2"],
    )


# Single table mapping public name -> source stimulus dict. Both the registry
# and the human-readable labels derive from this, so they cannot drift apart.
_SPECS: Dict[str, Mapping[str, Any]] = {
    "weat_c1": _weat_data.C1,
    "weat_c2": _weat_data.C2,
    "weat_c3": _weat_data.C3,
    "weat_c4": _weat_data.C4,
    "seat_c1": _seat_data.C1,
    "seat_c2": _seat_data.C2,
    "seat_c3": _seat_data.C3,
    "seat_c4": _seat_data.C4,
}

#: Every bundled word set, keyed by the name you would import.
WORD_SETS: Dict[str, WordSets] = {
    name: _to_word_sets(spec) for name, spec in _SPECS.items()
}

#: Human-readable label for each entry, taken from the source stimulus dicts.
WORD_SET_LABELS: Dict[str, str] = {
    name: str(spec["name"]) for name, spec in _SPECS.items()
}

# Bound as module attributes as well as registry entries, so both
# `from fairLMs.data import weat_c1` and `get_word_set("weat_c1")` work.
weat_c1 = WORD_SETS["weat_c1"]
weat_c2 = WORD_SETS["weat_c2"]
weat_c3 = WORD_SETS["weat_c3"]
weat_c4 = WORD_SETS["weat_c4"]

seat_c1 = WORD_SETS["seat_c1"]
seat_c2 = WORD_SETS["seat_c2"]
seat_c3 = WORD_SETS["seat_c3"]
seat_c4 = WORD_SETS["seat_c4"]


def list_word_sets() -> List[str]:
    """Names of the bundled word sets, in registry order.

    Mirrors :func:`fairLMs.metrics.list_metrics`.
    """
    return list(WORD_SETS)


def get_word_set(name: str) -> WordSets:
    """Return a bundled word set by name.

    Parameters
    ----------
    name:
        A key from :func:`list_word_sets`, e.g. ``"weat_c1"``.

    Raises
    ------
    KeyError
        If ``name`` is not bundled. The message lists what is available rather
        than falling back to a default set.
    """
    try:
        return WORD_SETS[name]
    except KeyError:
        raise KeyError(
            f"unknown word set {name!r}. Available: {', '.join(list_word_sets())}."
        ) from None
