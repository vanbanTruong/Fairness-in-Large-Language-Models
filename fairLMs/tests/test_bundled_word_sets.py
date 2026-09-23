"""Tests for ``fairLMs.data`` — the short-import wrappers around the bundled
WEAT/SEAT stimuli.

The point of the module is that a published association test needs no term
lists and no unpacking at the call site, so these tests pin the three things
that could quietly break that:

1. The exported names are importable and are real ``WordSets``.
2. They still match the stimulus files they re-export (no silent drift, no
   truncation).
3. They satisfy the constraints the metrics impose, so ``compute`` accepts them
   without the caller pre-checking anything.
"""

from __future__ import annotations

import pytest

from fairLMs.data import (
    WORD_SET_LABELS,
    WORD_SETS,
    get_word_set,
    list_word_sets,
    seat_c1,
    weat_c1,
)
from fairLMs.definition.encoder_only.intrinsic_bias.similarity_based.seat import (
    data as seat_data,
)
from fairLMs.definition.encoder_only.intrinsic_bias.similarity_based.weat import (
    data as weat_data,
)
from fairLMs.metrics import SEAT, WEAT, WordSets

ROLES = ("target_1", "target_2", "attribute_1", "attribute_2")


class TestExports:
    def test_short_import_yields_a_word_sets(self):
        """The documented one-liner import works and needs no unpacking."""
        assert isinstance(weat_c1, WordSets)
        assert isinstance(seat_c1, WordSets)

    def test_registry_and_module_attributes_are_the_same_objects(self):
        assert WORD_SETS["weat_c1"] is weat_c1
        assert WORD_SETS["seat_c1"] is seat_c1

    def test_every_registry_entry_is_a_word_sets(self):
        assert WORD_SETS, "expected at least one bundled word set"
        for name, word_set in WORD_SETS.items():
            assert isinstance(word_set, WordSets), name

    def test_list_word_sets_covers_the_registry(self):
        assert list_word_sets() == list(WORD_SETS)
        assert set(list_word_sets()) >= {"weat_c1", "seat_c1"}

    def test_labels_cover_every_entry(self):
        assert set(WORD_SET_LABELS) == set(WORD_SETS)
        for name, label in WORD_SET_LABELS.items():
            assert label.strip(), f"{name} has an empty label"

    def test_get_word_set_round_trips(self):
        for name in list_word_sets():
            assert get_word_set(name) is WORD_SETS[name]

    def test_unknown_name_raises_and_lists_alternatives(self):
        """No silent fallback to a default set — the message must be actionable."""
        with pytest.raises(KeyError) as excinfo:
            get_word_set("weat_c99")
        message = str(excinfo.value)
        assert "weat_c99" in message
        assert "weat_c1" in message


class TestFidelityToSource:
    """The wrappers must re-export the stimuli verbatim."""

    @pytest.mark.parametrize(
        "name, spec",
        [(f"weat_c{i}", spec) for i, spec in enumerate(weat_data.ALL_TESTS, start=1)]
        + [(f"seat_c{i}", spec) for i, spec in enumerate(seat_data.ALL_TESTS, start=1)],
    )
    def test_terms_match_the_stimulus_file(self, name, spec):
        word_set = get_word_set(name)
        assert tuple(word_set.target_1) == tuple(spec["t1"])
        assert tuple(word_set.target_2) == tuple(spec["t2"])
        assert tuple(word_set.attribute_1) == tuple(spec["a1"])
        assert tuple(word_set.attribute_2) == tuple(spec["a2"])
        assert WORD_SET_LABELS[name] == spec["name"]

    def test_both_families_are_bundled(self):
        """A regression here would mean one family stopped being re-exported."""
        names = list_word_sets()
        assert len([n for n in names if n.startswith("weat_")]) == len(
            weat_data.ALL_TESTS
        )
        assert len([n for n in names if n.startswith("seat_")]) == len(
            seat_data.ALL_TESTS
        )


class TestUsableByMetrics:
    def test_targets_are_balanced_so_seat_accepts_them(self):
        """SEAT compares targets pairwise and rejects unequal lists."""
        for name, word_set in WORD_SETS.items():
            word_set.require_balanced_targets(f"SEAT[{name}]")

    def test_immutable_so_sharing_module_level_objects_is_safe(self):
        """Containers coerce to tuples; a caller cannot corrupt the shared set."""
        for role in ROLES:
            assert isinstance(getattr(weat_c1, role), tuple)
        with pytest.raises(AttributeError):
            weat_c1.target_1 = ["mutated"]

    @pytest.mark.parametrize("metric_cls", [WEAT, SEAT])
    def test_accepted_as_the_data_argument(self, metric_cls, monkeypatch):
        """``compute(model, weat_c1)`` must pass container validation.

        The embedding step is stubbed out: this asserts the container is the
        shape the metric wants, not the numerical result.
        """
        seen = {}

        def fake_get_tokenizer_model(model, tokenizer=None, **kwargs):
            seen["model"] = model
            raise _StopBeforeEmbedding

        monkeypatch.setattr(
            "fairLMs.metrics.similarity_based.get_tokenizer_model",
            fake_get_tokenizer_model,
        )

        with pytest.raises(_StopBeforeEmbedding):
            metric_cls().compute("sentinel-model", weat_c1)
        assert seen["model"] == "sentinel-model"


class _StopBeforeEmbedding(Exception):
    """Marker: validation passed and the metric reached model resolution."""
