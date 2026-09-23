"""Strict validation and explicit-schema tests for representation evidence."""

from __future__ import annotations

import json
import math
from fractions import Fraction

import pandas as pd
import pytest

from fairLMs.diagnostics import (
    ReferenceDistribution,
    ReferencePurpose,
    RepresentationEvidence,
)

AXIS = "chromatic-family"
FIELD = "unfamiliar__slot_93"
LABEL_MAPPING = {
    "external/citrine": "amber",
    "external/indigo": "teal",
}


def _reference(probabilities=None):
    return ReferenceDistribution(
        axis=AXIS,
        probabilities=(
            {"amber": 0.5, "teal": 0.5}
            if probabilities is None
            else probabilities
        ),
        source="Synthetic census table v7",
        period="2040",
        geography="Example region",
        population="Synthetic residents",
        purpose=ReferencePurpose("population"),
    )


def test_reference_distribution_is_strict_and_defensively_copies_input():
    probabilities = {"amber": 0.375, "teal": 0.625}
    reference = _reference(probabilities)

    probabilities["amber"] = 1.0
    probabilities["teal"] = 0.0

    assert dict(reference.probabilities) == {"amber": 0.375, "teal": 0.625}


@pytest.mark.parametrize(
    "probabilities",
    [
        {"amber": 0.5, "teal": 0.517},  # never silently normalize 1.017
        {"amber": 0.6, "teal": -0.1, "green": 0.5},
        {"amber": math.nan, "teal": math.nan},
        {"amber": math.inf, "teal": -math.inf},
        {"amber": "0.5", "teal": 0.5},
        {"amber": Fraction(10**4000, 1), "teal": 0.0},
        {"amber": 1.0},
        {},
    ],
)
def test_reference_distribution_rejects_malformed_probabilities(probabilities):
    with pytest.raises((TypeError, ValueError)):
        _reference(probabilities)


@pytest.mark.parametrize(
    "counts",
    [
        {"amber": 1.0, "teal": 1},
        {"amber": True, "teal": 1},
        {"amber": -1, "teal": 2},
        {"amber": math.nan, "teal": 2},
        {"amber": math.inf, "teal": 2},
        {"amber": "1", "teal": 2},
        {"amber": 1},
        {"amber": 0, "teal": 0},
        {},
    ],
)
def test_representation_counts_require_integers_two_cells_and_positive_total(counts):
    with pytest.raises((TypeError, ValueError)):
        RepresentationEvidence(axis=AXIS, counts=counts, source="synthetic table")


def test_representation_counts_allow_a_zero_cell_and_copy_input():
    counts = {"amber": 4, "teal": 0}
    evidence = RepresentationEvidence(
        axis=AXIS, counts=counts, source="synthetic table"
    )
    counts["amber"] = 0
    counts["teal"] = 4

    assert dict(evidence.counts) == {"amber": 4, "teal": 0}
    assert evidence.total == 4


def test_from_records_requires_explicit_mapping_and_preserves_mapping_provenance():
    records = [
        {FIELD: "external/citrine", "unused": "x"},
        {FIELD: "external/indigo", "unused": "y"},
        {FIELD: "external/citrine", "unused": "z"},
    ]
    evidence = RepresentationEvidence.from_records(
        records,
        axis=AXIS,
        group_field=FIELD,
        support=("amber", "teal"),
        source="unregistered records",
        value_map=LABEL_MAPPING,
        provenance={"dataset_id": "unregistered-widget-v7"},
    )

    assert dict(evidence.counts) == {"amber": 2, "teal": 1}
    assert evidence.provenance["field_mapping"] == {"group": FIELD}
    assert evidence.provenance["value_map_used"] is True
    assert evidence.to_dict()["provenance"]["value_map"] == [
        {"canonical": "amber", "raw": "external/citrine"},
        {"canonical": "teal", "raw": "external/indigo"},
    ]
    assert evidence.provenance["adapter"] == "records"
    assert evidence.provenance["dataset_id"] == "unregistered-widget-v7"


def test_from_dataframe_uses_the_same_explicit_mapping_contract():
    frame = pd.DataFrame(
        {
            FIELD: ["external/indigo", "external/citrine", "external/indigo"],
            "irrelevant": [10, 20, 30],
        }
    )

    evidence = RepresentationEvidence.from_dataframe(
        frame,
        axis=AXIS,
        group_column=FIELD,
        support=("amber", "teal"),
        source="unregistered dataframe",
        value_map=LABEL_MAPPING,
        provenance={"dataset_id": "unregistered-frame-v2"},
    )

    assert dict(evidence.counts) == {"amber": 1, "teal": 2}
    assert evidence.provenance["field_mapping"] == {"group": FIELD}
    assert evidence.provenance["value_map_used"] is True
    assert evidence.to_dict()["provenance"]["value_map"] == [
        {"canonical": "amber", "raw": "external/citrine"},
        {"canonical": "teal", "raw": "external/indigo"},
    ]
    assert evidence.provenance["adapter"] == "dataframe"
    assert evidence.provenance["dataset_id"] == "unregistered-frame-v2"


def test_record_order_does_not_change_counts_or_provenance_mapping():
    records = [
        {FIELD: "external/citrine"},
        {FIELD: "external/indigo"},
        {FIELD: "external/citrine"},
        {FIELD: "external/citrine"},
    ]
    kwargs = {
        "axis": AXIS,
        "group_field": FIELD,
        "support": ("amber", "teal"),
        "source": "unregistered records",
        "value_map": LABEL_MAPPING,
        "provenance": {"dataset_id": "unregistered-order-check"},
    }

    forward = RepresentationEvidence.from_records(records, **kwargs)
    reverse = RepresentationEvidence.from_records(list(reversed(records)), **kwargs)

    assert dict(forward.counts) == dict(reverse.counts) == {"amber": 3, "teal": 1}
    assert dict(forward.provenance) == dict(reverse.provenance)


def test_value_map_provenance_supports_json_scalar_keys_and_is_deterministic():
    records = [
        {FIELD: "external"},
        {FIELD: 7},
        {FIELD: 1.5},
        {FIELD: False},
        {FIELD: None},
    ]
    value_map = {
        "external": "amber",
        1.5: "teal",
        None: "amber",
        7: "teal",
        False: "amber",
    }
    kwargs = {
        "axis": AXIS,
        "group_field": FIELD,
        "support": ("amber", "teal"),
        "source": "mixed scalar records",
    }

    evidence = RepresentationEvidence.from_records(
        records,
        value_map=value_map,
        **kwargs,
    )
    reversed_map_evidence = RepresentationEvidence.from_records(
        records,
        value_map=dict(reversed(tuple(value_map.items()))),
        **kwargs,
    )

    expected = [
        {"canonical": "amber", "raw": None},
        {"canonical": "amber", "raw": False},
        {"canonical": "teal", "raw": 7},
        {"canonical": "teal", "raw": 1.5},
        {"canonical": "amber", "raw": "external"},
    ]
    assert evidence.to_dict()["provenance"]["value_map"] == expected
    assert evidence.to_dict()["provenance"] == reversed_map_evidence.to_dict()[
        "provenance"
    ]
    json.dumps(evidence.to_dict(), allow_nan=False, sort_keys=True)


@pytest.mark.parametrize("raw_key", [("tuple",), b"bytes", math.nan, math.inf])
def test_value_map_rejects_nonportable_or_nonfinite_raw_keys(raw_key):
    with pytest.raises((TypeError, ValueError), match=r"(?i)value_map raw key"):
        RepresentationEvidence.from_records(
            [{FIELD: "external/citrine"}],
            axis=AXIS,
            group_field=FIELD,
            support=("amber", "teal"),
            source="unregistered records",
            value_map={raw_key: "amber"},
        )


@pytest.mark.parametrize("invalid_output", ["", "   ", "violet", 7])
def test_value_map_prevalidates_unobserved_outputs(invalid_output):
    value_map = {
        "external/citrine": "amber",
        "unused/raw/value": invalid_output,
    }

    with pytest.raises((TypeError, ValueError), match=r"(?i)value_map output"):
        RepresentationEvidence.from_records(
            [{FIELD: "external/citrine"}],
            axis=AXIS,
            group_field=FIELD,
            support=("amber", "teal"),
            source="unregistered records",
            value_map=value_map,
        )


@pytest.mark.parametrize(
    ("records", "row_index"),
    [
        ([{FIELD: "external/citrine"}, {"wrong_field": "external/indigo"}], 1),
        ([{FIELD: "external/citrine"}, {FIELD: "external/chartreuse"}], 1),
        ([{FIELD: "external/citrine"}, {FIELD: None}], 1),
        ([{FIELD: 7}, {FIELD: "external/indigo"}], 0),
    ],
)
def test_from_records_rejects_missing_unknown_or_uncoerced_values_with_row_index(
    records, row_index
):
    with pytest.raises(
        (TypeError, ValueError), match=rf"(?i)row(?: index)?[^0-9]*{row_index}"
    ):
        RepresentationEvidence.from_records(
            records,
            axis=AXIS,
            group_field=FIELD,
            support=("amber", "teal"),
            source="unregistered records",
            value_map=LABEL_MAPPING,
        )


def test_from_records_does_not_guess_a_mapping():
    with pytest.raises((TypeError, ValueError)):
        RepresentationEvidence.from_records(
            [{FIELD: "external/citrine"}, {FIELD: "external/indigo"}],
            axis=AXIS,
            group_field=FIELD,
            support=("amber", "teal"),
            source="unregistered records",
        )


def test_from_dataframe_does_not_drop_missing_values():
    frame = pd.DataFrame({FIELD: ["external/citrine", None, "external/indigo"]})

    with pytest.raises(
        (TypeError, ValueError), match=r"(?i)row(?: index)?[^0-9]*1"
    ):
        RepresentationEvidence.from_dataframe(
            frame,
            axis=AXIS,
            group_column=FIELD,
            support=("amber", "teal"),
            source="unregistered dataframe",
            value_map=LABEL_MAPPING,
        )


def test_from_dataframe_rejects_duplicate_mapped_group_column():
    frame = pd.DataFrame(
        [["external/citrine", "external/indigo"]],
        columns=[FIELD, FIELD],
    )

    with pytest.raises(ValueError, match=r"(?i)group column.*must be unique"):
        RepresentationEvidence.from_dataframe(
            frame,
            axis=AXIS,
            group_column=FIELD,
            support=("amber", "teal"),
            source="unregistered dataframe",
            value_map=LABEL_MAPPING,
        )


@pytest.mark.parametrize(
    ("values", "row_index"),
    [
        (["external/citrine", "external/chartreuse"], 1),
        ([7, "external/indigo"], 0),
    ],
)
def test_from_dataframe_rejects_unknown_or_uncoerced_values_with_row_index(
    values, row_index
):
    frame = pd.DataFrame({FIELD: values})

    with pytest.raises(
        (TypeError, ValueError), match=rf"(?i)row(?: index)?[^0-9]*{row_index}"
    ):
        RepresentationEvidence.from_dataframe(
            frame,
            axis=AXIS,
            group_column=FIELD,
            support=("amber", "teal"),
            source="unregistered dataframe",
            value_map=LABEL_MAPPING,
        )
