"""Strict evidence and adapter tests for row-level score audits."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from fairLMs.diagnostics import ScoredGroups

AXIS = "unfamiliar-cohort-axis"
GROUP_FIELD = "external__cohort_code"
SCORE_FIELD = "channel__score_17"
VALUE_MAP = {
    "source/amber": "amber",
    "source/teal": "teal",
}


def _evidence(**overrides):
    kwargs = {
        "axis": AXIS,
        "groups": ["amber", "teal", "amber"],
        "scores": [0.1, 0.7, 0.3],
        "score_name": "unfamiliar-score-channel",
        "source": "external score table v4",
        "score_range": (0.0, 1.0),
        "provenance": {"table_id": "outside-catalog-44"},
    }
    kwargs.update(overrides)
    return ScoredGroups(**kwargs)


def test_scored_groups_snapshots_rows_range_and_provenance():
    groups = ["amber", "teal", "amber"]
    scores = [0.1, 0.7, 0.3]
    score_range = [0.0, 1.0]
    provenance = {"nested": {"revision": 4}}
    evidence = _evidence(
        groups=groups,
        scores=scores,
        score_range=score_range,
        provenance=provenance,
    )

    groups[0] = "mutated"
    scores[0] = 99.0
    score_range[1] = 99.0
    provenance["nested"]["revision"] = 99

    assert evidence.groups == ("amber", "teal", "amber")
    assert evidence.scores == (0.1, 0.7, 0.3)
    assert evidence.support == ("amber", "teal")
    assert evidence.total == 3
    assert evidence.score_range == (0.0, 1.0)
    assert evidence.provenance["nested"]["revision"] == 4

    payload = evidence.to_dict()
    assert payload["support"] == ["amber", "teal"]
    assert payload["groups"] == ["amber", "teal", "amber"]
    assert payload["scores"] == [0.1, 0.7, 0.3]
    json.dumps(payload, allow_nan=False, sort_keys=True)
    payload["provenance"]["nested"]["revision"] = -1
    assert evidence.provenance["nested"]["revision"] == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("groups", "amber,teal"),
        ("groups", {"amber", "teal"}),
        ("groups", (group for group in ("amber", "teal"))),
        ("scores", "0.1,0.2"),
        ("scores", {0.1, 0.2}),
        ("scores", (score for score in (0.1, 0.2))),
    ],
)
def test_scored_groups_requires_ordered_public_sequences(field, value):
    with pytest.raises(TypeError, match="ordered sequence"):
        _evidence(**{field: value})


@pytest.mark.parametrize(
    ("groups", "scores"),
    [
        ([], []),
        (["amber"], [0.1]),
        (["amber", "amber"], [0.1, 0.2]),
        (["amber", ""], [0.1, 0.2]),
        (["amber", 7], [0.1, 0.2]),
        (["amber", "teal"], [0.1]),
        (["amber"], [0.1, 0.2]),
        (["amber", "teal"], [True, 0.2]),
        (["amber", "teal"], ["0.1", 0.2]),
        (["amber", "teal"], [math.nan, 0.2]),
        (["amber", "teal"], [math.inf, 0.2]),
    ],
)
def test_scored_groups_rejects_malformed_rows(groups, scores):
    with pytest.raises((TypeError, ValueError)):
        _evidence(groups=groups, scores=scores)


@pytest.mark.parametrize(
    "score_range",
    [
        "0,1",
        (0.0,),
        (0.0, 1.0, 2.0),
        (True, 1.0),
        ("0", 1.0),
        (math.nan, 1.0),
        (0.0, math.inf),
        (1.0, 0.0),
    ],
)
def test_scored_groups_rejects_malformed_score_ranges(score_range):
    with pytest.raises((TypeError, ValueError)):
        _evidence(score_range=score_range)


def test_score_range_is_optional_inclusive_and_not_a_normalization():
    unrestricted = _evidence(score_range=None, scores=[-4.0, 8.0, 0.0])
    endpoint_values = _evidence(scores=[0.0, 1.0, 0.5])
    constant = _evidence(scores=[0.5, 0.5, 0.5], score_range=(0.5, 0.5))

    assert unrestricted.score_range is None
    assert endpoint_values.score_range == (0.0, 1.0)
    assert constant.score_range == (0.5, 0.5)

    with pytest.raises(ValueError, match="outside"):
        _evidence(scores=[-0.01, 0.5, 0.25])


def test_scored_groups_rejects_nonportable_provenance():
    with pytest.raises(TypeError):
        _evidence(provenance={"bad": object()})
    with pytest.raises(TypeError):
        _evidence(provenance={1: "non-string key"})
    with pytest.raises(ValueError):
        _evidence(provenance={"bad": math.nan})


def test_scored_groups_accepts_finite_numpy_real_scalars():
    evidence = _evidence(
        groups=["amber", "teal", "amber"],
        scores=[np.float32(0.1), np.float64(0.7), np.int64(0)],
        score_range=(np.int64(0), np.float64(1.0)),
    )

    assert evidence.scores == pytest.approx((0.1, 0.7, 0.0))
    assert evidence.score_range == (0.0, 1.0)


def test_from_records_requires_explicit_fields_support_and_mapping():
    records = [
        {GROUP_FIELD: "source/amber", SCORE_FIELD: 0.1, "unused": "x"},
        {GROUP_FIELD: "source/teal", SCORE_FIELD: 0.8, "unused": "y"},
        {GROUP_FIELD: "source/amber", SCORE_FIELD: 0.3, "unused": "z"},
    ]
    evidence = ScoredGroups.from_records(
        records,
        axis=AXIS,
        group_field=GROUP_FIELD,
        score_field=SCORE_FIELD,
        support=("amber", "teal"),
        score_name="external-score",
        source="unregistered records",
        value_map=VALUE_MAP,
        score_range=(0.0, 1.0),
        provenance={"dataset_id": "unseen-score-table"},
    )

    assert evidence.groups == ("amber", "teal", "amber")
    assert evidence.scores == (0.1, 0.8, 0.3)
    assert evidence.provenance["adapter"] == "records"
    assert evidence.provenance["field_mapping"] == {
        "group": GROUP_FIELD,
        "score": SCORE_FIELD,
    }
    assert evidence.to_dict()["provenance"]["value_map"] == [
        {"canonical": "amber", "raw": "source/amber"},
        {"canonical": "teal", "raw": "source/teal"},
    ]


@pytest.mark.parametrize(
    ("records", "row_index"),
    [
        ([{GROUP_FIELD: "source/amber", SCORE_FIELD: 0.1}, {}], 1),
        (
            [
                {GROUP_FIELD: "source/amber", SCORE_FIELD: 0.1},
                {GROUP_FIELD: "source/unknown", SCORE_FIELD: 0.2},
            ],
            1,
        ),
        (
            [
                {GROUP_FIELD: "source/amber", SCORE_FIELD: 0.1},
                {GROUP_FIELD: "source/teal", SCORE_FIELD: "0.2"},
            ],
            1,
        ),
        (
            [
                {GROUP_FIELD: "source/amber", SCORE_FIELD: 0.1},
                {GROUP_FIELD: "source/teal", SCORE_FIELD: math.nan},
            ],
            1,
        ),
        (
            [
                {GROUP_FIELD: "source/amber", SCORE_FIELD: 0.1},
                {GROUP_FIELD: "source/teal", SCORE_FIELD: math.inf},
            ],
            1,
        ),
        (
            [
                {GROUP_FIELD: "source/amber", SCORE_FIELD: 0.1},
                {GROUP_FIELD: "source/teal", SCORE_FIELD: True},
            ],
            1,
        ),
        (
            [
                {GROUP_FIELD: "source/amber", SCORE_FIELD: 0.1},
                {GROUP_FIELD: None, SCORE_FIELD: 0.2},
            ],
            1,
        ),
    ],
)
def test_from_records_rejects_bad_rows_without_coercing_or_dropping(records, row_index):
    with pytest.raises((TypeError, ValueError), match=rf"(?i)row[^0-9]*{row_index}"):
        ScoredGroups.from_records(
            records,
            axis=AXIS,
            group_field=GROUP_FIELD,
            score_field=SCORE_FIELD,
            support=("amber", "teal"),
            score_name="external-score",
            source="unregistered records",
            value_map=VALUE_MAP,
        )


def test_from_records_rejects_unobserved_explicit_support():
    with pytest.raises(ValueError, match="missing categories"):
        ScoredGroups.from_records(
            [{GROUP_FIELD: "source/amber", SCORE_FIELD: 0.1}],
            axis=AXIS,
            group_field=GROUP_FIELD,
            score_field=SCORE_FIELD,
            support=("amber", "teal"),
            score_name="external-score",
            source="unregistered records",
            value_map=VALUE_MAP,
        )


def test_from_records_rejects_same_field_and_out_of_range_score_with_row_number():
    with pytest.raises(ValueError, match="distinct"):
        ScoredGroups.from_records(
            [{GROUP_FIELD: "source/amber"}],
            axis=AXIS,
            group_field=GROUP_FIELD,
            score_field=GROUP_FIELD,
            support=("amber", "teal"),
            score_name="external-score",
            source="ambiguous records",
            value_map=VALUE_MAP,
        )

    with pytest.raises(ValueError, match=r"(?i)row[^0-9]*1.*outside"):
        ScoredGroups.from_records(
            [
                {GROUP_FIELD: "source/amber", SCORE_FIELD: 0.1},
                {GROUP_FIELD: "source/teal", SCORE_FIELD: 1.1},
            ],
            axis=AXIS,
            group_field=GROUP_FIELD,
            score_field=SCORE_FIELD,
            support=("amber", "teal"),
            score_name="external-score",
            source="out-of-range records",
            value_map=VALUE_MAP,
            score_range=(0.0, 1.0),
        )


def test_from_records_rejects_non_string_group_without_a_value_map():
    with pytest.raises(ValueError, match=r"(?i)row[^0-9]*0.*non-string"):
        ScoredGroups.from_records(
            [
                {GROUP_FIELD: 7, SCORE_FIELD: 0.1},
                {GROUP_FIELD: "teal", SCORE_FIELD: 0.2},
            ],
            axis=AXIS,
            group_field=GROUP_FIELD,
            score_field=SCORE_FIELD,
            support=("amber", "teal"),
            score_name="external-score",
            source="strict records",
        )


def test_from_dataframe_uses_two_explicit_columns_and_preserves_provenance():
    frame = pd.DataFrame(
        {
            GROUP_FIELD: ["source/amber", "source/teal", "source/amber"],
            SCORE_FIELD: [0.2, 0.9, 0.4],
            "ignored": [1, 2, 3],
        }
    )
    evidence = ScoredGroups.from_dataframe(
        frame,
        axis=AXIS,
        group_column=GROUP_FIELD,
        score_column=SCORE_FIELD,
        support=("amber", "teal"),
        score_name="external-score",
        source="unregistered dataframe",
        value_map=VALUE_MAP,
        score_range=(0.0, 1.0),
        provenance={"split": "held-out"},
    )

    assert evidence.groups == ("amber", "teal", "amber")
    assert evidence.scores == (0.2, 0.9, 0.4)
    assert evidence.provenance["adapter"] == "dataframe"
    assert evidence.provenance["field_mapping"] == {
        "group": GROUP_FIELD,
        "score": SCORE_FIELD,
    }
    assert evidence.provenance["split"] == "held-out"


@pytest.mark.parametrize("column", [GROUP_FIELD, SCORE_FIELD])
def test_from_dataframe_rejects_duplicate_mapped_columns(column):
    other = SCORE_FIELD if column == GROUP_FIELD else GROUP_FIELD
    frame = pd.DataFrame(
        [["source/amber", "source/teal", 0.1]],
        columns=[column, column, other],
    )

    with pytest.raises(ValueError, match="must be unique"):
        ScoredGroups.from_dataframe(
            frame,
            axis=AXIS,
            group_column=GROUP_FIELD,
            score_column=SCORE_FIELD,
            support=("amber", "teal"),
            score_name="external-score",
            source="ambiguous dataframe",
            value_map=VALUE_MAP,
        )


def test_from_dataframe_does_not_drop_missing_or_coerce_score_strings():
    for scores in ([0.1, None], [0.1, "0.2"]):
        frame = pd.DataFrame(
            {
                GROUP_FIELD: ["source/amber", "source/teal"],
                SCORE_FIELD: scores,
            }
        )
        with pytest.raises((TypeError, ValueError), match=r"(?i)row[^0-9]*1"):
            ScoredGroups.from_dataframe(
                frame,
                axis=AXIS,
                group_column=GROUP_FIELD,
                score_column=SCORE_FIELD,
                support=("amber", "teal"),
                score_name="external-score",
                source="strict dataframe",
                value_map=VALUE_MAP,
            )
