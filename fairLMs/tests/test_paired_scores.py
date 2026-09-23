"""Strict evidence and adapter tests for paired scorer audits."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from fairLMs.diagnostics import PairedScores

AXIS = "unfamiliar-intervention-axis"
PAIR_FIELD = "external__match_key_41"
CONDITION_FIELD = "variant__code_8"
SCORE_FIELD = "channel__score_23"
ROLES = ("baseline", "intervention")
CONDITION_MAP = {
    "source/control": "baseline",
    "source/swap": "intervention",
}


def _evidence(**overrides):
    kwargs = {
        "axis": AXIS,
        "pair_ids": ["pair-b", "pair-a", "pair-b", "pair-a"],
        "conditions": ["intervention", "baseline", "baseline", "intervention"],
        "scores": [0.9, 0.2, 0.4, 0.5],
        "condition_roles": ROLES,
        "score_name": "unfamiliar-score-channel",
        "source": "external paired score table v1",
        "pairing_basis": "Pre-registered minimal identity-token substitution",
        "score_range": (0.0, 1.0),
        "provenance": {"table_id": "outside-catalog-pairs-17"},
    }
    kwargs.update(overrides)
    return PairedScores(**kwargs)


def test_paired_scores_snapshots_and_canonicalizes_pairs_and_roles():
    pair_ids = ["pair-b", "pair-a", "pair-b", "pair-a"]
    conditions = ["intervention", "baseline", "baseline", "intervention"]
    scores = [0.9, 0.2, 0.4, 0.5]
    roles = ["baseline", "intervention"]
    score_range = [0.0, 1.0]
    provenance = {"nested": {"revision": 4}}
    evidence = _evidence(
        pair_ids=pair_ids,
        conditions=conditions,
        scores=scores,
        condition_roles=roles,
        score_range=score_range,
        provenance=provenance,
    )

    pair_ids[0] = "mutated"
    conditions[0] = "mutated"
    scores[0] = 99.0
    roles[0] = "mutated"
    score_range[1] = 99.0
    provenance["nested"]["revision"] = 99

    assert evidence.pair_ids == ("pair-a", "pair-a", "pair-b", "pair-b")
    assert evidence.conditions == (
        "baseline",
        "intervention",
        "baseline",
        "intervention",
    )
    assert evidence.scores == (0.2, 0.5, 0.4, 0.9)
    assert evidence.condition_roles == ROLES
    assert evidence.total == 4
    assert evidence.pair_count == 2
    assert evidence.score_range == (0.0, 1.0)
    assert evidence.provenance["nested"]["revision"] == 4

    payload = evidence.to_dict()
    assert payload["pair_count"] == 2
    json.dumps(payload, allow_nan=False, sort_keys=True)
    payload["provenance"]["nested"]["revision"] = -1
    assert evidence.provenance["nested"]["revision"] == 4


def test_heterogeneous_json_scalar_pair_ids_have_deterministic_portable_order():
    evidence = _evidence(
        pair_ids=["7", 3, 3, "7"],
        conditions=["intervention", "baseline", "intervention", "baseline"],
        scores=[0.8, 0.1, 0.4, 0.2],
    )

    assert evidence.pair_ids == (3, 3, "7", "7")
    assert evidence.conditions == ROLES * 2
    json.dumps(evidence.to_dict(), allow_nan=False, sort_keys=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pair_ids", "pair-a,pair-b"),
        ("pair_ids", {"pair-a", "pair-b"}),
        ("pair_ids", (item for item in ("pair-a", "pair-b"))),
        ("conditions", "baseline,intervention"),
        ("conditions", {"baseline", "intervention"}),
        ("scores", "0.1,0.2"),
        ("scores", {0.1, 0.2}),
        ("condition_roles", "baseline,intervention"),
        ("condition_roles", {"baseline", "intervention"}),
    ],
)
def test_paired_scores_requires_ordered_public_sequences(field, value):
    with pytest.raises(TypeError, match="ordered sequence"):
        _evidence(**{field: value})


@pytest.mark.parametrize(
    "roles",
    [[], ["baseline"], ["baseline", "intervention", "third"], ["same", "same"]],
)
def test_condition_roles_must_be_exactly_two_distinct_labels(roles):
    with pytest.raises(ValueError, match="condition_roles"):
        _evidence(condition_roles=roles)


@pytest.mark.parametrize(
    "pair_ids",
    [
        [None, None, "pair-b", "pair-b"],
        [True, True, "pair-b", "pair-b"],
        ["", "", "pair-b", "pair-b"],
        [math.nan, math.nan, "pair-b", "pair-b"],
        [math.inf, math.inf, "pair-b", "pair-b"],
        [object(), object(), "pair-b", "pair-b"],
    ],
)
def test_pair_ids_must_be_nonmissing_portable_nonboolean_scalars(pair_ids):
    with pytest.raises((TypeError, ValueError), match="pair_id"):
        _evidence(pair_ids=pair_ids)


@pytest.mark.parametrize(
    ("pair_ids", "conditions", "scores", "message"),
    [
        (["p1", "p1"], ["baseline"], [0.1, 0.2], "same number"),
        (
            ["p1", "p1"],
            ["baseline", "unknown"],
            [0.1, 0.2],
            "outside",
        ),
        (
            ["p1", "p1"],
            ["baseline", "baseline"],
            [0.1, 0.2],
            "duplicate",
        ),
        (
            ["p1", "p2", "p2"],
            ["baseline", "baseline", "intervention"],
            [0.1, 0.2, 0.3],
            "missing",
        ),
    ],
)
def test_every_pair_requires_exactly_one_row_per_declared_role(
    pair_ids, conditions, scores, message
):
    with pytest.raises(ValueError, match=message):
        _evidence(pair_ids=pair_ids, conditions=conditions, scores=scores)


@pytest.mark.parametrize("bad_score", [True, "0.2", math.nan, math.inf])
def test_scores_are_strict_finite_reals_without_coercion(bad_score):
    with pytest.raises((TypeError, ValueError), match="score at row"):
        _evidence(scores=[bad_score, 0.2, 0.3, 0.4])


def test_score_range_is_inclusive_validation_metadata_only():
    endpoint_evidence = _evidence(scores=[0.0, 1.0, 1.0, 0.0])
    unrestricted = _evidence(scores=[-8.0, 12.0, 3.0, -4.0], score_range=None)

    assert endpoint_evidence.score_range == (0.0, 1.0)
    assert unrestricted.score_range is None
    with pytest.raises(ValueError, match="outside"):
        _evidence(scores=[-0.01, 0.2, 0.3, 0.4])


@pytest.mark.parametrize(
    "field",
    ["axis", "score_name", "source", "pairing_basis"],
)
def test_required_measurement_metadata_must_be_nonempty(field):
    with pytest.raises(ValueError, match=field):
        _evidence(**{field: " "})


def test_paired_scores_rejects_nonportable_provenance():
    with pytest.raises(TypeError):
        _evidence(provenance={"bad": object()})
    with pytest.raises(ValueError):
        _evidence(provenance={"bad": math.nan})


def test_paired_scores_accepts_finite_numpy_scalars():
    evidence = _evidence(
        pair_ids=[np.int64(2), np.int64(1), np.int64(1), np.int64(2)],
        conditions=["intervention", "baseline", "intervention", "baseline"],
        scores=[np.float32(0.8), np.float64(0.1), np.int64(0), np.float64(0.4)],
    )

    assert evidence.pair_ids == (1, 1, 2, 2)
    assert evidence.scores == pytest.approx((0.1, 0.0, 0.4, 0.8))


def test_from_records_maps_three_explicit_fields_and_preserves_provenance():
    rows = [
        {
            PAIR_FIELD: "p2",
            CONDITION_FIELD: "source/swap",
            SCORE_FIELD: 0.9,
            "ignored": "x",
        },
        {
            PAIR_FIELD: "p1",
            CONDITION_FIELD: "source/control",
            SCORE_FIELD: 0.2,
            "ignored": "y",
        },
        {
            PAIR_FIELD: "p2",
            CONDITION_FIELD: "source/control",
            SCORE_FIELD: 0.4,
            "ignored": "z",
        },
        {
            PAIR_FIELD: "p1",
            CONDITION_FIELD: "source/swap",
            SCORE_FIELD: 0.5,
            "ignored": "q",
        },
    ]
    evidence = PairedScores.from_records(
        rows,
        axis=AXIS,
        pair_id_field=PAIR_FIELD,
        condition_field=CONDITION_FIELD,
        score_field=SCORE_FIELD,
        condition_roles=ROLES,
        score_name="external-paired-score",
        source="unregistered paired records",
        pairing_basis="Reviewed minimal substitutions under protocol v2",
        condition_map=CONDITION_MAP,
        score_range=(0.0, 1.0),
        provenance={"split": "held-out"},
    )

    assert evidence.pair_ids == ("p1", "p1", "p2", "p2")
    assert evidence.conditions == ROLES * 2
    assert evidence.scores == (0.2, 0.5, 0.4, 0.9)
    assert evidence.provenance["adapter"] == "records"
    assert evidence.provenance["field_mapping"] == {
        "pair_id": PAIR_FIELD,
        "condition": CONDITION_FIELD,
        "score": SCORE_FIELD,
    }
    assert evidence.to_dict()["provenance"]["condition_map"] == [
        {"canonical": "baseline", "raw": "source/control"},
        {"canonical": "intervention", "raw": "source/swap"},
    ]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{PAIR_FIELD: "p1", CONDITION_FIELD: "source/control"}],
        [
            {
                PAIR_FIELD: "p1",
                CONDITION_FIELD: "source/unknown",
                SCORE_FIELD: 0.1,
            }
        ],
        [
            {
                PAIR_FIELD: "p1",
                CONDITION_FIELD: "source/control",
                SCORE_FIELD: "0.1",
            }
        ],
    ],
)
def test_from_records_rejects_bad_rows_without_dropping_or_coercing(rows):
    with pytest.raises((TypeError, ValueError)):
        PairedScores.from_records(
            rows,
            axis=AXIS,
            pair_id_field=PAIR_FIELD,
            condition_field=CONDITION_FIELD,
            score_field=SCORE_FIELD,
            condition_roles=ROLES,
            score_name="external-paired-score",
            source="strict paired records",
            pairing_basis="Reviewed minimal substitutions",
            condition_map=CONDITION_MAP,
        )


def test_from_records_rejects_nonmapping_rows_and_ambiguous_fields():
    common = dict(
        axis=AXIS,
        condition_roles=ROLES,
        score_name="external-paired-score",
        source="strict paired records",
        pairing_basis="Reviewed minimal substitutions",
        condition_map=CONDITION_MAP,
    )
    with pytest.raises(TypeError, match="must be a mapping"):
        PairedScores.from_records(
            ["not-a-row"],
            pair_id_field=PAIR_FIELD,
            condition_field=CONDITION_FIELD,
            score_field=SCORE_FIELD,
            **common,
        )
    with pytest.raises(ValueError, match="distinct"):
        PairedScores.from_records(
            [{PAIR_FIELD: "p1"}],
            pair_id_field=PAIR_FIELD,
            condition_field=PAIR_FIELD,
            score_field=SCORE_FIELD,
            **common,
        )


def test_from_dataframe_matches_records_and_preserves_dataframe_adapter():
    frame = pd.DataFrame(
        {
            PAIR_FIELD: ["p2", "p1", "p2", "p1"],
            CONDITION_FIELD: [
                "source/swap",
                "source/control",
                "source/control",
                "source/swap",
            ],
            SCORE_FIELD: [0.9, 0.2, 0.4, 0.5],
            "ignored": [1, 2, 3, 4],
        }
    )
    evidence = PairedScores.from_dataframe(
        frame,
        axis=AXIS,
        pair_id_column=PAIR_FIELD,
        condition_column=CONDITION_FIELD,
        score_column=SCORE_FIELD,
        condition_roles=ROLES,
        score_name="external-paired-score",
        source="unregistered paired dataframe",
        pairing_basis="Reviewed minimal substitutions under protocol v2",
        condition_map=CONDITION_MAP,
        score_range=(0.0, 1.0),
        provenance={"split": "held-out"},
    )

    assert evidence.pair_ids == ("p1", "p1", "p2", "p2")
    assert evidence.scores == (0.2, 0.5, 0.4, 0.9)
    assert evidence.provenance["adapter"] == "dataframe"
    assert evidence.provenance["split"] == "held-out"


@pytest.mark.parametrize("column", [PAIR_FIELD, CONDITION_FIELD, SCORE_FIELD])
def test_from_dataframe_rejects_duplicate_mapped_columns(column):
    others = [
        item for item in (PAIR_FIELD, CONDITION_FIELD, SCORE_FIELD) if item != column
    ]
    frame = pd.DataFrame([["x", "y", "z", 0.1]], columns=[column, column, *others])

    with pytest.raises(ValueError, match="must be unique"):
        PairedScores.from_dataframe(
            frame,
            axis=AXIS,
            pair_id_column=PAIR_FIELD,
            condition_column=CONDITION_FIELD,
            score_column=SCORE_FIELD,
            condition_roles=ROLES,
            score_name="external-paired-score",
            source="ambiguous dataframe",
            pairing_basis="Reviewed minimal substitutions",
            condition_map=CONDITION_MAP,
        )


def test_from_dataframe_requires_dataframe_and_all_three_named_columns():
    with pytest.raises(TypeError, match="DataFrame"):
        PairedScores.from_dataframe(
            [],
            axis=AXIS,
            pair_id_column=PAIR_FIELD,
            condition_column=CONDITION_FIELD,
            score_column=SCORE_FIELD,
            condition_roles=ROLES,
            score_name="external-paired-score",
            source="invalid input",
            pairing_basis="Reviewed minimal substitutions",
        )

    frame = pd.DataFrame({PAIR_FIELD: ["p1"], CONDITION_FIELD: ["source/control"]})
    with pytest.raises(ValueError, match="score column"):
        PairedScores.from_dataframe(
            frame,
            axis=AXIS,
            pair_id_column=PAIR_FIELD,
            condition_column=CONDITION_FIELD,
            score_column=SCORE_FIELD,
            condition_roles=ROLES,
            score_name="external-paired-score",
            source="missing score column",
            pairing_basis="Reviewed minimal substitutions",
            condition_map=CONDITION_MAP,
        )
