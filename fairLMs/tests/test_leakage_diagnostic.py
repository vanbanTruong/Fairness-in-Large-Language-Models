"""Scientific and applicability tests for the ``b_leak`` stereotype-leakage slice."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from fairLMs.diagnostics import (
    AssociationCounts,
    ComponentOverride,
    DatasetAuditSpec,
    DiagnosticStatus,
    LeakageExtractionConfig,
    LeakageExtractionRecord,
    LogBase,
    ReportStatus,
    StereotypeLeakage,
    SurfaceCooccurrenceExtractor,
    TargetKind,
    TextEvidence,
    audit_leakage,
)

AXIS = "unregistered-protected-axis"

GROUP_LEXICON = ("doctor", "nurse")
TRAIT_LEXICON = ("brilliant", "caring")

#: A corpus whose window co-occurrences can be counted by hand, so the count
#: matrix used by the parity test is never taken from the extractor itself.
PARITY_TEXTS = (
    "The doctor was brilliant and caring.",
    "A caring nurse stayed brilliant.",
    "The doctor is brilliant.",
    "The nurse is caring.",
    "Nobody here matches the declared lexicon at all.",
)
#: doctor@1 sees brilliant@3 and caring@5; nurse@2 sees caring@1 and
#: brilliant@4; then one further doctor/brilliant and one nurse/caring pair.
PARITY_PAIR_COUNTS = {
    ("doctor", "brilliant"): 2,
    ("doctor", "caring"): 1,
    ("nurse", "brilliant"): 1,
    ("nurse", "caring"): 2,
}
PARITY_COUNTING_BASIS = (
    "surface lexical co-occurrence within a symmetric +/-5-token window, "
    "group-anchored, self position excluded"
)

GOLDEN_FIXTURE_PATH = (
    Path(__file__).parent
    / "data"
    / "golden"
    / "diagnostics"
    / "b_leak"
    / "gap_paper_lexicon_v1.json"
)


def _spec(
    *,
    target_kind="benchmark_dataset",
    design_stance="population_proxy",
    task_family="free_text",
    target_name="unregistered-synthetic-benchmark",
    requested_components=("b_leak",),
    **extra,
):
    return DatasetAuditSpec(
        target_name=target_name,
        target_kind=target_kind,
        task_family=task_family,
        design_stance=design_stance,
        references={},
        requested_components=requested_components,
        **extra,
    )


def _counts(matrix, *, axis=AXIS, group_terms=None, trait_terms=None, **extra):
    groups = group_terms or sorted(matrix)
    traits = trait_terms or sorted(next(iter(matrix.values())))
    return AssociationCounts(
        axis=axis,
        group_terms=groups,
        trait_terms=traits,
        counts=matrix,
        source="Unregistered synthetic corpus",
        counting_basis="declared synthetic counting basis",
        **extra,
    )


def _record(**overrides):
    payload = {
        "extractor_id": "declared_external_extractor",
        "extractor_version": "external/1",
        "config_digest": "declared-config-digest",
        "text_digest": "declared-text-digest",
        "text_unit_count": 4,
        "token_count": 24,
        "matched_group_positions": 0,
        "matched_trait_positions": 0,
        "event_count": 0,
        "window": 5,
    }
    payload.update(overrides)
    return LeakageExtractionRecord(**payload)


def _config(**overrides):
    payload = {
        "group_lexicon": list(GROUP_LEXICON),
        "trait_lexicon": list(TRAIT_LEXICON),
    }
    payload.update(overrides)
    return LeakageExtractionConfig(**payload)


def _result(report):
    assert set(report.components) == {"b_leak"}
    return report.components["b_leak"]


def _value(matrix, *, diagnostic=None, **spec_kwargs):
    report = audit_leakage(_counts(matrix), _spec(**spec_kwargs), diagnostic)
    result = _result(report)
    assert result.status is DiagnosticStatus.READY
    return result.value


# ---------------------------------------------------------------------------
# Kernel behaviour
# ---------------------------------------------------------------------------


def test_uniform_matrix_carries_exactly_zero_information():
    result = _result(
        audit_leakage(
            _counts({"ga": {"t1": 10, "t2": 10}, "gb": {"t1": 10, "t2": 10}}),
            _spec(),
        )
    )

    assert result.status is DiagnosticStatus.READY
    assert result.value == 0.0
    assert result.details["mutual_information"] == 0.0
    assert result.details["entropy_group"] == pytest.approx(1.0, rel=0, abs=1e-15)
    assert result.details["entropy_trait"] == pytest.approx(1.0, rel=0, abs=1e-15)
    assert result.details["information_unit"] == "bits"


def test_independent_margins_produce_zero_mutual_information_and_zero_nmi():
    # Every group has the same trait profile, so the observed matrix has
    # exactly independent margins and stays independent under add-one
    # smoothing over the complete pair space.
    matrix = {
        "ga": {"t1": 4, "t2": 4, "t3": 4},
        "gb": {"t1": 12, "t2": 12, "t3": 12},
    }
    evidence = _counts(matrix)

    total = evidence.total_events
    for group in evidence.group_terms:
        for trait in evidence.trait_terms:
            expected = (
                evidence.group_margins[group] * evidence.trait_margins[trait] / total
            )
            assert evidence.counts[group][trait] == pytest.approx(expected)

    result = _result(audit_leakage(evidence, _spec()))

    assert result.status is DiagnosticStatus.READY
    assert result.details["mutual_information"] == pytest.approx(
        0.0, rel=0, abs=1e-12
    )
    assert result.value == pytest.approx(0.0, rel=0, abs=1e-12)
    assert result.value >= 0.0
    assert result.details["total_events"] == 48
    assert result.details["pair_space_size"] == 6


def test_smoothing_biases_a_generically_independent_matrix_upward():
    # Additive smoothing over the complete pair space is not distribution
    # preserving: a matrix whose margins are independent but whose row
    # profiles differ is pulled toward uniform, so the reported association
    # is a small positive number rather than an exact zero. The bias
    # vanishes as the smoothing constant does.
    matrix = {"ga": {"t1": 10, "t2": 20}, "gb": {"t1": 30, "t2": 60}}
    evidence = _counts(matrix)
    total = evidence.total_events
    for group in evidence.group_terms:
        for trait in evidence.trait_terms:
            assert evidence.counts[group][trait] == pytest.approx(
                evidence.group_margins[group] * evidence.trait_margins[trait] / total
            )

    add_one = _value(matrix)
    barely_smoothed = _value(
        matrix, diagnostic=StereotypeLeakage(smoothing_alpha=1e-300)
    )

    assert add_one > 0.0
    assert add_one == pytest.approx(3.252123637596724e-05, rel=0, abs=1e-15)
    assert barely_smoothed == 0.0


def test_deterministic_association_reaches_the_maximal_normalized_value():
    diagonal = {"ga": {"t1": 100, "t2": 0}, "gb": {"t1": 0, "t2": 100}}
    result = _result(
        audit_leakage(
            _counts(diagonal),
            _spec(),
            StereotypeLeakage(smoothing_alpha=1e-300),
        )
    )

    assert result.status is DiagnosticStatus.READY
    assert result.value == pytest.approx(1.0, rel=0, abs=1e-12)
    assert result.value <= 1.0
    assert result.details["nmi_definition"] == "2 * MI / (H_group + H_trait)"
    assert result.details["estimand"] == "normalized_mutual_information"
    assert result.details["unit"] == "dimensionless"

    three_by_three = {
        f"g{index}": {f"t{other}": (60 if index == other else 0) for other in range(3)}
        for index in range(3)
    }
    wider = _result(
        audit_leakage(
            _counts(three_by_three),
            _spec(),
            StereotypeLeakage(smoothing_alpha=1e-300),
        )
    )
    assert wider.value == pytest.approx(1.0, rel=0, abs=1e-12)
    assert wider.value <= 1.0


def test_paper_smoothing_shrinks_a_deterministic_association_below_the_maximum():
    diagonal = {"ga": {"t1": 100, "t2": 0}, "gb": {"t1": 0, "t2": 100}}
    unsmoothed = _value(diagonal, diagnostic=StereotypeLeakage(smoothing_alpha=1e-300))
    add_one = _value(diagonal)
    independent = _value({"ga": {"t1": 10, "t2": 10}, "gb": {"t1": 10, "t2": 10}})

    assert independent < add_one < unsmoothed
    assert add_one == pytest.approx(0.920509557616066, rel=0, abs=1e-15)


def test_normalized_value_is_invariant_to_the_log_base_but_information_is_not():
    matrix = {"ga": {"t1": 7, "t2": 2, "t3": 1}, "gb": {"t1": 1, "t2": 9, "t3": 3}}
    bits = _result(audit_leakage(_counts(matrix), _spec()))
    nats = _result(
        audit_leakage(
            _counts(matrix),
            _spec(),
            StereotypeLeakage(log_base=LogBase.NATURAL),
        )
    )

    assert nats.value == pytest.approx(bits.value, rel=0, abs=1e-15)
    assert nats.details["mutual_information"] != bits.details["mutual_information"]
    assert nats.details["mutual_information"] == pytest.approx(
        bits.details["mutual_information"] * math.log(2.0), rel=0, abs=1e-15
    )
    assert bits.details["information_unit"] == "bits"
    assert nats.details["information_unit"] == "nats"
    assert bits.details["log_base"] == "base_2"
    assert nats.details["log_base"] == "natural"


def test_relabelling_groups_and_reordering_rows_leave_the_value_unchanged():
    matrix = {"ga": {"t1": 7, "t2": 2, "t3": 1}, "gb": {"t1": 1, "t2": 9, "t3": 3}}
    baseline = _result(audit_leakage(_counts(matrix), _spec()))

    group_map = {"ga": "zeta", "gb": "alpha"}
    trait_map = {"t1": "u3", "t2": "u1", "t3": "u2"}
    relabelled = {
        group_map[group]: {trait_map[trait]: count for trait, count in row.items()}
        for group, row in matrix.items()
    }
    permuted = _result(audit_leakage(_counts(relabelled), _spec()))

    reordered_rows = {
        "gb": {"t3": 3, "t1": 1, "t2": 9},
        "ga": {"t2": 2, "t3": 1, "t1": 7},
    }
    reordered = _result(audit_leakage(_counts(reordered_rows), _spec()))

    assert permuted.value == baseline.value
    assert reordered.value == baseline.value
    assert reordered.details == baseline.details
    assert permuted.details["mutual_information"] == (
        baseline.details["mutual_information"]
    )
    assert permuted.details["group_term_count"] == 2
    assert permuted.details["trait_term_count"] == 3


def test_a_wider_declared_pair_space_shrinks_the_value_for_the_same_observations():
    observed = {"ga": {"t1": 7, "t2": 1}, "gb": {"t1": 1, "t2": 7}}
    widened = {
        "ga": {"t1": 7, "t2": 1, "t3": 0, "t4": 0},
        "gb": {"t1": 1, "t2": 7, "t3": 0, "t4": 0},
    }

    narrow = _result(audit_leakage(_counts(observed), _spec()))
    wide = _result(audit_leakage(_counts(widened), _spec()))

    assert narrow.details["total_events"] == wide.details["total_events"] == 16
    assert narrow.details["pair_space_size"] == 4
    assert wide.details["pair_space_size"] == 8
    assert wide.value < narrow.value
    assert narrow.details["smoothing_scheme"] == "additive_over_complete_pair_space"
    assert narrow.details["pair_space"] == "complete"


# ---------------------------------------------------------------------------
# Extraction and dual-path parity
# ---------------------------------------------------------------------------


def test_surface_extractor_reproduces_the_hand_counted_window_matrix():
    extractor = SurfaceCooccurrenceExtractor(config=_config())
    evidence = TextEvidence(
        axis=AXIS,
        texts=PARITY_TEXTS,
        source="Unregistered synthetic corpus",
    )

    counts = extractor.extract(evidence)

    assert extractor.tokenize(PARITY_TEXTS[0]) == (
        "the",
        "doctor",
        "was",
        "brilliant",
        "and",
        "caring",
    )
    assert {group: dict(row) for group, row in counts.counts.items()} == {
        "doctor": {"brilliant": 2, "caring": 1},
        "nurse": {"brilliant": 1, "caring": 2},
    }
    assert counts.total_events == 6
    assert counts.pair_space_size == 4
    assert counts.observed_cell_count == 4
    assert counts.counting_basis == PARITY_COUNTING_BASIS

    record = counts.extraction
    assert record is not None
    assert record.extractor_id == "surface_cooccurrence"
    assert record.extractor_version == "surface_cooccurrence/1"
    assert record.config_digest == extractor.config_digest
    assert record.text_digest == evidence.text_digest
    assert record.text_unit_count == len(PARITY_TEXTS)
    assert record.event_count == 6
    assert record.window == 5


def test_raw_text_and_supplied_count_matrix_paths_agree_exactly():
    config = _config()
    spec = _spec(leakage_extraction=config)
    text_evidence = TextEvidence(
        axis=AXIS,
        texts=PARITY_TEXTS,
        source="Unregistered synthetic corpus",
    )
    supplied = AssociationCounts.from_pair_counts(
        PARITY_PAIR_COUNTS,
        axis=AXIS,
        group_terms=GROUP_LEXICON,
        trait_terms=TRAIT_LEXICON,
        source="Unregistered synthetic corpus",
        counting_basis=PARITY_COUNTING_BASIS,
    )
    extracted = SurfaceCooccurrenceExtractor(config=config).extract(text_evidence)

    assert extracted.counts == supplied.counts
    assert extracted.matrix_digest == supplied.matrix_digest
    assert extracted.lexicon_digest == supplied.lexicon_digest

    from_text = _result(audit_leakage(text_evidence, spec))
    from_counts = _result(audit_leakage(supplied, spec))

    assert from_text.status is DiagnosticStatus.READY
    assert from_counts.status is DiagnosticStatus.READY
    assert from_text.value == from_counts.value
    assert from_text.value == pytest.approx(
        0.029049405545331364, rel=0, abs=1e-15
    )

    text_details = dict(from_text.details)
    count_details = dict(from_counts.details)
    assert text_details.pop("evidence_path") == "raw_text"
    assert count_details.pop("evidence_path") == "association_counts"
    assert text_details == count_details
    assert tuple(from_text.assumptions) == tuple(from_counts.assumptions)

    assert from_text.provenance["extractor"] == {
        "extractor_id": "surface_cooccurrence",
        "extractor_version": "surface_cooccurrence/1",
    }
    assert from_text.provenance["extraction_record"]["event_count"] == 6
    assert "extractor" not in from_counts.provenance
    assert "extraction_record" not in from_counts.provenance
    assert (
        from_text.provenance["extraction"] == from_counts.provenance["extraction"]
    )
    assert from_text.provenance["estimator"] == from_counts.provenance["estimator"]


def test_zero_lexical_hits_from_a_valid_extraction_is_the_canonical_zero():
    config = _config()
    spec = _spec(leakage_extraction=config)
    evidence = TextEvidence(
        axis=AXIS,
        texts=("Nobody here matches the declared lexicon at all.", "Still nothing."),
        source="Unregistered synthetic corpus",
    )

    report = audit_leakage(evidence, spec)
    result = _result(report)

    assert result.status is DiagnosticStatus.READY
    assert result.value == 0.0
    assert result.details["zero_lexical_hits"] is True
    assert result.details["total_events"] == 0
    assert result.details["observed_cell_count"] == 0
    assert result.provenance["extraction_record"]["event_count"] == 0
    assert result.provenance["extraction_record"]["text_unit_count"] == 2

    assert any("zero group-trait co-occurrences" in item for item in report.warnings)
    assert any(
        "not evidence that the dataset is free of stereotype association" in item
        for item in report.warnings
    )
    assert report.status is ReportStatus.SUCCESS


def test_all_zero_matrix_with_an_extraction_record_is_a_verified_zero():
    zeros = {"ga": {"t1": 0, "t2": 0}, "gb": {"t1": 0, "t2": 0}}
    report = audit_leakage(_counts(zeros, extraction=_record()), _spec())
    result = _result(report)

    assert result.status is DiagnosticStatus.READY
    assert result.value == 0.0
    assert result.details["zero_lexical_hits"] is True
    assert any("zero group-trait co-occurrences" in item for item in report.warnings)


def test_all_zero_matrix_without_an_extraction_record_is_blocked_not_zero():
    zeros = {"ga": {"t1": 0, "t2": 0}, "gb": {"t1": 0, "t2": 0}}
    report = audit_leakage(_counts(zeros), _spec())
    result = _result(report)

    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "zero_counts_without_extraction_record"
    assert result.details["total_events"] == 0
    assert "zero_lexical_hits" not in result.details
    assert report.status is ReportStatus.BLOCKED
    assert report.warnings == ()


def test_unverified_zero_requires_an_explicit_opt_in():
    zeros = {"ga": {"t1": 0, "t2": 0}, "gb": {"t1": 0, "t2": 0}}
    report = audit_leakage(
        _counts(zeros),
        _spec(),
        StereotypeLeakage(allow_unverified_zero=True),
    )
    result = _result(report)

    assert result.status is DiagnosticStatus.READY
    assert result.value == 0.0
    assert result.details["zero_lexical_hits"] is True
    assert any("zero group-trait co-occurrences" in item for item in report.warnings)


# ---------------------------------------------------------------------------
# Malformed or incomplete evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "counts",
    [
        {"ga": {"t1": 1, "t2": 2}, "gb": {"t1": 3}},
        {"ga": {"t1": 1, "t2": 2}, "gb": {"t1": 3, "t2": 4, "t3": 5}},
        {"ga": {"t1": 1, "t2": 2}},
        {"ga": {"t1": 1, "t2": 2}, "gb": {"t1": 3, "t2": 4}, "gc": {"t1": 0, "t2": 0}},
        {"ga": {"t1": 1, "t2": 2}, "gb": {"t1": -3, "t2": 4}},
        {"ga": {"t1": 1, "t2": 2}, "gb": {"t1": 3.5, "t2": 4}},
        {"ga": {"t1": 1, "t2": 2}, "gb": {"t1": True, "t2": 4}},
    ],
)
def test_incomplete_or_malformed_count_matrices_never_reach_the_kernel(counts):
    with pytest.raises((TypeError, ValueError)):
        AssociationCounts(
            axis=AXIS,
            group_terms=["ga", "gb"],
            trait_terms=["t1", "t2"],
            counts=counts,
            source="Unregistered synthetic corpus",
            counting_basis="declared synthetic counting basis",
        )


def test_an_extraction_record_that_disagrees_with_the_matrix_is_refused():
    with pytest.raises(ValueError, match="does not match the matrix total"):
        _counts(
            {"ga": {"t1": 1, "t2": 0}, "gb": {"t1": 0, "t2": 0}},
            extraction=_record(event_count=7),
        )


def test_raw_text_without_a_declared_extraction_is_blocked_not_zero():
    evidence = TextEvidence(
        axis=AXIS,
        texts=("The doctor was brilliant.",),
        source="Unregistered synthetic corpus",
    )
    report = audit_leakage(evidence, _spec())
    result = _result(report)

    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "missing_extraction_config"
    assert result.details["evidence_path"] == "raw_text"
    assert report.status is ReportStatus.BLOCKED


def test_lexicon_support_mismatch_blocks_instead_of_aligning_supports():
    foreign = AssociationCounts.from_pair_counts(
        {("x1", "y1"): 3},
        axis=AXIS,
        group_terms=["x1", "x2"],
        trait_terms=["y1", "y2"],
        source="Unregistered synthetic corpus",
        counting_basis="declared synthetic counting basis",
    )
    result = _result(
        audit_leakage(foreign, _spec(leakage_extraction=_config()))
    )

    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "lexicon_support_mismatch"
    assert tuple(result.details["evidence_group_terms"]) == ("x1", "x2")
    assert tuple(result.details["extraction_group_terms"]) == GROUP_LEXICON
    assert tuple(result.details["group_term_symmetric_difference"]) == (
        "doctor",
        "nurse",
        "x1",
        "x2",
    )


def test_extraction_configuration_mismatch_blocks_and_reports_both_digests():
    config = _config()
    evidence = TextEvidence(
        axis=AXIS,
        texts=PARITY_TEXTS,
        source="Unregistered synthetic corpus",
    )
    extracted = SurfaceCooccurrenceExtractor(config=config).extract(evidence)
    other = _config(window=3)

    result = _result(audit_leakage(extracted, _spec(leakage_extraction=other)))

    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "extraction_config_mismatch"
    assert result.details["evidence_config_digest"] == config.config_digest
    assert result.details["extraction_config_digest"] == other.config_digest
    assert config.config_digest != other.config_digest


def test_a_failing_extraction_is_failed_not_a_zero():
    class ExplodingExtractor:
        extractor_id = "exploding_extractor"
        extractor_version = "exploding/1"

        def __init__(self, config):
            self.config = config

        @property
        def config_digest(self):
            return self.config.config_digest

        def extract(self, evidence):
            raise RuntimeError("synthetic extraction failure")

        def to_dict(self):
            return {
                "extractor_id": self.extractor_id,
                "extractor_version": self.extractor_version,
            }

    evidence = TextEvidence(
        axis=AXIS,
        texts=("The doctor was brilliant.",),
        source="Unregistered synthetic corpus",
    )
    report = audit_leakage(
        evidence,
        _spec(),
        StereotypeLeakage(extractor=ExplodingExtractor(_config())),
    )
    result = _result(report)

    assert result.status is DiagnosticStatus.FAILED
    assert result.value is None
    assert result.reason_code == "extraction_failed"
    assert result.details["exception_type"] == "RuntimeError"
    assert report.status is ReportStatus.FAILED
    assert report.warnings == ()


def test_an_object_that_does_not_implement_the_extractor_protocol_is_refused():
    class NotAnExtractor:
        extractor_id = "incomplete"

    with pytest.raises(TypeError, match="LeakageExtractor protocol"):
        StereotypeLeakage(extractor=NotAnExtractor())


# ---------------------------------------------------------------------------
# Configuration sensitivity and provenance
# ---------------------------------------------------------------------------


def test_smoothing_alpha_changes_the_value_and_is_recorded_in_provenance():
    matrix = {"ga": {"t1": 7, "t2": 2, "t3": 1}, "gb": {"t1": 1, "t2": 9, "t3": 3}}
    values = {}
    for alpha in (0.01, 1.0, 10.0):
        result = _result(
            audit_leakage(
                _counts(matrix),
                _spec(),
                StereotypeLeakage(smoothing_alpha=alpha),
            )
        )
        values[alpha] = result.value
        assert result.details["smoothing_alpha"] == alpha
        assert result.provenance["estimator"]["smoothing_alpha"] == alpha
        assert result.provenance["estimator"]["smoothing_scheme"] == (
            "additive_over_complete_pair_space"
        )

    assert len(set(values.values())) == 3
    assert values[0.01] > values[1.0] > values[10.0]

    paper = _result(audit_leakage(_counts(matrix), _spec()))
    generalized = _result(
        audit_leakage(
            _counts(matrix),
            _spec(),
            StereotypeLeakage(smoothing_alpha=0.5),
        )
    )
    assert paper.details["paper_alignment"] == "paper_exact"
    assert generalized.details["paper_alignment"] == "generalized_estimator"


def test_window_size_changes_the_counts_the_value_and_the_provenance():
    texts = (
        "The doctor was brilliant.",
        "The nurse is caring.",
        "doctor one two three four five six seven caring",
    )
    evidence = TextEvidence(
        axis=AXIS,
        texts=texts,
        source="Unregistered synthetic corpus",
    )

    narrow_config = _config(window=5)
    wide_config = _config(window=8)
    narrow_counts = SurfaceCooccurrenceExtractor(config=narrow_config).extract(evidence)
    wide_counts = SurfaceCooccurrenceExtractor(config=wide_config).extract(evidence)

    assert narrow_counts.total_events == 2
    assert wide_counts.total_events == 3
    assert narrow_counts.counts != wide_counts.counts

    narrow = _result(
        audit_leakage(evidence, _spec(leakage_extraction=narrow_config))
    )
    wide = _result(audit_leakage(evidence, _spec(leakage_extraction=wide_config)))

    assert narrow.value != wide.value
    assert narrow.provenance["extraction"]["window"] == 5
    assert wide.provenance["extraction"]["window"] == 8
    assert narrow.provenance["extraction"]["paper_alignment"] == "paper_exact"
    assert wide.provenance["extraction"]["paper_alignment"] == "generalized_extraction"
    assert any("generalized package extension" in item for item in wide.assumptions)


def test_token_normalization_is_declared_and_collapses_are_recorded():
    config = LeakageExtractionConfig(
        group_lexicon=["Doctors", "doctor", "nurse"],
        trait_lexicon=["brilliant", "caring"],
        replace_substrings={"nurse practitioner": "nurse"},
        surface_map={"doctors": "doctor"},
    )

    assert config.canonical_group_terms == ("doctor", "nurse")
    assert config.collapsed_terms["doctor"] == ("Doctors", "doctor")
    assert config.pair_space_size == 4

    extractor = SurfaceCooccurrenceExtractor(config=config)
    assert extractor.tokenize("Two DOCTORS arrived") == ("two", "doctor", "arrived")


def test_provenance_preserves_every_declared_extraction_and_estimator_knob():
    config = _config()
    evidence = TextEvidence(
        axis=AXIS,
        texts=PARITY_TEXTS,
        source="Unregistered synthetic corpus",
    )
    result = _result(audit_leakage(evidence, _spec(leakage_extraction=config)))

    extraction = result.provenance["extraction"]
    assert tuple(extraction["group_lexicon"]) == GROUP_LEXICON
    assert tuple(extraction["trait_lexicon"]) == TRAIT_LEXICON
    assert tuple(extraction["canonical_group_terms"]) == GROUP_LEXICON
    assert tuple(extraction["canonical_trait_terms"]) == TRAIT_LEXICON
    assert extraction["window"] == 5
    assert extraction["window_rule"] == "symmetric_token_window_excluding_center"
    assert extraction["match_attribute"] == "surface"
    assert extraction["token_pattern"] == r"\b[\w_]+\b"
    assert extraction["lowercase"] is True
    assert extraction["replace_substrings"] == {}
    assert extraction["surface_map"] == {}
    assert extraction["counting_unit"] == "group_anchored_ordered_cooccurrence"
    assert extraction["config_digest"] == config.config_digest

    estimator = result.provenance["estimator"]
    assert estimator["smoothing_alpha"] == 1.0
    assert estimator["log_base"] == "base_2"
    assert estimator["information_unit"] == "bits"
    assert estimator["pair_space"] == "complete"
    assert estimator["paper_alignment"] == "paper_exact"

    assert result.provenance["extractor"]["extractor_version"] == (
        "surface_cooccurrence/1"
    )
    assert result.provenance["extraction_record"]["extractor_version"] == (
        "surface_cooccurrence/1"
    )
    assert result.provenance["evidence"]["counting_basis"] == PARITY_COUNTING_BASIS
    assert result.provenance["evidence"]["lexicon_digest"]
    assert result.provenance["evidence"]["matrix_digest"]


def test_top_pmi_pairs_cover_observed_cells_only_and_honour_the_limit():
    matrix = {"ga": {"t1": 7, "t2": 1, "t3": 0}, "gb": {"t1": 1, "t2": 7, "t3": 0}}
    full = _result(audit_leakage(_counts(matrix), _spec()))
    limited = _result(
        audit_leakage(_counts(matrix), _spec(), StereotypeLeakage(top_pmi_pairs=1))
    )

    assert full.details["top_pmi_pairs_scope"] == "observed_cells_only"
    assert len(full.details["top_pmi_pairs"]) == 4
    assert all(pair["count"] > 0 for pair in full.details["top_pmi_pairs"])
    pmi_values = [pair["pmi"] for pair in full.details["top_pmi_pairs"]]
    assert pmi_values == sorted(pmi_values, reverse=True)
    assert len(limited.details["top_pmi_pairs"]) == 1
    assert limited.details["top_pmi_pairs"][0] == full.details["top_pmi_pairs"][0]


def test_count_matrix_is_only_serialized_when_it_is_explicitly_requested():
    matrix = {"ga": {"t1": 7, "t2": 1}, "gb": {"t1": 1, "t2": 7}}
    default = _result(audit_leakage(_counts(matrix), _spec()))
    verbose = _result(
        audit_leakage(
            _counts(matrix), _spec(), StereotypeLeakage(include_count_matrix=True)
        )
    )

    assert "count_matrix" not in default.details
    assert verbose.details["count_matrix"]["ga"] == {"t1": 7, "t2": 1}


@pytest.mark.parametrize(
    "smoothing_alpha",
    [
        0.0,
        -1.0,
        True,
        5e-324,
        1e-310,
        Fraction(1, 10**4000),
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_smoothing_alpha_must_be_finite_positive_and_numeric(smoothing_alpha):
    with pytest.raises((TypeError, ValueError)):
        StereotypeLeakage(smoothing_alpha=smoothing_alpha)


def test_the_default_diagnostic_is_the_paper_parity_estimator():
    diagnostic = StereotypeLeakage()

    assert diagnostic.smoothing_alpha == 1.0
    assert diagnostic.log_base is LogBase.BASE_2
    assert diagnostic.information_unit == "bits"
    assert diagnostic.paper_alignment == "paper_exact"
    assert diagnostic.extractor is None
    assert diagnostic.allow_unverified_zero is False
    assert diagnostic.include_count_matrix is False


# ---------------------------------------------------------------------------
# Applicability: a non-ready component never carries a number
# ---------------------------------------------------------------------------


def test_unrequested_component_is_not_applicable_without_a_numeric_sentinel():
    result = _result(
        audit_leakage(
            _counts({"ga": {"t1": 3, "t2": 1}, "gb": {"t1": 1, "t2": 3}}),
            _spec(requested_components=("b_rep",)),
        )
    )

    assert result.status is DiagnosticStatus.NOT_APPLICABLE
    assert result.value is None
    assert result.reason_code == "component_not_requested"


def test_generated_output_is_relabelled_not_measured_as_dataset_leakage():
    report = audit_leakage(
        _counts({"ga": {"t1": 3, "t2": 1}, "gb": {"t1": 1, "t2": 3}}),
        _spec(target_kind=TargetKind.GENERATED_OUTPUT),
    )
    result = _result(report)

    assert result.status is DiagnosticStatus.NOT_APPLICABLE
    assert result.value is None
    assert result.reason_code == "target_kind_not_supported"
    assert "output association" in result.reason
    assert report.status is ReportStatus.NOT_APPLICABLE


def test_an_axis_outside_the_declared_protected_axes_is_blocked():
    result = _result(
        audit_leakage(
            _counts({"ga": {"t1": 3, "t2": 1}, "gb": {"t1": 1, "t2": 3}}),
            _spec(protected_axes=("some-other-axis",)),
        )
    )

    assert result.status is DiagnosticStatus.BLOCKED
    assert result.value is None
    assert result.reason_code == "axis_not_declared"
    assert tuple(result.details["protected_axes"]) == ("some-other-axis",)


def test_a_component_override_can_suppress_but_never_manufacture_a_value():
    override = ComponentOverride(
        component="b_leak",
        status="not_applicable",
        reason="The audit declares this axis out of scope for leakage.",
        declared_reason_code="axis_out_of_scope",
    )
    report = audit_leakage(
        _counts({"ga": {"t1": 3, "t2": 1}, "gb": {"t1": 1, "t2": 3}}),
        _spec(component_overrides={"b_leak": override}),
    )
    result = _result(report)

    assert result.status is DiagnosticStatus.NOT_APPLICABLE
    assert result.value is None
    assert result.reason_code == "applicability_override"
    assert result.details["declared_reason_code"] == "axis_out_of_scope"
    assert result.provenance["override"]["component"] == "b_leak"
    assert report.warnings == ()


# ---------------------------------------------------------------------------
# Report surface
# ---------------------------------------------------------------------------


def test_a_ready_report_warns_about_intent_and_about_a_stress_test_stance():
    matrix = {"ga": {"t1": 7, "t2": 1}, "gb": {"t1": 1, "t2": 7}}
    population = audit_leakage(_counts(matrix), _spec())
    stress = audit_leakage(_counts(matrix), _spec(design_stance="stress_test"))

    assert len(population.warnings) == 1
    assert "intended test signal" in population.warnings[0]
    assert len(stress.warnings) == 2
    assert stress.warnings[0] == population.warnings[0]
    assert "stress-test" in stress.warnings[1]
    assert _result(stress).value == _result(population).value
    assert any(
        item.startswith("Design stance: stress_test")
        for item in _result(stress).assumptions
    )


def test_class_and_functional_entry_points_agree():
    evidence = _counts({"ga": {"t1": 7, "t2": 1}, "gb": {"t1": 1, "t2": 7}})
    spec = _spec()
    diagnostic = StereotypeLeakage()

    direct = diagnostic.compute(evidence, spec)
    reported = _result(audit_leakage(evidence, spec, diagnostic))

    assert reported.to_dict() == direct.to_dict()


def test_the_report_serializes_to_json_without_a_numeric_sentinel():
    evidence = _counts({"ga": {"t1": 7, "t2": 1}, "gb": {"t1": 1, "t2": 7}})
    ready = json.loads(json.dumps(audit_leakage(evidence, _spec()).to_dict()))
    blocked = json.loads(
        json.dumps(
            audit_leakage(
                _counts({"ga": {"t1": 0, "t2": 0}, "gb": {"t1": 0, "t2": 0}}),
                _spec(),
            ).to_dict()
        )
    )

    assert ready["components"]["b_leak"]["status"] == "ready"
    assert ready["components"]["b_leak"]["value"] > 0.0
    assert blocked["components"]["b_leak"]["status"] == "blocked"
    assert blocked["components"]["b_leak"]["value"] is None
    assert blocked["components"]["b_leak"]["reason_code"] == (
        "zero_counts_without_extraction_record"
    )


def test_a_ready_computation_exception_becomes_a_failed_component():
    class FailingLeakage(StereotypeLeakage):
        def compute(self, evidence, spec):
            raise OverflowError("synthetic numeric failure")

    report = audit_leakage(
        _counts({"ga": {"t1": 7, "t2": 1}, "gb": {"t1": 1, "t2": 7}}),
        _spec(),
        FailingLeakage(),
    )
    result = _result(report)

    assert result.status is DiagnosticStatus.FAILED
    assert result.value is None
    assert result.reason_code == "computation_failed"
    assert result.details["exception_type"] == "OverflowError"
    assert report.status is ReportStatus.FAILED


@pytest.mark.parametrize(
    ("smoothing_alpha", "reason_code", "nonpositive_marginals", "smoothed_total"),
    [
        pytest.param(
            1e-160,
            "numeric_computation_failed",
            0,
            1_000_000.0,
            id="marginal_product_underflows_to_zero",
        ),
        pytest.param(
            1e308,
            "degenerate_normalization_denominator",
            4,
            None,
            id="smoothed_total_overflows_to_infinity",
        ),
    ],
)
def test_a_float_degenerate_smoothing_constant_fails_instead_of_raising(
    smoothing_alpha, reason_code, nonpositive_marginals, smoothed_total
):
    """A strictly positive alpha keeps every cell positive in the reals only.

    ``smoothing_alpha`` is validated over the whole of
    ``[sys.float_info.min, inf)``, and at either extreme IEEE-754 breaks the
    invariant: the marginal product underflows to ``0.0`` for a tiny alpha,
    and the smoothed total overflows to ``inf`` -- driving every cell and
    every marginal to ``0.0`` -- for a huge one. ``compute()`` must publish
    that as a ``failed`` component carrying the estimator settings, not raise
    ``ZeroDivisionError`` out of a pure kernel and lose them.
    """

    counts = _counts(
        {"doctor": {"brilliant": 1_000_000, "caring": 0}, "nurse": {"brilliant": 0, "caring": 0}}
    )
    diagnostic = StereotypeLeakage(smoothing_alpha=smoothing_alpha)

    result = diagnostic.compute(counts, _spec())

    assert result.status is DiagnosticStatus.FAILED
    assert result.value is None
    assert result.reason_code == reason_code
    assert result.details["smoothing_alpha"] == smoothing_alpha
    assert result.details["log_base"] == "base_2"
    assert result.details["total_events"] == 1_000_000
    assert result.details["pair_space_size"] == 4
    assert result.details["nonpositive_marginal_count"] == nonpositive_marginals
    assert result.details["smoothed_total"] == smoothed_total

    report = audit_leakage(counts, _spec(), diagnostic)
    reported = _result(report)

    assert reported.status is DiagnosticStatus.FAILED
    assert reported.value is None
    assert reported.reason_code == reason_code
    # Not swallowed into ``computation_failed`` by the module entry point.
    assert "exception_type" not in reported.details
    assert report.status is ReportStatus.FAILED
    json.dumps(report.to_dict())

    # The same matrix with a workable constant still measures the association,
    # so the failure is about the constant, not about the evidence.
    ready = StereotypeLeakage(smoothing_alpha=1e-100).compute(counts, _spec())
    assert ready.status is DiagnosticStatus.READY
    assert ready.value == pytest.approx(0.4957280095210252)


# ---------------------------------------------------------------------------
# Frozen paper golden fixture
# ---------------------------------------------------------------------------


def test_gap_paper_lexicon_golden_fixture_is_portable_and_matches_the_paper():
    fixture_text = GOLDEN_FIXTURE_PATH.read_text(encoding="utf-8")
    fixture = json.loads(fixture_text)

    assert "/Users/" not in fixture_text
    assert "Downloads" not in fixture_text
    assert fixture["fixture_schema_version"] == "1"
    assert fixture["fixture_id"] == "gap_paper_lexicon_v1"
    assert fixture["dataset"] == "GAP"
    assert not Path(fixture["oracle"]["script"]).is_absolute()
    assert fixture["oracle"]["replayable_in_ci"] is False
    assert "does not execute the external paper runner" in (
        fixture["oracle"]["verification_note"]
    )
    assert "does not bundle or replay the GAP corpus" in (
        fixture["evidence"]["provenance"]["limitation"]
    )

    parameters = fixture["parameters"]
    assert parameters["smoothing_alpha"] == 1.0
    assert parameters["log_base"] == "base_2"
    assert parameters["window"] == 5
    assert parameters["match_attribute"] == "surface"

    group_terms = fixture["evidence"]["group_terms"]
    trait_terms = fixture["evidence"]["trait_terms"]
    assert len(group_terms) == 47
    assert len(trait_terms) == 36
    assert not set(group_terms) & set(trait_terms)

    pair_counts = {
        (group, trait): count
        for group, trait, count in fixture["evidence"]["observed_pair_counts"]
    }
    assert len(pair_counts) == fixture["expected"]["observed_cell_count"]
    assert sum(pair_counts.values()) == fixture["expected"]["total_events"]
    assert fixture["expected"]["total_events"] == fixture["paper"]["total_pairs"]

    evidence = AssociationCounts.from_pair_counts(
        pair_counts,
        axis=fixture["axis"],
        group_terms=group_terms,
        trait_terms=trait_terms,
        source="Portable paper golden count matrix",
        counting_basis=fixture["evidence"]["counting_basis"],
    )
    assert evidence.pair_space_size == fixture["expected"]["pair_space_size"]
    assert evidence.observed_cell_count == fixture["expected"]["observed_cell_count"]

    spec = DatasetAuditSpec(
        target_name=fixture["dataset"],
        target_kind="benchmark_dataset",
        task_family="coreference_resolution",
        design_stance="stress_test",
        references={},
        requested_components=("b_leak",),
    )
    diagnostic = StereotypeLeakage(
        smoothing_alpha=parameters["smoothing_alpha"],
        log_base=parameters["log_base"],
        top_pmi_pairs=parameters["top_pmi_pairs"],
    )
    result = _result(audit_leakage(evidence, spec, diagnostic))

    assert result.status.value == fixture["expected"]["status"]
    assert result.value == fixture["expected"]["value"]
    assert result.details["mutual_information"] == (
        fixture["expected"]["mutual_information"]
    )
    assert result.details["information_unit"] == fixture["expected"]["information_unit"]
    assert result.details["paper_alignment"] == fixture["expected"]["paper_alignment"]
    assert result.details["total_events"] == fixture["expected"]["total_events"]
    assert result.details["pair_space_size"] == fixture["expected"]["pair_space_size"]

    tolerance = fixture["paper"]["tolerance"]
    assert result.value == pytest.approx(
        fixture["paper"]["nmi"], rel=0, abs=tolerance
    )
    assert result.details["mutual_information"] == pytest.approx(
        fixture["paper"]["mi"], rel=0, abs=tolerance
    )

    expected_pairs = fixture["expected"]["top_pmi_pairs"]
    computed_pairs = result.details["top_pmi_pairs"]
    assert len(computed_pairs) == len(expected_pairs) == parameters["top_pmi_pairs"]
    for computed, expected in zip(computed_pairs, expected_pairs):
        assert computed["group"] == expected["group"]
        assert computed["trait"] == expected["trait"]
        assert computed["count"] == expected["count"]
        assert computed["pmi"] == pytest.approx(expected["pmi"], rel=0, abs=1e-12)
