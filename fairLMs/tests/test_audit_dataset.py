"""Contract tests for ``audit_dataset`` and the ``fairLMs.diagnostics`` surface.

Everything asserted here is part of the published contract -- component status,
reason code, serialized payload, numeric value -- never an implementation
detail.  The fixture is a deliberately unregistered dataset whose column names
("seam", "phrasing", "cast", "node", "slate", "stall", "wording") match no
loader in this package, so no component can be recognized, selected or
configured by inference.

The tests are offline and deterministic: no network, no model download, no
randomness.
"""

from __future__ import annotations

import json
import pathlib
import re
from types import MappingProxyType

import pytest

import fairLMs.diagnostics as diagnostics_package
from fairLMs.diagnostics import registry as registry_module
from fairLMs.diagnostics import (
    BACKEND_CONSTRUCTION_SLOTS,
    CONSTRUCTION_SLOTS,
    DIAGNOSTIC_REGISTRY,
    DIAGNOSTIC_SCHEMA_VERSION,
    LIGHTWEIGHT_CONSTRUCTION_SLOTS,
    SELF_IDENTIFICATION_FRAME,
    ComponentPlan,
    ComponentResult,
    DatasetAudit,
    DatasetAuditSpec,
    DatasetDiagnostic,
    DatasetEvidence,
    DiagnosticReport,
    DiagnosticStatus,
    FramingDisparity,
    GroupedTexts,
    IdentityMaskConfig,
    LeakageExtractionConfig,
    LengthDisparity,
    MinimalPairResidual,
    OptionItems,
    OptionLengthBias,
    OptionRoleContrast,
    PairedTexts,
    ReferenceDistribution,
    ReportStatus,
    RepresentationEvidence,
    ScoredGroups,
    ScorerMeanGap,
    TemplateGroups,
    TemplateImbalance,
    TextEvidence,
    audit_dataset,
    audit_scores,
    construction_vector,
    get_diagnostic,
    list_diagnostics,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

AXIS = "lithic-tradition"
SOURCE = "quarry_probe.csv @ 2026-02-11"
TARGET = "unlisted-quarry-probe"

_ROWS = (
    {"seam": "basalt", "phrasing": "the basalt blade cuts clean", "cast": "arc-1"},
    {"seam": "basalt", "phrasing": "the basalt blade splits", "cast": "arc-1"},
    {"seam": "obsidian", "phrasing": "the obsidian core cuts clean", "cast": "arc-2"},
    {"seam": "obsidian", "phrasing": "the obsidian core splits wide", "cast": "arc-3"},
)

_PAIR_ROWS = (
    {"node": "p1", "slate": "rough", "phrasing": "the basalt blade cuts clean"},
    {"node": "p1", "slate": "refined", "phrasing": "the obsidian blade cuts clean"},
    {"node": "p2", "slate": "rough", "phrasing": "the basalt blade splits"},
    {"node": "p2", "slate": "refined", "phrasing": "the obsidian blade splits"},
)

_OPTION_ROWS = (
    {"stall": "q1", "slate": "rough", "wording": "the basalt blade"},
    {"stall": "q1", "slate": "refined", "wording": "the obsidian blade"},
    {"stall": "q2", "slate": "rough", "wording": "basalt"},
    {"stall": "q2", "slate": "refined", "wording": "obsidian core"},
)

#: Every diagnostic that is actually implemented, and the class that implements
#: it.  A stub or an unimplemented slot must never appear here or in
#: ``DIAGNOSTIC_REGISTRY``; the three backend-dependent slots are real classes
#: that block by name until a backend is supplied.
_IMPLEMENTED_DIAGNOSTICS = {
    "b_diff_dep": "DependencyDepthDisparity",
    "b_diff_len": "LengthDisparity",
    "b_equiv": "SemanticEquivalence",
    "b_frame": "FramingDisparity",
    "b_gram": "GrammarConsistency",
    "b_leak": "StereotypeLeakage",
    "b_min": "MinimalPairResidual",
    "b_opt": "OptionLengthBias",
    "b_rep": "RepresentativenessBias",
    "b_temp": "TemplateImbalance",
    "score_counterfactual_sensitivity": "ScorerCounterfactualSensitivity",
    "score_mean_gap": "ScorerMeanGap",
    "score_rate_gap": "ScorerRateGap",
    "score_wasserstein_1_gap": "ScorerWasserstein1Gap",
}


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _representation():
    return RepresentationEvidence(
        axis=AXIS, counts={"basalt": 2, "obsidian": 2}, source=SOURCE
    )


def _evidence(*, target_name=TARGET, **overrides) -> DatasetEvidence:
    views = {
        "representation": {AXIS: _representation()},
        "texts": {
            AXIS: TextEvidence.from_records(
                _ROWS, axis=AXIS, text_field="phrasing", source=SOURCE
            )
        },
        "grouped_texts": {
            AXIS: GroupedTexts.from_records(
                _ROWS,
                axis=AXIS,
                group_field="seam",
                text_field="phrasing",
                declared_groups=("basalt", "obsidian"),
                source=SOURCE,
            )
        },
        "paired_texts": {
            AXIS: PairedTexts.from_records(
                _PAIR_ROWS,
                axis=AXIS,
                pair_id_field="node",
                condition_field="slate",
                text_field="phrasing",
                condition_roles=("rough", "refined"),
                pairing_basis="Rows sharing node differ only in the seam token.",
                source=SOURCE,
            )
        },
        "option_items": {
            AXIS: OptionItems.from_records(
                _OPTION_ROWS,
                axis=AXIS,
                item_id_field="stall",
                role_field="slate",
                option_field="wording",
                declared_roles=("rough", "refined"),
                question_family="multiple_choice_stereotype_contrast",
                source=SOURCE,
            )
        },
        "template_groups": {
            AXIS: TemplateGroups.from_records(
                _ROWS,
                axis=AXIS,
                group_field="seam",
                template_id_field="cast",
                declared_groups=("basalt", "obsidian"),
                template_identity_rule="Exact cast string as authored upstream.",
                source=SOURCE,
            )
        },
    }
    views.update(overrides)
    return DatasetEvidence(target_name=target_name, **views)


def _thin_evidence() -> DatasetEvidence:
    """Only the representation view: every other geometry is genuinely absent."""
    return _evidence(
        texts={},
        grouped_texts={},
        paired_texts={},
        option_items={},
        template_groups={},
    )


def _extraction() -> LeakageExtractionConfig:
    return LeakageExtractionConfig(
        group_lexicon=("basalt", "obsidian"), trait_lexicon=("blade", "core")
    )


def _spec(
    *,
    target_name=TARGET,
    task_family="paired_sentence",
    requested_components=("b_rep", "b_leak", *CONSTRUCTION_SLOTS),
    target_kind="benchmark_dataset",
    design_stance="stress_test",
    protected_axes=(AXIS,),
    leakage_extraction=True,
) -> DatasetAuditSpec:
    return DatasetAuditSpec(
        target_name=target_name,
        target_kind=target_kind,
        task_family=task_family,
        design_stance=design_stance,
        protected_axes=protected_axes,
        references={
            AXIS: ReferenceDistribution(
                axis=AXIS,
                probabilities={"basalt": 0.5, "obsidian": 0.5},
                source="Balanced design target stated in the dataset card.",
                purpose="design_target",
                population="Intended balanced counterfactual design",
            )
        },
        requested_components=requested_components,
        leakage_extraction=_extraction() if leakage_extraction else None,
    )


def _configured():
    """Fully configured instances for the five lightweight slots."""
    return (
        MinimalPairResidual(
            identity_mask=IdentityMaskConfig(identity_terms=("basalt", "obsidian"))
        ),
        LengthDisparity(),
        OptionLengthBias(
            role_contrast=OptionRoleContrast(
                stereotype_role="rough", anti_stereotype_role="refined"
            )
        ),
        FramingDisparity(predicate=SELF_IDENTIFICATION_FRAME),
        TemplateImbalance(),
    )


def _assert_value_discipline(report: DiagnosticReport) -> None:
    """A non-ready component never carries a number, and never a zero."""
    for name, result in report.components.items():
        if result.status is DiagnosticStatus.READY:
            assert isinstance(result.value, float), name
            assert result.reason_code is None, name
        else:
            assert result.value is None, name
            assert result.reason_code, name
            assert result.reason, name


# --------------------------------------------------------------------------
# Suitable evidence runs; unsuitable evidence is reported, never skipped
# --------------------------------------------------------------------------


def test_requested_components_with_suitable_evidence_actually_run():
    report = audit_dataset(
        _evidence(), _spec(), axis=AXIS, diagnostics=_configured()
    ).run()

    ready = {
        name
        for name, result in report.components.items()
        if result.status is DiagnosticStatus.READY
    }
    assert ready == {
        "b_rep",
        "b_leak",
        "b_min",
        "b_diff_len",
        "b_frame",
        "b_opt",
        "b_temp",
    }
    assert report.components["b_rep"].value == 0.0
    assert report.components["b_min"].value == 0.0
    assert report.components["b_frame"].value == 0.0
    assert report.components["b_diff_len"].value == pytest.approx(
        0.5 / 4.75, rel=0, abs=1e-15
    )
    assert report.components["b_opt"].value == -0.5
    assert report.components["b_temp"].value == 1.0
    assert report.components["b_leak"].value == pytest.approx(
        0.18872187554086717, rel=0, abs=1e-12
    )
    # Only the three backend slots are unready, so the report is partial.
    assert report.status is ReportStatus.PARTIAL
    _assert_value_discipline(report)


def test_requested_component_without_suitable_evidence_is_reported_not_skipped():
    report = audit_dataset(
        _thin_evidence(),
        _spec(requested_components=("b_rep", "b_leak", "b_min", "b_opt")),
        axis=AXIS,
    ).run()

    # Nothing vanished: every requested component is in the report.
    assert {"b_rep", "b_leak", "b_min", "b_opt"} <= set(report.components)

    leakage = report.components["b_leak"]
    assert leakage.status is DiagnosticStatus.BLOCKED
    assert leakage.reason_code == "missing_association_evidence"
    assert list(leakage.details["required_views"]) == ["association_counts", "texts"]

    for slot in ("b_min", "b_opt"):
        result = report.components[slot]
        assert result.status is DiagnosticStatus.NOT_APPLICABLE, slot
        assert result.reason_code == "evidence_view_not_supplied", slot
        assert result.details["axis"] == AXIS
    assert report.components["b_min"].details["required_view"] == "paired_texts"
    assert report.components["b_opt"].details["required_view"] == "option_items"

    _assert_value_discipline(report)


def test_absent_representation_evidence_is_not_applicable_rather_than_zero():
    report = audit_dataset(
        _evidence(representation={}),
        _spec(requested_components=("b_rep",)),
        axis=AXIS,
    ).run()

    result = report.components["b_rep"]
    assert result.status is DiagnosticStatus.NOT_APPLICABLE
    assert result.reason_code == "evidence_view_not_supplied"
    assert result.value is None
    assert report.status is ReportStatus.NOT_APPLICABLE


def test_a_component_whose_configuration_is_missing_blocks_with_a_precise_code():
    # No diagnostics are supplied, so every configurable slot is
    # default-constructed and must block for its own missing estimand
    # declaration rather than guess one.
    report = audit_dataset(_evidence(), _spec(), axis=AXIS).run()

    expected = {
        "b_min": "missing_identity_mask",
        "b_opt": "missing_option_role_contrast",
        "b_frame": "missing_frame_predicate",
    }
    for slot, reason_code in expected.items():
        result = report.components[slot]
        assert result.status is DiagnosticStatus.BLOCKED, slot
        assert result.reason_code == reason_code, slot
        assert result.value is None, slot

    # A missing configuration blocks only its own slot.
    assert report.components["b_diff_len"].status is DiagnosticStatus.READY
    assert report.components["b_temp"].status is DiagnosticStatus.READY
    _assert_value_discipline(report)


def test_a_supplied_view_without_its_declared_reference_blocks_b_rep():
    # The representation view is present, but the comparison it must be scored
    # against is not declared, so the component blocks instead of scoring
    # against an invented uniform reference.
    spec = DatasetAuditSpec(
        target_name=TARGET,
        target_kind="benchmark_dataset",
        task_family="paired_sentence",
        design_stance="stress_test",
        protected_axes=(AXIS,),
        references={},
        requested_components=("b_rep",),
    )
    result = audit_dataset(_evidence(), spec, axis=AXIS).run().components["b_rep"]

    assert result.status is DiagnosticStatus.BLOCKED
    assert result.reason_code == "missing_reference"
    assert result.value is None


def test_text_evidence_without_an_extraction_configuration_blocks_b_leak():
    report = audit_dataset(
        _evidence(),
        _spec(requested_components=("b_leak",), leakage_extraction=False),
        axis=AXIS,
    ).run()

    result = report.components["b_leak"]
    assert result.status is DiagnosticStatus.BLOCKED
    assert result.reason_code == "missing_extraction_config"
    assert result.value is None
    assert result.details["evidence_path"] == "raw_text"


# --------------------------------------------------------------------------
# Only the request selects components
# --------------------------------------------------------------------------


def test_a_component_absent_from_requested_components_never_runs():
    evidence = _evidence()
    report = audit_dataset(
        evidence, _spec(requested_components=("b_rep",)), axis=AXIS
    ).run()

    # Every view is present, but only the requested component is planned.
    assert set(evidence.available_views) == {
        "grouped_texts",
        "option_items",
        "paired_texts",
        "representation",
        "template_groups",
        "texts",
    }
    assert set(report.components) == {"b_rep"}
    assert dict(report.provenance["views_used"]) == {"b_rep": "representation"}
    assert list(report.provenance["unused_evidence_views"]) == [
        "grouped_texts",
        "option_items",
        "paired_texts",
        "template_groups",
        "texts",
    ]
    # No construction slot is in play, so no slot metadata is emitted.
    assert "slot_order" not in report.provenance


def test_supplying_a_diagnostic_for_an_unrequested_component_is_refused():
    with pytest.raises(ValueError, match="is not in spec.requested_components"):
        audit_dataset(
            _evidence(),
            _spec(requested_components=("b_rep",)),
            axis=AXIS,
            diagnostics=(LengthDisparity(),),
        )


def test_an_unrequested_construction_slot_is_present_but_never_run():
    report = audit_dataset(
        _evidence(),
        _spec(requested_components=("b_temp",)),
        axis=AXIS,
        diagnostics=(TemplateImbalance(),),
    ).run()

    # Rule 2: whenever any slot is in play, all eight are reported.
    assert set(report.components) == set(CONSTRUCTION_SLOTS)
    assert report.components["b_temp"].status is DiagnosticStatus.READY
    for slot in CONSTRUCTION_SLOTS:
        if slot == "b_temp":
            continue
        result = report.components[slot]
        assert result.status is DiagnosticStatus.NOT_APPLICABLE, slot
        assert result.reason_code == "component_not_requested", slot
        assert result.value is None, slot
    _assert_value_discipline(report)


def test_construction_vector_keeps_the_declared_slot_order():
    report = audit_dataset(
        _evidence(), _spec(), axis=AXIS, diagnostics=_configured()
    ).run()

    assert tuple(
        result.component for result in construction_vector(report)
    ) == CONSTRUCTION_SLOTS
    assert tuple(report.provenance["slot_order"]) == CONSTRUCTION_SLOTS
    assert tuple(report.provenance["backend_slots"]) == BACKEND_CONSTRUCTION_SLOTS
    for slot in BACKEND_CONSTRUCTION_SLOTS:
        result = report.components[slot]
        assert result.status is DiagnosticStatus.BLOCKED, slot
        assert result.value is None, slot
        assert result.reason_code.endswith("_backend_unavailable"), slot


def test_unknown_and_vector_component_names_are_refused_by_name():
    with pytest.raises(ValueError, match="Unknown requested component 'b_novel'"):
        audit_dataset(
            _evidence(), _spec(requested_components=("b_novel",)), axis=AXIS
        )

    with pytest.raises(ValueError, match="b_constr is a component vector"):
        audit_dataset(
            _evidence(), _spec(requested_components=("b_constr",)), axis=AXIS
        )


# --------------------------------------------------------------------------
# Registry membership is inert
# --------------------------------------------------------------------------


def test_registry_membership_alone_never_causes_a_component_to_run():
    # Eleven diagnostics are registered; requesting one runs exactly one.
    assert len(DIAGNOSTIC_REGISTRY) > 1
    report = audit_dataset(
        _evidence(), _spec(requested_components=("b_rep",)), axis=AXIS
    ).run()
    assert set(report.components) == {"b_rep"}


def test_an_empty_registry_does_not_change_what_audit_dataset_runs(monkeypatch):
    baseline = audit_dataset(
        _evidence(), _spec(), axis=AXIS, diagnostics=_configured()
    ).run()

    empty: MappingProxyType = MappingProxyType({})
    monkeypatch.setattr(registry_module, "DIAGNOSTIC_REGISTRY", empty)
    monkeypatch.setattr(diagnostics_package, "DIAGNOSTIC_REGISTRY", empty)

    report = audit_dataset(
        _evidence(), _spec(), axis=AXIS, diagnostics=_configured()
    ).run()

    assert set(report.components) == set(baseline.components)
    assert {name: result.value for name, result in report.components.items()} == {
        name: result.value for name, result in baseline.components.items()
    }
    assert report.to_json() == baseline.to_json()


# --------------------------------------------------------------------------
# Nothing branches on the dataset's identity
# --------------------------------------------------------------------------


def _component_payload(report: DiagnosticReport) -> dict:
    return {name: result.to_dict() for name, result in report.components.items()}


def test_two_specs_differing_only_in_target_name_are_indistinguishable():
    first = audit_dataset(
        _evidence(target_name="unlisted-quarry-probe"),
        _spec(target_name="unlisted-quarry-probe"),
        axis=AXIS,
        diagnostics=_configured(),
    ).run()
    second = audit_dataset(
        _evidence(target_name="winobias"),
        _spec(target_name="winobias"),
        axis=AXIS,
        diagnostics=_configured(),
    ).run()

    assert _component_payload(first) == _component_payload(second)
    assert first.warnings == second.warnings
    assert first.status is second.status
    # The name reaches provenance and nothing else.
    assert first.provenance["target_name"] == "unlisted-quarry-probe"
    assert second.provenance["target_name"] == "winobias"


def test_task_family_does_not_change_any_component_outcome():
    first = audit_dataset(
        _evidence(),
        _spec(task_family="paired_sentence"),
        axis=AXIS,
        diagnostics=_configured(),
    ).run()
    second = audit_dataset(
        _evidence(),
        _spec(task_family="coreference_resolution"),
        axis=AXIS,
        diagnostics=_configured(),
    ).run()

    assert _component_payload(first) == _component_payload(second)


# --------------------------------------------------------------------------
# audit_scores stays a separate, unchanged entry point
# --------------------------------------------------------------------------


def test_audit_scores_remains_reachable_and_unchanged():
    scores = ScoredGroups(
        axis=AXIS,
        groups=("basalt", "basalt", "obsidian", "obsidian"),
        scores=(0.2, 0.4, 0.6, 0.8),
        score_name="toxicity",
        source=SOURCE,
    )
    spec = DatasetAuditSpec(
        target_name=TARGET,
        target_kind="score_table",
        task_family="free_text",
        design_stance="population_proxy",
        requested_components=("score_mean_gap",),
    )

    report = audit_scores(scores, spec, ScorerMeanGap())

    assert set(report.components) == {"score_mean_gap"}
    result = report.components["score_mean_gap"]
    assert result.status is DiagnosticStatus.READY
    assert result.value == pytest.approx(0.4, rel=0, abs=1e-12)
    assert dict(report.provenance) == {"diagnostic": "score_mean_gap"}
    assert report.status is ReportStatus.SUCCESS


def test_a_score_component_requested_from_audit_dataset_points_back_at_audit_scores():
    report = audit_dataset(
        _evidence(),
        _spec(requested_components=("b_rep", "score_mean_gap")),
        axis=AXIS,
    ).run()

    result = report.components["score_mean_gap"]
    assert result.status is DiagnosticStatus.NOT_APPLICABLE
    assert result.reason_code == "component_requires_score_evidence"
    assert result.value is None
    assert "audit_scores" in result.reason
    assert result.details["entry_point"] == "audit_scores"


# --------------------------------------------------------------------------
# Planning, serialization and determinism
# --------------------------------------------------------------------------


def test_plan_is_inspectable_and_agrees_with_the_run():
    audit = audit_dataset(_evidence(), _spec(), axis=AXIS, diagnostics=_configured())

    assert isinstance(audit, DatasetAudit)
    plans = audit.plan()
    assert tuple(plans) == audit.component_order
    assert all(isinstance(plan, ComponentPlan) for plan in plans.values())
    # Planning is pure: a second call yields the same decisions.
    assert {name: plan.to_dict() for name, plan in audit.plan().items()} == {
        name: plan.to_dict() for name, plan in plans.items()
    }

    report = audit.run()
    for name, plan in plans.items():
        result = report.components[name]
        assert isinstance(result, ComponentResult)
        if plan.status is not DiagnosticStatus.READY:
            assert result.status is plan.status, name
            assert result.reason_code == plan.reason_code, name
            assert result.value is None, name


def test_report_serializes_deterministically_and_round_trips_through_json():
    report = audit_dataset(
        _evidence(), _spec(), axis=AXIS, diagnostics=_configured()
    ).run()

    first = report.to_json()
    second = report.to_json()
    assert first == second

    payload = json.loads(first)
    assert payload == report.to_dict()
    assert payload["schema_version"] == DIAGNOSTIC_SCHEMA_VERSION
    assert list(payload["components"]) == sorted(payload["components"])
    assert payload["provenance"]["entry_point"] == "audit_dataset"
    assert payload["provenance"]["axis"] == AXIS

    # Two independently resolved audits over the same declared inputs
    # serialize byte-identically, and one audit re-run is stable.
    audit = audit_dataset(_evidence(), _spec(), axis=AXIS, diagnostics=_configured())
    assert audit.run().to_json() == first
    assert audit.run().to_json() == first

    for name, component in payload["components"].items():
        if component["status"] == "ready":
            assert isinstance(component["value"], float), name
        else:
            assert component["value"] is None, name


def test_resolved_selection_serializes_for_an_audit_trail():
    audit = audit_dataset(_evidence(), _spec(), axis=AXIS, diagnostics=_configured())
    payload = audit.to_dict()

    assert payload["target_name"] == TARGET
    assert payload["axis"] == AXIS
    assert payload["component_order"] == list(audit.component_order)
    assert payload["diagnostics"]["b_diff_len"] == "LengthDisparity"
    assert payload["spec"]["requested_components"] == list(
        audit.spec.requested_components
    )
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


# --------------------------------------------------------------------------
# Registry and public surface
# --------------------------------------------------------------------------


def test_registry_holds_exactly_the_implemented_diagnostics_and_no_stub():
    assert dict(DIAGNOSTIC_REGISTRY.items()) == {
        name: DIAGNOSTIC_REGISTRY[name] for name in _IMPLEMENTED_DIAGNOSTICS
    }
    assert set(DIAGNOSTIC_REGISTRY) == set(_IMPLEMENTED_DIAGNOSTICS)
    assert list_diagnostics() == sorted(_IMPLEMENTED_DIAGNOSTICS)

    for name, class_name in _IMPLEMENTED_DIAGNOSTICS.items():
        registered = DIAGNOSTIC_REGISTRY[name]
        assert registered.__name__ == class_name
        assert issubclass(registered, DatasetDiagnostic)
        assert registered.name == name
        # D032: a registered diagnostic is real, so it constructs with no
        # arguments and is instantiable through the registry.
        instance = get_diagnostic(name)
        assert isinstance(instance, registered)
        assert callable(instance.plan) and callable(instance.compute)

    # The three backend-dependent slots are real classes and are registered;
    # default-constructed they block by name for their missing backend.
    assert set(BACKEND_CONSTRUCTION_SLOTS) <= set(DIAGNOSTIC_REGISTRY)
    for slot in BACKEND_CONSTRUCTION_SLOTS:
        assert getattr(get_diagnostic(slot), "backend", "missing") is None
    dataset_components = {
        name for name in DIAGNOSTIC_REGISTRY if not name.startswith("score_")
    }
    assert dataset_components == {"b_rep", "b_leak", *CONSTRUCTION_SLOTS}


def test_every_public_symbol_named_in_all_actually_imports():
    exported = list(diagnostics_package.__all__)

    assert len(exported) == len(set(exported)), "__all__ contains duplicates"
    assert exported == sorted(exported), "__all__ is not in sorted order"

    namespace: dict = {}
    exec("from fairLMs.diagnostics import *", namespace)  # noqa: S102
    assert [name for name in exported if name not in namespace] == []

    for name in exported:
        assert not name.startswith("_"), name
        assert getattr(diagnostics_package, name) is namespace[name], name

    # D032: nothing unimplemented is advertised on the public surface.
    for unshipped in (
        "FramePredicateLike",
        "OutputAssociation",
        "_resolve_extractor",
    ):
        assert unshipped not in exported


def test_every_registered_diagnostic_class_is_publicly_exported():
    exported = set(diagnostics_package.__all__)
    for class_name in _IMPLEMENTED_DIAGNOSTICS.values():
        assert class_name in exported, class_name


# --------------------------------------------------------------------------
# Report-level warnings
# --------------------------------------------------------------------------


def test_report_warnings_are_deduplicated_and_in_the_declared_order():
    report = audit_dataset(
        _evidence(), _spec(), axis=AXIS, diagnostics=_configured()
    ).run()

    expected_prefixes = (
        "An observed group-trait association",
        "Construction bias is reported as a vector",
        "Component(s) b_equiv, b_gram, b_diff_dep are blocked",
        "b_diff_len measured length in declared surface tokens",
        "The audit has a stress-test design stance",
    )
    assert len(report.warnings) == len(set(report.warnings))
    assert len(report.warnings) == len(expected_prefixes)
    for warning, prefix in zip(report.warnings, expected_prefixes):
        assert warning.startswith(prefix)


def test_a_non_stress_test_stance_drops_only_the_stress_test_warning():
    report = audit_dataset(
        _evidence(),
        _spec(design_stance="population_proxy"),
        axis=AXIS,
        diagnostics=_configured(),
    ).run()

    assert not any(
        warning.startswith("The audit has a stress-test design stance")
        for warning in report.warnings
    )
    assert any(
        warning.startswith("Construction bias is reported as a vector")
        for warning in report.warnings
    )


def test_an_axis_with_no_evidence_yields_no_numbers_at_all():
    report = audit_dataset(
        _evidence(),
        _spec(protected_axes=(AXIS, "cortex-stage")),
        axis="cortex-stage",
        diagnostics=_configured(),
    ).run()

    assert all(result.value is None for result in report.components.values())
    assert report.status is ReportStatus.BLOCKED
    assert report.components["b_leak"].reason_code == "missing_association_evidence"
    for slot in CONSTRUCTION_SLOTS:
        assert report.components[slot].reason_code == "evidence_view_not_supplied"
    assert dict(report.provenance["views_used"]) == {}
    _assert_value_discipline(report)


# --------------------------------------------------------------------------
# The documented precedence for the backend-dependent construction slots
# --------------------------------------------------------------------------
#
# ``_backend_slot_result`` checks the shared applicability precedence and then
# the evidence-view geometry *before* the backend, so a backend slot is BLOCKED
# only when it was requested and its required view is present. Prose that
# promises "always blocked" invites a reader to write
# ``status is BLOCKED and reason_code.endswith("_backend_unavailable")`` as a
# backend probe, which silently matches nothing in the two most common setups.

#: Every place that describes what a backend-dependent slot reports.
_BACKEND_SLOT_DOC_SITES = (
    "README.md",
    "docs/index.md",
    "docs/guides/dataset-audit.md",
    "docs/registry/diagnostics.md",
    "scripts/gen_registry_docs.py",
    "diagnostics/registry.py",
)

#: Sites that must name the authority for "which slots need a backend".
_BACKEND_SLOT_POINTER_SITES = (
    "README.md",
    "docs/guides/dataset-audit.md",
    "docs/registry/diagnostics.md",
    "scripts/gen_registry_docs.py",
    "diagnostics/registry.py",
)

_ALWAYS_BLOCKED = re.compile(r"always[^.]{0,80}blocked", re.IGNORECASE)


def _flattened_prose(relative_path: str) -> str:
    """Return the file with runs of whitespace collapsed, so a claim that is
    split over two source lines is still one searchable sentence."""

    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", text)


def test_no_doc_site_claims_a_backend_slot_is_always_blocked():
    offenders = {}
    for site in _BACKEND_SLOT_DOC_SITES:
        match = _ALWAYS_BLOCKED.search(_flattened_prose(site))
        if match is not None:
            offenders[site] = match.group(0)
    assert offenders == {}, (
        "A backend-dependent construction slot is BLOCKED only when it was "
        "requested and its required evidence view is present; otherwise it is "
        "NOT_APPLICABLE. These sites still promise an invariant: "
        f"{offenders!r}"
    )


def test_every_backend_slot_doc_site_states_the_not_applicable_alternative():
    missing = [
        site
        for site in _BACKEND_SLOT_DOC_SITES
        if "not_applicable" not in _flattened_prose(site)
    ]
    assert missing == [], (
        "Each site that describes the backend slots must also state the "
        f"not-applicable outcome; these do not: {missing!r}"
    )


def test_backend_slot_doc_sites_point_at_the_backend_slot_constants():
    missing = [
        site
        for site in _BACKEND_SLOT_POINTER_SITES
        if "BACKEND_CONSTRUCTION_SLOTS" not in _flattened_prose(site)
    ]
    assert missing == [], (
        "Status alone cannot identify a backend slot, so the docs must point "
        f"at BACKEND_CONSTRUCTION_SLOTS; these do not: {missing!r}"
    )


def test_backend_slot_status_follows_geometry_before_the_backend():
    """The executable form of the sentence the doc sites now carry."""

    not_requested = audit_dataset(
        _evidence(),
        _spec(requested_components=LIGHTWEIGHT_CONSTRUCTION_SLOTS),
        axis=AXIS,
        diagnostics=_configured(),
    ).run()
    for slot in BACKEND_CONSTRUCTION_SLOTS:
        result = not_requested.components[slot]
        assert result.status is DiagnosticStatus.NOT_APPLICABLE
        assert result.reason_code == "component_not_requested"
        assert result.value is None

    view_absent = audit_dataset(
        _thin_evidence(),
        _spec(requested_components=CONSTRUCTION_SLOTS),
        axis=AXIS,
    ).run()
    for slot in BACKEND_CONSTRUCTION_SLOTS:
        result = view_absent.components[slot]
        assert result.status is DiagnosticStatus.NOT_APPLICABLE
        assert result.reason_code == "evidence_view_not_supplied"
        assert result.value is None

    supplied = audit_dataset(
        _evidence(),
        _spec(requested_components=CONSTRUCTION_SLOTS),
        axis=AXIS,
        diagnostics=_configured(),
    ).run()
    for slot in BACKEND_CONSTRUCTION_SLOTS:
        result = supplied.components[slot]
        assert result.status is DiagnosticStatus.BLOCKED
        assert result.reason_code.endswith("_backend_unavailable")
        assert result.value is None
