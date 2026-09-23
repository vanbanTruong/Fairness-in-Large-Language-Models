"""Acceptance tests for the multi-evidence ``audit_dataset`` entry point."""

from __future__ import annotations

import json
from types import MappingProxyType

import pytest

from fairLMs.diagnostics import (
    BACKEND_CONSTRUCTION_SLOTS,
    CONSTRUCTION_SLOTS,
    ComponentOverride,
    ComponentPlan,
    DatasetAudit,
    DatasetAuditSpec,
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
    PairedScores,
    PairedTexts,
    ReferenceDistribution,
    RepresentationEvidence,
    ReportStatus,
    SELF_IDENTIFICATION_FRAME,
    ScoredGroups,
    ScorerMeanGap,
    StereotypeLeakage,
    SurfaceCooccurrenceExtractor,
    TemplateGroups,
    TemplateImbalance,
    TextEvidence,
    audit_dataset,
    audit_scores,
    construction_vector,
)

AXIS = "chromatic-family"
SOURCE = "Unregistered synthetic extract v7"

# Deliberately unfamiliar field names: the package must never read semantics
# out of a column name, a loader identity, or a dataset name.
_ROWS = (
    {"cohort": "amber", "utterance": "the amber one tends to soothe", "arc": "a1"},
    {"cohort": "teal", "utterance": "the teal one tends to soothe", "arc": "a1"},
    {"cohort": "amber", "utterance": "I am amber and I direct the room", "arc": "a2"},
    {"cohort": "teal", "utterance": "I am teal and I direct", "arc": "a2"},
)

_PAIR_ROWS = (
    {"knot": "k1", "variant": "primed", "utterance": "the amber one soothes"},
    {"knot": "k1", "variant": "counter", "utterance": "the teal one soothes"},
    {"knot": "k2", "variant": "primed", "utterance": "the amber one directs"},
    {"knot": "k2", "variant": "counter", "utterance": "the teal one directs"},
)

_OPTION_ROWS = (
    {"slot": "q1", "stance": "primed", "wording": "the soothing one"},
    {"slot": "q1", "stance": "counter", "wording": "the directing one indeed"},
    {"slot": "q2", "stance": "primed", "wording": "amber"},
    {"slot": "q2", "stance": "counter", "wording": "teal for certain"},
)


def _extraction() -> LeakageExtractionConfig:
    return LeakageExtractionConfig(
        group_lexicon=("amber", "teal"),
        trait_lexicon=("soothe", "direct"),
    )


def _evidence(**overrides) -> DatasetEvidence:
    views = {
        "target_name": "unregistered-synthetic-benchmark",
        "representation": {
            AXIS: RepresentationEvidence(
                axis=AXIS, counts={"amber": 2, "teal": 2}, source=SOURCE
            )
        },
        "texts": {
            AXIS: TextEvidence.from_records(
                _ROWS, axis=AXIS, text_field="utterance", source=SOURCE
            )
        },
        "grouped_texts": {
            AXIS: GroupedTexts.from_records(
                _ROWS,
                axis=AXIS,
                group_field="cohort",
                text_field="utterance",
                declared_groups=("amber", "teal"),
                source=SOURCE,
            )
        },
        "paired_texts": {
            AXIS: PairedTexts.from_records(
                _PAIR_ROWS,
                axis=AXIS,
                pair_id_field="knot",
                condition_field="variant",
                text_field="utterance",
                condition_roles=("primed", "counter"),
                pairing_basis="Rows sharing knot differ only in the cohort token.",
                source=SOURCE,
            )
        },
        "option_items": {
            AXIS: OptionItems.from_records(
                _OPTION_ROWS,
                axis=AXIS,
                item_id_field="slot",
                role_field="stance",
                option_field="wording",
                declared_roles=("primed", "counter"),
                question_family="multiple_choice_stereotype_contrast",
                source=SOURCE,
            )
        },
        "template_groups": {
            AXIS: TemplateGroups.from_records(
                _ROWS,
                axis=AXIS,
                group_field="cohort",
                template_id_field="arc",
                declared_groups=("amber", "teal"),
                template_identity_rule="Exact arc string as authored upstream.",
                source=SOURCE,
            )
        },
    }
    views.update(overrides)
    return DatasetEvidence(**views)


def _spec(
    *,
    requested_components=("b_rep", "b_leak", *CONSTRUCTION_SLOTS),
    target_kind="benchmark_dataset",
    design_stance="stress_test",
    protected_axes=(AXIS,),
    leakage_extraction=True,
    component_overrides=None,
    references=True,
) -> DatasetAuditSpec:
    return DatasetAuditSpec(
        target_name="unregistered-synthetic-benchmark",
        target_kind=target_kind,
        task_family="paired_sentence",
        design_stance=design_stance,
        protected_axes=protected_axes,
        references=(
            {
                AXIS: ReferenceDistribution(
                    axis=AXIS,
                    probabilities={"amber": 0.5, "teal": 0.5},
                    source="Design target stated in the dataset card.",
                    purpose="design_target",
                    population="Intended balanced counterfactual design",
                )
            }
            if references
            else {}
        ),
        requested_components=requested_components,
        leakage_extraction=_extraction() if leakage_extraction else None,
        component_overrides=component_overrides or {},
    )


def _configured_diagnostics():
    return (
        MinimalPairResidual(
            identity_mask=IdentityMaskConfig(identity_terms=("amber", "teal"))
        ),
        LengthDisparity(),
        OptionLengthBias(
            role_contrast=OptionRoleContrast(
                stereotype_role="primed", anti_stereotype_role="counter"
            )
        ),
        FramingDisparity(predicate=SELF_IDENTIFICATION_FRAME),
        TemplateImbalance(),
    )


def test_unfamiliar_dataset_plans_and_runs_every_requested_component():
    audit = audit_dataset(
        _evidence(), _spec(), axis=AXIS, diagnostics=_configured_diagnostics()
    )

    assert isinstance(audit, DatasetAudit)
    plan = audit.plan()
    assert list(plan) == [
        "b_rep",
        "b_leak",
        *CONSTRUCTION_SLOTS,
    ]
    assert all(isinstance(item, ComponentPlan) for item in plan.values())

    report = audit.run()
    assert isinstance(report, DiagnosticReport)
    assert set(report.components) == set(plan)
    assert report.status is ReportStatus.PARTIAL

    ready = {
        name
        for name, result in report.components.items()
        if result.status is DiagnosticStatus.READY
    }
    assert {"b_rep", "b_leak", "b_min", "b_diff_len", "b_frame", "b_opt", "b_temp"} <= ready

    # Nothing that is not ready may carry a number.
    for result in report.components.values():
        if result.status is not DiagnosticStatus.READY:
            assert result.value is None
            assert result.reason_code and result.reason

    provenance = report.to_dict()["provenance"]
    assert provenance["entry_point"] == "audit_dataset"
    assert provenance["axis"] == AXIS
    assert provenance["slot_order"] == list(CONSTRUCTION_SLOTS)
    assert provenance["backend_slots"] == list(BACKEND_CONSTRUCTION_SLOTS)
    assert provenance["views_used"]["b_leak"] == "texts"
    assert provenance["views_used"]["b_min"] == "paired_texts"

    # Strict, deterministic JSON with no non-standard numbers.
    encoded = report.to_json(indent=None)
    assert encoded == report.to_json(indent=None)
    json.loads(encoded)


def test_component_selection_ignores_the_registry_and_the_dataset_name(monkeypatch):
    import fairLMs.diagnostics.registry as registry_module

    monkeypatch.setattr(
        registry_module, "DIAGNOSTIC_REGISTRY", MappingProxyType({})
    )

    spec = _spec(requested_components=("b_rep", "b_leak"))
    report = audit_dataset(_evidence(), spec, axis=AXIS).run()

    assert set(report.components) == {"b_rep", "b_leak"}
    assert report.components["b_rep"].status is DiagnosticStatus.READY
    assert report.components["b_leak"].status is DiagnosticStatus.READY


def test_registry_membership_never_auto_runs_an_unrequested_component():
    spec = _spec(requested_components=("b_rep",))
    report = audit_dataset(_evidence(), spec, axis=AXIS).run()

    # ``b_leak`` and the construction slots are all registered, and all of
    # their evidence is present, but none of them was requested.
    assert set(report.components) == {"b_rep"}


def test_requesting_one_construction_slot_yields_the_full_eight_slot_vector():
    spec = _spec(requested_components=("b_temp",))
    report = audit_dataset(
        _evidence(), spec, axis=AXIS, diagnostics=(TemplateImbalance(),)
    ).run()

    assert set(report.components) == set(CONSTRUCTION_SLOTS)
    vector = construction_vector(report)
    assert [result.component for result in vector] == list(CONSTRUCTION_SLOTS)
    assert report.components["b_temp"].status is DiagnosticStatus.READY
    for slot in CONSTRUCTION_SLOTS:
        if slot == "b_temp":
            continue
        result = report.components[slot]
        assert result.value is None
        # Precedence step 1 outranks the backend check, so an unrequested
        # backend slot never advertises a missing dependency.
        assert result.status is DiagnosticStatus.NOT_APPLICABLE
        assert result.reason_code == "component_not_requested"

    # When every slot is requested the three backend slots block on their
    # own missing backend and nothing else changes.
    full = audit_dataset(
        _evidence(),
        _spec(requested_components=CONSTRUCTION_SLOTS),
        axis=AXIS,
        diagnostics=_configured_diagnostics(),
    ).run()
    for slot in BACKEND_CONSTRUCTION_SLOTS:
        result = full.components[slot]
        assert result.status is DiagnosticStatus.BLOCKED
        assert result.value is None
        assert result.reason_code.endswith("_backend_unavailable")


def test_score_component_names_are_routed_to_audit_scores():
    spec = _spec(requested_components=("b_rep", "score_mean_gap"))
    report = audit_dataset(_evidence(), spec, axis=AXIS).run()

    result = report.components["score_mean_gap"]
    assert result.status is DiagnosticStatus.NOT_APPLICABLE
    assert result.reason_code == "component_requires_score_evidence"
    assert result.value is None
    assert "audit_scores" in result.reason


def test_unknown_requested_component_lists_the_available_names():
    spec = _spec(requested_components=("b_rep", "b_novel"))
    with pytest.raises(ValueError) as excinfo:
        audit_dataset(_evidence(), spec, axis=AXIS)

    message = str(excinfo.value)
    assert "b_novel" in message
    assert "b_rep" in message
    assert "b_leak" in message


def test_b_constr_is_named_as_a_vector_rather_than_a_component():
    spec = _spec(requested_components=("b_constr",))
    with pytest.raises(ValueError) as excinfo:
        audit_dataset(_evidence(), spec, axis=AXIS)

    assert "b_constr is a component vector, not a component" in str(excinfo.value)


def test_count_matrix_wins_over_text_and_the_unused_view_is_recorded():
    config = _extraction()
    text = TextEvidence.from_records(
        _ROWS, axis=AXIS, text_field="utterance", source=SOURCE
    )
    counts = SurfaceCooccurrenceExtractor(config=config).extract(text)
    evidence = _evidence(association_counts={AXIS: counts})

    spec = _spec(requested_components=("b_leak",))
    report = audit_dataset(evidence, spec, axis=AXIS).run()

    result = report.components["b_leak"]
    assert result.status is DiagnosticStatus.READY
    assert result.details["evidence_path"] == "association_counts"
    provenance = report.to_dict()["provenance"]
    assert provenance["views_used"]["b_leak"] == "association_counts"
    assert "texts" in provenance["unused_evidence_views"]


def test_absent_evidence_is_reported_rather_than_skipped_or_zeroed():
    evidence = DatasetEvidence(
        target_name="sparse-extract",
        representation={
            AXIS: RepresentationEvidence(
                axis=AXIS, counts={"amber": 2, "teal": 2}, source=SOURCE
            )
        },
    )
    spec = _spec(
        requested_components=("b_rep", "b_leak", "b_min", "b_temp"),
        leakage_extraction=False,
    )
    report = audit_dataset(evidence, spec, axis=AXIS).run()

    assert set(report.components) == {"b_rep", "b_leak", *CONSTRUCTION_SLOTS}
    assert report.components["b_rep"].status is DiagnosticStatus.READY

    leakage = report.components["b_leak"]
    assert leakage.status is DiagnosticStatus.BLOCKED
    assert leakage.reason_code == "missing_association_evidence"
    assert leakage.value is None

    for slot in ("b_min", "b_temp"):
        result = report.components[slot]
        assert result.status is DiagnosticStatus.NOT_APPLICABLE
        assert result.reason_code == "evidence_view_not_supplied"
        assert result.value is None


def test_an_override_can_only_suppress_and_never_manufacture_a_value():
    override = ComponentOverride(
        component="b_rep",
        status="not_applicable",
        reason="This extract has no declared comparison population.",
        declared_reason_code="no_declared_population",
    )
    spec = _spec(
        requested_components=("b_rep",),
        component_overrides={"b_rep": override},
    )
    report = audit_dataset(_evidence(), spec, axis=AXIS).run()

    result = report.components["b_rep"]
    assert result.status is DiagnosticStatus.NOT_APPLICABLE
    assert result.reason_code == "applicability_override"
    assert result.value is None
    assert result.details["applicability_override"] is True
    assert result.details["declared_reason_code"] == "no_declared_population"
    # The full override map is snapshotted into the report's spec.
    assert report.to_dict()["spec"]["component_overrides"]["b_rep"]["reason"]


def test_an_undeclared_axis_blocks_rather_than_silently_measuring():
    spec = _spec(
        requested_components=("b_rep",),
        protected_axes=("some-other-axis",),
        references=False,
    )
    report = audit_dataset(_evidence(), spec, axis=AXIS).run()

    result = report.components["b_rep"]
    assert result.status is DiagnosticStatus.BLOCKED
    assert result.reason_code == "axis_not_declared"
    assert result.value is None


def test_a_non_benchmark_target_kind_is_not_applicable_everywhere():
    spec = _spec(
        requested_components=("b_rep", "b_leak"),
        target_kind="generated_output",
        references=False,
        protected_axes=(),
    )
    report = audit_dataset(_evidence(), spec, axis=AXIS).run()

    for name in ("b_rep", "b_leak"):
        result = report.components[name]
        assert result.status is DiagnosticStatus.NOT_APPLICABLE
        assert result.reason_code == "target_kind_not_supported"
        assert result.value is None
    assert "output association" in report.components["b_leak"].reason


def test_supplied_diagnostics_are_validated_against_the_request():
    evidence = _evidence()

    with pytest.raises(TypeError, match="must be a DatasetDiagnostic"):
        audit_dataset(evidence, _spec(), axis=AXIS, diagnostics=("b_min",))

    spec = _spec(requested_components=("b_rep",))
    with pytest.raises(ValueError, match="not in spec.requested_components"):
        audit_dataset(evidence, spec, axis=AXIS, diagnostics=(TemplateImbalance(),))

    with pytest.raises(ValueError, match="duplicate component names"):
        audit_dataset(
            evidence,
            _spec(requested_components=("b_temp",)),
            axis=AXIS,
            diagnostics=(TemplateImbalance(), TemplateImbalance()),
        )

    with pytest.raises(ValueError, match="audit_scores"):
        audit_dataset(
            evidence,
            _spec(requested_components=("score_mean_gap",)),
            axis=AXIS,
            diagnostics=(ScorerMeanGap(),),
        )


def test_row_level_scorer_evidence_is_refused_by_the_dataset_container():
    scored = ScoredGroups(
        axis=AXIS,
        groups=("amber", "teal"),
        scores=(0.1, 0.2),
        source=SOURCE,
        score_name="toxicity",
    )
    with pytest.raises(TypeError, match="audit_scores"):
        DatasetEvidence(target_name="mixed", representation={AXIS: scored})


def test_audit_scores_remains_a_separate_entry_point():
    scored = ScoredGroups(
        axis=AXIS,
        groups=("amber", "amber", "teal", "teal"),
        scores=(0.1, 0.3, 0.2, 0.6),
        source=SOURCE,
        score_name="toxicity",
    )
    spec = DatasetAuditSpec(
        target_name="unregistered-synthetic-benchmark",
        target_kind="score_table",
        task_family="free_text",
        design_stance="population_proxy",
        requested_components=("score_mean_gap",),
    )
    report = audit_scores(scored, spec)

    assert set(report.components) == {"score_mean_gap"}
    assert report.components["score_mean_gap"].status is DiagnosticStatus.READY
    assert isinstance(scored, ScoredGroups)
    assert PairedScores is not None


def test_audit_requires_an_explicit_axis_and_typed_inputs():
    evidence = _evidence()
    spec = _spec()

    with pytest.raises(TypeError, match="must be a DatasetEvidence"):
        audit_dataset({}, spec, axis=AXIS)
    with pytest.raises(TypeError, match="must be a DatasetAuditSpec"):
        audit_dataset(evidence, object(), axis=AXIS)
    with pytest.raises(ValueError, match="axis must be a non-empty string"):
        audit_dataset(evidence, spec, axis="   ")


def test_resolved_selection_is_serializable_for_an_audit_trail():
    audit = audit_dataset(
        _evidence(),
        _spec(requested_components=("b_rep", "b_leak")),
        axis=AXIS,
    )
    payload = audit.to_dict()

    assert set(payload) == {
        "target_name",
        "axis",
        "component_order",
        "diagnostics",
        "evidence_views",
        "views_used",
        "spec",
    }
    assert payload["component_order"] == ["b_rep", "b_leak"]
    assert payload["diagnostics"]["b_leak"] == "StereotypeLeakage"
    json.dumps(payload, allow_nan=False)


def test_a_default_constructed_component_blocks_for_its_own_configuration():
    spec = _spec(requested_components=("b_min",))
    report = audit_dataset(_evidence(), spec, axis=AXIS).run()

    result = report.components["b_min"]
    assert result.status is DiagnosticStatus.BLOCKED
    assert result.reason_code == "missing_identity_mask"
    assert result.value is None


def test_a_conflicting_extraction_configuration_is_refused():
    other = LeakageExtractionConfig(
        group_lexicon=("amber", "teal"),
        trait_lexicon=("soothe", "direct"),
        window=3,
    )
    spec = _spec(requested_components=("b_leak",))
    with pytest.raises(ValueError, match="does not match spec.leakage_extraction"):
        audit_dataset(
            _evidence(),
            spec,
            axis=AXIS,
            diagnostics=(
                StereotypeLeakage(
                    extractor=SurfaceCooccurrenceExtractor(config=other)
                ),
            ),
        )
