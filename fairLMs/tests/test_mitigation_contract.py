"""Conformance suite: contract checks every mitigator in the registry satisfies.

The mitigation analogue of ``tests/test_common.py``. A new mitigator that
forgets the parameter protocol, the declaration protocol, or the payload
guarantee of its category fails here rather than surprising a user.
"""

import inspect

import pytest

from fairLMs.applicability import (
    ACCESS_LEVELS,
    CAPABILITIES,
    AccessLevel,
    TASK_PROFILES,
    access_rank,
    validate_declaration,
)
from fairLMs.mitigation import (
    CATEGORIES,
    MITIGATOR_REGISTRY,
    MitigationResult,
    Mitigator,
    get_mitigator,
    list_by_category,
    list_mitigators,
)

ALL = list_mitigators()

#: The fourteen names, copied from the manuscript's Bias Mitigation section and
#: Table 1. If this list and the registry disagree, one of them is wrong.
MANUSCRIPT_NAMES = {
    "counterfactual_data_augmentation",
    "group_label_reweighting",
    "identity_term_augmentation",
    "debiasing_prompt",
    "adversarial_debiasing",
    "counterfactual_invariance_loss",
    "group_regularized_objective",
    "influence_guided_suppression",
    "subspace_projection",
    "iterative_nullspace_projection",
    "self_debiasing",
    "score_calibration",
    "group_aware_thresholding",
    "output_reranking",
}

#: Table 1's footer lists exactly these as architecture-agnostic.
ARCHITECTURE_AGNOSTIC = {
    "counterfactual_data_augmentation",
    "group_label_reweighting",
    "identity_term_augmentation",
    "score_calibration",
    "group_aware_thresholding",
}

#: Named in the paper or book as extension points, and deliberately not shipped.
OUT_OF_SCOPE = {
    "rlhf",
    "dpo",
    "constitutional_ai",
    "unidetox",
    "gedi",
    "rad",
    "dexperts",
    "fairsteer",
    "argre",
    "lsdm",
    "fairmed",
    "conceptor_debiasing",
    "gender_constrained_beam_search",
    "model_based_rewriting",
    "co2pt",
    "biasunlearn",
    "counterfactual_data_substitution",
    "pruning",
    "ablation",
    "data_filtering",
    "output_filtering",
    "refusal_policies",
}


def _satisfying_model(mitigator):
    """Return a model that satisfies *mitigator*'s declarations, or ``None``.

    Lets a test exercise one check in isolation: without this, a mitigator that
    needs a model is refused on the capability check before its evidence is ever
    looked at, and the test would be asserting the wrong refusal.
    """
    if not mitigator.requires and not mitigator.architectures:
        return None
    for task, profile in TASK_PROFILES.items():
        if (
            mitigator.requires <= profile.capabilities
            and profile.architecture in mitigator.architectures
            and access_rank(profile.access) >= access_rank(mitigator.access)
        ):
            return profile
    raise AssertionError(f"no deployment satisfies {mitigator.name}")


class TestRegistry:
    def test_registry_holds_exactly_the_fourteen_named_components(self):
        assert set(ALL) == MANUSCRIPT_NAMES
        assert len(ALL) == 14

    def test_the_category_split_is_four_four_three_three(self):
        assert len(list_by_category("pre")) == 4
        assert len(list_by_category("in")) == 4
        assert len(list_by_category("intra")) == 3
        assert len(list_by_category("post")) == 3

    def test_no_out_of_scope_method_is_advertised_as_available(self):
        import fairLMs.mitigation as package

        assert not (set(ALL) & OUT_OF_SCOPE)
        exported = {name.lower() for name in package.__all__}
        assert not (exported & OUT_OF_SCOPE)

    @pytest.mark.parametrize("name", ALL)
    def test_registry_name_matches_the_class_attribute(self, name):
        assert MITIGATOR_REGISTRY[name].name == name

    @pytest.mark.parametrize("name", ALL)
    def test_public_import_from_the_package_root(self, name):
        import fairLMs.mitigation as package

        cls = MITIGATOR_REGISTRY[name]
        assert getattr(package, cls.__name__) is cls
        assert cls.__name__ in package.__all__

    def test_unknown_name_lists_alternatives(self):
        with pytest.raises(KeyError) as exc:
            get_mitigator("not_a_mitigator")
        assert "score_calibration" in str(exc.value)


class TestParameterProtocol:
    """Same sklearn conventions as the metrics; config only in ``__init__``."""

    @pytest.mark.parametrize("name", ALL)
    def test_instantiable_with_no_arguments(self, name):
        assert isinstance(get_mitigator(name), Mitigator)

    @pytest.mark.parametrize("name", ALL)
    def test_init_stores_params_verbatim(self, name):
        cls = MITIGATOR_REGISTRY[name]
        mitigator = cls()
        for param in cls._param_names():
            assert hasattr(mitigator, param), (
                f"{name}: __init__ parameter {param!r} is not stored as "
                f"self.{param}, so get_params() cannot see it"
            )

    @pytest.mark.parametrize("name", ALL)
    def test_get_params_roundtrips(self, name):
        mitigator = get_mitigator(name)
        params = mitigator.get_params(deep=False)
        assert type(mitigator)(**params).get_params(deep=False) == params

    @pytest.mark.parametrize("name", ALL)
    def test_works_with_sklearn_clone(self, name):
        sklearn_base = pytest.importorskip("sklearn.base")
        mitigator = get_mitigator(name)
        cloned = sklearn_base.clone(mitigator)
        assert type(cloned) is type(mitigator)
        assert cloned is not mitigator

    @pytest.mark.parametrize("name", ALL)
    def test_set_params_rejects_unknown(self, name):
        with pytest.raises(ValueError, match="Invalid parameter"):
            get_mitigator(name).set_params(definitely_not_real=1)

    @pytest.mark.parametrize("name", ALL)
    def test_no_data_argument_in_init(self, name):
        # Data goes to apply(), never to __init__. Catches a mitigator that
        # takes its evidence at construction time.
        forbidden = {"data", "evidence", "records", "scores", "model", "corpus"}
        assert not (set(MITIGATOR_REGISTRY[name]._param_names()) & forbidden)

    @pytest.mark.parametrize("name", ALL)
    def test_init_does_not_validate(self, name):
        # sklearn forbids validation in __init__: a nonsense value must be
        # stored and only rejected when apply() runs.
        cls = MITIGATOR_REGISTRY[name]
        params = cls()._param_names()
        if not params:
            pytest.skip(f"{name} has no configuration parameters")
        cls(**{params[0]: "NONSENSE_SENTINEL"})  # must not raise


class TestDeclarationProtocol:
    @pytest.mark.parametrize("name", ALL)
    def test_all_five_attributes_are_declared_on_the_class(self, name):
        cls = MITIGATOR_REGISTRY[name]
        for attribute in ("category", "access", "architectures", "requires", "accepts"):
            assert attribute in vars(cls), f"{name} does not declare {attribute}"

    @pytest.mark.parametrize("name", ALL)
    def test_declarations_are_in_the_vocabulary(self, name):
        validate_declaration(MITIGATOR_REGISTRY[name])

    @pytest.mark.parametrize("name", ALL)
    def test_category_is_known(self, name):
        assert MITIGATOR_REGISTRY[name].category in CATEGORIES

    @pytest.mark.parametrize("name", ALL)
    def test_access_is_known_and_spelled_gray(self, name):
        access = AccessLevel(MITIGATOR_REGISTRY[name].access)
        assert access in ACCESS_LEVELS
        assert "grey" not in access.value

    @pytest.mark.parametrize("name", ALL)
    def test_accepts_is_never_empty(self, name):
        assert MITIGATOR_REGISTRY[name].accepts, f"{name} declares no containers"

    @pytest.mark.parametrize("name", ALL)
    def test_requires_is_a_subset_of_the_vocabulary(self, name):
        assert MITIGATOR_REGISTRY[name].requires <= CAPABILITIES

    @pytest.mark.parametrize("name", sorted(ARCHITECTURE_AGNOSTIC))
    def test_table_1_footer_methods_are_architecture_agnostic(self, name):
        assert set(MITIGATOR_REGISTRY[name].architectures) == {
            "encoder_only",
            "decoder_only",
            "encoder_decoder",
        }

    @pytest.mark.parametrize("name", ALL)
    def test_a_declared_combination_is_satisfiable_by_some_deployment(self, name):
        cls = MITIGATOR_REGISTRY[name]
        assert any(
            cls.requires <= profile.capabilities
            and profile.architecture in cls.architectures
            and access_rank(profile.access) >= access_rank(cls.access)
            for profile in TASK_PROFILES.values()
        ), f"{name} declares a combination no deployment satisfies"

    def test_access_level_matches_the_categorys_definition(self):
        # The paper fixes these: pre/post are black-box, in is white-box,
        # intra is gray-box.
        expected = {
            "pre": AccessLevel.BLACK_BOX,
            "post": AccessLevel.BLACK_BOX,
            "in": AccessLevel.WHITE_BOX,
            "intra": AccessLevel.GRAY_BOX,
        }
        for name, cls in MITIGATOR_REGISTRY.items():
            assert AccessLevel(cls.access) is expected[cls.category], name


class TestApplyContract:
    @pytest.mark.parametrize("name", ALL)
    def test_apply_has_the_uniform_signature(self, name):
        params = [
            p.name
            for p in inspect.signature(
                MITIGATOR_REGISTRY[name].apply
            ).parameters.values()
            if p.name != "self"
        ]
        assert params == ["model", "evidence"], f"{name}.apply{params}"

    @pytest.mark.parametrize("name", ALL)
    def test_wrong_evidence_is_refused_by_name(self, name):
        mitigator = get_mitigator(name)
        with pytest.raises(TypeError) as exc:
            mitigator.apply(_satisfying_model(mitigator), object())
        # The refusal must name what was accepted, not just say "invalid".
        assert any(
            cls.__name__ in str(exc.value) for cls in mitigator.accepts
        ), f"{name} refusal does not name its accepted containers: {exc.value}"

    @pytest.mark.parametrize("name", ALL)
    def test_documented_with_a_docstring(self, name):
        cls = MITIGATOR_REGISTRY[name]
        assert cls.__doc__ and cls.__doc__.strip(), f"{name} has no docstring"

    @pytest.mark.parametrize("name", ALL)
    def test_has_a_runnable_example(self, name):
        cls = MITIGATOR_REGISTRY[name]
        # Every component carries an example somewhere in its own docs or in
        # the doctest of the helper it delegates to.
        assert ">>>" in (cls.__doc__ or "") or ">>>" in (
            inspect.getmodule(cls).__doc__ or ""
        ), f"{name} has no runnable example"


class TestMitigationResult:
    def test_is_not_float_convertible(self):
        # Unlike MetricResult this is not a scalar, and inviting it to be
        # averaged into a summary number would be a category error.
        result = MitigationResult(
            mitigator="score_calibration", category="post", result={"a": 1}
        )
        with pytest.raises(TypeError):
            float(result)

    def test_an_intra_result_must_be_a_model_adapter(self):
        with pytest.raises(TypeError, match="must be a ModelAdapter"):
            MitigationResult(
                mitigator="subspace_projection",
                category="intra",
                result={"not": "an adapter"},
            )

    def test_an_in_result_must_be_callable(self):
        with pytest.raises(TypeError, match="must be a callable"):
            MitigationResult(
                mitigator="adversarial_debiasing", category="in", result={"nope": 1}
            )

    def test_an_unknown_category_is_refused(self):
        with pytest.raises(ValueError, match="category must be one of"):
            MitigationResult(mitigator="x", category="during", result=1)

    def test_a_none_payload_is_refused_rather_than_returned_empty(self):
        with pytest.raises(ValueError, match="must not be None"):
            MitigationResult(mitigator="x", category="post", result=None)

    def test_serializes_and_roundtrips(self):
        import json

        result = MitigationResult(
            mitigator="score_calibration",
            category="post",
            result={"method": "platt"},
            provenance={"seed": 0, "fairLMs_version": "0.1"},
        )
        restored = json.loads(result.to_json())
        assert restored["result"] == {"method": "platt"}
        assert restored["result_serializable"] is True
        assert restored["provenance"]["seed"] == 0

    def test_a_live_payload_is_recorded_as_a_stub_not_stringified(self):
        # A report containing "<function _loss at 0x7f...>" would look like data
        # and be worthless. Record the type and say it is not serializable.
        result = MitigationResult(
            mitigator="adversarial_debiasing", category="in", result=lambda *a: 0.0
        )
        payload = result.to_dict()
        assert payload["result"] is None
        assert payload["result_serializable"] is False
        assert payload["result_type"] == "function"
        assert "0x" not in result.to_json()


class TestProvenance:
    @pytest.mark.parametrize("name", ALL)
    def test_provenance_records_config_and_version(self, name):
        # Built through the shared helper, so checking one instance per class
        # verifies the wiring without needing valid evidence for all fourteen.
        mitigator = get_mitigator(name)
        provenance = mitigator._provenance()
        assert provenance["mitigator"] == name
        assert provenance["category"] == mitigator.category
        assert "fairLMs_version" in provenance
        assert set(provenance["config"]) == set(mitigator.get_params())

    def test_a_callable_config_value_is_recorded_without_leaking_the_object(self):
        from fairLMs.mitigation import SubspaceProjection

        provenance = SubspaceProjection(encode=lambda m, t: t)._provenance()
        assert provenance["config"]["encode"] == "<function>"
