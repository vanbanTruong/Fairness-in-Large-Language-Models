"""The declaration mechanism: capabilities, profiles, and named refusal.

The manuscript's central claim is that every component declares the model
capabilities and evidence it needs, runs against whatever satisfies them, and is
*refused by name* otherwise. These tests pin that claim down: refusal happens,
it happens before any compute, and the message names the specific missing thing
rather than saying "not applicable".
"""

import re

import pytest
import torch

from fairLMs.definitions.core.applicability import (
    ACCESS_LEVELS,
    ARCHITECTURES,
    CAPABILITIES,
    TASK_PROFILES,
    AccessLevel,
    ApplicabilityError,
    ModelProfile,
    access_rank,
    check_applicability,
    describe_model,
    validate_declaration,
)
from fairLMs.definitions import METRIC_REGISTRY, SEAT, WordSets
from fairLMs.definitions.models.base import LoadedModel
from fairLMs.definitions.models.huggingface import HuggingFaceModel
from fairLMs.definitions.models.openai import OpenAIModel


def _loaded(task):
    return LoadedModel(
        name="stub",
        tokenizer=object(),
        model=object(),
        device=torch.device("cpu"),
        task=task,
    )


class TestVocabulary:
    def test_every_capability_is_lowercase_snake_case(self):
        # These strings are public API: a refusal message names them.
        for capability in CAPABILITIES:
            assert capability == capability.lower()
            assert " " not in capability

    def test_access_levels_are_ordered_black_gray_white(self):
        assert [level.value for level in ACCESS_LEVELS] == [
            "black_box",
            "gray_box",
            "white_box",
        ]
        assert access_rank("black_box") < access_rank("gray_box")
        assert access_rank("gray_box") < access_rank("white_box")

    def test_spelling_is_gray_not_grey(self):
        assert AccessLevel.GRAY_BOX.value == "gray_box"
        assert not any("grey" in level.value for level in ACCESS_LEVELS)

    def test_every_task_profile_declares_a_known_architecture(self):
        for task, profile in TASK_PROFILES.items():
            assert profile.architecture in ARCHITECTURES, task
            assert profile.capabilities <= CAPABILITIES, task

    def test_an_unknown_capability_is_refused_not_absorbed(self):
        with pytest.raises(ValueError, match="unknown capabilities"):
            ModelProfile(
                architecture="encoder_only",
                capabilities=frozenset({"telepathy"}),
                access=AccessLevel.WHITE_BOX,
            )


class TestModelProfiles:
    @pytest.mark.parametrize(
        "task,architecture",
        [
            ("mlm", "encoder_only"),
            ("encoder", "encoder_only"),
            ("sequence_classification", "encoder_only"),
            ("seq2seq", "encoder_decoder"),
            ("causal", "decoder_only"),
            ("openai", "decoder_only"),
        ],
    )
    def test_architecture_is_derived_from_the_existing_task_tag(
        self, task, architecture
    ):
        assert describe_model(_loaded(task)).architecture == architecture

    def test_an_api_decoder_exposes_no_activations(self):
        profile = describe_model(OpenAIModel())
        assert "hidden_states" not in profile.capabilities
        assert "attentions" not in profile.capabilities
        assert "gradients" not in profile.capabilities
        assert profile.access is AccessLevel.BLACK_BOX

    def test_a_local_checkpoint_grants_white_box_access(self):
        profile = describe_model(HuggingFaceModel("bert-base-uncased", task="mlm"))
        assert profile.access is AccessLevel.WHITE_BOX
        assert "gradients" in profile.capabilities

    def test_an_unknown_deployment_is_not_second_guessed(self):
        # The (tokenizer, model) escape hatch carries no task. Refusing on
        # missing information would break it.
        assert describe_model(None) is None
        assert describe_model((object(), object())) is None
        assert describe_model(_loaded(None)) is None

    def test_an_adapter_may_declare_its_own_profile(self):
        declared = ModelProfile(
            architecture="decoder_only",
            capabilities=frozenset({"free_generation"}),
            access=AccessLevel.BLACK_BOX,
        )

        class Custom:
            profile = declared

        assert describe_model(Custom()) is declared


class _Component:
    """A minimal declaring component, standing in for a metric or mitigator."""

    def __init__(self, **declarations):
        self.requires = declarations.get("requires", frozenset())
        self.accepts = declarations.get("accepts", ())
        self.architectures = declarations.get("architectures", ())
        self.access = declarations.get("access")


class TestRefusalNamesTheMissingItem:
    def test_a_missing_capability_is_named(self):
        component = _Component(requires=frozenset({"hidden_states"}))
        with pytest.raises(ApplicabilityError) as exc:
            check_applicability(component, OpenAIModel())
        message = str(exc.value)
        assert "hidden_states" in message
        assert "free_generation" in message  # what it does provide

    def test_a_wrong_architecture_is_named(self):
        component = _Component(architectures=("encoder_only",))
        with pytest.raises(ApplicabilityError) as exc:
            check_applicability(component, _loaded("causal"))
        assert "encoder_only" in str(exc.value)
        assert "decoder_only" in str(exc.value)

    def test_insufficient_access_is_named(self):
        component = _Component(access=AccessLevel.WHITE_BOX)
        with pytest.raises(ApplicabilityError) as exc:
            check_applicability(component, OpenAIModel())
        message = str(exc.value)
        assert "white_box" in message and "black_box" in message

    def test_a_wrong_evidence_container_is_named(self):
        component = _Component(accepts=(WordSets,))
        with pytest.raises(ApplicabilityError) as exc:
            check_applicability(component, None, evidence=[1, 2, 3])
        assert "WordSets" in str(exc.value)
        assert "list" in str(exc.value)

    def test_a_required_capability_with_no_model_is_refused(self):
        component = _Component(requires=frozenset({"hidden_states"}))
        with pytest.raises(ApplicabilityError, match="no model was given"):
            check_applicability(component, None)

    def test_refusal_is_a_type_error_so_existing_guards_keep_working(self):
        component = _Component(requires=frozenset({"hidden_states"}))
        with pytest.raises(TypeError):
            check_applicability(component, OpenAIModel())


class TestEmptyDeclarationsNeverRefuse:
    """Safe defaults: the mechanism can be added without changing behaviour."""

    def test_a_component_declaring_nothing_accepts_anything(self):
        component = _Component()
        assert check_applicability(component, OpenAIModel(), evidence=[1]) is not None
        assert check_applicability(component, None, evidence=object()) is None

    def test_an_unknown_model_is_never_refused(self):
        component = _Component(
            requires=frozenset({"gradients"}),
            architectures=("encoder_only",),
            access=AccessLevel.WHITE_BOX,
        )
        # A raw tuple carries no task; silence is required to keep the escape
        # hatch that the rest of the suite depends on.
        assert check_applicability(component, (object(), object())) is None

    def test_evidence_is_not_checked_when_none_is_supplied(self):
        # Metrics resolve their data separately and may legitimately pass None
        # through this path.
        component = _Component(accepts=(WordSets,))
        assert check_applicability(component, None, evidence=None) is None


class TestDeclarationValidation:
    def test_a_typo_in_a_capability_is_caught(self):
        with pytest.raises(ValueError, match="unknown capabilities"):
            validate_declaration(_Component(requires=frozenset({"hiden_states"})))

    def test_a_typo_in_an_architecture_is_caught(self):
        with pytest.raises(ValueError, match="unknown families"):
            validate_declaration(_Component(architectures=("encoder-only",)))

    def test_a_bare_type_in_accepts_is_caught(self):
        with pytest.raises(TypeError, match="not a bare type"):
            validate_declaration(_Component(accepts=WordSets))


class TestMetricRetrofit:
    """The paper's claim covers the metrics too, not only the mitigators."""

    @pytest.mark.parametrize("name,cls", sorted(METRIC_REGISTRY.items()))
    def test_a_declared_capability_is_reachable_on_some_deployment(self, name, cls):
        # A metric requiring a combination no profile can satisfy would be
        # permanently unusable, which the declaration should not permit.
        assert any(
            cls.requires <= profile.capabilities
            and profile.architecture in cls.architectures
            for profile in TASK_PROFILES.values()
        ), f"{name} declares a combination no deployment satisfies"

    def test_the_motivating_refusal_from_the_paper(self):
        # "A user pointing OpenAIModel at SEAT should be told SEAT requires
        # hidden_states before spending GPU time."
        with pytest.raises(TypeError) as exc:
            SEAT().compute(
                OpenAIModel(), WordSets(["man"], ["woman"], ["career"], ["family"])
            )
        message = str(exc.value)
        assert "hidden_states" in message
        assert "free_generation" in message

    @pytest.mark.parametrize(
        "adapter_cls,kwargs,expected",
        [
            # An API decoder cannot produce activations at all: the capability
            # check speaks, naming the quantity.
            (OpenAIModel, {}, "hidden_states"),
            # A local checkpoint with the wrong head could produce them; the
            # task check speaks instead, because "reload with task=encoder" is
            # the actionable fix and "no hidden_states" would be misleading.
            (
                HuggingFaceModel,
                {"model_name": "bert-base-uncased", "task": "causal"},
                "task='encoder'",
            ),
        ],
    )
    def test_refusal_lands_before_the_model_is_loaded(
        self, adapter_cls, kwargs, expected
    ):
        # load() downloads weights or opens a client. A refusal that arrives
        # after that has already cost the user the thing it promised to save.
        class Exploding(adapter_cls):
            def load(self):
                raise AssertionError("load() must not be reached")

        with pytest.raises(TypeError, match=re.escape(expected)):
            SEAT().compute(
                Exploding(**kwargs),
                WordSets(["man"], ["woman"], ["career"], ["family"]),
            )


# --- regression: tokenizer-dependent metrics are refused by name over an API --
#
# Four metrics declared only `free_generation` or `token_logprobs`, both of
# which the openai profile grants, yet their implementations call
# resolve.get_tokenizer_model and then tokenize themselves. They were admitted
# by the matcher and died later on a bare
# `AttributeError: 'OpenAILoadedModel' object has no attribute 'tokenizer'`,
# which is precisely the deep failure the declaration contract exists to
# replace.


@pytest.mark.parametrize(
    "metric_name",
    [
        "cooccurrence_association",
        "demographic_next_token_proportion",
        "demographic_representation_divergence",
        "stereotypical_log_likelihood",
    ],
)
def test_tokenizer_dependent_metrics_are_refused_over_an_api(metric_name):
    from fairLMs.definitions.core.applicability import ApplicabilityError, check_applicability
    from fairLMs.definitions import METRIC_REGISTRY
    from fairLMs.definitions.models.openai import OpenAILoadedModel

    served = OpenAILoadedModel(name="gpt-x", client=object(), model="gpt-x")

    with pytest.raises(ApplicabilityError) as excinfo:
        check_applicability(METRIC_REGISTRY[metric_name](), served)
    assert "local_tokenizer" in str(excinfo.value)


def test_local_profiles_all_grant_a_local_tokenizer():
    """Declaring local_tokenizer must not refuse any locally loaded model."""
    from fairLMs.definitions.core.applicability import TASK_PROFILES

    for task, profile in TASK_PROFILES.items():
        if task == "openai":
            assert "local_tokenizer" not in profile.capabilities
        else:
            assert "local_tokenizer" in profile.capabilities, task


def test_every_metric_that_resolves_a_tokenizer_declares_it():
    """The vocabulary describes the code: if a metric calls
    get_tokenizer_model and requires a specific head, it declares the need."""
    import inspect

    from fairLMs.definitions import METRIC_REGISTRY

    for name, cls in sorted(METRIC_REGISTRY.items()):
        module = inspect.getmodule(cls)
        try:
            source = inspect.getsource(module)
        except OSError:  # pragma: no cover
            continue
        if "get_tokenizer_model" not in source:
            continue
        if getattr(cls, "required_task", None) is None:
            continue  # model-optional: runs on precomputed outputs too
        assert "local_tokenizer" in (getattr(cls, "requires", ()) or ()), name
