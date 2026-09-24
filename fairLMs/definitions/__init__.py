"""Public sklearn-style metric API.

Every metric exposes ``compute(model, dataset=None, **kwargs) -> MetricResult``.

Modules are grouped by metric family (e.g. ``similarity_based`` for WEAT/SEAT/CEAT).
"""

from fairLMs.definitions.algorithmic_disparity import (
    LexicalFrequencyProportion,
    MorphologicalChoiceDivergence,
)
from fairLMs.definitions.attention_head import (
    GradientBasedBiasEstimation,
    NaturalIndirectEffect,
)
from fairLMs.definitions.base import FairnessMetric, MetricResult
from fairLMs.definitions.data import (
    ConceptSpec,
    ContextSets,
    ContrastSpec,
    DemographicPrompts,
    GroupPredictions,
    GroupProperties,
    GroupWordPairs,
    LabeledSentences,
    OccupationTriples,
    ProbeSet,
    PromptPairs,
    QuerySpec,
    ScorePair,
    SentenceTriples,
    StereotypeLabelled,
    VectorSets,
    WordSets,
)
from fairLMs.definitions.counterfactual_fairness import (
    CounterfactualFairnessScore,
    CounterfactualRobustness,
)
from fairLMs.definitions.demographic_representation import (
    DemographicNextTokenProportion,
    DemographicRepresentationDivergence,
)
from fairLMs.definitions.encoder_decoder_extrinsic import (
    CounterfactualAucScore,
    InferenceBiasScore,
    NormalizedPositionDistance,
    TranslationSimilarityScore,
)
from fairLMs.definitions.encoder_decoder_stereotypical import (
    StereotypicalDivergence,
    StereotypicalValueAttribution,
)
from fairLMs.definitions.encoder_extrinsic import (
    ContextBasedDisparityScore,
    EqualOpportunityGap,
    FairInferenceScore,
)
from fairLMs.definitions.masked_token import (
    ContrastBasedScore,
    DiscoveryOfCorrelationsScore,
    LogProbabilityBiasScore,
)
from fairLMs.definitions.performance_disparity import (
    AccuracyDisparity,
    BiasAmplifierScore,
    SensitiveNameSimilarity,
)
from fairLMs.definitions.pseudo_log_likelihood import (
    AllUnmaskedLikelihoodAttentionScore,
    AllUnmaskedLikelihoodScore,
    ContextAssociationTestScore,
    CrowSPairsScore,
    PseudoLogLikelihoodScore,
)
from fairLMs.definitions.similarity_based import CEAT, SEAT, WEAT
from fairLMs.definitions.stereotypical_association import (
    CooccurrenceAssociation,
    StereotypicalLogLikelihood,
)
from fairLMs.definitions import core, models, resources
from fairLMs.definitions.core import (
    AccessLevel,
    ApplicabilityError,
    ModelProfile,
    check_applicability,
)
from fairLMs.definitions.models import (
    HuggingFaceModel,
    LoadedModel,
    ModelAdapter,
    OpenAILoadedModel,
    OpenAIModel,
    get_openai_client,
    load_causal_lm,
    load_encoder,
    load_masked_lm,
    load_seq2seq,
    load_sequence_classifier,
    resolve_device,
)
from fairLMs.definitions.resources import (
    WORD_SETS,
    WORD_SET_LABELS,
    get_word_set,
    list_word_sets,
    seat_c1,
    seat_c2,
    seat_c3,
    seat_c4,
    weat_c1,
    weat_c2,
    weat_c3,
    weat_c4,
)
from fairLMs.definitions import functional
from fairLMs.definitions.functional import (
    accuracy_disparity,
    context_based_disparity,
    equal_opportunity_gap,
    fair_inference_score,
    inference_bias_score,
)

# Short aliases used in papers / book chapters
CPS = CrowSPairsScore
LPBS = LogProbabilityBiasScore
CAT = ContextAssociationTestScore
DisCo = DiscoveryOfCorrelationsScore
CBS = ContrastBasedScore
PLL = PseudoLogLikelihoodScore
AUL = AllUnmaskedLikelihoodScore
AULA = AllUnmaskedLikelihoodAttentionScore

METRIC_REGISTRY = {
    CrowSPairsScore.name: CrowSPairsScore,
    LogProbabilityBiasScore.name: LogProbabilityBiasScore,
    WEAT.name: WEAT,
    SEAT.name: SEAT,
    CEAT.name: CEAT,
    PseudoLogLikelihoodScore.name: PseudoLogLikelihoodScore,
    AllUnmaskedLikelihoodScore.name: AllUnmaskedLikelihoodScore,
    AllUnmaskedLikelihoodAttentionScore.name: AllUnmaskedLikelihoodAttentionScore,
    ContextAssociationTestScore.name: ContextAssociationTestScore,
    DiscoveryOfCorrelationsScore.name: DiscoveryOfCorrelationsScore,
    ContrastBasedScore.name: ContrastBasedScore,
    FairInferenceScore.name: FairInferenceScore,
    EqualOpportunityGap.name: EqualOpportunityGap,
    ContextBasedDisparityScore.name: ContextBasedDisparityScore,
    CounterfactualAucScore.name: CounterfactualAucScore,
    InferenceBiasScore.name: InferenceBiasScore,
    TranslationSimilarityScore.name: TranslationSimilarityScore,
    NormalizedPositionDistance.name: NormalizedPositionDistance,
    LexicalFrequencyProportion.name: LexicalFrequencyProportion,
    MorphologicalChoiceDivergence.name: MorphologicalChoiceDivergence,
    StereotypicalDivergence.name: StereotypicalDivergence,
    StereotypicalValueAttribution.name: StereotypicalValueAttribution,
    CounterfactualRobustness.name: CounterfactualRobustness,
    CounterfactualFairnessScore.name: CounterfactualFairnessScore,
    DemographicNextTokenProportion.name: DemographicNextTokenProportion,
    DemographicRepresentationDivergence.name: DemographicRepresentationDivergence,
    AccuracyDisparity.name: AccuracyDisparity,
    BiasAmplifierScore.name: BiasAmplifierScore,
    SensitiveNameSimilarity.name: SensitiveNameSimilarity,
    GradientBasedBiasEstimation.name: GradientBasedBiasEstimation,
    NaturalIndirectEffect.name: NaturalIndirectEffect,
    CooccurrenceAssociation.name: CooccurrenceAssociation,
    StereotypicalLogLikelihood.name: StereotypicalLogLikelihood,
}


def list_metrics():
    """Return sorted public metric names."""
    return sorted(METRIC_REGISTRY)


def get_metric(name: str) -> FairnessMetric:
    """Instantiate a metric by registry name."""
    try:
        cls = METRIC_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown metric {name!r}. Available: {', '.join(list_metrics())}"
        ) from exc
    return cls()


__all__ = [
    "AUL",
    "AULA",
    "AccuracyDisparity",
    "AllUnmaskedLikelihoodAttentionScore",
    "AllUnmaskedLikelihoodScore",
    "BiasAmplifierScore",
    "CAT",
    "CBS",
    "CEAT",
    "CPS",
    "CooccurrenceAssociation",
    "ContextAssociationTestScore",
    "ContextBasedDisparityScore",
    "ContextSets",
    "ContrastBasedScore",
    "CounterfactualAucScore",
    "CounterfactualFairnessScore",
    "CounterfactualRobustness",
    "CrowSPairsScore",
    "DemographicNextTokenProportion",
    "DemographicRepresentationDivergence",
    "DisCo",
    "DiscoveryOfCorrelationsScore",
    "EqualOpportunityGap",
    "FairInferenceScore",
    "FairnessMetric",
    "GradientBasedBiasEstimation",
    "InferenceBiasScore",
    "LPBS",
    "LexicalFrequencyProportion",
    "LogProbabilityBiasScore",
    "METRIC_REGISTRY",
    "MetricResult",
    "MorphologicalChoiceDivergence",
    "NaturalIndirectEffect",
    "NormalizedPositionDistance",
    "PLL",
    "PseudoLogLikelihoodScore",
    "SEAT",
    "SensitiveNameSimilarity",
    "StereotypicalDivergence",
    "StereotypicalLogLikelihood",
    "StereotypicalValueAttribution",
    "TranslationSimilarityScore",
    "VectorSets",
    "WEAT",
    "WordSets",
    # Data containers
    "ConceptSpec",
    "ContrastSpec",
    "DemographicPrompts",
    "GroupPredictions",
    "GroupProperties",
    "GroupWordPairs",
    "LabeledSentences",
    "OccupationTriples",
    "ProbeSet",
    "PromptPairs",
    "QuerySpec",
    "ScorePair",
    "SentenceTriples",
    "StereotypeLabelled",
    # sklearn.metrics-style functions (model-free metrics)
    "accuracy_disparity",
    "context_based_disparity",
    "equal_opportunity_gap",
    "fair_inference_score",
    "functional",
    "inference_bias_score",
    "get_metric",
    "list_metrics",
    # Shared contracts, model adapters, and bundled definition evidence
    "AccessLevel",
    "ApplicabilityError",
    "ModelProfile",
    "check_applicability",
    "core",
    "models",
    "resources",
    "HuggingFaceModel",
    "LoadedModel",
    "ModelAdapter",
    "OpenAILoadedModel",
    "OpenAIModel",
    "get_openai_client",
    "load_causal_lm",
    "load_encoder",
    "load_masked_lm",
    "load_seq2seq",
    "load_sequence_classifier",
    "resolve_device",
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
