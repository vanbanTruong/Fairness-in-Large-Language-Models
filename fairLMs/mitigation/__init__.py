"""Bias mitigation: fourteen components across four intervention categories.

Mitigators adopt the same contract as metrics. Each declares an intervention
category (pre-, in-, intra- or post-processing), an access level, supported
architectures, capabilities and evidence containers, and applicability is
decided from those declarations by the one shared matcher in
:mod:`fairLMs.applicability` - exactly as it is for metrics. An unsatisfiable
pairing is **refused by name**, never given a proxy.

.. code-block:: python

    from fairLMs.mitigation import MITIGATOR_REGISTRY

    mitigator = MITIGATOR_REGISTRY["iterative_nullspace_projection"](n_iterations=8)
    outcome = mitigator.apply(model, evidence)   # -> MitigationResult

By category: **pre** 4, **in** 4, **intra** 3, **post** 3.

======================================  =========  ============
Component                               Category   Access
======================================  =========  ============
``counterfactual_data_augmentation``    pre        black_box
``group_label_reweighting``             pre        black_box
``identity_term_augmentation``          pre        black_box
``debiasing_prompt``                    pre        black_box
``adversarial_debiasing``               in         white_box
``counterfactual_invariance_loss``      in         white_box
``group_regularized_objective``         in         white_box
``influence_guided_suppression``        in         white_box
``subspace_projection``                 intra      gray_box
``iterative_nullspace_projection``      intra      gray_box
``self_debiasing``                      intra      gray_box
``score_calibration``                   post       black_box
``group_aware_thresholding``            post       black_box
``output_reranking``                    post       black_box
======================================  =========  ============

**Out of scope, deliberately.** Named in the paper as extension points and
**not** provided: RLHF, DPO, Constitutional AI and other preference-optimized
training loops; UniDetox; the influence-estimation half of IF-Guide; GeDi, RAD,
DExperts, FairSteer, ARGRE and other guided-decoding methods; LSDM; FairMed;
gender-constrained beam search; model-based rewriting. Named in the book but not
provided here: conceptor debiasing; pruning and ablation; the no-attribute
(name-proxy) setting; Co2PT; BiasUnlearn; data filtering and toxicity removal;
dataset curation and balanced collection; output filtering and safety
classification; refusal policies and content moderation; counterfactual data
**substitution**. The book's "bias diagnosis and data auditing" family is
covered, but by :mod:`fairLMs.diagnostics`.

No mandatory heavy dependency: spaCy, sentence-transformers, LanguageTool and
Java stay optional. The base install runs all fourteen components.
"""

from fairLMs.applicability import (
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
from fairLMs.diagnostics.evidence import LabeledScoredGroups

from .base import (
    CATEGORIES,
    MITIGATION_SCHEMA_VERSION,
    MitigationResult,
    Mitigator,
)
from .evidence import (
    AttributeLabeledVectors,
    CandidateSets,
    CorpusWithBenignPool,
    CorpusWithLexicon,
    GroupLabeledRecords,
    InfluenceScoredCorpus,
    PromptSpec,
    SwapLexicon,
    TextRecords,
)
from .harness import (
    ComparisonReport,
    MetricDelta,
    MetricEvaluation,
    compare_before_after,
)
from .inprocessing import (
    AdversarialDebiasing,
    CounterfactualInvarianceLoss,
    GroupRegularizedObjective,
    InfluenceGuidedSuppression,
)
from .intraprocessing import (
    IterativeNullspaceProjection,
    ProjectedModelAdapter,
    SelfDebiasedModelAdapter,
    SelfDebiasing,
    SubspaceProjection,
)
from .postprocessing import (
    GroupAwareThresholding,
    OutputReranking,
    ScoreCalibration,
)
from .preprocessing import (
    CounterfactualDataAugmentation,
    DebiasingPrompt,
    GroupLabelReweighting,
    IdentityTermAugmentation,
)
from .registry import (
    MITIGATOR_REGISTRY,
    get_mitigator,
    list_by_category,
    list_mitigators,
)

__all__ = [
    # Applicability mechanism
    "ACCESS_LEVELS",
    "ARCHITECTURES",
    "CAPABILITIES",
    "TASK_PROFILES",
    "AccessLevel",
    "ApplicabilityError",
    "ModelProfile",
    "access_rank",
    "check_applicability",
    "describe_model",
    "validate_declaration",
    # Contracts
    "CATEGORIES",
    "MITIGATION_SCHEMA_VERSION",
    "MITIGATOR_REGISTRY",
    "MitigationResult",
    "Mitigator",
    "get_mitigator",
    "list_by_category",
    "list_mitigators",
    # Evidence
    "AttributeLabeledVectors",
    "CandidateSets",
    "CorpusWithBenignPool",
    "CorpusWithLexicon",
    "GroupLabeledRecords",
    "InfluenceScoredCorpus",
    "LabeledScoredGroups",
    "PromptSpec",
    "SwapLexicon",
    "TextRecords",
    # Pre-processing
    "CounterfactualDataAugmentation",
    "DebiasingPrompt",
    "GroupLabelReweighting",
    "IdentityTermAugmentation",
    # In-processing
    "AdversarialDebiasing",
    "CounterfactualInvarianceLoss",
    "GroupRegularizedObjective",
    "InfluenceGuidedSuppression",
    # Intra-processing
    "IterativeNullspaceProjection",
    "ProjectedModelAdapter",
    "SelfDebiasedModelAdapter",
    "SelfDebiasing",
    "SubspaceProjection",
    # Post-processing
    "GroupAwareThresholding",
    "OutputReranking",
    "ScoreCalibration",
    # Before/after harness
    "ComparisonReport",
    "MetricDelta",
    "MetricEvaluation",
    "compare_before_after",
]
