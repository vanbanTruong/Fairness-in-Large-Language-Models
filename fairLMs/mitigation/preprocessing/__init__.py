"""Pre-processing mitigators: black-box, applied to the data before training.

These transform evidence, not models. Each records the lexicon, filter or target
distribution it used in provenance, and each **refuses** a row it cannot handle
rather than dropping it: a corpus that silently shrank during augmentation would
change the very distribution the transform claims to be correcting.

One mitigator per module:

======================================  ==========================================
``counterfactual_data_augmentation``    :mod:`~.counterfactual_data_augmentation`
``group_label_reweighting``             :mod:`~.group_label_reweighting`
``identity_term_augmentation``          :mod:`~.identity_term_augmentation`
``debiasing_prompt``                    :mod:`~.debiasing_prompt`
======================================  ==========================================
"""

from __future__ import annotations

from ._shared import _ALL_ARCHITECTURES
from .counterfactual_data_augmentation import (
    CounterfactualDataAugmentation,
    _match_case,
    _swap,
    _WORD,
)
from .debiasing_prompt import DebiasingPrompt
from .group_label_reweighting import GroupLabelReweighting
from .identity_term_augmentation import IdentityTermAugmentation

__all__ = [
    "CounterfactualDataAugmentation",
    "DebiasingPrompt",
    "GroupLabelReweighting",
    "IdentityTermAugmentation",
]
