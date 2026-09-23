"""Identity-term augmentation: add benign examples carrying the same terms."""

from __future__ import annotations

from typing import Any

from fairLMs.mitigation.base import MitigationResult, Mitigator
from fairLMs.mitigation.evidence import CorpusWithBenignPool, GroupLabeledRecords
from fairLMs.mitigation.preprocessing._shared import _ALL_ARCHITECTURES

__all__ = ["IdentityTermAugmentation"]


class IdentityTermAugmentation(Mitigator):
    """Add benign identity-term examples to break the term-to-label shortcut.

    A classifier trained on a corpus where an identity term appears mostly in
    toxic examples learns the term as the signal. Adding benign examples that
    carry the same terms breaks that correlation at the data level.

    The benign pool is supplied by the caller and labelled with the declared
    ``benign_label``: which outcome counts as benign is a property of the task,
    not something to infer.

    Parameters
    ----------
    benign_label:
        The outcome label attached to every added example.

    Examples
    --------
    >>> from fairLMs.mitigation import (
    ...     CorpusWithBenignPool, GroupLabeledRecords, IdentityTermAugmentation,
    ...     TextRecords)
    >>> corpus = GroupLabeledRecords(
    ...     axis="religion",
    ...     groups=["muslim", "muslim", "christian"],
    ...     labels=["toxic", "toxic", "clean"],
    ...     label_name="toxicity", source="doctest",
    ...     texts=["a", "b", "c"],
    ... )
    >>> pool = TextRecords(
    ...     texts=["My muslim neighbour bakes bread."], source="doctest")
    >>> out = IdentityTermAugmentation(benign_label="clean").apply(
    ...     None, CorpusWithBenignPool(corpus=corpus, pool=pool))
    >>> out.result.n_rows
    4
    """

    name = "identity_term_augmentation"
    category = "pre"
    access = "black_box"
    architectures = _ALL_ARCHITECTURES
    requires = frozenset()
    accepts = (CorpusWithBenignPool,)

    def __init__(self, benign_label: str = "", benign_group: str = ""):
        self.benign_label = benign_label
        self.benign_group = benign_group

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        corpus, pool = evidence.corpus, evidence.pool
        if not self.benign_label:
            raise ValueError(
                f"{self.name}: benign_label must be declared. Which outcome counts "
                f"as benign is a property of the task; observed labels are "
                f"{sorted(set(corpus.labels))!r}."
            )
        if self.benign_label not in set(corpus.labels):
            raise ValueError(
                f"{self.name}: benign_label {self.benign_label!r} does not occur in "
                f"the corpus; observed labels are {sorted(set(corpus.labels))!r}."
            )
        group = self.benign_group or corpus.groups[0]
        if group not in set(corpus.groups):
            raise ValueError(
                f"{self.name}: benign_group {group!r} does not occur in the corpus; "
                f"observed groups are {sorted(set(corpus.groups))!r}."
            )

        augmented = GroupLabeledRecords(
            axis=corpus.axis,
            groups=list(corpus.groups) + [group] * pool.n_rows,
            labels=list(corpus.labels) + [self.benign_label] * pool.n_rows,
            label_name=corpus.label_name,
            source=f"{corpus.source}+identity_terms",
            texts=list(corpus.texts) + list(pool.texts),
        )
        return self._result(
            augmented,
            axis=corpus.axis,
            benign_label=self.benign_label,
            benign_group=group,
            pool_source=pool.source,
            n_input_rows=corpus.n_rows,
            n_added=pool.n_rows,
            n_output_rows=augmented.n_rows,
        )
