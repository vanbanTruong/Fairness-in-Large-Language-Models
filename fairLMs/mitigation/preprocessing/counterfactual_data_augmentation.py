"""Counterfactual data augmentation: swap sensitive surface forms via a lexicon."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from fairLMs.mitigation.base import MitigationResult, Mitigator
from fairLMs.mitigation.evidence import CorpusWithLexicon, TextRecords
from fairLMs.mitigation.preprocessing._shared import _ALL_ARCHITECTURES

__all__ = ["CounterfactualDataAugmentation"]

#: Word-boundary tokenizer used for surface-form swaps. Deliberately simple and
#: declared: a mitigator must not pull in spaCy to split a string.
_WORD = re.compile(r"\b\w+\b", re.UNICODE)


def _match_case(source: str, replacement: str) -> str:
    """Carry *source*'s casing onto *replacement*."""
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _swap(text: str, table: Dict[str, str]) -> Tuple[str, int]:
    """Return the swapped text and how many terms were replaced."""
    count = 0

    def replace(match: "re.Match") -> str:
        nonlocal count
        word = match.group(0)
        target = table.get(word.lower())
        if target is None:
            return word
        count += 1
        return _match_case(word, target)

    return _WORD.sub(replace, text), count


class CounterfactualDataAugmentation(Mitigator):
    """Swap sensitive surface forms via an explicit lexicon; emit original + swapped.

    This is counterfactual data **augmentation**: the swapped copy is added
    alongside the original. Counterfactual data *substitution*, which replaces
    rather than adds, is deliberately not provided - see the out-of-scope list
    in the package docstring.

    A record containing no lexicon term cannot be rewritten. That is refused,
    not dropped: silently emitting only the augmentable subset would skew the
    corpus toward exactly the rows that mention the protected attribute.

    Parameters
    ----------
    on_unrewritable:
        ``"refuse"`` (default) raises on the first record carrying no lexicon
        term. ``"keep"`` passes it through unswapped, which the caller must ask
        for explicitly and which is recorded in provenance.

    Examples
    --------
    >>> from fairLMs.mitigation import (
    ...     CorpusWithLexicon, CounterfactualDataAugmentation, SwapLexicon,
    ...     TextRecords)
    >>> evidence = CorpusWithLexicon(
    ...     records=TextRecords(texts=["He is a nurse."], source="doctest"),
    ...     lexicon=SwapLexicon(
    ...         axis="gender", pairs=[("he", "she")], source="doctest"),
    ... )
    >>> result = CounterfactualDataAugmentation().apply(None, evidence)
    >>> tuple(result.result.texts)
    ('He is a nurse.', 'She is a nurse.')
    """

    name = "counterfactual_data_augmentation"
    category = "pre"
    access = "black_box"
    architectures = _ALL_ARCHITECTURES
    requires = frozenset()
    accepts = (CorpusWithLexicon,)

    def __init__(self, on_unrewritable: str = "refuse"):
        self.on_unrewritable = on_unrewritable

    def _apply(self, model: Any, evidence: Any) -> MitigationResult:
        records, lexicon = evidence.records, evidence.lexicon
        if self.on_unrewritable not in ("refuse", "keep"):
            raise ValueError(
                "on_unrewritable must be 'refuse' or 'keep'; got "
                f"{self.on_unrewritable!r}."
            )
        table = lexicon.mapping()

        texts: List[str] = []
        ids: List[str] = []
        n_swapped = 0
        n_untouched = 0
        for index, text in enumerate(records.texts):
            row_id = records.ids[index] if records.ids else str(index)
            swapped, count = _swap(text, table)
            texts.append(text)
            ids.append(f"{row_id}:original")
            if count == 0:
                n_untouched += 1
                if self.on_unrewritable == "refuse":
                    raise ValueError(
                        f"{self.name}: record {row_id!r} contains no term from the "
                        f"{lexicon.axis!r} lexicon, so it cannot be rewritten: "
                        f"{text!r}. Extend the lexicon, remove the record "
                        f"deliberately, or pass on_unrewritable='keep'. It will "
                        f"not be dropped silently."
                    )
                texts.append(text)
                ids.append(f"{row_id}:kept")
            else:
                n_swapped += 1
                texts.append(swapped)
                ids.append(f"{row_id}:swapped")

        augmented = TextRecords(
            texts=texts,
            source=f"{records.source}+cda",
            ids=ids,
            provenance={
                "transform": self.name,
                "axis": lexicon.axis,
                "lexicon_source": lexicon.source,
            },
        )
        return self._result(
            augmented,
            axis=lexicon.axis,
            lexicon_source=lexicon.source,
            lexicon_pairs=[list(p) for p in lexicon.pairs],
            n_input_rows=records.n_rows,
            n_output_rows=augmented.n_rows,
            n_swapped=n_swapped,
            n_unrewritable=n_untouched,
            on_unrewritable=self.on_unrewritable,
        )
