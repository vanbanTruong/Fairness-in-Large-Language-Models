"""The new evidence containers: validate at construction, never coerce.

Every container here refuses malformed input rather than dropping, coercing or
silently repairing it. These tests are mostly refusals, which is the point.
"""

import pytest

from fairLMs.datasets.diagnostics import LabeledScoredGroups, ScoredGroups
from fairLMs.mitigation import (
    AttributeLabeledVectors,
    CandidateSets,
    GroupLabeledRecords,
    InfluenceScoredCorpus,
    PromptSpec,
    SwapLexicon,
    TextRecords,
)


def _scored(**overrides):
    kwargs = dict(
        axis="gender",
        groups=["f", "f", "m", "m"],
        scores=[0.9, 0.2, 0.8, 0.1],
        score_name="p_hire",
        source="unit-test",
        score_range=[0.0, 1.0],
    )
    kwargs.update(overrides)
    return ScoredGroups(**kwargs)


def _labeled(**overrides):
    kwargs = dict(
        scored=_scored(),
        labels=["yes", "no", "yes", "no"],
        label_name="outcome",
        positive_label="yes",
    )
    kwargs.update(overrides)
    return LabeledScoredGroups(**kwargs)


class TestLabeledScoredGroups:
    def test_composition_inherits_the_scored_groups_validation(self):
        # Not re-implemented and not relaxed: the composed type still refuses.
        with pytest.raises(ValueError, match="same number of rows"):
            _labeled(scored=_scored(groups=["f", "m"], scores=[0.1]))
        with pytest.raises(ValueError, match="at least two observed categories"):
            _labeled(scored=_scored(groups=["f", "f", "f", "f"]))
        with pytest.raises(ValueError, match="outside the declared score_range"):
            _labeled(scored=_scored(scores=[9.0, 0.2, 0.8, 0.1]))

    def test_labels_must_align_with_scores(self):
        with pytest.raises(ValueError, match="same number of rows"):
            _labeled(labels=["yes", "no"])

    def test_positive_label_must_occur_in_labels(self):
        with pytest.raises(ValueError, match="does not occur in labels"):
            _labeled(positive_label="maybe")

    def test_at_least_two_distinct_labels_are_required(self):
        with pytest.raises(ValueError, match="at least two distinct outcomes"):
            _labeled(labels=["yes"] * 4)

    def test_positive_label_is_mandatory_and_never_inferred(self):
        # Equal opportunity is defined for the positive class, so which label
        # is positive is part of the question, not a property of the data.
        with pytest.raises(TypeError):
            LabeledScoredGroups(
                scored=_scored(),
                labels=["yes", "no", "yes", "no"],
                label_name="outcome",
            )

    def test_scored_groups_itself_was_not_widened(self):
        # The standing decision on the diagnostics side: labels and pair
        # geometry do not widen ScoredGroups.
        assert not hasattr(_scored(), "labels")
        with pytest.raises(TypeError):
            ScoredGroups(
                axis="gender",
                groups=["f", "m"],
                scores=[0.1, 0.2],
                score_name="s",
                source="t",
                labels=["a", "b"],
            )

    def test_views_and_iteration_are_deterministic(self):
        evidence = _labeled()
        assert evidence.n_rows == 4
        assert evidence.support == ("f", "m")
        assert evidence.label_support == ("no", "yes")
        assert evidence.positives() == (True, False, True, False)
        assert [g for g, _, _ in evidence.iter_groups()] == ["f", "m"]

    def test_is_frozen(self):
        with pytest.raises(Exception):
            _labeled().label_name = "other"

    def test_roundtrips_to_json_safe_data(self):
        import json

        payload = _labeled().to_dict()
        assert json.loads(json.dumps(payload))["positive_label"] == "yes"


class TestSwapLexicon:
    def test_a_self_mapping_pair_is_refused(self):
        with pytest.raises(ValueError, match="maps .* to itself"):
            SwapLexicon(axis="gender", pairs=[("he", "He")], source="t")

    def test_a_reused_term_is_refused_as_ambiguous(self):
        with pytest.raises(ValueError, match="reuses the term"):
            SwapLexicon(
                axis="gender", pairs=[("he", "she"), ("he", "they")], source="t"
            )

    def test_mapping_is_bidirectional_and_case_insensitive(self):
        table = SwapLexicon(axis="gender", pairs=[("he", "she")], source="t").mapping()
        assert table["he"] == "she" and table["she"] == "he"

    def test_an_empty_lexicon_is_refused(self):
        with pytest.raises(ValueError, match="must not be empty"):
            SwapLexicon(axis="gender", pairs=[], source="t")


class TestPromptSpec:
    def test_a_template_without_a_query_placeholder_is_refused(self):
        with pytest.raises(ValueError, match=r"\{query\}"):
            PromptSpec(templates=["Be fair."])

    def test_render_substitutes_every_template(self):
        spec = PromptSpec(templates=["A: {query}", "B: {query}"])
        assert spec.render("x") == ("A: x", "B: x")


class TestInfluenceScoredCorpus:
    def test_a_flagged_index_outside_the_corpus_is_refused(self):
        with pytest.raises(ValueError, match="outside the corpus range"):
            InfluenceScoredCorpus(
                n_examples=2, flagged=[5], influence=[1.0], source="t"
            )

    def test_scores_must_align_with_flagged_indices(self):
        with pytest.raises(ValueError, match="one score per flagged example"):
            InfluenceScoredCorpus(
                n_examples=4, flagged=[0, 1], influence=[1.0], source="t"
            )

    def test_duplicate_flagged_indices_are_refused(self):
        with pytest.raises(ValueError, match="duplicate indices"):
            InfluenceScoredCorpus(
                n_examples=4, flagged=[1, 1], influence=[1.0, 2.0], source="t"
            )

    def test_a_non_finite_score_is_refused(self):
        with pytest.raises(ValueError, match="must be finite"):
            InfluenceScoredCorpus(
                n_examples=2, flagged=[0], influence=[float("nan")], source="t"
            )


class TestAttributeLabeledVectors:
    def test_a_one_dimensional_array_is_refused(self):
        with pytest.raises(ValueError, match="must be a 2-D"):
            AttributeLabeledVectors(
                axis="gender", vectors=[1.0, 2.0], labels=["f", "m"], source="t"
            )

    def test_labels_must_align_with_rows(self):
        with pytest.raises(ValueError, match="same number of rows"):
            AttributeLabeledVectors(
                axis="gender", vectors=[[1.0], [2.0]], labels=["f"], source="t"
            )

    def test_a_constant_attribute_is_refused(self):
        with pytest.raises(ValueError, match="at least two observed attribute"):
            AttributeLabeledVectors(
                axis="gender", vectors=[[1.0], [2.0]], labels=["f", "f"], source="t"
            )

    def test_a_non_finite_vector_is_refused(self):
        with pytest.raises(ValueError, match="must be finite"):
            AttributeLabeledVectors(
                axis="gender",
                vectors=[[float("inf")], [2.0]],
                labels=["f", "m"],
                source="t",
            )

    def test_vectors_are_immutable(self):
        evidence = AttributeLabeledVectors(
            axis="gender", vectors=[[1.0], [2.0]], labels=["f", "m"], source="t"
        )
        with pytest.raises(ValueError):
            evidence.vectors[0, 0] = 99.0


class TestGroupLabeledRecords:
    def test_groups_and_labels_must_align(self):
        with pytest.raises(ValueError, match="same number of rows"):
            GroupLabeledRecords(
                axis="gender",
                groups=["f", "m"],
                labels=["yes"],
                label_name="outcome",
                source="t",
            )

    def test_a_single_group_is_refused(self):
        with pytest.raises(ValueError, match="at least two observed categories"):
            GroupLabeledRecords(
                axis="gender",
                groups=["f", "f"],
                labels=["yes", "no"],
                label_name="outcome",
                source="t",
            )

    def test_cells_are_the_observed_joint(self):
        records = GroupLabeledRecords(
            axis="gender",
            groups=["f", "m"],
            labels=["yes", "no"],
            label_name="outcome",
            source="t",
        )
        assert records.cells == (("no", "m"), ("yes", "f"))


class TestCandidateSets:
    def test_candidate_lists_must_align_with_queries(self):
        with pytest.raises(ValueError, match="one candidate list per query"):
            CandidateSets(
                queries=["a", "b"],
                candidates=[["x"]],
                scorer=lambda q, c: 0.0,
                scorer_name="t",
            )

    def test_duplicate_candidates_are_refused(self):
        with pytest.raises(ValueError, match="contains duplicates"):
            CandidateSets(
                queries=["a"],
                candidates=[["x", "x"]],
                scorer=lambda q, c: 0.0,
                scorer_name="t",
            )

    def test_a_non_callable_scorer_is_refused(self):
        with pytest.raises(TypeError, match="must be callable"):
            CandidateSets(
                queries=["a"],
                candidates=[["x"]],
                scorer="not callable",
                scorer_name="t",
            )


class TestTextRecords:
    def test_ids_must_align_and_be_unique(self):
        with pytest.raises(ValueError, match="same number of rows"):
            TextRecords(texts=["a", "b"], source="t", ids=["1"])
        with pytest.raises(ValueError, match="must not contain duplicates"):
            TextRecords(texts=["a", "b"], source="t", ids=["1", "1"])

    def test_an_empty_corpus_is_refused(self):
        with pytest.raises(ValueError, match="must not be empty"):
            TextRecords(texts=[], source="t")
