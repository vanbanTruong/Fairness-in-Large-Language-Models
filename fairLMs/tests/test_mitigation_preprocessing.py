"""Pre-processing components: transforms, provenance, and refusals."""

import pytest

from fairLMs.mitigation import (
    CorpusWithBenignPool,
    CorpusWithLexicon,
    CounterfactualDataAugmentation,
    DebiasingPrompt,
    GroupLabeledRecords,
    GroupLabelReweighting,
    IdentityTermAugmentation,
    PromptSpec,
    SwapLexicon,
    TextRecords,
)

LEXICON = SwapLexicon(
    axis="gender",
    pairs=[("he", "she"), ("his", "her"), ("man", "woman")],
    source="unit-test",
)


class TestCounterfactualDataAugmentation:
    def test_emits_the_original_and_the_swapped_copy(self):
        records = TextRecords(texts=["He is a nurse."], source="t")
        result = CounterfactualDataAugmentation().apply(
            None, CorpusWithLexicon(records=records, lexicon=LEXICON)
        )
        assert list(result.result.texts) == ["He is a nurse.", "She is a nurse."]

    def test_output_is_exactly_twice_the_input(self):
        records = TextRecords(
            texts=["He is a nurse.", "The man left.", "His idea."], source="t"
        )
        result = CounterfactualDataAugmentation().apply(
            None, CorpusWithLexicon(records=records, lexicon=LEXICON)
        )
        assert result.result.n_rows == 2 * records.n_rows

    def test_casing_is_carried_onto_the_replacement(self):
        records = TextRecords(texts=["He and HIS and man"], source="t")
        result = CounterfactualDataAugmentation().apply(
            None, CorpusWithLexicon(records=records, lexicon=LEXICON)
        )
        assert result.result.texts[1] == "She and HER and woman"

    def test_swapping_is_involutive_on_a_fully_covered_sentence(self):
        # Swapping twice must return the original: the lexicon is bidirectional.
        records = TextRecords(texts=["he left"], source="t")
        once = CounterfactualDataAugmentation().apply(
            None, CorpusWithLexicon(records=records, lexicon=LEXICON)
        )
        swapped = TextRecords(texts=[once.result.texts[1]], source="t")
        twice = CounterfactualDataAugmentation().apply(
            None, CorpusWithLexicon(records=swapped, lexicon=LEXICON)
        )
        assert twice.result.texts[1] == "he left"

    def test_a_substring_match_does_not_trigger_a_swap(self):
        # "shed" contains "she" but is not the term. Word boundaries matter.
        records = TextRecords(texts=["The shed by the manor. He left."], source="t")
        result = CounterfactualDataAugmentation().apply(
            None, CorpusWithLexicon(records=records, lexicon=LEXICON)
        )
        assert "shed" in result.result.texts[1]
        assert "manor" in result.result.texts[1]

    def test_an_unrewritable_record_is_refused_not_dropped(self):
        records = TextRecords(texts=["He left.", "Nothing here."], source="t")
        with pytest.raises(ValueError, match="cannot be rewritten"):
            CounterfactualDataAugmentation().apply(
                None, CorpusWithLexicon(records=records, lexicon=LEXICON)
            )

    def test_keeping_an_unrewritable_record_must_be_asked_for(self):
        records = TextRecords(texts=["Nothing here."], source="t")
        result = CounterfactualDataAugmentation(on_unrewritable="keep").apply(
            None, CorpusWithLexicon(records=records, lexicon=LEXICON)
        )
        assert result.result.n_rows == 2
        assert result.provenance["n_unrewritable"] == 1
        assert result.provenance["on_unrewritable"] == "keep"

    def test_provenance_records_the_lexicon_and_roundtrips(self):
        import json

        records = TextRecords(texts=["He left."], source="t")
        result = CounterfactualDataAugmentation().apply(
            None, CorpusWithLexicon(records=records, lexicon=LEXICON)
        )
        restored = json.loads(result.to_json())
        assert restored["provenance"]["lexicon_source"] == "unit-test"
        assert ["he", "she"] in restored["provenance"]["lexicon_pairs"]
        assert restored["provenance"]["axis"] == "gender"

    def test_a_transposed_pairing_is_refused_at_construction(self):
        records = TextRecords(texts=["He left."], source="t")
        with pytest.raises(TypeError, match="records must be a TextRecords"):
            CorpusWithLexicon(records=LEXICON, lexicon=records)


class TestGroupLabelReweighting:
    def test_weights_are_one_when_label_and_group_are_already_independent(self):
        # Known value: a balanced 2x2 table needs no reweighting.
        records = GroupLabeledRecords(
            axis="gender",
            groups=["f", "f", "m", "m"],
            labels=["yes", "no", "yes", "no"],
            label_name="outcome",
            source="t",
        )
        weights = GroupLabelReweighting().apply(None, records).result["weights"]
        assert weights == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_an_over_represented_cell_is_down_weighted(self):
        records = GroupLabeledRecords(
            axis="gender",
            groups=["f", "f", "f", "m"],
            labels=["yes", "yes", "yes", "no"],
            label_name="outcome",
            source="t",
        )
        result = GroupLabelReweighting().apply(None, records).result
        # (yes, f) is the dominant cell and must be pushed down relative to the
        # independence target.
        assert result["cells"]["yes|f"]["weight"] < 1.0

    def test_reweighting_removes_the_label_group_association(self):
        # The defining invariant: under the new weights the joint factorises.
        records = GroupLabeledRecords(
            axis="gender",
            groups=["f", "f", "f", "m", "m", "m"],
            labels=["yes", "yes", "no", "yes", "no", "no"],
            label_name="outcome",
            source="t",
        )
        weights = GroupLabelReweighting().apply(None, records).result["weights"]
        total = sum(weights)
        joint = {}
        for w, label, group in zip(weights, records.labels, records.groups):
            joint[(label, group)] = joint.get((label, group), 0.0) + w / total
        for (label, group), mass in joint.items():
            p_label = sum(m for (lab, _), m in joint.items() if lab == label)
            p_group = sum(m for (_, grp), m in joint.items() if grp == group)
            assert mass == pytest.approx(p_label * p_group, abs=1e-9)

    def test_weights_are_strictly_positive(self):
        records = GroupLabeledRecords(
            axis="gender",
            groups=["f", "m"],
            labels=["yes", "no"],
            label_name="outcome",
            source="t",
        )
        weights = GroupLabelReweighting().apply(None, records).result["weights"]
        assert all(w > 0 for w in weights)

    def test_uniform_target_equalises_the_cells(self):
        records = GroupLabeledRecords(
            axis="gender",
            groups=["f", "f", "f", "m"],
            labels=["yes", "yes", "yes", "no"],
            label_name="outcome",
            source="t",
        )
        result = GroupLabelReweighting(target="uniform").apply(None, records).result
        masses = [
            cell["observed"] * cell["weight"] for cell in result["cells"].values()
        ]
        assert masses == pytest.approx([masses[0]] * len(masses))

    def test_an_unknown_target_is_refused(self):
        records = GroupLabeledRecords(
            axis="gender",
            groups=["f", "m"],
            labels=["yes", "no"],
            label_name="outcome",
            source="t",
        )
        with pytest.raises(ValueError, match="must be 'independence' or 'uniform'"):
            GroupLabelReweighting(target="balanced").apply(None, records)


class TestIdentityTermAugmentation:
    def _corpus(self):
        return GroupLabeledRecords(
            axis="religion",
            groups=["muslim", "muslim", "christian"],
            labels=["toxic", "toxic", "clean"],
            label_name="toxicity",
            source="t",
            texts=["a", "b", "c"],
        )

    def test_adds_the_pool_with_the_declared_benign_label(self):
        pool = TextRecords(texts=["kind words", "more kind words"], source="pool")
        result = IdentityTermAugmentation(benign_label="clean").apply(
            None, CorpusWithBenignPool(corpus=self._corpus(), pool=pool)
        )
        assert result.result.n_rows == 5
        assert list(result.result.labels)[-2:] == ["clean", "clean"]

    def test_augmentation_weakens_the_term_to_label_correlation(self):
        corpus = self._corpus()
        pool = TextRecords(texts=["kind"] * 4, source="pool")
        result = IdentityTermAugmentation(
            benign_label="clean", benign_group="muslim"
        ).apply(None, CorpusWithBenignPool(corpus=corpus, pool=pool))

        def toxic_rate(records, group):
            rows = [i for i, g in enumerate(records.groups) if g == group]
            return sum(records.labels[i] == "toxic" for i in rows) / len(rows)

        assert toxic_rate(corpus, "muslim") == 1.0
        assert toxic_rate(result.result, "muslim") < 1.0

    def test_the_benign_label_must_be_declared(self):
        pool = TextRecords(texts=["kind"], source="pool")
        with pytest.raises(ValueError, match="benign_label must be declared"):
            IdentityTermAugmentation().apply(
                None, CorpusWithBenignPool(corpus=self._corpus(), pool=pool)
            )

    def test_an_unobserved_benign_label_is_refused(self):
        pool = TextRecords(texts=["kind"], source="pool")
        with pytest.raises(ValueError, match="does not occur in the corpus"):
            IdentityTermAugmentation(benign_label="neutral").apply(
                None, CorpusWithBenignPool(corpus=self._corpus(), pool=pool)
            )

    def test_a_corpus_without_texts_is_refused(self):
        corpus = GroupLabeledRecords(
            axis="religion",
            groups=["a", "b"],
            labels=["toxic", "clean"],
            label_name="toxicity",
            source="t",
        )
        pool = TextRecords(texts=["kind"], source="pool")
        with pytest.raises(ValueError, match="must carry texts"):
            CorpusWithBenignPool(corpus=corpus, pool=pool)


class TestDebiasingPrompt:
    def test_renders_every_template_over_every_query(self):
        spec = PromptSpec(templates=["A: {query}", "B: {query}"], queries=["x", "y"])
        prompts = DebiasingPrompt().apply(None, spec).result["prompts"]
        assert prompts == [["A: x", "B: x"], ["A: y", "B: y"]]

    def test_a_spec_with_no_queries_is_refused(self):
        with pytest.raises(ValueError, match="carries no queries"):
            DebiasingPrompt().apply(None, PromptSpec(templates=["A: {query}"]))

    def test_an_encoder_only_model_is_refused_as_non_generative(self):
        from fairLMs.applicability import TASK_PROFILES

        spec = PromptSpec(templates=["A: {query}"], queries=["x"])
        with pytest.raises(TypeError, match="decoder_only"):
            DebiasingPrompt().apply(TASK_PROFILES["mlm"], spec)

    def test_a_generative_model_is_accepted(self):
        from fairLMs.applicability import TASK_PROFILES

        spec = PromptSpec(templates=["A: {query}"], queries=["x"])
        assert DebiasingPrompt().apply(TASK_PROFILES["causal"], spec).category == "pre"
