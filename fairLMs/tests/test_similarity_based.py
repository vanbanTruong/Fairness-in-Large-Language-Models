"""Focused tests for the refactored similarity-based family (WEAT / SEAT / CEAT).

Covers the three things the refactor was meant to buy:

1. Malformed input fails immediately with an actionable message.
2. Configuration declared in ``__init__`` actually reaches the computation
   (several of these params used to be silently ignored).
3. The legacy keyword API still works, but warns.
"""

import warnings

import numpy as np
import pytest

from fairLMs.metrics import CEAT, SEAT, WEAT, ContextSets, VectorSets, WordSets

TERMS = (
    ["Adam", "Chip", "Harry", "Josh"],
    ["Alonzo", "Jamel", "Lerone", "Percell"],
    ["caress", "freedom", "health", "love"],
    ["abuse", "crash", "filth", "murder"],
)


# --------------------------------------------------------------------------
# Container validation
# --------------------------------------------------------------------------
class TestWordSets:
    def test_rejects_bare_string(self):
        with pytest.raises(TypeError, match="Wrap it in a list"):
            WordSets("Adam", ["a"], ["b"], ["c"])

    def test_rejects_empty_role(self):
        with pytest.raises(ValueError, match="at least one term"):
            WordSets([], ["a"], ["b"], ["c"])

    def test_rejects_non_string_members(self):
        with pytest.raises(TypeError, match="only strings"):
            WordSets([1], ["a"], ["b"], ["c"])

    def test_normalizes_to_tuples(self):
        ws = WordSets(*TERMS)
        assert isinstance(ws.target_1, tuple)

    def test_balanced_targets_check(self):
        WordSets(["a", "b"], ["c", "d"], ["e"], ["f"]).require_balanced_targets("X")
        with pytest.raises(ValueError, match="same length"):
            WordSets(["a", "b"], ["c"], ["e"], ["f"]).require_balanced_targets("X")


class TestVectorSets:
    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            VectorSets([1, 2], [3, 4], [5, 6], [7, 8])

    def test_rejects_mismatched_dims(self):
        with pytest.raises(ValueError, match="share an embedding dimension"):
            VectorSets([[1, 2]], [[1, 2, 3]], [[1, 2]], [[1, 2]])

    def test_reports_n_dims(self):
        rng = np.random.default_rng(0)
        assert VectorSets(*(rng.normal(size=(3, 16)) for _ in range(4))).n_dims == 16


class TestContextSets:
    def test_rejects_flat_list(self):
        with pytest.raises(TypeError, match="must be a mapping"):
            ContextSets(["a"], ["b"], ["c"], ["d"])

    def test_min_contexts(self, context_sets):
        assert context_sets.min_contexts() == 2

    def test_require_contexts(self, context_sets):
        context_sets.require_contexts(2, "CEAT")
        with pytest.raises(ValueError, match="sample_size=5"):
            context_sets.require_contexts(5, "CEAT")


# --------------------------------------------------------------------------
# Config actually reaches the computation
# --------------------------------------------------------------------------
class TestConfigIsHonoured:
    def test_weat_n_samples_threaded(self, vector_sets):
        """``n_samples`` used to be hardcoded to 10_000 inside compute_weat."""
        result = WEAT(n_samples=250).compute(None, vector_sets)
        assert result.details["n_samples"] == 250

    def test_weat_is_deterministic_for_fixed_vectors(self, vector_sets):
        a = WEAT(n_samples=200).compute(None, vector_sets).score
        b = WEAT(n_samples=200).compute(None, vector_sets).score
        assert a == b

    @pytest.mark.parametrize("pooling", ["mean", "cls"])
    def test_seat_pooling_reported(self, encoder_model, pooling):
        result = SEAT(n_samples=100, pooling=pooling).compute(
            encoder_model, WordSets(*TERMS)
        )
        assert result.details["pooling"] == pooling

    def test_seat_pooling_changes_result(self, encoder_model):
        """``pooling`` was accepted but hardcoded to "mean"; it must now matter."""
        ws = WordSets(*TERMS)
        mean = SEAT(n_samples=100, pooling="mean").compute(encoder_model, ws).score
        cls_ = SEAT(n_samples=100, pooling="cls").compute(encoder_model, ws).score
        assert mean != cls_

    def test_ceat_seed_makes_runs_reproducible(self, encoder_model, context_sets):
        kw = dict(sample_size=2, n_trials=5, seed=7)
        a = CEAT(**kw).compute(encoder_model, context_sets).score
        b = CEAT(**kw).compute(encoder_model, context_sets).score
        assert a == b


# --------------------------------------------------------------------------
# Reproducibility is declared config, not a global side effect
#
# WEAT/SEAT p-values are sampled, but until `seed` existed the only way to pin
# one was `np.random.seed(...)` at the call site: reproducibility via mutation
# of interpreter-wide state, invisible to `get_params()`, and shared with every
# other consumer of the global RNG. CEAT already took a `seed`; these tests hold
# its two siblings to the same contract.
#
# `n_samples` is kept below C(2n, n) throughout — 70 for the four-term sets —
# so the sampled branch runs. Above that threshold the p-value is computed by
# exact enumeration and no seed could matter.
# --------------------------------------------------------------------------
class TestSeedIsDeclaredConfig:
    @pytest.mark.parametrize("metric_cls", [WEAT, SEAT])
    def test_seed_is_introspectable_config(self, metric_cls):
        """The claim the paper makes about keyword-only config, for `seed`."""
        metric = metric_cls(n_samples=20, seed=0)
        assert metric.get_params()["seed"] == 0
        # Reconstructible from its own params, so sklearn.clone works.
        assert type(metric)(**metric.get_params()).get_params() == metric.get_params()

    def test_seed_rejected_positionally(self):
        """Config stays keyword-only; a bare 0 must not land in `seed`."""
        with pytest.raises(TypeError):
            SEAT(0)  # noqa: B018

    def test_weat_seed_makes_runs_reproducible(self, vector_sets):
        kw = dict(n_samples=20, seed=7)
        a = WEAT(**kw).compute(None, vector_sets).details["p_value"]
        b = WEAT(**kw).compute(None, vector_sets).details["p_value"]
        assert a == b

    def test_seat_seed_makes_runs_reproducible(self, encoder_model):
        kw = dict(n_samples=50, seed=7)
        ws = WordSets(*TERMS)
        a = SEAT(**kw).compute(encoder_model, ws).details["p_value"]
        b = SEAT(**kw).compute(encoder_model, ws).details["p_value"]
        assert a == b

    @pytest.mark.parametrize("metric_cls", [WEAT, SEAT])
    def test_seed_is_reported_in_details(self, metric_cls, vector_sets, encoder_model):
        """A result must carry the seed that produced it, or it is not evidence."""
        if metric_cls is WEAT:
            result = WEAT(n_samples=20, seed=3).compute(None, vector_sets)
        else:
            result = SEAT(n_samples=50, seed=3).compute(
                encoder_model, WordSets(*TERMS)
            )
        assert result.details["seed"] == 3

    def test_seeded_result_ignores_global_numpy_state(self, vector_sets):
        """Seeding numpy globally must not be able to change a seeded result."""
        import numpy as np

        np.random.seed(1)
        a = WEAT(n_samples=20, seed=5).compute(None, vector_sets).details["p_value"]
        np.random.seed(999)
        b = WEAT(n_samples=20, seed=5).compute(None, vector_sets).details["p_value"]
        assert a == b

    def test_compute_does_not_disturb_the_global_rng(self, vector_sets):
        """The metric must not consume global entropy others are relying on."""
        import numpy as np

        np.random.seed(0)
        expected = np.random.random(3).tolist()

        np.random.seed(0)
        WEAT(n_samples=20, seed=5).compute(None, vector_sets)
        after = np.random.random(3).tolist()

        assert after == expected

    def test_different_seeds_sample_different_permutations(self):
        """Otherwise `seed` would be accepted and silently ignored."""
        import numpy as np

        from fairLMs.utils import permutation_pval

        rng = np.random.default_rng(42)
        s_t1 = rng.normal(1.0, size=12)
        s_t2 = rng.normal(0.0, size=12)
        seen = {
            permutation_pval(s_t1, s_t2, n_samples=200, seed=seed)
            for seed in range(8)
        }
        assert len(seen) > 1

    def test_exact_branch_is_seed_invariant(self):
        """C(2n, n) <= n_samples enumerates every partition; the seed is moot."""
        from fairLMs.utils import permutation_pval

        s_t1 = [3.0, 2.0, 1.0, 0.0]
        s_t2 = [0.0, 1.0, 2.0, 3.0]
        seen = {
            permutation_pval(s_t1, s_t2, n_samples=10_000, seed=seed)
            for seed in range(5)
        }
        assert len(seen) == 1


# --------------------------------------------------------------------------
# Argument handling
# --------------------------------------------------------------------------
class TestArgumentHandling:
    def test_unknown_kwarg_raises(self, vector_sets):
        with pytest.raises(TypeError, match="n_bootstrp"):
            WEAT().compute(None, vector_sets, n_bootstrp=20)

    def test_weat_without_model_needs_vectors(self):
        with pytest.raises(ValueError, match="needs a model"):
            WEAT().compute(None, WordSets(*TERMS))

    def test_weat_with_no_data_at_all(self):
        with pytest.raises(ValueError, match="four term sets"):
            WEAT().compute(None)

    def test_wrong_container_type_is_reported(self, encoder_model):
        with pytest.raises(TypeError, match="expects a ContextSets"):
            CEAT().compute(encoder_model, WordSets(*TERMS))

    def test_mapping_is_accepted(self, vector_sets):
        """A dict keyed by role aliases coerces to a container."""
        t1, t2, a1, a2 = TERMS
        ws = {"t1": t1, "t2": t2, "a1": a1, "a2": a2}
        with pytest.raises(ValueError, match="needs a model"):
            WEAT().compute(None, ws)  # coerced to WordSets, then needs a model

    def test_incomplete_mapping_names_missing_roles(self):
        with pytest.raises(ValueError, match="missing attribute_1"):
            WEAT().compute(None, {"t1": ["a"], "t2": ["b"]})


# --------------------------------------------------------------------------
# Legacy keyword API
# --------------------------------------------------------------------------
class TestLegacyApi:
    def test_legacy_terms_still_work_and_warn(self, encoder_model):
        t1, t2, a1, a2 = TERMS
        with pytest.warns(DeprecationWarning, match="deprecated"):
            legacy = WEAT(n_samples=100).compute(
                model=encoder_model, T1_terms=t1, T2_terms=t2, A_terms=a1, B_terms=a2
            )
        modern = WEAT(n_samples=100).compute(encoder_model, WordSets(*TERMS))
        assert legacy.score == modern.score

    def test_legacy_a1_a2_aliases(self, encoder_model):
        t1, t2, a1, a2 = TERMS
        with pytest.warns(DeprecationWarning):
            result = SEAT(n_samples=100).compute(
                model=encoder_model, T1_terms=t1, T2_terms=t2, A1_terms=a1, A2_terms=a2
            )
        assert isinstance(result.score, float)

    def test_legacy_vecs_path(self, vector_sets):
        with pytest.warns(DeprecationWarning, match="VectorSets"):
            result = WEAT(n_samples=100).compute(
                None,
                T1_vecs=vector_sets.target_1,
                T2_vecs=vector_sets.target_2,
                A_vecs=vector_sets.attribute_1,
                B_vecs=vector_sets.attribute_2,
            )
        assert isinstance(result.score, float)

    def test_legacy_contexts_path(self, encoder_model, context_sets):
        with pytest.warns(DeprecationWarning, match="ContextSets"):
            result = CEAT(sample_size=2, n_trials=3, seed=0).compute(
                model=encoder_model,
                T1_contexts=dict(context_sets.target_1),
                T2_contexts=dict(context_sets.target_2),
                A1_contexts=dict(context_sets.attribute_1),
                A2_contexts=dict(context_sets.attribute_2),
            )
        assert isinstance(result.score, float)

    def test_modern_path_does_not_warn(self, vector_sets):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            WEAT(n_samples=100).compute(None, vector_sets)
