"""The before/after harness, and the aggregations it refuses to perform."""

import json
import numpy as np
import torch
from fairLMs.definitions.models.base import LoadedModel, ModelAdapter
from fairLMs.mitigation.intraprocessing import ProjectedModelAdapter

import pytest

from fairLMs.definitions import GroupPredictions, ScorePair
from fairLMs.mitigation import (
    ComparisonReport,
    MetricDelta,
    MetricEvaluation,
    compare_before_after,
)
from fairLMs.definitions import AccuracyDisparity, EqualOpportunityGap


class TestReporting:
    def test_runs_a_declared_metric_set_against_both_systems(self):
        report = compare_before_after(
            None,
            None,
            metrics={
                "accuracy_disparity": MetricEvaluation(
                    metric=AccuracyDisparity(),
                    data=ScorePair([1.0, 0.0], [0.0, 1.0]),
                    after_data=ScorePair([1.0, 0.0], [0.0, 1.0]),
                )
            },
        )
        assert len(report.fairness) == 1
        delta = report.fairness[0]
        assert delta.metric == "accuracy_disparity"
        assert delta.before is not None and delta.after is not None
        assert delta.delta == pytest.approx(0.0)

    def test_delta_is_after_minus_before(self):
        delta = MetricDelta(metric="m", bias_type="intrinsic", before=0.5, after=0.2)
        assert delta.delta == pytest.approx(-0.3)

    def test_a_failed_metric_is_recorded_not_dropped(self):
        # A partial comparison must be visibly partial.
        report = compare_before_after(
            None, None, metrics={"equal_opportunity_gap": "not valid evidence"}
        )
        delta = report.fairness[0]
        assert delta.error is not None
        assert delta.before is None and delta.delta is None

    def test_unknown_metric_names_are_refused_up_front(self):
        with pytest.raises(KeyError, match="unknown metric"):
            compare_before_after(None, None, metrics={"not_a_metric": []})


class TestSeparationOfConcerns:
    def _report(self):
        return compare_before_after(
            None,
            None,
            metrics={
                "accuracy_disparity": MetricEvaluation(
                    metric=AccuracyDisparity(),
                    data=ScorePair([1.0, 0.0], [0.0, 1.0]),
                    after_data=ScorePair([1.0, 0.0], [0.0, 1.0]),
                ),
                "equal_opportunity_gap": MetricEvaluation(
                    metric=EqualOpportunityGap(),
                    data=GroupPredictions([1, 1, 0], [1, 0, 0], ["A", "B", "A"]),
                    after_data=GroupPredictions([1, 1, 0], [1, 0, 0], ["A", "B", "A"]),
                ),
            },
            utility={"accuracy": lambda model: 0.9},
        )

    def test_fairness_and_utility_are_reported_separately(self):
        report = self._report()
        payload = report.to_dict()
        assert "fairness" in payload and "utility" in payload
        assert [d.metric for d in report.utility] == ["accuracy"]
        # Utility never appears among the fairness deltas.
        assert "accuracy" not in [d.metric for d in report.fairness]

    def test_intrinsic_and_extrinsic_are_reported_separately(self):
        report = self._report()
        payload = report.to_dict()["fairness"]
        assert set(payload) == {"intrinsic", "extrinsic"}
        assert all(d.bias_type == "extrinsic" for d in report.extrinsic)
        assert all(d.bias_type == "intrinsic" for d in report.intrinsic)

    def test_there_is_no_composite_effectiveness_score(self):
        report = self._report()
        # Not float-convertible, and no aggregate anywhere in the payload.
        with pytest.raises(TypeError):
            float(report)
        # Checked against the data, not the prose: the explanatory note names
        # the thing it is refusing to compute.
        payload = report.to_dict()
        payload.pop("note")
        serialized = json.dumps(payload)
        for forbidden in ("overall", "composite", "effectiveness", "total_score"):
            assert forbidden not in serialized
        assert set(payload) == {"fairness", "utility", "provenance"}

    def test_the_report_says_why_it_does_not_aggregate(self):
        assert "no composite" in self._report().to_dict()["note"].lower()

    def test_report_serializes_deterministically(self):
        report = self._report()
        assert json.loads(report.to_json()) == json.loads(report.to_json())

    def test_an_empty_report_is_representable(self):
        assert ComparisonReport().to_dict()["utility"] == []


# --- regression: the base model must not be mutated across metrics -----------
#
# An intra-processing adapter installs a forward hook on the very nn.Module the
# base adapter caches. Scoring one metric all the way through (before, then
# after) used to leave that hook attached, so every later metric read its
# "before" from the already mitigated model and reported a delta of exactly
# zero. The first metric was right and every one after it was silently wrong.


class _TinyModel(torch.nn.Module):
    """Three-dimensional embedding table with a locatable input embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(4, 3)
        with torch.no_grad():
            self.emb.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
                )
            )

    def get_input_embeddings(self):
        return self.emb

    def forward(self, ids):
        return self.emb(ids)


class _TinyAdapter(ModelAdapter):
    name = "tiny"
    task = "encoder"

    def __init__(self) -> None:
        self._loaded = None
        self._module = _TinyModel()

    def load(self):
        if self._loaded is None:
            self._loaded = LoadedModel(
                name=self.name,
                tokenizer=None,
                model=self._module,
                device="cpu",
                task=self.task,
            )
        return self._loaded


def _row_sum(adapter):
    """Stand in for a metric: read something off the model's own forward pass."""
    model = adapter.load().model
    return float(model(torch.tensor([0, 1, 2, 3])).detach().abs().sum())


def _projected_pair():
    base = _TinyAdapter()
    mitigated = ProjectedModelAdapter(
        base,
        np.diag([1.0, 1.0, 0.0]),
        method="subspace_projection",
        axis="gender",
        probe_family="test",
        layer="input_embeddings",
    )
    return base, mitigated


def test_every_measure_sees_the_unmitigated_baseline():
    base, mitigated = _projected_pair()

    report = compare_before_after(
        base,
        mitigated,
        metrics={},
        utility={"first": _row_sum, "second": _row_sum, "third": _row_sum},
    )

    assert [d.metric for d in report.utility] == ["first", "second", "third"]
    befores = {d.before for d in report.utility}
    assert befores == {6.0}, f"baseline drifted across measures: {befores}"
    for delta in report.utility:
        assert delta.after == 4.0
        assert delta.after - delta.before == -2.0


def test_comparison_restores_the_caller_s_base_model():
    base, mitigated = _projected_pair()
    untouched = _row_sum(base)

    compare_before_after(base, mitigated, metrics={}, utility={"probe": _row_sum})

    assert _row_sum(base) == untouched


def test_preloaded_hook_is_removed_before_baseline_even_when_candidate_fails():
    base, mitigated = _projected_pair()
    assert _row_sum(mitigated) == 4.0
    assert _row_sum(base) == 4.0  # shared model currently has the hook

    def measure(adapter):
        value = _row_sum(adapter)
        if adapter is mitigated:
            raise ValueError("candidate failed after installing its hook")
        return value

    report = compare_before_after(
        base, mitigated, metrics={}, utility={"probe": measure}
    )
    assert report.utility[0].before == 6.0
    assert report.utility[0].after is None
    assert "candidate failed" in report.utility[0].error
    assert _row_sum(base) == 6.0


def test_comparison_tolerates_an_irreversible_mitigated_model():
    """A mitigated object with no remove() must still produce a report."""

    class _Opaque:
        pass

    report = compare_before_after(
        _TinyAdapter(), _Opaque(), metrics={}, utility={"probe": _row_sum}
    )
    assert report.utility[0].before == 6.0
    assert report.utility[0].error is not None
