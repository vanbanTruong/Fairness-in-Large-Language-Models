"""CrowS-Pairs orientation: ``sent_more`` is the stereotype in every row.

The official metric (Nangia et al., 2020, ``metric.py``) counts a preference
for ``sent_more`` for stereo and anti-stereo rows alike, because ``sent_more``
is the more stereotyping sentence in both cases; the ``stereo_antistereo``
label describes the pair, not the side. An earlier loader swapped the sides
for anti-stereo rows, which inverted 218 of the 1,508 pairs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fairLMs.datasets import CrowSPairs
from fairLMs.datasets._paths import resolve_crows_pairs_csv


@pytest.fixture(scope="module")
def frame():
    return pd.read_csv(resolve_crows_pairs_csv(None))


def test_anti_stereo_rows_keep_sent_more_as_the_stereotype(frame):
    rows = CrowSPairs().load()
    assert len(rows) == len(frame) == 1508
    by_label = {"stereo": 0, "antistereo": 0}
    for row, (_, source) in zip(rows, frame.iterrows()):
        assert row["stereotype"] == source["sent_more"]
        assert row["anti_stereotype"] == source["sent_less"]
        assert row["stereo_antistereo"] == source["stereo_antistereo"]
        by_label[source["stereo_antistereo"]] += 1
    assert by_label == {"stereo": 1290, "antistereo": 218}


def test_the_documented_anti_stereo_example_is_oriented_like_the_paper(frame):
    """Row 2 of the bundled CSV is the paper's anti-stereotype illustration."""
    example = next(
        row for row in CrowSPairs().load() if row["stereo_antistereo"] == "antistereo"
    )
    assert "he would come forward" in example["stereotype"]
    assert "she would come forward" in example["anti_stereotype"]


def test_bias_type_filter_does_not_change_orientation(frame):
    gender = CrowSPairs(bias_type="gender").load()
    expected = frame[frame["bias_type"] == "gender"]
    assert len(gender) == len(expected)
    assert [row["stereotype"] for row in gender] == list(expected["sent_more"])
