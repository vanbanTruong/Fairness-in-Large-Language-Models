"""StereoSet dataset loader."""

from __future__ import annotations

from typing import List, Optional, Sequence

from fairLMs.datasets.base import FairnessDataset, optional_limit

# Official StereoSet gold_label convention (Hugging Face / McGill-NLP):
#   0 = anti-stereotype, 1 = stereotype, 2 = unrelated
_OFFICIAL_LABEL = {0: "anti", 1: "stereo", 2: "unrelated"}


class StereoSet(FairnessDataset):
    """Load StereoSet as pairs or triples.

    Parameters
    ----------
    config:
        ``\"intersentence\"`` (default) or ``\"intrasentence\"``.
    as_triples:
        If True, require stereotype / anti_stereotype / unrelated.
        If False, return stereotype / anti_stereotype pairs only.
    hf_path:
        Hugging Face dataset id. Defaults try both common ids.
    label_map:
        Optional override of gold_label → role mapping. Defaults to the
        official StereoSet mapping ``{0: anti, 1: stereo, 2: unrelated}``.

    Each pair example::

        {"stereotype", "anti_stereotype", "bias_type"}

    Each triple example::

        {"stereotype", "anti_stereotype", "unrelated", "bias_type"}
    """

    name = "stereoset"
    data_origin = "Hugging Face Hub, downloaded at first use"

    def __init__(
        self,
        config: str = "intersentence",
        split: str = "validation",
        as_triples: bool = False,
        n_max: Optional[int] = None,
        hf_path: Optional[str] = None,
        label_map: Optional[dict] = None,
        revision: Optional[str] = None,
    ):
        self.config = config
        self.split = split
        self.as_triples = as_triples
        self.n_max = n_max
        self.hf_path = hf_path
        self.revision = revision
        self.label_map = dict(label_map) if label_map is not None else None
        self._cache: Optional[List[dict]] = None

    def _load_hf(self):
        from datasets import load_dataset

        errors = []
        candidates = (
            [self.hf_path] if self.hf_path else ["stereoset", "McGill-NLP/stereoset"]
        )
        for path in candidates:
            if not path:
                continue
            try:
                loaded = load_dataset(
                    path, self.config, split=self.split, revision=self.revision
                )
                self._resolved_hf_path = path
                self._resolved_fingerprint = getattr(loaded, "_fingerprint", None)
                return loaded
            except Exception as exc:  # noqa: BLE001 - try next candidate
                errors.append(f"{path}: {exc}")
        raise RuntimeError(
            "Failed to load StereoSet from Hugging Face. Tried: " + "; ".join(errors)
        )

    def load(self) -> Sequence[dict]:
        if self._cache is not None:
            return optional_limit(self._cache, self.n_max)

        if self.config not in ("intersentence", "intrasentence"):
            raise ValueError("StereoSet config must be intersentence or intrasentence.")
        if self.n_max is not None and (
            isinstance(self.n_max, bool)
            or not isinstance(self.n_max, int)
            or self.n_max < 0
        ):
            raise ValueError("n_max must be a non-negative integer or None.")
        if self.n_max == 0:
            return []
        ds = self._load_hf()
        examples: List[dict] = []

        # Prefer the dataset's declared ClassLabel vocabulary when available.
        label_names = None
        try:
            feature = ds.features["sentences"]
            fields = feature.feature if hasattr(feature, "feature") else feature
            label_names = fields["gold_label"].names
        except (AttributeError, KeyError, TypeError):
            pass
        canonical_roles = {
            "stereotype": "stereo",
            "anti-stereotype": "anti",
            "unrelated": "unrelated",
        }
        label_map = self.label_map
        if label_map is None:
            label_map = (
                {i: canonical_roles.get(name) for i, name in enumerate(label_names)}
                if label_names is not None
                else dict(_OFFICIAL_LABEL)
            )
        for index, row in enumerate(ds):
            sentences = row["sentences"]["sentence"]
            labels = row["sentences"]["gold_label"]
            if len(sentences) != len(labels):
                raise ValueError(
                    f"StereoSet row {index}: sentence/label lengths differ."
                )
            bucket = {}
            for sent, label in zip(sentences, labels):
                role = (
                    label_map.get(label)
                    if not isinstance(label, str)
                    else canonical_roles.get(label)
                )
                if role not in ("stereo", "anti", "unrelated"):
                    raise ValueError(
                        f"StereoSet row {index}: unknown gold_label {label!r}."
                    )
                if role in bucket or not isinstance(sent, str) or not sent.strip():
                    raise ValueError(
                        f"StereoSet row {index}: duplicate role or invalid sentence."
                    )
                bucket[role] = sent
            required = (
                ("stereo", "anti", "unrelated")
                if self.as_triples
                else ("stereo", "anti")
            )
            if not all(role in bucket for role in required):
                raise ValueError(f"StereoSet row {index}: missing required role.")
            context = row.get("context")
            if self.config == "intersentence" and (
                not isinstance(context, str) or not context.strip()
            ):
                raise ValueError(
                    f"StereoSet row {index}: intersentence context is required."
                )
            example = {
                "stereotype": bucket["stereo"],
                "anti_stereotype": bucket["anti"],
                "bias_type": row.get("bias_type"),
                "id": row.get("id"),
                "target": row.get("target"),
                "context": context,
                "config": self.config,
                # Candidate texts remain separate. Scorers condition on context
                # without masking/scoring context tokens for intersentence data.
                "scoring_context": context if self.config == "intersentence" else None,
            }
            if self.as_triples:
                example["unrelated"] = bucket["unrelated"]
            examples.append(example)
            if self.n_max is not None and len(examples) >= self.n_max:
                break

        self._cache = examples
        return optional_limit(examples, self.n_max)
