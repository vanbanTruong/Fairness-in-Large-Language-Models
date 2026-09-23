"""JSON-safe, non-secret run metadata shared by metric and comparison results."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping

_SECRET_KEYS = {"api_key", "token", "access_token", "password", "authorization"}


def json_safe(value):
    """Normalize numerical results; unsupported objects are identified, not repr'd."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
            if str(k).lower() not in _SECRET_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if dataclasses.is_dataclass(value):
        return {
            f.name: json_safe(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    if callable(value):
        return {
            "callable": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__name__)}",
            "serialized": False,
        }
    return {"type": type(value).__name__, "serialized": False}


def model_provenance(model):
    if model is None:
        return {"kind": "precomputed"}
    metadata = {
        "type": type(model).__name__,
        "name": getattr(model, "name", None),
        "task": getattr(model, "task", None),
    }
    kwargs = getattr(model, "model_kwargs", {})
    metadata["requested_revision"] = (
        kwargs.get("revision") if isinstance(kwargs, Mapping) else None
    )
    loaded = getattr(model, "_loaded", None)
    underlying = (
        getattr(loaded, "model", None)
        if loaded is not None
        else getattr(model, "model", None)
    )
    if underlying is None and isinstance(model, (tuple, list)) and len(model) > 1:
        underlying = model[1]
    metadata["resolved_revision"] = getattr(
        getattr(underlying, "config", None), "_commit_hash", None
    )
    tokenizer_kwargs = getattr(model, "tokenizer_kwargs", {})
    metadata["tokenizer_requested_revision"] = (
        tokenizer_kwargs.get("revision", metadata["requested_revision"])
        if isinstance(tokenizer_kwargs, Mapping)
        else None
    )
    if getattr(model, "base", None) is not None and model.base is not model:
        metadata["base"] = model_provenance(model.base)
        metadata["intervention"] = {
            key: json_safe(getattr(model, key))
            for key in (
                "method",
                "layer",
                "axis",
                "probe_family",
                "decay",
                "epsilon",
                "seed",
                "max_new_tokens",
                "spec",
            )
            if hasattr(model, key)
        }
        if hasattr(model, "projection"):
            metadata["intervention"]["projection_sha256"] = hashlib.sha256(
                json.dumps(
                    json_safe(model.projection), sort_keys=True, allow_nan=False
                ).encode()
            ).hexdigest()
    return metadata


def evidence_provenance(data):
    metadata = {"type": type(data).__name__}
    if data is None:
        metadata["sha256"] = None
        return metadata
    content = data
    if hasattr(data, "load"):
        for key in (
            "name",
            "split",
            "config",
            "n_max",
            "categories",
            "context_condition",
            "revision",
            "_resolved_hf_path",
            "_resolved_fingerprint",
        ):
            if hasattr(data, key):
                metadata[key] = json_safe(getattr(data, key))
        # Never perform I/O for metadata. Hash precisely the materialized slice.
        content = getattr(data, "_cache", None)
        if content is None:
            metadata["sha256"] = None
            return metadata
        limit = getattr(data, "n_max", None)
        if limit is not None:
            content = content[:limit]
    if (
        not isinstance(content, (Mapping, list, tuple, str))
        and not dataclasses.is_dataclass(content)
        and not hasattr(content, "tolist")
    ):
        metadata["sha256"] = None
        return metadata
    payload = json.dumps(
        json_safe(content), ensure_ascii=False, sort_keys=True, allow_nan=False
    ).encode()
    metadata["sha256"] = hashlib.sha256(payload).hexdigest()
    return metadata
