"""BBQ protocol helpers for Context-Based Disparity (from e532af2)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

UNKNOWN_MARGIN = 0.05


def _stereo_target_idx(row: dict) -> Optional[int]:
    stereo = {
        str(g).strip().lower()
        for g in row.get("additional_metadata", {}).get("stereotyped_groups", [])
    }
    if not stereo:
        return None
    info = row.get("answer_info", {})
    for i in (0, 1, 2):
        meta = info.get(f"ans{i}")
        if isinstance(meta, (list, tuple)) and len(meta) > 1:
            tag = str(meta[1]).strip().lower()
            if tag in stereo:
                return i
    return None


def _unknown_idx(row: dict) -> Optional[int]:
    info = row.get("answer_info", {})
    for i in (0, 1, 2):
        meta = info.get(f"ans{i}")
        if isinstance(meta, (list, tuple)) and len(meta) > 1:
            if "unknown" in str(meta[1]).lower():
                return i
    return None


def bbq_rows_to_protocol(rows: Sequence[dict]) -> List[Dict[str, Any]]:
    """Convert BBQ rows into S_DIS/S_AMB protocol dicts using gold labels.

    This restores the original BBQ jsonl protocol. Model-free gold outputs are
    used so the demo exercises real BBQ stimuli without a heavy MLM scorer.
    For model-based scoring, replace ``output`` with model predictions.
    """
    outputs = []
    for row in rows:
        cond = row.get("context_condition")
        if cond not in ("disambig", "ambig"):
            continue
        label = row.get("label")
        unk = _unknown_idx(row)
        stereo_idx = _stereo_target_idx(row)
        if cond == "ambig":
            expected = "UNKNOWN"
            # gold for ambig is typically the unknown option
            output = "UNKNOWN" if label == unk else ("biased" if label == stereo_idx else "other")
            # Map to protocol labels used by compute_s_*
            if output != "UNKNOWN":
                output = "target" if label == stereo_idx else "nontarget"
        else:
            # disambig: expected is the gold non-unknown answer polarity vs stereotype
            if label == unk or label not in (0, 1, 2):
                expected = "UNKNOWN"
            elif stereo_idx is not None and label == stereo_idx:
                expected = "target"
            else:
                expected = "nontarget"
            output = expected  # gold oracle demo on real BBQ rows
        outputs.append({"cond": cond, "output": output, "expected": expected, "category": row.get("category")})
    return outputs
