"""Descriptive peer comparisons included in court and single-LLM context."""

from __future__ import annotations

import bisect

import math

from typing import Any, Dict, List, Optional

def _quantile(sorted_xs: List[float], q: float) -> Optional[float]:
    """Interpolate a quantile from a sorted list."""
    n = len(sorted_xs)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_xs[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac)

def peer_compare(exhibit_id: str, value: Any, cohort_stats: Dict[str, List[float]]) -> Dict[str, Any]:
    """Compare a strategy measurement with the same field in the peer population.

    Return the peer median, quantiles and candidate mid-rank percentile. Missing
    or nonnumeric inputs receive an availability note.
    """
    col = cohort_stats.get(exhibit_id) if cohort_stats else None
    if not col:
        return {"exhibit_id": exhibit_id, "n": 0, "note": "no cohort data for this exhibit"}
    xs = sorted(col)
    n = len(xs)
    out = {"exhibit_id": exhibit_id, "value": value, "n": n,
           "cohort_median": _quantile(xs, 0.5), "cohort_p10": _quantile(xs, 0.1),
           "cohort_p90": _quantile(xs, 0.9)}
    try:
        v = float(value)
    except (TypeError, ValueError):
        out["note"] = "champion value is not numeric for this exhibit"
        return out
    if not math.isfinite(v):
        out["note"] = "champion value is not finite for this exhibit"
        return out
    lo = bisect.bisect_left(xs, v)                                  # Give tied observations their midpoint rank.
    hi = bisect.bisect_right(xs, v)
    out["percentile"] = (lo + hi) / (2.0 * n)
    return out

def build_cohort_stats(case_files: List[Any]) -> Dict[str, List[float]]:
    """Collect numeric fields from model-visible strategy profiles.

    Exclude Boolean flags and fields withheld by to_llm_dict().
    """
    cols: Dict[str, List[float]] = {}
    for cf in case_files:
        d = cf.to_llm_dict() if hasattr(cf, "to_llm_dict") else dict(cf)
        for k, v in d.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if math.isfinite(float(v)):
                cols.setdefault(k, []).append(float(v))
    return cols
