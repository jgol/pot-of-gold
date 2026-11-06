"""Numerical conventions shared by training diagnostics and reference comparisons."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import math

import numpy as np

def _cvar_left(vals: Sequence[float], q: float = 0.05) -> Optional[float]:
    """Mean of the worst ceil(q * n) values; return None for an empty sample.

    Higher values indicate a less severe loss tail.
    """
    a = np.sort(np.asarray(vals, dtype=float))
    if a.size == 0:
        return None
    k = max(1, int(np.ceil(q * a.size)))
    return float(np.mean(a[:k]))


def cvar_left(values, q=0.05):
    """Return the lower-tail mean, using zero for an empty development portfolio."""
    result = _cvar_left(values, q)
    return 0.0 if result is None else result


def _rejects(statistic: float, threshold: float, reject_when: str) -> bool:
    """Apply the calibrated rejection side; equality remains accepted."""
    if reject_when == "below":
        return statistic < threshold
    if reject_when == "above":
        return statistic > threshold
    raise ValueError(f"reject_when must be 'below'|'above', got {reject_when!r}")


TAIL_Q = 0.05


def _finite(xs) -> List[float]:
    return [float(x) for x in (xs or [])
            if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]


def _max_drawdown(series: Sequence[float]) -> float:
    """Return peak-to-trough loss as a nonpositive value on cumulative daily P&L."""
    cum = peak = mdd = 0.0
    for x in series:
        cum += x
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return float(mdd)


def metrics(series: Sequence[float]) -> Dict[str, Any]:
    xs = _finite(series)
    if not xs:
        return {"n_days": 0, "cvar5": None, "mean": None, "loss_day_rate": None, "max_drawdown": None}
    cv = _cvar_left(xs, TAIL_Q)
    return {"n_days": len(xs),
            "cvar5": (round(float(cv), 6) if cv is not None else None),
            "mean": round(sum(xs) / len(xs), 6),
            "loss_day_rate": round(sum(1 for x in xs if x < 0) / len(xs), 4),
            "max_drawdown": round(_max_drawdown(xs), 6)}


def portfolio_series(day_pnl_by_champ: Dict[str, Dict[Any, float]], kept: Sequence[str],
                     days: Sequence[Any], mode: str = "mean") -> List[float]:
    """Aggregate strategy P&L on a fixed calendar.

    In mean mode, average over retained strategies whose held-out fold includes
    the date. In sum mode, add their P&L. Keep dates with no eligible strategies
    as zero so changing membership does not change the calendar.
    """
    kept = list(kept)
    out: List[float] = []
    for d in days:
        vals = [day_pnl_by_champ[c][d] for c in kept if d in day_pnl_by_champ.get(c, {})]
        if not vals:
            out.append(0.0)
        elif mode == "sum":
            out.append(float(sum(vals)))
        else:
            out.append(float(sum(vals) / len(vals)))
    return out
