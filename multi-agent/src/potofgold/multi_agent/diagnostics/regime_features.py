"""Summarize calendar and volatility exposure on training entry days.

Compare entry dates with the available training regime mix. Exclude causal
volatility-rank warm-up from both entry and base distributions. Calendar
axes remain available independently.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

# Minimum prior observations for a causal percentile to enter concentration.
VIX_RANK_MIN_HISTORY = 20


def _to_date(d):
    import pandas as pd
    return pd.Timestamp(d).date()


def _dow_tv(fired_rows: Sequence, base_rows: Sequence) -> Optional[float]:
    """Total-variation distance in [0,1] between the fired-day and base day-of-week distributions. 0 = the
    fires track the base weekday mix; →1 = the rule concentrates on particular weekdays."""
    if not fired_rows:
        return None

    def hist(rows):
        h = np.zeros(5, dtype=np.float64)
        for r in rows:
            w = r[0]
            if 0 <= w <= 4:
                h[w] += 1.0
        s = h.sum()
        return (h / s) if s > 0 else h
    return float(0.5 * np.abs(hist(fired_rows) - hist(base_rows)).sum())


def regime_summary(fired_days, all_days) -> Dict:
    """Compare training entry dates with the available calendar regime mix.

    Return exposure fractions, differences from base rates and weekday
    total-variation distance, without raw counts or dates.
    """
    from potofgold.multi_agent.diagnostics.calendar_features import calendar_row
    alld = sorted({_to_date(d) for d in all_days})
    if not alld:
        return {"available": False}
    fired = {_to_date(d) for d in fired_days}
    rows = {d: calendar_row(d) for d in alld}
    fired_rows = [rows[d] for d in alld if d in fired]        # Restrict entries to training dates.
    base_rows = [rows[d] for d in alld]

    def _frac(rws, pred) -> Optional[float]:
        return (sum(1 for r in rws if pred(r)) / len(rws)) if rws else None

    preds = {"opex": lambda r: r[1] == 1, "tom": lambda r: r[2] == 1,
             "fomc_near": lambda r: abs(r[3]) <= 3}
    out: Dict = {"available": True, "fires_present": bool(fired_rows)}
    for k, pred in preds.items():
        f = _frac(fired_rows, pred)
        b = _frac(base_rows, pred)
        out[f"{k}_fire_frac"] = f
        out[f"{k}_base_frac"] = b
        out[f"{k}_lift"] = (f - b) if (f is not None and b is not None) else None
    out["dow_tv_distance"] = _dow_tv(fired_rows, base_rows)
    return out


def _causal_expanding_pctile(vals: Sequence[float], *, min_history: int = 0) -> List[Optional[float]]:
    """Inclusive causal percentile among finite present/past training observations.

    Missing values do not enter the reference distribution or its denominator.
    Warm-up observations update the history but yield None until ``min_history``
    previous valid observations exist. Appending future rows cannot change a rank.
    The default preserves the original static summary convention; the executable
    concentration diagnostic supplies its frozen positive warm-up explicitly.
    """
    if not isinstance(min_history, int) or isinstance(min_history, bool) or min_history < 0:
        raise ValueError("min_history must be a nonnegative integer")
    history = []
    out = []
    for value in vals:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float("nan")
        if not np.isfinite(numeric):
            out.append(None)
            continue
        ready = len(history) >= min_history
        history.append(numeric)
        out.append(sum(v <= numeric for v in history) / len(history) if ready else None)
    return out


def _iqr(sorted_vals: Sequence[float]) -> Optional[float]:
    s = list(sorted_vals)
    if len(s) < 2:
        return None

    def q(p):
        pos = (len(s) - 1) * p
        lo = int(pos)
        return s[lo] + (s[lo + 1] - s[lo]) * (pos - lo) if lo + 1 < len(s) else s[lo]
    return q(0.75) - q(0.25)


def load_vix_rows(path: Optional[str] = None) -> Dict:
    """Load date-indexed VIX, term slope and ATM-IV observations.

    Require all three columns. Drop rows with unparseable or nonfinite values
    or nonpositive ATM IV.
    """
    import math
    import pandas as pd
    p = path or os.environ.get("CAT_RUNNER_VIX_STATE")
    if not p:
        raise ValueError("Supply the 10:30 volatility-state CSV through CAT_RUNNER_VIX_STATE or path")
    df = pd.read_csv(p)
    required = ("date", "vix_1030", "vix_term_slope_1030", "atm_iv_1030")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"vol-state CSV {p} missing required column(s): {missing}")
    rows: Dict = {}
    for _, r in df.iterrows():
        try:
            vix, slope, iv = float(r["vix_1030"]), float(r["vix_term_slope_1030"]), float(r["atm_iv_1030"])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(vix) and math.isfinite(slope) and math.isfinite(iv) and iv > 0):
            continue
        rows[pd.Timestamp(r["date"]).date()] = (vix, slope, iv)
    return rows


def vol_regime_summary(fired_days, all_days, vix_rows: Dict, *, min_fires: int = 10) -> Dict:
    """Summarize volatility conditions on training entry days relative to available days.

    Return mean differences, causal VIX percentile means and percentile spread
    using population standard deviation and IQR. Require enough observations
    for dispersion summaries and keep raw counts internal.
    """
    train = sorted({_to_date(d) for d in all_days})
    cov = [(d, vix_rows[d]) for d in train if d in vix_rows]
    fired = {_to_date(d) for d in fired_days}
    total_fired = len(fired & set(train))
    if not cov:
        return {"vol_available": False, "vol_fires_present": False}
    dates = [d for d, _ in cov]
    vix = [r[0] for _, r in cov]
    slope = [r[1] for _, r in cov]
    iv = [r[2] for _, r in cov]
    pct = _causal_expanding_pctile(vix)                          # Rank within the available training timeline.
    idx = [i for i, d in enumerate(dates) if d in fired]
    out: Dict = {"vol_available": True, "vol_fires_present": bool(idx),
                 "_vol_covered_fired": len(idx), "_vol_total_fired": total_fired}
    if not idx:
        return out

    def _lift(arr):
        return sum(arr[i] for i in idx) / len(idx) - sum(arr) / len(arr)
    out["vix_level_lift"] = _lift(vix)
    out["atm_iv_lift"] = _lift(iv)
    out["term_slope_lift"] = _lift(slope)
    out["vix_pctile_mean"] = sum(pct[i] for i in idx) / len(idx)
    # Require enough entries before summarizing percentile dispersion.
    if len(idx) >= max(2, min_fires):
        fp = sorted(pct[i] for i in idx)
        m = sum(fp) / len(fp)
        out["vix_pctile_spread"] = (sum((x - m) ** 2 for x in fp) / len(fp)) ** 0.5
        out["vix_pctile_iqr"] = _iqr(fp)
    else:
        out["vix_pctile_spread"] = None
        out["vix_pctile_iqr"] = None
    return out


# Total variation compares exposure with the base mix; HHI is separate context.
def _normalized_hhi(counts: Sequence[float]) -> Optional[float]:
    """Normalize Herfindahl concentration across reachable base buckets.

    Use (sum(p_k^2) - 1/K) / (1 - 1/K), where K is the number of buckets
    occupied by the training base. Return None for fewer than two reachable
    buckets or no observations.
    """
    k = len(counts)
    total = float(sum(counts))
    if k < 2 or total <= 0:
        return None
    hhi = sum((c / total) ** 2 for c in counts)
    lo = 1.0 / k
    return float((hhi - lo) / (1.0 - lo))


def _tv_from_base(fired_counts: Sequence[float], base_counts: Sequence[float]) -> Optional[float]:
    """Return half the absolute distribution difference between aligned fired/base buckets.

    The result lies in [0, 1]. Return None without entries or when fewer than
    two base buckets are reachable.
    """
    if len(fired_counts) < 2:
        return None
    nf = float(sum(fired_counts))
    nb = float(sum(base_counts))
    if nf <= 0 or nb <= 0:
        return None
    return float(0.5 * sum(abs(f / nf - b / nb) for f, b in zip(fired_counts, base_counts)))


def _vix_pctile_bucket_counts(fired_days, all_days, vix_rows: Dict, *, n_bins: int = 5,
                             min_history: int = VIX_RANK_MIN_HISTORY):
    """Aligned fired/base counts after causal-rank warm-up and missing-data removal.

    The same eligible dates define both denominators. Warm-up fires never enter
    the volatility numerator, and calendar diagnostics can still use those dates.
    Bins have equal percentile width; their historical frequency need not be equal.
    """
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least two")
    train = sorted({_to_date(d) for d in all_days})
    cov = [(d, vix_rows[d]) for d in train if d in vix_rows]
    if not cov:
        return None, None, 0
    dates = [d for d, _ in cov]
    pct = _causal_expanding_pctile([r[0] for _, r in cov], min_history=min_history)
    valid = [(d, min(int(p * n_bins), n_bins - 1))
             for d, p in zip(dates, pct) if p is not None]
    if not valid:
        return None, None, 0
    fired = {_to_date(d) for d in fired_days}
    occupied = sorted({b for _, b in valid})
    fcounts = {b: 0 for b in occupied}
    bcounts = {b: 0 for b in occupied}
    for d, b in valid:
        bcounts[b] += 1
        if d in fired:
            fcounts[b] += 1
    return [fcounts[b] for b in occupied], [bcounts[b] for b in occupied], len(occupied)


def _calendar_axis_concentrations(fired_days, all_days) -> Dict[str, Optional[float]]:
    """Measure entry/base total-variation distance on each calendar partition.

    Use weekday, OPEX week, turn of month and proximity to FOMC decisions.
    Axes with fewer than two occupied base buckets return None.
    """
    from potofgold.multi_agent.diagnostics.calendar_features import calendar_row
    alld = sorted({_to_date(d) for d in all_days})
    fired = {_to_date(d) for d in alld} & {_to_date(d) for d in fired_days}
    rows = {d: calendar_row(d) for d in alld}

    def _axis(bucket_of) -> Optional[float]:
        base_buckets = sorted({bucket_of(rows[d]) for d in alld})
        fcounts = {b: 0 for b in base_buckets}
        bcounts = {b: 0 for b in base_buckets}
        for d in alld:
            b = bucket_of(rows[d])
            bcounts[b] += 1
            if d in fired:
                fcounts[b] += 1
        return _tv_from_base([fcounts[b] for b in base_buckets], [bcounts[b] for b in base_buckets])

    return {
        "dow": _axis(lambda r: r[0]),                           # Weekday partition.
        "opex": _axis(lambda r: int(r[1] == 1)),               # OPEX-week partition.
        "tom": _axis(lambda r: int(r[2] == 1)),                # Turn-of-month partition.
        "fomc_near": _axis(lambda r: int(abs(r[3]) <= 3)),     # Within three calendar days of an FOMC decision.
    }


def regime_concentration_summary(fired_days, all_days, vix_rows: Optional[Dict] = None, *,
                                 min_fires: int = 10, n_vix_bins: int = 5,
                                 vix_min_history: int = VIX_RANK_MIN_HISTORY) -> Dict:
    """Combine entry/base exposure differences across available regime axes.

    Use max(volatility TV, mean available calendar TVs). The volatility axis
    excludes rank warm-up and requires min_fires covered entries. This combined
    summary belongs to the same evidence family as its component axes.
    """
    train = sorted({_to_date(d) for d in all_days})
    fired = {_to_date(d) for d in fired_days} & set(train)
    n_fired = len(fired)
    if not train or n_fired < max(1, min_fires):
        return {"available": False, "reason": "too_few_fires_or_no_train_days",
                "_n_fired": n_fired, "min_fires": int(min_fires)}

    cal_axes = _calendar_axis_concentrations(fired, train)
    cal_vals = [v for v in cal_axes.values() if v is not None]
    cal_conc = float(sum(cal_vals) / len(cal_vals)) if cal_vals else None

    vix_conc = None
    vix_hhi = None
    n_vix_bins_occupied = 0
    if vix_rows:
        fcounts, bcounts, n_vix_bins_occupied = _vix_pctile_bucket_counts(fired, train, vix_rows, n_bins=n_vix_bins,
                                                                         min_history=vix_min_history)
        if fcounts is not None and sum(fcounts) >= max(1, min_fires):
            vix_conc = _tv_from_base(fcounts, bcounts)
            vix_hhi = _normalized_hhi(fcounts)

    axis_vals = [v for v in (vix_conc, cal_conc) if v is not None]
    if not axis_vals:                                           # Without a usable axis, concentration is undefined.
        return {"available": False, "reason": "no_usable_regime_axis",
                "_n_fired": n_fired, "vol_available": bool(vix_rows)}

    primary = float(max(axis_vals))
    return {
        "available": True,
        "fire_regime_concentration": primary,                  # Primary numeric measurement.
        # Per-axis context is kept outside the primary measurement.
        "vix_pctile_concentration": vix_conc,
        "vix_pctile_hhi": vix_hhi,
        "calendar_concentration": cal_conc,
        "calendar_axis_concentration": cal_axes,
        "combine": "max(vix, calendar); calendar=mean(available subaxes); axis=TV-from-base",
        "vix_rank_min_history": int(vix_min_history),
        "vol_warmup_policy": "exclude undefined ranks from both fired and base distributions",
        "interpretation": "training exposure concentration; no implied future loss",
        "n_vix_bins_occupied": int(n_vix_bins_occupied),
        "vol_available": bool(vix_rows) and vix_conc is not None,
        "_n_fired": n_fired,
    }


# Training replay and exposure summaries.
def regime_for_champion(genome: dict, prep: dict, *, stored_day_pnl=None, fidelity_tol: float = 1e-4,
                        vix_csv: Optional[str] = None, vix_rows: Optional[Dict] = None) -> Dict:
    """Reconstruct training entries and summarize calendar and volatility exposure.

    Check split separation and reconstruction fidelity. Reuse supplied VIX
    rows or load them once from vix_csv. Missing volatility data leaves calendar
    summaries available.
    """
    from potofgold.evolutionary_search.search.genetics import individual_from_genome_dict
    from potofgold.multi_agent.diagnostics.perturbation_prober import _run_trades, _fired_days
    from potofgold.multi_agent.diagnostics.decision_tracer import reconstructed_day_pnl, fidelity_vs_stored, assert_disjoint_days
    if prep.get("train_days") is not None and prep.get("heldout_days") is not None:
        assert_disjoint_days(prep["train_days"], prep["heldout_days"])
    ind = individual_from_genome_dict(genome)
    dates = np.asarray(prep["train_df"]["date"].values)
    trades = _run_trades(ind, prep)
    fired = _fired_days(trades, dates)
    train_days = prep.get("train_days") or list(dates)           # Causal ranks use the training timeline.
    out = regime_summary(fired, train_days)
    if vix_rows is None and vix_csv is not None:
        try:
            vix_rows = load_vix_rows(vix_csv)
        except FileNotFoundError:
            vix_rows = None
    if vix_rows:
        out.update(vol_regime_summary(fired, train_days, vix_rows))
    else:
        out["vol_available"] = False                            # Without volatility data, return calendar summaries only.
    if stored_day_pnl is not None:
        fid = fidelity_vs_stored(reconstructed_day_pnl(trades, dates), stored_day_pnl, fidelity_tol)
        out["fidelity_ok"] = fid["ok"]
        out["_fidelity_maxdelta"] = fid["maxdelta"]
    out["n_trades"] = int(len(trades))
    return out
