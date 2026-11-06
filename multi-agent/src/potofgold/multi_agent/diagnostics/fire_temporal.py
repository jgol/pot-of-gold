"""Summarize the timing of training entries and their decision margins.

fire_day_gap_gini measures uneven spacing between unique entry days in
training-day units. behavior_half_consistency compares mean absolute
margins in the first and second chronological halves using 1 - |a-b|/(a+b).
Return scalar summaries without dates or realized P&L.
"""
import numpy as np

_MIN_FIRES = 6


def _gini(x) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0 or a.sum() <= 0:
        return 0.0
    a = np.sort(a)
    n = a.size
    return float((2.0 * np.sum((np.arange(1, n + 1)) * a) / (n * a.sum())) - (n + 1.0) / n)


def fire_temporal_metrics(decision_bars, binding_margins, bar_dates) -> dict:
    """Summarize entry-day spacing and chronological margin consistency.

    Decision bars index bar_dates; binding_margins must align with those bars.
    Return None for insufficient or invalid input. Invalid margins affect only
    the margin summary, not valid day-spacing measurements.
    """
    none = {"fire_day_gap_gini": None, "behavior_half_consistency": None}
    b = np.asarray(decision_bars)
    bd = np.asarray(bar_dates)
    # Day-spacing summaries do not require valid margin values.
    if b.size < _MIN_FIRES or bd.size == 0 or int(b.min()) < 0 or int(b.max()) >= bd.size:
        return none
    order = np.argsort(b)
    b = b[order]
    fired_dates = bd[b]

    # Measure gaps between unique entry days in training-day units.
    train_days = sorted(set(bd.tolist()))
    rank = {d: i for i, d in enumerate(train_days)}
    fired_day_ranks = np.array(sorted({rank[d] for d in fired_dates.tolist()}))
    gaps = np.diff(fired_day_ranks) if fired_day_ranks.size > 1 else np.array([])
    gap_gini = _gini(gaps) if gaps.size else 0.0

    # Margin consistency requires one margin per entry, in chronological order.
    m = np.asarray(binding_margins, dtype=float)
    cons = None
    if m.size == b.size:
        m = m[order]
        h = m.size // 2
        a = np.abs(m[:h]); a = a[np.isfinite(a)]
        c = np.abs(m[h:]); c = c[np.isfinite(c)]
        if a.size and c.size:
            ma, mc = float(a.mean()), float(c.mean())
            # Two zero-magnitude halves have undefined consistency.
            cons = 1.0 - abs(ma - mc) / (ma + mc) if (ma + mc) > 0 else None
    return {"fire_day_gap_gini": round(gap_gini, 4),
            "behavior_half_consistency": (round(cons, 4) if cons is not None else None)}
