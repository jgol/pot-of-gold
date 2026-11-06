"""Derive weekday, OPEX-week, turn-of-month and FOMC-proximity inputs from dates.

Each feature is constant within a trading day. Both replay and regime diagnostics
use these functions and the bundled FOMC calendar.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path as _Path

import numpy as np
import pandas as pd

# Source and coverage are recorded in the bundled CSV.
FOMC_CSV = _Path(__file__).resolve().parent / "fomc_dates.csv"


def load_fomc_dates(csv_path: "_Path | str" = FOMC_CSV):
    """Load decision dates from the bundled FOMC calendar; reject a missing or empty file."""
    df = pd.read_csv(csv_path, comment="#")
    dates = sorted(_dt.date.fromisoformat(str(s).strip()) for s in df["fomc_date"].dropna())
    if not dates:
        raise ValueError(f"[calendar_features] no FOMC dates loaded from {csv_path}")
    return dates


# Names used by the GP grammar and terminal preparation.
CALENDAR_TERMINALS = ("CalDayOfWeek", "CalIsOpexWeek", "CalTurnOfMonth", "CalDaysToFomc")

_FOMC_SORTED = load_fomc_dates()


def _third_friday(year: int, month: int) -> _dt.date:
    """Return the third Friday of the month."""
    first = _dt.date(year, month, 1)
    first_friday = 1 + ((4 - first.weekday()) % 7)   # Python weekdays start at Monday=0.
    return _dt.date(year, month, first_friday + 14)


def _is_opex_week(d: _dt.date) -> int:
    """Return whether the date falls in the week containing the third Friday."""
    tf = _third_friday(d.year, d.month)
    monday = tf - _dt.timedelta(days=tf.weekday())
    return int(monday <= d <= monday + _dt.timedelta(days=4))


def _days_to_fomc(d: _dt.date) -> int:
    """Return signed calendar days to the nearest FOMC decision, clipped to [-10, 10].

    Positive values precede the decision; negative values follow it.
    """
    nearest = min(((f - d).days for f in _FOMC_SORTED), key=abs)
    return int(np.clip(nearest, -10, 10))


def calendar_row(d: _dt.date):
    """Return the four date features in CALENDAR_TERMINALS order."""
    return (d.weekday(),
            _is_opex_week(d),
            int(d.day <= 3 or d.day >= 26),               # Turn of month: calendar days 1-3 or 26 onward.
            _days_to_fomc(d))


def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing calendar feature columns to df in place.

    Compute values once per unique date and map them to the rows.
    """
    if all(c in df.columns for c in CALENDAR_TERMINALS):
        return df
    if "date" not in df.columns:
        raise KeyError("add_calendar_columns: df has no 'date' column")
    dates = pd.to_datetime(df["date"]).dt.date.to_numpy()
    cache = {d: calendar_row(d) for d in pd.unique(dates)}
    rows = [cache[d] for d in dates]
    for i, name in enumerate(CALENDAR_TERMINALS):
        df[name] = np.asarray([r[i] for r in rows],
                              dtype=np.int16 if name != "CalDaysToFomc" else np.int32)
    return df
