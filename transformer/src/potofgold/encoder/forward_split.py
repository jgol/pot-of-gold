"""Trading-date embargo and input indicators shared by training and evaluation."""
import numpy as np


def apply_embargo(dates, train_mask, val_mask, embargo_days):
    """Drop validation windows in the first embargo_days trading dates after training.
    
    Returns a new validation mask without changing train_mask. Nonpositive
    embargo_days leaves validation unchanged; a long embargo can exclude it all.
    """
    val_mask = np.asarray(val_mask).copy()
    train_mask = np.asarray(train_mask)
    if embargo_days is None or embargo_days <= 0 or not train_mask.any() or not val_mask.any():
        return val_mask
    dates = np.asarray(dates)
    # Use Python max for ISO date strings; NumPy max does not support them.
    last_train = max(dates[train_mask].tolist())
    # Count distinct observed dates, not windows or calendar days.
    after = np.array(sorted(d for d in set(dates.tolist()) if d > last_train))
    if len(after) <= embargo_days:
        return val_mask & False                      # No observed date remains after the embargo.
    cutoff = after[embargo_days]                     # First date after the excluded trading dates.
    return val_mask & (dates >= cutoff)


# Anchor indicators shared by target residualization and indicator baselines.
SCALAR_REGRESSORS = {
    "ATM_IV": 41, "PRV15": 111, "PRV30": 112, "PRV60": 113, "VIXSpot": 119,
    "VIXTermSlope": 130, "MinutesToClose": 114, "session_logret": 105,
    "logret_5m": 108, "logret_30m": 110, "HL_range": 106,
}
_SKEW_DIFF_RAW = (17, 65)   # skew_25d = raw column 17 - raw column 65 (appended as a difference column)


def build_residualizer_scalars(X, active_features):
    """Build indicator regressors from the last time step of preprocessed windows.
    
    X is (N, T, V); active_features maps feature positions to raw indices.
    Returns (scalars, names), skipping absent features. Appends put-minus-call
    skew only if both IV features are present. The OLS fit adds the intercept.
    """
    active = [int(i) for i in active_features]

    def _pos(raw):
        return active.index(raw) if raw in active else None

    # Use the observed input anchor, never a future target bar. These values are
    # already transformed like encoder inputs; raw indices only identify columns.
    anchor = np.asarray(X)[:, -1, :]
    cols, names = [], []
    for nm, raw in SCALAR_REGRESSORS.items():
        p = _pos(raw)
        if p is not None:
            cols.append(p); names.append(nm)
    D = anchor[:, cols]
    p17, p65 = _pos(_SKEW_DIFF_RAW[0]), _pos(_SKEW_DIFF_RAW[1])
    if p17 is not None and p65 is not None:
        D = np.concatenate([D, (anchor[:, p17] - anchor[:, p65])[:, None]], axis=1)
        names.append("skew_25d")
    return D, names
