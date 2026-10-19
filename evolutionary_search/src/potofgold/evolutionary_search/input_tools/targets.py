"""Six unresidualized financial labels; supervised fitting belongs to each search fold."""
import numpy as np

from potofgold.analysis.forward_targets import compute_forward_targets, MIN_FORWARD_BARS

TARGET_NAMES = ("volatility_difference", "volatility_scaled_return", "upward_excursion",
               "downward_excursion", "time_to_upward_extremum", "time_to_downward_extremum")


def day_targets(frame):
    """Reuse the original forward formulas over the full supplied session.

    Labels use strictly future bars (anchor, final row], population volatility
    annualized by session length * 252, and normalized first-extremum timing.
    The last five rows are invalid. No scaling, residualization or basis is fit.
    """
    n = len(frame)
    bars = frame["bar_position"].to_numpy(dtype=np.int64)
    if not np.array_equal(bars, np.arange(n)):
        raise ValueError("Target construction requires a complete day starting at bar zero")
    prices = frame["SPXClose"].to_numpy(dtype=np.float64)
    iv = frame["ATM_IV"].to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Target construction requires finite positive SPXClose")
    log_price = np.log(prices)
    targets = np.zeros((n, len(TARGET_NAMES)), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for anchor in range(n):
        forward, ok = compute_forward_targets(log_price, iv, anchor, n - 1, n)
        if not ok or n - 1 - anchor < MIN_FORWARD_BARS:
            continue
        future = log_price[anchor + 1:]
        timing = [(np.argmax(future) + 1) / len(future), (np.argmin(future) + 1) / len(future)]
        values = np.asarray([*forward[:4], *timing], dtype=np.float32)
        if np.isfinite(values).all():
            targets[anchor] = values
            valid[anchor] = True
    return targets, valid
