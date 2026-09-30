"""Forward targets from the window anchor through the same-day close.

Returns realized annualized volatility minus anchor IV, the forward log
return divided by IV * sqrt(years to close), maximum and minimum log-return
excursions, and anchor IV as a control. IV is annualized fractional volatility.
Earlier window bars and other days are not read.
"""
import numpy as np

from ..encoder.config import RAW_SESSION_LOG_RET_INDEX, RAW_ATM_IV_INDEX

# Conservative same-day cutoff in the stored daily panel.
LAST_LIVE_BAR = 389

TRADING_DAYS_PER_YEAR = 252.0
# Require enough future bars to estimate realized volatility.
MIN_FORWARD_BARS = 5
# Forward-target names are results JSON keys, in the calculation order below:
#   volatility_difference: realized annualized volatility minus anchor implied volatility
#   volatility_scaled_return: forward return divided by implied volatility * sqrt(years to close)
#   upward_excursion / downward_excursion: maximum excursions from the anchor
#   anchor_implied_volatility: anchor implied-volatility control
FORWARD_TARGET_NAMES = ("volatility_difference", "volatility_scaled_return", "upward_excursion",
                        "downward_excursion", "anchor_implied_volatility")
N_FORWARD_TARGETS = len(FORWARD_TARGET_NAMES)
_INVALID = np.zeros(N_FORWARD_TARGETS, dtype=np.float64)


def compute_forward_targets(session_log_ret, atm_iv, last_bar_pos: int, close_bar: int, bars_per_day: int):
    """Return (targets[5], valid) for one window.
    
    session_log_ret and atm_iv are one-dimensional day arrays. last_bar_pos
    is the window anchor; close_bar is the last live bar, bounded by the caller.
    bars_per_day sets the volatility annualization. Invalid spans return a
    placeholder vector and False; callers must retain the validity mask.
    """
    iv_t = float(atm_iv[last_bar_pos])
    anchor_iv_control = iv_t if np.isfinite(iv_t) else 0.0   # anchor_implied_volatility control target
    # Keep anchor IV but skip future targets if the horizon or IV is invalid.
    if (close_bar - last_bar_pos) < MIN_FORWARD_BARS or not np.isfinite(iv_t) or iv_t <= 1e-6:
        out = _INVALID.copy(); out[4] = anchor_iv_control
        return out, False
    p_t = float(session_log_ret[last_bar_pos])
    fwd = np.asarray(session_log_ret[last_bar_pos + 1: close_bar + 1], dtype=np.float64)  # Future bars through the same-day close.
    if fwd.size == 0 or not np.isfinite(p_t) or not np.all(np.isfinite(fwd)):
        out = _INVALID.copy(); out[4] = anchor_iv_control
        return out, False
    p_close = float(fwd[-1])
    # Differencing session log returns gives one-bar log returns. Include the
    # anchor only as the starting point of the first future return.
    path = np.concatenate(([p_t], fwd))
    rets = np.diff(path)
    # The experiment uses population standard deviation and the supplied full
    # session length for annualization, even when the forward span is shorter.
    realized_ann_vol = float(np.std(rets)) * np.sqrt(bars_per_day * TRADING_DAYS_PER_YEAR)
    vrp_residual = realized_ann_vol - iv_t                                # Realized minus anchor implied volatility.
    tau = (close_bar - last_bar_pos) / (bars_per_day * TRADING_DAYS_PER_YEAR)   # years to close
    vol_scaled_surprise = (p_close - p_t) / (iv_t * np.sqrt(tau)) if tau > 0 else 0.0   # volatility_scaled_return
    # Extrema exclude the anchor, so excursions can be negative.
    max_favorable_excursion = float(np.max(fwd)) - p_t                    # upward_excursion from anchor
    max_adverse_excursion = p_t - float(np.min(fwd))                      # downward_excursion from anchor
    # Keep the original target order; downstream column selection depends on it.
    out = np.array([vrp_residual, vol_scaled_surprise, max_favorable_excursion,
                    max_adverse_excursion, anchor_iv_control], dtype=np.float64)
    return out, bool(np.all(np.isfinite(out)))


def load_evaluation_days(data_path, start_date, end_date, log1p_indices=(101, 102, 104, 132, 139),
                         *, T=None, stride=None):
    """Load encoder windows and collect financial targets in the same row order.

    The shared loader supplies each day after preprocessing and VIX forward-fill.
    Targets use the full daily feature array and remain separate from model inputs.
    """
    from ..encoder.data import load_ssl_days

    chunks = {"fwd_predictive": [], "fwd_predictive_valid": []}

    def collect(day, window_length, window_stride):
        n_bars = len(day)
        starts = range(0, n_bars - window_length + 1, window_stride)
        close_bar = min(LAST_LIVE_BAR, n_bars - 1)
        # Store labels in float32 before later residualization, as in the experiment.
        predictive = np.zeros((len(starts), N_FORWARD_TARGETS), dtype=np.float32)
        predictive_valid = np.zeros(len(starts), dtype=bool)
        for row, start in enumerate(starts):
            anchor = start + window_length - 1
            predictive[row], predictive_valid[row] = compute_forward_targets(
                day[:, RAW_SESSION_LOG_RET_INDEX], day[:, RAW_ATM_IV_INDEX],
                anchor, close_bar, n_bars)
        chunks["fwd_predictive"].append(predictive)
        chunks["fwd_predictive_valid"].append(predictive_valid)

    data = load_ssl_days(data_path, start_date, end_date, log1p_indices=log1p_indices,
                         T=T, stride=stride, on_day=collect)
    for key, parts in chunks.items():
        empty = np.empty((0, N_FORWARD_TARGETS), dtype=np.float32) if key == "fwd_predictive" else np.empty(0, dtype=bool)
        data[key] = np.concatenate(parts) if parts else empty
        if len(data[key]) != len(data["X_windows"]):
            raise ValueError(f"Targets {key} do not align with the input windows")
    valid = data["fwd_predictive_valid"]
    n_valid = int(valid.sum())
    iv_values = data["fwd_predictive"][valid, 4] if n_valid else np.array([np.nan])
    median_iv = float(np.nanmedian(iv_values))
    print(f"  Forward-target valid windows: {n_valid}/{len(valid)} ({n_valid/max(len(valid), 1):.1%})")
    if n_valid and not (0.01 <= median_iv <= 3.0):
        print(f"  WARNING: median anchor IV={median_iv:.4f} is outside the expected "
              "annualized fractional-volatility range; check the corpus IV units.")
    else:
        print(f"  Anchor-IV units check: median anchor_implied_volatility={median_iv:.4f} (annualized-fractional range)")
    return data
