"""Data loading, preprocessing, and training splits.

Loads and cleans daily NPZ arrays, constructs overlapping windows, prepares
date splits, and fits input normalization on training observations.
"""

import gc
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .config import (
    TOTAL_VARIATES, GAMMA_INDICES, OPTIONS_GRID_END, RELIABILITY_END,
    ACTIVE_SSL_FEATURES, ACTIVE_N_SSL_FEATURES,
    SSL_HYPERPARAMS, SSL_LOG1P_VARIATES, WINDOW_STRIDE,
    TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
)


# Preprocessing

def should_skip_day(X):
    # Treat all-zero and all-NaN columns as dead, including flow aggregates.
    all_zero = np.all(X == 0, axis=0)
    all_nan = np.all(np.isnan(X), axis=0)
    dead = int(np.sum(all_zero | all_nan))
    return dead > 130


def preprocess_day(X, log1p_indices=(101, 102, 104, 132, 139)):
    """Clean a daily feature array for encoder input.

    Inputs and outputs have shape (n_bars, n_raw_variates). Apply VIX forward-fill
    separately before selecting encoder features. log1p_indices are raw column
    indices given sign(x)*log1p(|x|); defaults match the published experiment.
    Feature selection and training-fitted normalization happen later.
    """
    n_bars = X.shape[0]
    X = X.astype(np.float32)

    # Older 140-column arrays lack the unused final column of the base layout.
    if X.shape[1] < TOTAL_VARIATES:
        pad_width = TOTAL_VARIATES - X.shape[1]
        X = np.pad(X, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)

    # Gamma clamp [0, 100]
    for gi in GAMMA_INDICES:
        X[:, gi] = np.clip(X[:, gi], 0.0, 100.0)

    # Stored columns 132 and 137 already have signed-log compression. The
    # experiment applies a second transform to 132; preserve that preprocessing.
    # Other selected columns contain strike summaries and order-flow changes.
    for vi in log1p_indices:
        col = X[:, vi]
        X[:, vi] = np.sign(col) * np.log1p(np.abs(col))

    # The stored panel includes a settlement tail. Find its last nonzero options
    # bar before carrying quotes and related features into that tail.
    grid_data = X[:, :OPTIONS_GRID_END]
    any_nonzero = np.any(grid_data != 0, axis=1)
    nonzero_bars = np.where(any_nonzero)[0]
    cutoff_int = int(nonzero_bars[-1]) if len(nonzero_bars) > 0 else 0

    # Forward-fill options, strike aggregates, and order flow after the last live bar.
    if cutoff_int < n_bars - 1:
        X[cutoff_int + 1:, :RELIABILITY_END] = X[cutoff_int, :RELIABILITY_END]
        X[cutoff_int + 1:, 99:105] = X[cutoff_int, 99:105]    # strike aggregates
        X[cutoff_int + 1:, 131:140] = X[cutoff_int, 131:140]   # order flow

    return X


def apply_vix_forwardfill(X):
    """Forward-fill VIX futures and recompute derived variates in place."""
    n_bars = X.shape[0]

    # Zero denotes a missing futures quote. Carry only earlier quotes forward;
    # leading zeros stay zero, so no later quote fills an earlier observation.
    for vi in [123, 124]:
        col = X[:, vi]
        last_good = 0.0
        for b in range(n_bars):
            if col[b] != 0.0:
                last_good = col[b]
            elif last_good != 0.0:
                col[b] = last_good
        X[:, vi] = col

    # Recompute derived VIX variates (128=futures_spread, 129=f1_vix_basis, 130=term_structure_slope)
    vix = X[:, 119]
    f1 = X[:, 123]
    f2 = X[:, 124]
    X[:, 128] = f1 - f2
    safe_vix = np.where(vix > 0, vix, 1.0)
    X[:, 129] = np.where(vix > 0, (f1 - vix) / safe_vix, 0.0)

    # Term structure slope: linear regression of [vix9d, vix, vix3m, vix6m] on [9,30,90,180]
    tenors = np.array([9.0, 30.0, 90.0, 180.0])
    x_c = tenors - tenors.mean()
    denom = (x_c ** 2).sum()
    for b in range(n_bars):
        levels = np.array([X[b, 120], X[b, 119], X[b, 121], X[b, 122]])
        if levels.sum() > 0:
            y_c = levels - levels.mean()
            X[b, 130] = np.dot(x_c, y_c) / denom


def iter_trading_dates(start, end):
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)
    d = start
    while d <= end:
        # Holidays remain candidates; the loader skips dates absent from the corpus manifest.
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


# Window loading

def load_ssl_days(data_path, start_date, end_date, log1p_indices=(101, 102, 104, 132, 139),
                  *, T=None, stride=None, on_day=None):
    """Read daily arrays directly from an NPZ and build windows in date order.

    The archive contains a JSON _manifest and d_YYYY-MM-DD arrays.
    X_windows is float32 with shape (N, T, n_features); window_dates identifies
    each row. Normalization is fitted separately. When supplied, on_day receives
    each full-width preprocessed daily array, window length, and stride, allowing
    callers to collect aligned information without reloading the corpus.
    """
    T = SSL_HYPERPARAMS["seq_len"] if T is None else int(T)
    stride = WINDOW_STRIDE if stride is None else int(stride)
    if T < 1 or stride < 1:
        raise ValueError("Window length and stride must be positive")
    n_ssl = ACTIVE_N_SSL_FEATURES

    all_dates = list(iter_trading_dates(start_date, end_date))
    est_bars = len(all_dates) * 400
    est_windows = est_bars // stride + 256

    X_windows = np.empty((est_windows, T, n_ssl), dtype=np.float32)
    window_dates_list = []

    win_idx = 0
    n_days = 0
    n_skipped = 0

    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Corpus not found: {data_path}. Pass a valid --data NPZ file.")
    # Read one day at a time and close the archive when loading finishes or fails.
    with np.load(data_path, allow_pickle=False) as corpus:
        manifest = json.loads(bytes(corpus["_manifest"]).decode("utf-8"))
        available_dates = set(manifest["dates"])
        for d in all_dates:
            ds = d.isoformat()
            if ds not in available_dates:
                continue
            raw = corpus[f"d_{ds}"].astype(np.float32)
            # Match the input width used by the day-validity check before preprocessing.
            if raw.shape[1] < TOTAL_VARIATES:
                raw = np.pad(raw, ((0, 0), (0, TOTAL_VARIATES - raw.shape[1])),
                             mode="constant", constant_values=0.0)

            if should_skip_day(raw):
                n_skipped += 1
                continue

            data = preprocess_day(raw, log1p_indices)
            del raw
            apply_vix_forwardfill(data)

            n_bars = data.shape[0]
            n_days += 1

            # Select the configured input order; the experiment uses 147 features.
            day_ssl = data[:, ACTIVE_SSL_FEATURES]  # (n_bars, n_ssl)
            # Assemble every input window, including those late in the session.
            for start_pos in range(0, n_bars - T + 1, stride):
                end_pos = start_pos + T
                # Extend capacity when the initial window estimate is too small.
                if win_idx >= len(X_windows):
                    ext_x = np.empty((est_windows, T, n_ssl), dtype=np.float32)
                    X_windows = np.concatenate([X_windows, ext_x], axis=0)

                X_windows[win_idx] = day_ssl[start_pos:end_pos]
                window_dates_list.append(ds)
                win_idx += 1

            if on_day is not None:
                on_day(data, T, stride)

            del data, day_ssl
            gc.collect()

            if n_days % 50 == 0:
                print(f"  {n_days} days loaded, {win_idx} windows...")

    # Remove unused capacity before returning the window arrays.
    X_windows = X_windows[:win_idx]
    window_dates = np.array(window_dates_list)

    gb = X_windows.nbytes / 1e9
    print(f"Done. {n_days} days ({n_skipped} skipped), "
          f"{win_idx} windows, {gb:.2f} GB (inputs)")
    return dict(X_windows=X_windows, window_dates=window_dates)


# Input normalization and date splits

def prepare_input_splits(data, embargo_days=0, eval_window="val",
                               date_overrides=None, z_stats=None):
    """Return normalized inputs, date masks, and training normalization statistics.

    eval_window chooses validation or test dates for val_mask. Input transforms
    are fitted on training windows or use saved checkpoint statistics. Embargoes
    exclude the first configured observed trading dates after training.
    """
    X_all = data["X_windows"].copy()  # Copy before preprocessing so cached windows stay unchanged.
    dates = data["window_dates"]

    _D = date_overrides or {}
    _train_end = _D.get("TRAIN_END", TRAIN_END)
    _val_start, _val_end = _D.get("VAL_START", VAL_START), _D.get("VAL_END", VAL_END)
    _test_start, _test_end = _D.get("TEST_START", TEST_START), _D.get("TEST_END", TEST_END)
    train_mask = dates <= _train_end
    if eval_window == "test":
        val_mask = (dates >= _test_start) & (dates <= _test_end)
    elif eval_window == "val":
        val_mask = (dates >= _val_start) & (dates <= _val_end)
    else:
        raise ValueError(f"eval_window must be 'val' or 'test', got {eval_window!r}")

    from .forward_split import apply_embargo
    val_mask = apply_embargo(dates, train_mask, val_mask, embargo_days)

    # These feature-level transforms follow the raw-column transforms in
    # preprocess_day. Convert raw indices to positions in the selected layout.
    log1p_cols = sorted(
        ACTIVE_SSL_FEATURES.index(v)
        for v in SSL_LOG1P_VARIATES if v in ACTIVE_SSL_FEATURES
    )
    if log1p_cols:
        print(f"  Log1p compression: Applying log1p to {len(log1p_cols)} heavy-tailed variates "
              f"(raw: {sorted(SSL_LOG1P_VARIATES & set(ACTIVE_SSL_FEATURES))})")
        X_all[:, :, log1p_cols] = np.sign(X_all[:, :, log1p_cols]) * np.log1p(np.abs(X_all[:, :, log1p_cols]))

    # Fit normalization on training windows, ignoring NaNs per column. Flattening
    # overlapping windows counts a bar once for each window in which it appears.
    X_train_raw = X_all[train_mask]
    V = X_train_raw.shape[2]
    flat = X_train_raw.reshape(-1, V)
    global_mean = np.nanmean(flat, axis=0)
    global_std = np.nanstd(flat, axis=0)
    # Warn if any column was fully-NaN (all samples NaN on that variate)
    fully_nan = np.isnan(global_mean)
    if fully_nan.any():
        bad_positions = np.where(fully_nan)[0].tolist()
        bad_raws = [ACTIVE_SSL_FEATURES[i] for i in bad_positions]
        print(f"  WARNING: {fully_nan.sum()} variate(s) have all-NaN training "
              f"data; falling back to mean=0, std=1. raw: {bad_raws}")
        global_mean = np.where(fully_nan, 0.0, global_mean)
        global_std = np.where(fully_nan, 1.0, global_std)
    # Warnings on raw stats before flooring (so near-zero check isn't masked)
    raw_std = global_std.copy()
    for i, fidx in enumerate(ACTIVE_SSL_FEATURES):
        if abs(global_mean[i]) > 10:
            print(f"  WARNING: SSL feature v{fidx} extreme mean={global_mean[i]:.2f}")
        if raw_std[i] < 1e-4:
            print(f"  WARNING: SSL feature v{fidx} near-zero std={raw_std[i]:.6f}")

    # Floor standard deviations at 0.01 to limit amplification of tiny values.
    MIN_STD_FLOOR = 0.01
    n_floored = int((global_std < MIN_STD_FLOOR).sum())
    global_std = np.where(global_std < MIN_STD_FLOOR, MIN_STD_FLOOR, global_std)
    print(f"  SSL z-score: mean [{global_mean.min():.4f}, {global_mean.max():.4f}], "
          f"std [{global_std.min():.4f}, {global_std.max():.4f}]")
    if n_floored > 0:
        floored_vars = [f"v{ACTIVE_SSL_FEATURES[i]}" for i in range(V) if raw_std[i] < MIN_STD_FLOOR]
        print(f"  Std floor: {n_floored} variates floored at std={MIN_STD_FLOOR}: {floored_vars}")

    # Evaluation must use the checkpoint's training normalization. Recomputed
    # statistics above serve only as a comparison when saved statistics are supplied.
    if z_stats is not None:
        ck_mean = np.asarray(z_stats["mean"], dtype=np.float64)
        ck_std = np.asarray(z_stats["std"], dtype=np.float64)
        _stored = z_stats.get("log1p_cols")
        if _stored is not None and sorted(int(c) for c in _stored) != log1p_cols:
            raise ValueError(f"checkpoint log1p_cols {sorted(int(c) for c in _stored)} "
                             f"!= computed feature positions {log1p_cols}")
        dm = float(np.nanmax(np.abs(ck_mean - global_mean)))
        dsd = float(np.nanmax(np.abs(ck_std - global_std)))
        print(f"  z-score: OVERRIDE with checkpoint stats; cross-check max|Δ| mean={dm:.4g} std={dsd:.4g}")
        global_mean, global_std = ck_mean, ck_std
        MIN_STD_FLOOR = float(z_stats.get("min_std_floor", MIN_STD_FLOOR))

    # Z-score and clip.
    X_normed = (X_all - global_mean) / global_std
    X_normed = np.clip(X_normed, -5.0, 5.0)

    return {
        "X_normed": X_normed,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "z_stats": {
            "mean": global_mean,
            "std": global_std,
            "log1p_cols": log1p_cols,
            "min_std_floor": MIN_STD_FLOOR,
        },
    }
