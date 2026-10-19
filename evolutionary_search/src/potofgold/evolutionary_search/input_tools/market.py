"""Join a raw feature panel to explicit market prices, without fitting anything."""
import numpy as np

from .schema import GRID_IV_COLUMNS, GRID_IV_INDICES, repair_market_iv


def market_day(raw, supplied, *, previous_close=None, iv_repair_mult=4.0):
    """Supply GP scalars from raw columns, preserving named market overrides.

    SPXClose is required externally: absolute prices cannot be recovered from
    session returns alone. A supplied column takes precedence over a derived
    scalar, which permits exact reuse of reconciled historical market inputs.
    No embedding column is accepted through this path.
    """
    n = len(raw)
    if raw.ndim != 2 or raw.shape[1] < 159 or not np.isfinite(raw[:, :159]).all():
        raise ValueError("Expected finite raw daily features with at least 159 columns")
    if len(supplied) != n or not np.array_equal(supplied["bar_position"].to_numpy(), np.arange(n)):
        raise ValueError("Market rows must match every corpus bar, numbered zero through day length minus one")
    if "SPXClose" not in supplied:
        raise ValueError("Market input requires absolute SPXClose prices")
    if any(str(name).startswith("EMB_") for name in supplied):
        raise ValueError("Use --embeddings to reuse vectors; market input must contain only market data")
    price = supplied["SPXClose"].to_numpy(dtype=np.float64)
    if not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("SPXClose must be finite and positive")
    # A wrong date/row join can retain valid-looking prices. Session returns must
    # describe the same path; float16 corpus quantization permits a small error.
    offset = np.log(price) - raw[:, 105].astype(np.float64)
    if np.ptp(offset) > 0.002:
        raise ValueError("Market price changes disagree with corpus session returns; check row alignment")
    spread = raw[:, 46].astype(np.float64).copy()
    for bar in range(n):
        if spread[bar] > 0.035:
            spread[bar] = spread[bar - 1] if bar else 0.035
    delta1 = np.zeros(n)
    delta5 = np.zeros(n)
    delta1[1:] = np.diff(spread)
    delta5[5:] = spread[5:] - spread[:-5]
    # v112 stores unannualized Parkinson standard deviation, not variance.
    rv = raw[:, 112].astype(np.float64).copy()
    hl2 = np.square(np.maximum(raw[:, 106].astype(np.float64), 0.0))
    for bar in range(n):
        if rv[bar] <= 1e-9:
            rv[bar] = np.sqrt(hl2[max(0, bar - 29):bar + 1].mean() / (4 * np.log(2)))
    mtc = raw[:, 114].astype(np.float64).copy()
    if n > 1 and mtc[0] == 0 and mtc[1] > 0:
        mtc[0] = mtc[1] + 1
    high, low = np.maximum.accumulate(price), np.minimum.accumulate(price)
    position = np.divide(price - low, high - low, out=np.full(n, 0.5), where=(high - low) > 0.01)
    opening = price[0] / np.exp(float(raw[0, 105]))
    gap = np.log(opening / previous_close) if previous_close is not None else 0.0
    vix = raw[:, 119].astype(np.float64)
    gamma = raw[:, 137].astype(np.float64).copy()  # already signed-log1p in the corpus
    if iv_repair_mult > 0:
        corrupt = (vix / 100 > 0.01) & (raw[:, 41] > iv_repair_mult * vix / 100)
        gamma[corrupt] = 0.0
    derived = dict(
        ATM_IV=np.minimum(raw[:, 41].astype(np.float64), 0.8), RawSpread=spread,
        DeltaSpread1=delta1, DeltaSpread5=delta5, MinutesToClose=mtc, VIXSpot=vix,
        VIXTermSlope=raw[:, 120].astype(np.float64) - vix, RealizedVol30m=rv,
        BarOfDay=np.arange(n), PutCallSkew=raw[:, 17].astype(float) - raw[:, 65].astype(float),
        SessionPosition=position, OvernightGap=np.full(n, gap),
        GridReliability=(raw[:, 88:99] > 0).mean(axis=1), NetGammaProxy=gamma,
    )
    derived.update({name: raw[:, index].astype(np.float64) for name, index in zip(GRID_IV_COLUMNS, GRID_IV_INDICES)})
    frame = supplied.copy().reset_index(drop=True)
    for name, values in derived.items():
        if name not in frame:
            frame[name] = values
    repaired = repair_market_iv(frame, iv_repair_mult)
    required = ["SPXClose", *derived]
    values = frame[required].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Prepared market inputs contain non-finite values")
    if (frame["ATM_IV"] <= 0).any() or (frame["VIXSpot"] <= 0).any():
        raise ValueError("ATM_IV and VIXSpot must be positive")
    if (frame["MinutesToClose"] < 0).any() or (np.diff(frame["MinutesToClose"]) > 0).any():
        raise ValueError("MinutesToClose must be nonnegative and nonincreasing within a day")
    if (frame["RawSpread"] < 0).any():
        raise ValueError("RawSpread must be nonnegative")
    return frame, repaired
