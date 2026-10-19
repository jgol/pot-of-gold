"""Column order, alignment checks and the historical IV repair contract."""
import hashlib
import json
from pathlib import Path

import numpy as np

HISTORICAL_CHECKPOINT_SHA256 = "93a428ebe0fd8bbb29f9adbed31470046037d155389a6fd28fc00aaf7d62a6ca"
METADATA_KEY = b"potofgold.evolutionary_search.inputs"
GRID_NAMES = ("5Dp", "10Dp", "25Dp", "40Dp", "ATMp", "ATM", "ATMc", "40Dc", "25Dc", "10Dc", "5Dc")
GRID_IV_COLUMNS = tuple(f"IV_{name}" for name in GRID_NAMES)
GRID_IV_INDICES = tuple(1 + 8 * i for i in range(11))
GROUP_RANGES = {
    "EMB_GRID": (0, 88), "EMB_STRIKE": (88, 94), "EMB_SPX": (94, 108),
    "EMB_VIX": (108, 120), "EMB_FLOW_RAW": (120, 129), "EMB_FLOW_AGG": (129, 147),
    "EMB_GRID_PUT": (0, 32), "EMB_GRID_ATM": (32, 56), "EMB_GRID_CALL": (56, 88),
}
VARIATE_READOUTS = {"EMB_VAR_ATM_IV": 41, "EMB_VAR_ATM_SPREAD": 46, "EMB_VAR_SPX_RET": 105}
EMBEDDING_COLUMNS = ("EMB_SHARED", *GROUP_RANGES, *VARIATE_READOUTS)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_metadata(array):
    return json.loads(bytes(np.asarray(array, dtype=np.uint8)).decode("utf-8"))


def validate_keys(frame, label="market"):
    """Require real ISO dates and unique, nonnegative integer minute positions."""
    import pandas as pd
    if not {"date", "bar_position"} <= set(frame.columns):
        raise ValueError(f"{label} requires date and bar_position")
    frame = frame.copy()
    frame["date"] = frame["date"].astype(str)
    parsed = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    if not (parsed.dt.strftime("%Y-%m-%d") == frame["date"]).all():
        raise ValueError(f"{label} dates must use YYYY-MM-DD")
    bars = pd.to_numeric(frame["bar_position"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(bars).all() or (bars < 0).any() or (bars != np.floor(bars)).any():
        raise ValueError(f"{label} bar_position must be finite nonnegative integers")
    frame["bar_position"] = bars.astype(np.int64)
    if frame.duplicated(["date", "bar_position"]).any():
        raise ValueError(f"{label} contains duplicate date/bar_position keys")
    return frame.sort_values(["date", "bar_position"], kind="stable").reset_index(drop=True)


def repair_raw_iv(day, mult):
    """Repair raw grid IV in place before encoding; VIX is in volatility points."""
    if mult <= 0:
        return 0
    vix = day[:, 119].astype(np.float64) / 100.0
    count = 0
    for index in GRID_IV_INDICES:
        bad = (vix > 0.01) & (day[:, index] > mult * vix)
        count += int(bad.sum())
        day[bad, index] = vix[bad]
    return count


def repair_market_iv(frame, mult):
    """Use the same rule for GP scalar terminals, pricing and target construction."""
    if mult <= 0:
        return 0
    vix = frame["VIXSpot"].to_numpy(dtype=np.float64) / 100.0
    count = 0
    changed_skew = np.zeros(len(frame), dtype=bool)
    for name in ("ATM_IV", *GRID_IV_COLUMNS):
        values = frame[name].to_numpy(dtype=np.float64, copy=True)
        bad = (vix > 0.01) & (values > mult * vix)
        count += int(bad.sum())
        if name in ("IV_25Dp", "IV_25Dc"):
            changed_skew |= bad
        values[bad] = vix[bad]
        frame[name] = values
    if changed_skew.any():
        frame.loc[changed_skew, "PutCallSkew"] = (frame.loc[changed_skew, "IV_25Dp"]
                                                  - frame.loc[changed_skew, "IV_25Dc"])
    return count
