"""Load minute panels and expose aligned scalar and embedding terminals."""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

_IV_REPAIR_COLS = (
    "ATM_IV",
    "IV_5Dp",
    "IV_10Dp",
    "IV_25Dp",
    "IV_40Dp",
    "IV_ATMp",
    "IV_ATM",
    "IV_ATMc",
    "IV_40Dc",
    "IV_25Dc",
    "IV_10Dc",
    "IV_5Dc",
)


def repair_corrupt_iv_grid(
    df: pd.DataFrame, mult: Optional[float] = None, verbose: bool = True
) -> pd.DataFrame:
    if mult is None:
        mult = float(os.environ.get("GP_MARK_IV_VIX_MULT", "4"))
    if mult <= 0 or "VIXSpot" not in df.columns:
        return df
    vix = df["VIXSpot"].to_numpy(dtype=np.float64) / 100.0
    valid = vix > 0.01
    cap = mult * vix
    n_cells = 0
    cols_hit: List[str] = []
    _skew25_changed = np.zeros(len(df), dtype=bool)
    for col in _IV_REPAIR_COLS:
        if col not in df.columns:
            continue
        v = df[col].to_numpy(dtype=np.float64).copy()
        m = valid & (v > cap)
        k = int(m.sum())
        if k:
            v[m] = vix[m]
            df[col] = v.astype(df[col].dtype, copy=False)
            n_cells += k
            cols_hit.append(f"{col}:{k}")
            if col in ("IV_25Dp", "IV_25Dc"):
                _skew25_changed |= m
    if _skew25_changed.any() and {"PutCallSkew", "IV_25Dp", "IV_25Dc"} <= set(
        df.columns
    ):
        pcs = df["PutCallSkew"].to_numpy(dtype=np.float64).copy()
        p25 = df["IV_25Dp"].to_numpy(dtype=np.float64)
        c25 = df["IV_25Dc"].to_numpy(dtype=np.float64)
        pcs[_skew25_changed] = p25[_skew25_changed] - c25[_skew25_changed]
        df["PutCallSkew"] = pcs.astype(df["PutCallSkew"].dtype, copy=False)
        cols_hit.append(f"PutCallSkew(recomp):{int(_skew25_changed.sum())}")
    if verbose and n_cells:
        print(
            f"[iv-repair] replaced {n_cells} non-physical IV cells (> {mult:g}x VIX) with market VIX across {len(cols_hit)} col(s): {', '.join(cols_hit)}",
            flush=True,
        )
    return df


TYPED_VECTOR_COLUMNS: Tuple[str, ...] = (
    "EMB_SHARED",
    "EMB_GRID",
    "EMB_STRIKE",
    "EMB_SPX",
    "EMB_VIX",
    "EMB_FLOW_RAW",
    "EMB_FLOW_AGG",
    "EMB_GRID_PUT",
    "EMB_GRID_ATM",
    "EMB_GRID_CALL",
    "EMB_VAR_ATM_IV",
    "EMB_VAR_ATM_SPREAD",
    "EMB_VAR_SPX_RET",
)
PROBE_SCALAR_COLUMNS: Tuple[str, ...] = (
    "PredRV15",
    "PredRV30",
    "PredRegime",
    "PredSpread",
)
PROBE_SCALAR_COLUMNS_V2: Tuple[str, ...] = PROBE_SCALAR_COLUMNS + (
    "PredGammaAccel",
    "PredSmileConvexity",
    "PredJump",
    "PredFlowToxicity",
)
BYPASS_SCALAR_COLUMNS: Tuple[str, ...] = (
    "RawSpread",
    "DeltaSpread1",
    "DeltaSpread5",
    "ATM_IV",
    "MinutesToClose",
    "VIXSpot",
    "VIXTermSlope",
    "RealizedVol30m",
)
REGIME_PROB_COLUMNS: Tuple[str, ...] = tuple((f"RegimeProb{k}" for k in range(4)))
PRICE_COLUMN = "SPXClose"


@dataclass
class L1ParquetSchema:
    n_rows: int
    n_dates: int
    typed_vector_dim: int
    has_typed_vectors: bool
    has_probe_scalars: bool
    has_bypass_scalars: bool
    has_price_column: bool


def load_minute_parquet(path: str | Path) -> Tuple[pd.DataFrame, L1ParquetSchema]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Minute Parquet not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    if "date" not in df.columns:
        raise ValueError(f"Minute Parquet missing 'date' column: {list(df.columns)}")
    if PRICE_COLUMN not in df.columns:
        raise ValueError(f"Minute Parquet missing price column '{PRICE_COLUMN}'")
    if df[PRICE_COLUMN].isna().any():
        n_nan = int(df[PRICE_COLUMN].isna().sum())
        raise ValueError(f"Minute Parquet has {n_nan} NaN value(s) in '{PRICE_COLUMN}'")
    present_bypass = [c for c in BYPASS_SCALAR_COLUMNS if c in df.columns]
    if not present_bypass:
        raise ValueError(
            f"Minute Parquet has no bypass scalar columns. Expected some of: {BYPASS_SCALAR_COLUMNS}"
        )
    present_vecs = [c for c in TYPED_VECTOR_COLUMNS if c in df.columns]
    present_probes = [c for c in PROBE_SCALAR_COLUMNS if c in df.columns]
    vec_dim = 0
    if present_vecs:
        sample = df[present_vecs[0]].iloc[0]
        if isinstance(sample, (list, np.ndarray)):
            vec_dim = len(sample)
        for col in present_vecs:
            df[col] = df[col].apply(
                lambda x, _d=vec_dim: (
                    np.asarray(x, dtype=np.float32)
                    if isinstance(x, (list, np.ndarray))
                    else np.zeros(_d, dtype=np.float32)
                )
            )
    repair_corrupt_iv_grid(df)
    schema = L1ParquetSchema(
        n_rows=len(df),
        n_dates=df["date"].nunique(),
        typed_vector_dim=vec_dim,
        has_typed_vectors=len(present_vecs) > 0,
        has_probe_scalars=len(present_probes) > 0,
        has_bypass_scalars=True,
        has_price_column=True,
    )
    return (df, schema)
