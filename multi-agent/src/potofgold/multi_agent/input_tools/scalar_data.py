"""Scalar replay defaults and the option-grid input check."""
import os

from potofgold.evolutionary_search.search.simulator import GRID_IV_COLUMNS

DECISION_OFFSET_REAL = 2

N_FOLDS = 5

FOLD_PARTITION_SEED = 0

def assert_grid_present(df) -> None:
    """Require the option-IV grid used to calibrate the pricing proxy.

    Without these columns, the evaluator would use a different pricing path
    with credit factors calibrated for the grid-based calculation.
    """
    if os.environ.get("GP_V4_CLEAN_DATA", "0") != "0":
        raise ValueError("GP_V4_CLEAN_DATA is not supported by the released experiment; "
                         "use the shared evolutionary_search normalization")
    missing = [c for c in GRID_IV_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"[scalar_data] option-IV grid absent ({missing}) -> the evaluator would use "
                         f"the BS fallback, where _BASE_CREDIT_FACTORS are mis-calibrated (audit Issue #2). "
                         f"Use a parquet carrying the IV grid (e.g. l1_minute_scalars).")


def assert_gamma_present(df) -> None:
    """Require the gamma column supplied by evolutionary_search market preparation."""
    if "NetGammaProxy" not in df.columns:
        raise ValueError("Supply NetGammaProxy in the market parquet using "
                         "evolutionary_search input preparation")
