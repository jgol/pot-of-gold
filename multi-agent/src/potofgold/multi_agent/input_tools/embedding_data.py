"""Load embedding inputs and fit the recorded projections using training rows only."""

from __future__ import annotations

import os

import numpy as np

from potofgold.evolutionary_search.search.data import load_minute_parquet, TYPED_VECTOR_COLUMNS
from potofgold.evolutionary_search.search.templates import base_template_by_name
from potofgold.evolutionary_search.search.simulator import prepare_terminal_data
from potofgold.evolutionary_search.search.validation import kfold_day_split, slice_by_mask
from potofgold.evolutionary_search.search.readouts import (
    fit_lean_vector_bases_masked, fit_target_residualizer, apply_target_residual,
    assert_basis_fit_is_causal, PROJECTION_COMPONENT_COUNTS,
)
from potofgold.evolutionary_search.search.pipeline import (
    _scalar_baseline, _min_realized_bank, _lean_k_provenance, load_targets, file_sha256,
)
from potofgold.multi_agent.input_tools.scalar_data import assert_grid_present

DECISION_OFFSET_REAL = 2

N_FOLDS = 5

FOLD_PARTITION_SEED = 0

_EMB_GROUPS = TYPED_VECTOR_COLUMNS

_EMB_LIVENESS_MIN_CROSS_DAY_STD = 0.05      # Minimum cross-day variation for a usable embedding group.

_EMB_LIVENESS_MIN_INTRADAY_NZ_FRAC = 0.005  # Minimum within-day fraction of changing embedding values.

def assert_emb_live(df, *, fatal: bool = True) -> list:
    """Check for missing or constant embedding inputs before fitting readouts."""
    missing = [g for g in _EMB_GROUPS if g not in df.columns]
    if missing:
        raise SystemExit(f"[embedding_data] EMB groups absent ({missing}) -> the strategy has no embedding "
                         f"inputs. Supply all 13 EMB_* columns.")
    dcol = df["date"].values
    udays = list(dict.fromkeys(dcol.tolist()))
    dead = []
    for g in _EMB_GROUPS:
        arr = np.stack(df[g].tolist()).astype(np.float64)
        # Measure variation across daily mean vectors.
        day_means = np.stack([arr[dcol == d].mean(axis=0) for d in udays])
        cross_day_std = float(day_means.std(axis=0).max()) if len(udays) > 1 else 0.0
        # Use the first day to measure update cadence without overnight jumps.
        d0 = arr[dcol == udays[0]]
        nz_frac = float((np.abs(np.diff(d0, axis=0)).sum(axis=1) > 1e-9).mean()) if d0.shape[0] > 1 else 0.0
        if cross_day_std < _EMB_LIVENESS_MIN_CROSS_DAY_STD or nz_frac < _EMB_LIVENESS_MIN_INTRADAY_NZ_FRAC:
            dead.append(f"{g}(cross_day_std={cross_day_std:.3f}, intraday_nz={nz_frac:.4f})")
    if dead:
        _m = "[embedding_data] Low raw embedding variation: " + "; ".join(dead)
        if fatal:
            raise SystemExit(_m)
        print(_m, flush=True)
    return dead

_D1_RESIDUALIZE = os.environ.get("GP_D1_RESIDUALIZE", "0") == "1"

_D1_SCOPE = os.environ.get("GP_D1_RESIDUALIZE_SCOPE", "all").strip().lower()

if _D1_RESIDUALIZE and _D1_SCOPE not in ("all", "primary"):
    raise SystemExit(f"[embedding_data] GP_D1_RESIDUALIZE_SCOPE must be 'all' or 'primary', got {_D1_SCOPE!r}")

_D1_ESTIMATOR = os.environ.get("GP_D1_ESTIMATOR", "ridge").strip().lower()

if _D1_RESIDUALIZE and _D1_ESTIMATOR != "ridge":
    raise SystemExit(f"[embedding_data] GP_D1_ESTIMATOR must be 'ridge' for the released experiment, got {_D1_ESTIMATOR!r}")

_D1_SURFACE_CACHE: dict = {}

def _d1_priced_surface(df):
    """Cache the normalized indicator surface used for training-only target residualization."""
    key = id(df)
    surf = _D1_SURFACE_CACHE.get(key)
    if surf is None:
        surf = _scalar_baseline(df)
        _D1_SURFACE_CACHE[key] = surf
    return surf


def _residualize_targets_for_basis(targets, surface, train_mask, scope, primary_target_idx,
                                   estimator="ridge"):
    """Remove the indicator-predictable target component using training rows only."""
    if scope not in ("all", "primary"):
        raise ValueError(f"Unsupported residualization scope: {scope!r}")
    if estimator != "ridge":
        raise ValueError(f"Unsupported residualizer {estimator!r}; the released experiment uses ridge")
    out = np.array(targets, dtype=np.float64, copy=True)
    cols = [int(primary_target_idx)] if scope == "primary" else range(out.shape[1])
    for j in cols:
        fitted = fit_target_residualizer(out[:, j], surface, train_mask)
        out[:, j] = apply_target_residual(out[:, j], surface, fitted)
        if fitted is None:
            import warnings
            warnings.warn(f"Target {j} has too few finite training rows for ridge residualization; "
                          "its raw values are retained", RuntimeWarning)
    return out


def _fit_fold_basis(df, train_mask, basis_method, targets, targets_valid, primary_target_idx=0):
    """Fit shared evolutionary-search readouts on this fold's training rows and record their settings."""
    n_rows = int(np.asarray(train_mask, bool).sum())
    n_days = int(df.loc[np.asarray(train_mask, bool), "date"].nunique())
    if basis_method != "lean":
        raise ValueError(f"Unsupported basis {basis_method!r}; the released experiment uses lean readouts")
    if targets is None or targets_valid is None:
        raise ValueError("Lean readouts require minute-aligned targets and a valid-row mask")
    if _D1_RESIDUALIZE:
        targets = _residualize_targets_for_basis(
            targets, _d1_priced_surface(df), train_mask, _D1_SCOPE, primary_target_idx,
            estimator=_D1_ESTIMATOR)
    bases = fit_lean_vector_bases_masked(df, targets, targets_valid, train_mask,
                                         primary_target_idx=primary_target_idx)
    info = {
        "basis_method": basis_method, "n_basis_fit_rows": n_rows, "n_basis_fit_days": n_days,
        "per_group_k": dict(PROJECTION_COMPONENT_COUNTS), "basis_causal": True,
        "d1_residualize": bool(_D1_RESIDUALIZE),
        "d1_residualize_scope": _D1_SCOPE if _D1_RESIDUALIZE else None,
        "d1_estimator": _D1_ESTIMATOR if _D1_RESIDUALIZE else None,
        "emb_leaf_frac": os.environ.get("GP_EMB_LEAF_FRAC"),
        "emb_gen0_floor": os.environ.get("GP_EMB_GEN0_FLOOR"),
        "lean_proto_nonlinear": os.environ.get("GP_LEAN_PROTO_NONLINEAR"),
        "emb_bank_cap": _min_realized_bank(bases),
        **_lean_k_provenance(bases, int(np.asarray(targets).shape[1])),
    }
    return bases, info

def _load_minute_targets(path, df):
    """Load row-aligned targets, checking their dates, bars and primary target when supplied."""
    if not path:
        return None, None, 0
    n_rows = len(df)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    names = [str(x) for x in data.get("target_names", [])]
    if names and not any(name.startswith("T1_vrp") for name in names):
        return load_targets(path, df)
    targets = data["targets"]
    valid = data["valid"].astype(bool)
    if targets.shape[0] != n_rows or valid.shape[0] != n_rows:
        raise SystemExit(f"[embedding_data] targets {targets.shape}/{valid.shape} not row-aligned to "
                         f"the {n_rows}-row parquet. Rebuild minute-aligned targets (D-1 arm b).")
    keys = set(data)
    if {"date", "bar_position"} <= keys:
        if "date" not in df.columns or "bar_position" not in df.columns:
            raise SystemExit("[embedding_data] parquet lacks date/bar_position — cannot assert target alignment.")
        t_dates = np.asarray(data["date"]).astype(str)
        t_bars = np.asarray(data["bar_position"]).astype(np.int64)
        if not (np.array_equal(t_dates, df["date"].astype(str).to_numpy())
                and np.array_equal(t_bars, df["bar_position"].to_numpy().astype(np.int64))):
            raise SystemExit("[embedding_data] targets (date,bar_position) index does NOT match the parquet "
                             "row order. Regenerate targets for this parquet.")
    primary_idx = int(data["primary_target_idx"]) if "primary_target_idx" in keys else 0
    if "target_names" in keys:
        names = [str(x) for x in np.asarray(data["target_names"])]
        if not (0 <= primary_idx < len(names)) or not names[primary_idx].startswith("T1_vrp"):
            raise SystemExit(f"[embedding_data] primary target '{names[primary_idx] if names else '?'}' is not "
                             f"the VRP residual (T1_vrp*) — a priced/reordered target set would tercile on the "
                             f"wrong axis (spec §1.4).")
    return targets, valid, primary_idx

def _load_and_prepare(data_path):
    """Prepare the ordered corpus once; callers slice it into training folds."""
    from potofgold.multi_agent.diagnostics.calendar_features import add_calendar_columns
    from potofgold.multi_agent.input_tools.scalar_data import assert_gamma_present

    df, _ = load_minute_parquet(data_path)
    df.attrs["prepared_parquet_sha256"] = file_sha256(data_path)
    assert_grid_present(df)
    assert_emb_live(df, fatal=False)
    add_calendar_columns(df)
    assert_gamma_present(df)
    return df, prepare_terminal_data(df), N_FOLDS
