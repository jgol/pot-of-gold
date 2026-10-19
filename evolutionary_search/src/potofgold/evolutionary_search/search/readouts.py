"""Fit training-fold target residuals and embedding readouts for strategy search."""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

SVD_SOLVER = "full"
RANDOM_STATE = 42
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
EMB_COLS: List[str] = [
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
]
PROJECTION_COMPONENT_COUNTS: Dict[str, int] = {
    "EMB_VIX": 3,
    "EMB_STRIKE": 7,
    "EMB_FLOW_AGG": 14,
    "EMB_SHARED": 15,
    "EMB_SPX": 15,
    "EMB_GRID": 15,
    "EMB_FLOW_RAW": 15,
    "EMB_GRID_PUT": 12,
    "EMB_GRID_ATM": 12,
    "EMB_GRID_CALL": 12,
    "EMB_VAR_ATM_IV": 5,
    "EMB_VAR_ATM_SPREAD": 5,
    "EMB_VAR_SPX_RET": 5,
}


def canonicalize_signs(components: np.ndarray) -> np.ndarray:
    out = components.copy()
    for k in range(out.shape[0]):
        argmax_abs = int(np.argmax(np.abs(out[k])))
        if out[k, argmax_abs] < 0:
            out[k] = -out[k]
    return out


def _compute_effective_rank(arr: np.ndarray) -> float:
    centered = arr - arr.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigs = s**2 / max(centered.shape[0] - 1, 1)
    eigs = eigs[eigs > 0]
    if len(eigs) == 0:
        return 0.0
    p = eigs / eigs.sum()
    H = -float(np.sum(p * np.log(p)))
    return float(np.exp(H))


def _unit_scale_targets(Y):
    Y = np.asarray(Y, dtype=np.float64)
    s = Y.std(axis=0)
    return Y / np.where(s > 1e-12, s, 1.0)


def fit_target_residualizer(
    y: np.ndarray,
    X: np.ndarray,
    train_mask: np.ndarray,
    alphas: np.ndarray = RIDGE_ALPHAS,
) -> Optional[Dict]:
    """Fit the indicator-based target component on finite training rows only."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError(f"X {X.shape} not row-aligned to y {y.shape}")
    tr = (
        np.asarray(train_mask, dtype=bool)
        & np.isfinite(y)
        & np.all(np.isfinite(X), axis=1)
    )
    if int(tr.sum()) < X.shape[1] + 2:
        return None
    scaler = StandardScaler().fit(X[tr])
    model = RidgeCV(alphas=alphas).fit(scaler.transform(X[tr]), y[tr])
    return {"scaler": scaler, "model": model, "n_train": int(tr.sum())}


def apply_target_residual(
    y: np.ndarray, X: np.ndarray, fitted: Optional[Dict]
) -> np.ndarray:
    """Subtract the fitted indicator prediction without fitting on held-out rows."""
    y = np.asarray(y, dtype=np.float64)
    if fitted is None:
        return y.copy()
    X = np.asarray(X, dtype=np.float64)
    finite = np.all(np.isfinite(X), axis=1)
    Xf = np.where(finite[:, None], X, 0.0)
    pred = fitted["model"].predict(fitted["scaler"].transform(Xf))
    pred = np.where(finite, pred, 0.0)
    return y - pred


def assert_basis_fit_is_causal(
    train_mask: np.ndarray, test_mask: np.ndarray, df: pd.DataFrame
) -> None:
    train_mask = np.asarray(train_mask, dtype=bool)
    test_mask = np.asarray(test_mask, dtype=bool)
    if train_mask.shape != test_mask.shape or train_mask.shape[0] != len(df):
        raise ValueError(
            f"assert_basis_fit_is_causal: mask shapes {train_mask.shape}/{test_mask.shape} must both equal the df row count ({len(df)})."
        )
    if np.any(train_mask & test_mask):
        raise RuntimeError(
            f"Readout fitting and held-out masks overlap on {int(np.sum(train_mask & test_mask))} rows. "
            "Fit each fold's readouts using training rows only."
        )
    train_days = set(df.loc[train_mask, "date"].tolist())
    test_days = set(df.loc[test_mask, "date"].tolist())
    overlap_days = train_days & test_days
    if overlap_days:
        raise RuntimeError(
            f"Training and held-out masks share {len(overlap_days)} day(s), "
            f"including {sorted(map(str, overlap_days))[:3]}. "
            "Keep every day entirely within one partition when fitting readouts."
        )
    if not np.all(train_mask ^ test_mask):
        n_neither = int(np.sum(~(train_mask | test_mask)))
        raise RuntimeError(
            f"Training and held-out masks leave {n_neither} rows unassigned. "
            "Assign every row to exactly one partition."
        )


K_MAX_SUPERVISED = 7
B_MAX = K_MAX_SUPERVISED + 1


def _lean_k_for_group(eff_rank: float, n_targets: int) -> int:
    return int(
        max(1, min(K_MAX_SUPERVISED, int(n_targets), int(np.ceil(float(eff_rank)))))
    )


def fit_lean_vector_bases_masked(
    df: pd.DataFrame,
    targets: np.ndarray,
    targets_valid: np.ndarray,
    train_mask: np.ndarray,
    k_max: int = K_MAX_SUPERVISED,
    primary_target: Optional[np.ndarray] = None,
    primary_target_idx: int = 0,
) -> Dict:
    train_mask = np.asarray(train_mask, dtype=bool)
    targets_all = np.asarray(targets)
    targets_valid = np.asarray(targets_valid, dtype=bool)
    if train_mask.shape[0] != len(df):
        raise ValueError(f"train_mask len {train_mask.shape[0]} != df rows {len(df)}")
    if targets_all.shape[0] != len(df) or targets_valid.shape[0] != len(df):
        raise RuntimeError("targets / targets_valid must be row-aligned to df.")
    _n_targets = int(targets_all.shape[1])
    if _n_targets < 4:
        raise RuntimeError(f"Targets array has {_n_targets} columns, need >= 4.")
    if primary_target is None:
        primary_target = targets_all[:, primary_target_idx]
    primary_target = np.asarray(primary_target, dtype=np.float64)
    if primary_target.shape[0] != len(df):
        raise RuntimeError("primary_target must be row-aligned to df.")
    _cols_to_fit = [c for c in EMB_COLS if c in df.columns]
    train_idx = train_mask & targets_valid
    n_pls_fit = int(train_idx.sum())
    if n_pls_fit < 100:
        raise RuntimeError(
            f"Only {n_pls_fit} valid PLS training rows — too few for a stable lean fit."
        )
    Y_train = targets_all[train_idx].astype(np.float64)
    pt_train = primary_target[train_idx]
    if not np.all(np.isfinite(pt_train)):
        raise RuntimeError("primary_target has non-finite values on train rows.")
    (lo_thr, hi_thr) = np.percentile(pt_train, [100.0 / 3.0, 200.0 / 3.0])
    if hi_thr - lo_thr < 1e-09:
        raise RuntimeError(
            "primary_target is near-constant on this fold's train rows; the tercile centroids would be degenerate. Supply a harvestable target with cross-day spread."
        )
    top_mask = pt_train >= hi_thr
    bot_mask = pt_train <= lo_thr
    if int(top_mask.sum()) < 10 or int(bot_mask.sum()) < 10:
        raise RuntimeError(
            f"Tercile centroids need >= 10 train rows each; got top={int(top_mask.sum())} bot={int(bot_mask.sum())}. Primary target may be near-constant on this fold."
        )
    result: Dict = {
        "_train_end": None,
        "_n_fit_windows": int(train_mask.sum()),
        "_n_pls_fit_windows": n_pls_fit,
        "_method": "lean_vector",
        "_n_pls_directions": _n_targets,
        "_fit_mode": "per_fold_masked",
        "_k_max_supervised": int(k_max),
        "_svd_solver": SVD_SOLVER,
        "_random_state": RANDOM_STATE,
    }
    for col in _cols_to_fit:
        arr_all = np.stack(df[col].tolist()).astype(np.float64)
        arr_train_full = arr_all[train_mask]
        arr_train_pls = arr_all[train_idx]
        if np.any(np.isnan(arr_train_full)):
            raise RuntimeError(
                f"NaN in {col} training embeddings; cannot fit lean basis."
            )
        eff_rank = _compute_effective_rank(arr_train_full)
        K = _lean_k_for_group(eff_rank, min(_n_targets, int(k_max)))
        train_mean = arr_train_full.mean(axis=0)
        X_centered_pls = arr_train_pls - train_mean
        X_centered_full = arr_train_full - train_mean
        if float(np.max(np.abs(X_centered_full))) < 1e-09:
            raise RuntimeError(
                f"Embedding group {col!r} is constant on train rows (rank-0); cannot fit a lean basis (check encoder output)."
            )
        pls = PLSRegression(n_components=K, scale=False, max_iter=500)
        pls.fit(X_centered_pls, _unit_scale_targets(Y_train))
        (Q, _R) = np.linalg.qr(pls.x_rotations_.copy(), mode="reduced")
        pls_ortho = Q.T
        pca_anchor = PCA(
            n_components=1, svd_solver=SVD_SOLVER, random_state=RANDOM_STATE
        )
        pca_anchor.fit(X_centered_full)
        dir_bank = canonicalize_signs(np.vstack([pls_ortho, pca_anchor.components_]))
        components = dir_bank[:K]
        bank_projections = X_centered_full @ dir_bank.T
        dir_scale = bank_projections.std(axis=0)
        if not np.all(np.isfinite(dir_scale)) or np.any(dir_scale <= 0.0):
            raise RuntimeError(
                f"Lean bank direction std for {col!r} invalid: {dir_scale.tolist()}."
            )
        if np.any(dir_scale < 1e-06):
            deg = np.where(dir_scale < 1e-06)[0]
            raise RuntimeError(
                f"Lean basis for {col} near-zero std on bank rows {deg.tolist()} (rank-deficient embedding — check encoder quality)."
            )
        proto_bank = np.vstack(
            [arr_train_pls[top_mask].mean(axis=0), arr_train_pls[bot_mask].mean(axis=0)]
        )
        pseudo_ev = (dir_scale[:K] ** 2).astype(np.float64)
        result[col] = {
            "components": components.astype(np.float32),
            "mean_": train_mean.astype(np.float32),
            "explained_variance_": pseudo_ev.astype(np.float32),
            "explained_variance_ratio": [],
            "n_features_in_": 384,
            "k_used": int(K),
            "effective_rank": float(eff_rank),
            "n_pls_directions": int(K),
            "n_residual_pca": 0,
            "pc_std_": np.sqrt(pseudo_ev + 1e-08).astype(np.float32),
            "dir_bank_": dir_bank.astype(np.float32),
            "dir_mean_": train_mean.astype(np.float32),
            "dir_scale_": dir_scale.astype(np.float32),
            "proto_bank_": proto_bank.astype(np.float32),
            "proto_mean_": train_mean.astype(np.float32),
            "n_bank": int(dir_bank.shape[0]),
            "n_proto": int(proto_bank.shape[0]),
        }
    return result
