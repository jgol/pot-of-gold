"""Prepare financial evaluation targets using statistics fitted on training rows."""
import numpy as np

from ..encoder.config import ACTIVE_SSL_FEATURES
from ..encoder.data import prepare_input_splits
from ..encoder.forward_split import build_residualizer_scalars


def compute_train_target_stats(targets, valid, train_mask):
    """Compute target means and standard deviations from valid, finite training rows.
    
    targets and valid are (N, n_targets); train_mask is (N,). Fewer than two
    usable rows gives (0, 1). Constant targets use a standard deviation of 1.
    """
    targets = np.asarray(targets, dtype=np.float64)
    valid = np.asarray(valid).astype(bool)
    train_mask = np.asarray(train_mask).astype(bool)
    n = targets.shape[1]
    mean = np.zeros(n, dtype=np.float64)
    std = np.ones(n, dtype=np.float64)
    for j in range(n):
        col = targets[train_mask & valid[:, j], j]
        col = col[np.isfinite(col)]
        if col.size >= 2:
            mean[j] = float(col.mean())
            s = float(col.std())
            std[j] = s if s > 1e-8 else 1.0
    return mean, std


def standardize_targets(targets, mean, std):
    """Standardize targets by column; evaluation uses validity masks to exclude invalid rows."""
    return (np.asarray(targets, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) \
        / np.asarray(std, dtype=np.float64)


# Indicator normalization and residualization fitted on training rows.
def train_scalar_stats(scalars, train_mask):
    """Compute indicator means and standard deviations from finite training values.
    
    Constant columns use a standard deviation of 1.
    """
    scalars = np.asarray(scalars, dtype=np.float64)
    train_mask = np.asarray(train_mask).astype(bool)
    m = scalars.shape[1]
    mean = np.zeros(m, dtype=np.float64)
    std = np.ones(m, dtype=np.float64)
    for j in range(m):
        col = scalars[train_mask, j]
        col = col[np.isfinite(col)]
        if col.size >= 2:
            mean[j] = float(col.mean())
            s = float(col.std())
            std[j] = s if s > 1e-8 else 1.0
    return mean, std


def fit_scalar_residualizer(targets, scalars, valid, train_mask, scalar_mean, scalar_std):
    """Fit each target on standardized indicators and an intercept using training rows.
    
    Returns (n_scalars+1, n_targets) coefficients. A target with fewer than
    n_scalars+2 valid, finite rows gets zero coefficients.
    """
    targets = np.asarray(targets, dtype=np.float64)
    scalars = np.asarray(scalars, dtype=np.float64)
    valid = np.asarray(valid).astype(bool)
    train_mask = np.asarray(train_mask).astype(bool)
    z = (scalars - np.asarray(scalar_mean)) / np.asarray(scalar_std)
    m = z.shape[1]
    n_t = targets.shape[1]
    coef = np.zeros((m + 1, n_t), dtype=np.float64)
    for j in range(n_t):
        # Target validity can differ by column; all predictors must be finite
        # on each fitted row. No validation or test observations enter this fit.
        rows = (train_mask & valid[:, j] & np.isfinite(targets[:, j])
                & np.all(np.isfinite(z), axis=1))
        if int(rows.sum()) < m + 2:
            continue                                   # Zero coefficients leave this target unchanged.
        # The final coefficient is the intercept; keep this column order on apply.
        X = np.concatenate([z[rows], np.ones((int(rows.sum()), 1))], axis=1)
        beta, *_ = np.linalg.lstsq(X, targets[rows, j], rcond=None)
        coef[:, j] = beta
    return coef


def apply_scalar_residual(targets, scalars, coef, scalar_mean, scalar_std):
    """Subtract indicator predictions using fitted coefficients.
    
    Rows with missing indicators keep their original targets. The caller masks
    invalid targets; a placeholder value must not become an observed label.
    """
    targets = np.asarray(targets, dtype=np.float64)
    scalars = np.asarray(scalars, dtype=np.float64)
    z = (scalars - np.asarray(scalar_mean)) / np.asarray(scalar_std)
    finite = np.all(np.isfinite(z), axis=1)
    zf = np.where(finite[:, None], z, 0.0)
    X = np.concatenate([zf, np.ones((zf.shape[0], 1))], axis=1)
    pred = X @ coef
    pred[~finite] = 0.0                                # Missing indicators leave the target unchanged.
    return targets - pred


def make_evaluation_splits(data, embargo_days=0, eval_window="val",
                            date_overrides=None, z_stats=None):
    """Return (inputs, targets, validity) arrays for train and held-out dates.

    Inputs and targets retain the experiment's float32 storage boundary; validity
    is boolean. Residualization and standardization fit training observations
    only. The "val" entry contains the requested validation or test period.
    """
    inputs = prepare_input_splits(data, embargo_days=embargo_days, eval_window=eval_window,
                                 date_overrides=date_overrides, z_stats=z_stats)
    X_normed = inputs["X_normed"]
    train_mask, val_mask = inputs["train_mask"], inputs["val_mask"]

    # The evaluator scores the first four targets. Anchor IV remains available
    # in the raw targets for the corpus units diagnostic.
    fwd_all = np.asarray(data["fwd_predictive"])[:, :4]
    fwd_Tvalid = np.repeat(np.asarray(data["fwd_predictive_valid"])[:, None], 4, axis=1)
    n_tr_valid = int(fwd_Tvalid[train_mask].any(axis=1).sum())
    print(f"  forward targets: train-valid windows={n_tr_valid}, embargo_days={embargo_days}")

    # Fit the residualizer on training rows, apply its fixed coefficients to
    # every split, then standardize the residuals with training-only statistics.
    # Indicators use the same normalized scale as encoder inputs. These are
    # observable financial targets, separate from the trainer's latent targets.
    scalars, reg_names = build_residualizer_scalars(X_normed, ACTIVE_SSL_FEATURES)
    s_mean, s_std = train_scalar_stats(scalars, train_mask)
    coef = fit_scalar_residualizer(fwd_all, scalars, fwd_Tvalid, train_mask, s_mean, s_std)
    fwd_all = apply_scalar_residual(fwd_all, scalars, coef, s_mean, s_std)
    print(f"  Forward targets residualized against {len(reg_names)} indicator variables: {reg_names}")
    t_mean, t_std = compute_train_target_stats(fwd_all, fwd_Tvalid, train_mask)
    fwd_T_std = standardize_targets(fwd_all, t_mean, t_std)
    splits = {}
    for name, mask in (("train", train_mask), ("val", val_mask)):
        splits[name] = (X_normed[mask].astype(np.float32),
                        fwd_T_std[mask].astype(np.float32),
                        fwd_Tvalid[mask].astype(bool))
        print(f"  {name}: {int(mask.sum())} windows")
    return splits
