"""Forecast skill, embedding geometry, and metrics for whole-day resampling.

Forecast metrics pool valid rows; geometry metrics use rows as observations
and columns as embedding dimensions.
"""
import numpy as np


def _mask(pred, tgt, valid):
    """Select finite prediction/target pairs from aligned row arrays and a mask."""
    pred = np.asarray(pred, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    m = np.asarray(valid).astype(bool) & np.isfinite(pred) & np.isfinite(tgt)
    return pred[m], tgt[m]


def rank_ic(pred, tgt, valid):
    """Spearman correlation: agreement in predicted and observed ordering.

    Return NaN with fewer than three usable rows or either input constant.
    """
    from scipy.stats import spearmanr
    p, t = _mask(pred, tgt, valid)
    if p.size < 3 or np.all(p == p[0]) or np.all(t == t[0]):
        return float("nan")
    return float(spearmanr(p, t).correlation)


def ic(pred, tgt, valid):
    """Pearson correlation: linear association on finite, valid row pairs.

    Return NaN with fewer than three usable rows; constant inputs also give NaN.
    """
    from scipy.stats import pearsonr
    p, t = _mask(pred, tgt, valid)
    if p.size < 3:
        return float("nan")
    return float(pearsonr(p, t)[0])


def rankme(Z, eps=1e-7):
    """Entropy-based effective rank of centered (N, D) embeddings.

    Exponentiate the entropy of singular values normalized by their sum plus
    eps. A flatter spectrum means more active directions. This uses singular
    values, not the squared values used by covariance-based metrics. An
    all-zero spectrum returns 1 under this convention.
    """
    Z = np.asarray(Z, dtype=np.float64)
    s = np.linalg.svd(Z - Z.mean(0, keepdims=True), compute_uv=False)
    p = s / (s.sum() + eps)
    # Zero singular values contribute no entropy; skip log(0).
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def covariance_decay_exponent(Z, eps=1e-12):
    """Estimate spectral decay from centered (N, D) embeddings.

    Fit log covariance eigenvalue against log rank and negate the slope.
    Larger values mean variance is more concentrated in leading directions;
    fewer than four eigenvalues above eps gives NaN.
    """
    Z = np.asarray(Z, dtype=np.float64)
    Zc = Z - Z.mean(0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    lam = (s ** 2) / max(1, Zc.shape[0] - 1)          # covariance eigenvalues (descending)
    # Exclude near-zero eigenvalues before taking logs.
    lam = lam[lam > eps]
    if lam.size < 4:
        return float("nan")
    k = np.arange(1, lam.size + 1)
    slope = np.polyfit(np.log(k), np.log(lam), 1)[0]
    return float(-slope)


def vicreg_variance_hinge(Z, gamma=1.0, eps=1e-4):
    """Mean shortfall from a standard-deviation target for (N, D) embeddings.

    Each dimension contributes max(0, gamma - sqrt(sample variance + eps)).
    Zero means every stabilized standard deviation reaches gamma. Sample
    variance requires at least two observations.
    """
    Z = np.asarray(Z, dtype=np.float64)
    std = np.sqrt(Z.var(axis=0, ddof=1) + eps)
    return float(np.maximum(0.0, gamma - std).mean())


def participation_ratio(Z, eps=1e-8):
    """Effective number of covariance directions in (N, D) embeddings.

    Compute (sum λ)² / (sum λ² + eps), where λ are centered covariance
    eigenvalues. These are proportional to squared singular values, so this
    weights dominant directions more strongly than RankMe. NumPy keeps
    geometry resampling independent of torch.
    """
    X = np.asarray(Z, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    n = max(X.shape[0] - 1, 1)
    cov = (X.T @ X) / n
    lam = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    s = lam.sum()
    return float((s * s) / ((lam ** 2).sum() + eps))

def r2(pred, tgt, valid):
    """R² for aligned row arrays, relative to the finite, valid target mean.

    One is a perfect fit; zero matches the mean baseline; negative values
    mean greater squared error than that baseline. Fewer than two usable rows
    or constant targets give NaN, unlike sklearn's default finite replacements
    for constant targets.
    """
    m = np.asarray(valid, dtype=bool) & np.isfinite(pred) & np.isfinite(tgt)
    if int(m.sum()) < 2:
        return float("nan")
    y, yhat = tgt[m], pred[m]
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot

def seed_interval(vals, conf=0.95):
    """Mean and Student-t interval across finite seed-level values.

    Seeds are the replication units. Return (mean, [lo, hi], n), omitting
    non-finite values. With one seed, [mean, mean] is a placeholder, not an
    uncertainty estimate; with no seeds, both mean and bounds are NaN.
    """
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    n = int(a.size)
    if n == 0:
        return float("nan"), [float("nan"), float("nan")], 0
    m = float(a.mean())
    if n == 1:
        return m, [m, m], 1
    from scipy.stats import t as _tdist
    se = float(a.std(ddof=1)) / np.sqrt(n)
    h = float(_tdist.ppf(0.5 + conf / 2.0, n - 1)) * se
    return m, [m - h, m + h], n

def weighted_ranks(inverse, weights):
    """Ranks of rows repeated according to nonnegative integer weights.

    ``inverse`` maps each row to its sorted unique-value group, as returned by
    np.unique(..., return_inverse=True). Sum weights within each tied group to
    reproduce ranks without materializing copies. These midranks are 0.5 below
    the usual one-based ranks, a constant shift that leaves correlation intact.
    """
    counts = np.bincount(inverse, weights=weights)
    return (np.cumsum(counts) - 0.5 * counts)[inverse]

def metric_function(predictions, targets, valid, day_ids):
    """Prepare a callable that scores fixed predictions under day resampling.

    Predictions have shape (runs, rows, targets); targets and the Boolean
    validity mask have shape (rows, targets). Integer day_ids has shape (rows,)
    and indexes day_counts, a (days,) array of nonnegative integer multiplicities.
    Non-finite targets are omitted; predictions must be finite on retained rows.

    The callable returns (runs, targets, 2), ordered as rank-IC then R². Counts
    repeat all valid rows of each selected day, pooling rows across days. The
    resulting metrics equal explicitly repeating those rows; models stay fixed.
    Each target needs selected rows with nonconstant targets and predictions.
    """
    prepared = []
    for j in range(targets.shape[1]):
        # Cache tie groups once: day resampling changes multiplicities, not values.
        mask = valid[:, j] & np.isfinite(targets[:, j])
        y, pred = targets[mask, j], predictions[:, mask, j]
        if not np.isfinite(pred).all():
            raise ValueError("Predictions must be finite on every valid row")
        prepared.append((day_ids[mask], y, pred, np.unique(y, return_inverse=True)[1],
                         [np.unique(p, return_inverse=True)[1] for p in pred]))

    def metrics(day_counts):
        result = np.empty((len(predictions), targets.shape[1], 2))
        for j, (ids, y, pred, y_inverse, p_inverse) in enumerate(prepared):
            weights = day_counts[ids]
            n = weights.sum()
            # Weighted centering gives the same Pearson correlation of ranks
            # as Spearman correlation on the repeated rows, including ties.
            yr = weighted_ranks(y_inverse, weights)
            yr -= weights @ yr / n
            pr = np.stack([weighted_ranks(inv, weights) for inv in p_inverse])
            pr -= (pr @ weights / n)[:, None]
            denom = np.sqrt((pr ** 2 @ weights) * (weights @ yr ** 2))
            if np.any(denom <= 0):
                raise ValueError("Rank-IC is undefined for a constant target or prediction")
            result[:, j, 0] = pr @ (weights * yr) / denom
            # Recompute the target-mean baseline for each resample, using the
            # same row multiplicities for both baseline and prediction errors.
            sst = weights @ (y - weights @ y / n) ** 2
            result[:, j, 1] = 1 - ((pred - y) ** 2 @ weights) / sst
        return result
    return metrics
