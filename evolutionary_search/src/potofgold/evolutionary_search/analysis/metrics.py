"""Risk statistics from saved daily strategy outcomes.

PSR uses daily moments and converts annual Sharpe benchmarks to daily units.
The surviving-front PBO diagnostic resplits archived outcomes; it does not
reconstruct every candidate considered during the original search.
"""
from __future__ import annotations
from itertools import combinations
from math import sqrt
import numpy as np
from scipy.special import ndtr
from scipy.stats import rankdata
YEAR = 252

def psr_summary(daily_returns, benchmark_annual=1.0, periods_per_year=YEAR,
                min_obs=3):
    """Return moments and PSR, with all Sharpe arithmetic in DAILY units.

    Uses sample standard deviation (ddof=1) and central-moment plug-in
    skewness/Pearson kurtosis. A stated annual benchmark of 1 is converted to
    1/sqrt(252), not used as a daily Sharpe of 1. Nonfinite inputs and constant
    returns are unassessable; they never silently become high-confidence wins.
    """
    x = np.asarray(daily_returns, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("daily_returns must be one-dimensional")
    if periods_per_year <= 0 or not np.isfinite(benchmark_annual):
        raise ValueError("periods_per_year must be positive and benchmark finite")
    n = int(x.size)
    out = dict(n_obs=n, n_nonfinite=int(np.sum(~np.isfinite(x))),
               benchmark_annual=float(benchmark_annual),
               benchmark_daily=float(benchmark_annual / sqrt(periods_per_year)),
               periods_per_year=int(periods_per_year), valid=False,
               status="too_few_observations", mean=float("nan"), sd=float("nan"),
               sharpe_daily=float("nan"), sharpe_annual=float("nan"),
               skewness=float("nan"), kurtosis_pearson=float("nan"),
               variance_factor=float("nan"), psr_z=float("nan"),
               psr=float("nan"), autocorrelation_lag1=float("nan"),
               approximation="IID asymptotic plug-in; no serial-correlation correction")
    if out["n_nonfinite"]:
        out["status"] = "nonfinite_observations"
        return out
    if n:
        out["mean"] = float(x.mean())
    if n < max(3, int(min_obs)):
        return out
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    out["sd"] = sd
    # Relative guard preserves positive-scale invariance, including tiny P&L.
    scale = float(np.max(np.abs(x)))
    if sd == 0 or sd <= np.finfo(float).eps * scale:
        out["status"] = "constant_returns"
        return out
    # Standardize before powers to avoid unnecessary overflow in moments.
    z = (x - mean) / sd
    m2 = float(np.mean(z * z))
    g3 = float(np.mean(z ** 3) / m2 ** 1.5)
    g4 = float(np.mean(z ** 4) / m2 ** 2)
    sr = mean / sd
    variance = 1.0 - g3 * sr + (g4 - 1.0) * sr * sr / 4.0
    out.update(sharpe_daily=float(sr), sharpe_annual=float(sr * sqrt(periods_per_year)),
               skewness=g3, kurtosis_pearson=g4, variance_factor=float(variance))
    if np.std(x[:-1]) > 0 and np.std(x[1:]) > 0:
        out["autocorrelation_lag1"] = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    if not np.isfinite(variance) or variance <= 0:
        out["status"] = "invalid_variance_factor"
        return out
    statistic = (sr - out["benchmark_daily"]) * sqrt(n - 1) / sqrt(variance)
    out.update(valid=True, status="ok", psr_z=float(statistic), psr=float(ndtr(statistic)))
    return out


def _partition_sharpes(x):
    """Rank-only Sharpe convention: all-zero cash=0; nonzero flat=+/-inf.

    Sparse strategies often have no trades in a CSCV half. Preserve them as
    cash for ranking, and expose how often this happens. PSR never uses this
    convention: its constant-series result is explicitly unassessable.
    """
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    tolerance = np.finfo(float).eps * np.max(np.abs(x), axis=0)
    nonflat = sd > tolerance
    scores = np.zeros(x.shape[1], dtype=float)
    np.divide(mean, sd, out=scores, where=nonflat)
    flat_positive = (~nonflat) & (mean > 0)
    flat_negative = (~nonflat) & (mean < 0)
    scores[flat_positive] = np.inf
    scores[flat_negative] = -np.inf
    return scores, int(np.sum(~nonflat)), int(np.sum(flat_positive | flat_negative))


def cscv_front_pbo(returns_matrix, n_blocks=8):
    """Conditional surviving-front CSCV diagnostic, input [days, strategies].

    Split the ordered days into an even number of contiguous blocks and test
    every symmetric half/half partition. No IS-positive condition is applied.
    Globally nonfinite/flat columns are excluded and exact duplicate return
    vectors collapsed; require >=3 distinct nonflat vectors. On each split,
    average equally over exact best-IS ties. OOS ties receive average ranks;
    rank/(N+1) keeps relative ranks strictly between zero and one.

    pbo uses the conventional logit<=0 event. pbo_strict_below_median and
    median_tie_fraction separately disclose equality rather than concealing
    rank ties. Partitions overlap, so no binomial confidence interval applies.
    This resplits supplied OOS records; it does not recreate GP evolution or
    chronological future deployments, and cannot estimate full-search PBO.
    """
    x = np.asarray(returns_matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("returns_matrix must have shape [days, strategies]")
    if int(n_blocks) != n_blocks or n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even integer >=2")
    n_days, n_input = x.shape
    out = dict(status="too_few_observations", pbo=float("nan"),
               pbo_strict_below_median=float("nan"), median_tie_fraction=float("nan"),
               n_days=int(n_days), n_input_strategies=int(n_input),
               n_nonfinite_excluded=0, n_flat_excluded=0, n_duplicate_vectors_excluded=0,
               n_distinct_nonflat=0, n_blocks=int(n_blocks), n_partitions=0,
               n_partitions_with_best_is_ties=0, n_partition_flat_columns=0,
               n_partition_nonzero_flat_columns=0, mean_oos_relative_rank=float("nan"),
               interpretation="conditional on surviving return vectors; full-search PBO not identified",
               flat_partition_ranking="all-zero cash=0; nonzero flat returns rank at signed infinity",
               partitions=[])
    if n_days < 2 * n_blocks or n_input == 0:
        return out
    finite = np.all(np.isfinite(x), axis=0)
    out["n_nonfinite_excluded"] = int(np.sum(~finite))
    x = x[:, finite]
    if x.shape[1]:
        nonflat = np.ptp(x, axis=0) > np.finfo(float).eps * np.max(np.abs(x), axis=0)
        out["n_flat_excluded"] = int(np.sum(~nonflat))
        x = x[:, nonflat]
        n_before = x.shape[1]
        x = np.unique(x, axis=1)
        out["n_duplicate_vectors_excluded"] = int(n_before - x.shape[1])
    n_strategies = x.shape[1]
    out["n_distinct_nonflat"] = int(n_strategies)
    if n_strategies < 3:
        out["status"] = "fewer_than_three_distinct_nonflat_strategies"
        return out
    blocks = np.array_split(np.arange(n_days), int(n_blocks))
    # Each half appears once as training data and once as its complement's test data.
    half_stats = {
        selected: _partition_sharpes(x[np.concatenate([blocks[i] for i in selected])])
        for selected in combinations(range(int(n_blocks)), int(n_blocks // 2))
    }
    below, equal, relative_ranks = [], [], []
    for train_blocks, (train_sr, nflat_train, nonzero_flat_train) in half_stats.items():
        test_blocks = tuple(i for i in range(int(n_blocks)) if i not in train_blocks)
        test_sr, nflat_test, nonzero_flat_test = half_stats[test_blocks]
        winners = np.flatnonzero(train_sr == train_sr.max())
        ranks = rankdata(test_sr, method="average") / (n_strategies + 1.0)
        selected_ranks = ranks[winners]
        p_below = float(np.mean(selected_ranks < 0.5))
        p_equal = float(np.mean(selected_ranks == 0.5))
        below.append(p_below)
        equal.append(p_equal)
        relative_ranks.append(float(selected_ranks.mean()))
        out["n_partitions_with_best_is_ties"] += int(winners.size > 1)
        out["n_partition_flat_columns"] += nflat_train + nflat_test
        out["n_partition_nonzero_flat_columns"] += nonzero_flat_train + nonzero_flat_test
        out["partitions"].append(dict(train_blocks=list(train_blocks),
                                      n_best_is_ties=int(winners.size),
                                      p_strict_below_median=p_below, p_at_median=p_equal,
                                      mean_oos_relative_rank=float(selected_ranks.mean())))
    out.update(status="ok", pbo=float(np.mean(np.asarray(below) + equal)),
               pbo_strict_below_median=float(np.mean(below)),
               median_tie_fraction=float(np.mean(equal)),
               mean_oos_relative_rank=float(np.mean(relative_ranks)),
               n_partitions=len(below))
    return out
