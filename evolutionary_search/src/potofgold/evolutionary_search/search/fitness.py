"""Calculate strategy objectives, tail losses and feasibility penalties."""

from __future__ import annotations
import os
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from .simulator import vectorized_backtest

INFEASIBLE: Tuple[float, float, float] = (1000000000.0, 1000000000.0, 1000000000.0)
DEFAULT_MIN_TRADES = 15
DEFAULT_CVAR_FRAC = 0.05


def _env_float_or_none(name: str) -> Optional[float]:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else None


def _env_int_or_none(name: str) -> Optional[int]:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else None


DEFAULT_CVAR_TAIL_MAX: Optional[float] = _env_float_or_none("GP_CVAR_TAIL_MAX")
DEFAULT_EXPECTANCY_TRIM: Optional[float] = _env_float_or_none("GP_EXPECTANCY_TRIM")
DEFAULT_MIN_COVERAGE: Optional[float] = _env_float_or_none("GP_MIN_COVERAGE")
DEFAULT_MIN_DAYS: Optional[int] = _env_int_or_none("GP_MIN_DAYS")


def derived_min_days(cvar_frac: float = DEFAULT_CVAR_FRAC) -> int:
    return int(np.ceil(1.0 / float(cvar_frac)))


DEFAULT_OBJ1_ELIGIBLE_DAYS: bool = os.environ.get(
    "GP_OBJ1_ELIGIBLE_DAYS", ""
).strip().lower() in ("1", "true", "yes")
_ONESIDED_UPPER_TRIM = os.environ.get("GP_ONESIDED_TRIM", "") not in (
    "",
    "0",
    "false",
    "False",
)
_OBJ1_RAW_MEAN = os.environ.get("GP_OBJ1_RAW_MEAN", "") not in (
    "",
    "0",
    "false",
    "False",
)
CONCENTRATION_TRIM_DEFAULT: float = 0.2


def _effective_expectancy_trim() -> float:
    return (
        DEFAULT_EXPECTANCY_TRIM
        if DEFAULT_EXPECTANCY_TRIM is not None
        else CONCENTRATION_TRIM_DEFAULT
    )


def expectancy_trim_mismatch_warning() -> Optional[str]:
    if DEFAULT_EXPECTANCY_TRIM is None:
        return None
    from .validation import SIGNFLIP_TRIM_FRACTION

    if abs(float(DEFAULT_EXPECTANCY_TRIM) - float(SIGNFLIP_TRIM_FRACTION)) > 1e-09:
        return f"[capture_fitness] GP_EXPECTANCY_TRIM={DEFAULT_EXPECTANCY_TRIM} != verdict SIGNFLIP_TRIM_FRACTION={SIGNFLIP_TRIM_FRACTION}: the fitness (train) and the day-permutation verdict (OOS) will trim DIFFERENT fractions. Set GP_EXPECTANCY_TRIM={SIGNFLIP_TRIM_FRACTION} to keep the training objective and the significance statistic matched."
    return None


def _trimmed_mean_1d(x, frac: Optional[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return float("nan")
    if _OBJ1_RAW_MEAN:
        return float(x.mean())
    k = int(np.floor(frac * n)) if frac is not None and frac > 0.0 else 0
    if k == 0:
        return float(x.mean())
    if _ONESIDED_UPPER_TRIM:
        if n - k < 1:
            return float(x.mean())
        return float(np.sort(x)[: n - k].mean())
    if n - 2 * k < 1:
        return float(x.mean())
    return float(np.sort(x)[k : n - k].mean())


Objectives = Tuple[float, float, float]


def cvar_left_tail(
    pnls: Sequence[float], cvar_frac: float = DEFAULT_CVAR_FRAC
) -> float:
    arr = np.asarray(pnls, dtype=np.float64)
    n = arr.size
    if n == 0:
        return 0.0
    k = max(1, int(np.ceil(cvar_frac * n)))
    worst = np.sort(arr)[:k]
    return float(max(-worst.mean(), 0.0))


_GATED_NULL_PERMS = 256
_GATED_NULL_SEED = 0


def _raw_gated_gain(pnl: np.ndarray, fired: np.ndarray, cvar_frac: float) -> float:
    return cvar_left_tail(pnl[~fired], cvar_frac) - cvar_left_tail(
        pnl[fired], cvar_frac
    )


def signal_gated_cvar_gain(
    per_day_pnl: Sequence[float],
    fired_mask: Sequence[bool],
    cvar_frac: float = DEFAULT_CVAR_FRAC,
    n_null: int = _GATED_NULL_PERMS,
) -> float:
    pnl = np.asarray(per_day_pnl, dtype=np.float64)
    fired = np.asarray(fired_mask, dtype=bool)
    if pnl.shape != fired.shape:
        raise ValueError(
            f"per_day_pnl {pnl.shape} and fired_mask {fired.shape} must align"
        )
    raw = _raw_gated_gain(pnl, fired, cvar_frac)
    n = pnl.size
    n_f = int(fired.sum())
    if n_f == 0 or n_f == n or n_null <= 0:
        return raw
    n_q = n - n_f
    k_f = max(1, int(np.ceil(cvar_frac * n_f)))
    k_q = max(1, int(np.ceil(cvar_frac * n_q)))
    rng = np.random.default_rng(_GATED_NULL_SEED)
    perms = np.argsort(rng.random((n_null, n)), axis=1)
    fired_pnl = pnl[perms[:, :n_f]]
    quiet_pnl = pnl[perms[:, n_f:]]
    cvar_f = np.maximum(-np.sort(fired_pnl, axis=1)[:, :k_f].mean(axis=1), 0.0)
    cvar_q = np.maximum(-np.sort(quiet_pnl, axis=1)[:, :k_q].mean(axis=1), 0.0)
    null = cvar_q - cvar_f
    null_std = float(null.std())
    if null_std < 1e-09:
        return 0.0
    return (raw - float(null.mean())) / null_std


def per_day_sum_pnl(trades, dates) -> Dict:
    out: Dict = {}
    for t in trades:
        d = dates[t.entry_bar]
        out[d] = out.get(d, 0.0) + float(t.pnl)
    return out


def capture_objectives_from_trades(
    trades,
    dates,
    n_nodes: int,
    *,
    min_trades: int = DEFAULT_MIN_TRADES,
    cvar_frac: float = DEFAULT_CVAR_FRAC,
    cvar_tail_max: Optional[float] = None,
    trim_frac: Optional[float] = None,
    min_coverage_frac: Optional[float] = None,
    min_days: Optional[int] = None,
    obj1_eligible_days: Optional[bool] = None,
) -> Objectives:
    per_trade = np.array([float(t.pnl) for t in trades], dtype=np.float64)
    if per_trade.size < min_trades or not np.all(np.isfinite(per_trade)):
        return INFEASIBLE
    day_pnl = per_day_sum_pnl(trades, dates)
    per_day = np.array(list(day_pnl.values()), dtype=np.float64)
    eff_min_days = DEFAULT_MIN_DAYS if min_days is None else min_days
    if eff_min_days is not None and per_day.size < int(eff_min_days):
        return INFEASIBLE
    n_fold_days = int(np.unique(np.asarray(dates)).size)
    eff_min_cov = (
        DEFAULT_MIN_COVERAGE if min_coverage_frac is None else min_coverage_frac
    )
    if eff_min_cov is not None:
        if n_fold_days > 0 and per_day.size / n_fold_days < float(eff_min_cov):
            return INFEASIBLE
    _trim = trim_frac if trim_frac is not None else _effective_expectancy_trim()
    _obj1_elig = (
        DEFAULT_OBJ1_ELIGIBLE_DAYS
        if obj1_eligible_days is None
        else bool(obj1_eligible_days)
    )
    if _obj1_elig and n_fold_days > 0:
        coverage = per_day.size / n_fold_days
        neg_expectancy = -(_trimmed_mean_1d(per_day, _trim) * coverage)
    else:
        neg_expectancy = -_trimmed_mean_1d(per_day, _trim)
    tail = cvar_left_tail(per_day, cvar_frac)
    eff_tail_max = DEFAULT_CVAR_TAIL_MAX if cvar_tail_max is None else cvar_tail_max
    if eff_tail_max is not None and tail > eff_tail_max:
        return INFEASIBLE
    return (neg_expectancy, tail, float(n_nodes))


def _day_subfold_ids(dates, k_sub: int, seed: int) -> Dict:
    uniq = np.unique(np.asarray(dates))
    n = uniq.size
    if n < int(k_sub):
        return {}
    perm = np.random.default_rng(int(seed)).permutation(n)
    sub_of_uniq = np.empty(n, dtype=int)
    sub_of_uniq[perm] = np.arange(n) % int(k_sub)
    return {day: int(sub_of_uniq[i]) for (i, day) in enumerate(uniq)}


def cross_fold_agreement(
    trades,
    dates,
    *,
    k_sub: int = 2,
    partition_seed: int = 0,
    trim_frac: Optional[float] = None,
    cvar_frac: float = DEFAULT_CVAR_FRAC,
    min_days: Optional[int] = None,
) -> float:
    day_pnl = per_day_sum_pnl(trades, dates)
    if not day_pnl:
        return 0.0
    sub_of_day = _day_subfold_ids(list(day_pnl.keys()), int(k_sub), int(partition_seed))
    if not sub_of_day:
        return 0.0
    _trim = trim_frac if trim_frac is not None else _effective_expectancy_trim()
    eff_min_days = (
        max(5, int(np.ceil(derived_min_days(cvar_frac) / int(k_sub))))
        if min_days is None
        else int(min_days)
    )
    exps = []
    for h in range(int(k_sub)):
        vals = np.array(
            [v for (d, v) in day_pnl.items() if sub_of_day.get(d) == h],
            dtype=np.float64,
        )
        if vals.size < eff_min_days:
            return 0.0
        exps.append(_trimmed_mean_1d(vals, _trim))
    if any((e <= 0.0 for e in exps)):
        return 0.0
    return float(max(min(exps), 0.0))


def capture_objectives_from_trades_nested(
    trades,
    dates,
    n_nodes: int,
    *,
    k_sub: int,
    lam: float,
    partition_seed: int = 0,
    min_trades: int = DEFAULT_MIN_TRADES,
    cvar_frac: float = DEFAULT_CVAR_FRAC,
    cvar_tail_max: Optional[float] = None,
    trim_frac: Optional[float] = None,
    min_coverage_frac: Optional[float] = None,
    min_days: Optional[int] = None,
    min_subfold_days: Optional[int] = None,
) -> Objectives:
    per_trade = np.array([float(t.pnl) for t in trades], dtype=np.float64)
    if per_trade.size < min_trades or not np.all(np.isfinite(per_trade)):
        return INFEASIBLE
    if int(k_sub) < 2:
        return INFEASIBLE
    per_day_map = per_day_sum_pnl(trades, dates)
    per_day = np.array(list(per_day_map.values()), dtype=np.float64)
    eff_min_days = DEFAULT_MIN_DAYS if min_days is None else min_days
    if eff_min_days is not None and per_day.size < int(eff_min_days):
        return INFEASIBLE
    eff_min_cov = (
        DEFAULT_MIN_COVERAGE if min_coverage_frac is None else min_coverage_frac
    )
    if eff_min_cov is not None:
        n_fold_days = int(np.unique(np.asarray(dates)).size)
        if n_fold_days > 0 and per_day.size / n_fold_days < float(eff_min_cov):
            return INFEASIBLE
    sub_of_day = _day_subfold_ids(dates, int(k_sub), int(partition_seed))
    if not sub_of_day:
        return INFEASIBLE
    groups = [[] for _ in range(int(k_sub))]
    for day, pnl in per_day_map.items():
        sid = sub_of_day.get(day)
        if sid is not None:
            groups[sid].append(float(pnl))
    _base_min = int(eff_min_days) if eff_min_days is not None else DEFAULT_MIN_TRADES
    eff_min_sub = (
        max(5, int(np.ceil(_base_min / int(k_sub))))
        if min_subfold_days is None
        else int(min_subfold_days)
    )
    if any((len(g) < eff_min_sub for g in groups)):
        return INFEASIBLE
    _trim = trim_frac if trim_frac is not None else _effective_expectancy_trim()
    e_k = np.array(
        [_trimmed_mean_1d(np.array(g, dtype=np.float64), _trim) for g in groups],
        dtype=np.float64,
    )
    mean_e = float(np.mean(e_k))
    std_e = float(np.std(e_k, ddof=1)) if e_k.size > 1 else 0.0
    sigma_daily = float(np.std(per_day, ddof=1)) if per_day.size > 1 else 0.0
    n_bar = float(np.mean([len(g) for g in groups]))
    sigma_noise = sigma_daily / np.sqrt(n_bar) if n_bar > 0 else 0.0
    penalty = float(lam) * max(std_e - sigma_noise, 0.0)
    neg_expectancy = -(mean_e - penalty)
    tail = cvar_left_tail(per_day, cvar_frac)
    eff_tail_max = DEFAULT_CVAR_TAIL_MAX if cvar_tail_max is None else cvar_tail_max
    if eff_tail_max is not None and tail > eff_tail_max:
        return INFEASIBLE
    return (neg_expectancy, tail, float(n_nodes))


def genome_trades(
    individual,
    data,
    template,
    terminal_data,
    *,
    entry_decision_mask=None,
    warmup_bars=None,
    pca_bases=None,
    emb_proj_cache=None,
    entry_floor_within_day=None,
):
    stop = getattr(individual, "stop_mult", None)
    stop_kw = {} if stop is None else {"stop_loss_credit_multiple": float(stop)}
    warm_kw = {} if warmup_bars is None else {"warmup_bars": int(warmup_bars)}
    pca_kw = {} if pca_bases is None else {"pca_bases": pca_bases}
    cache_kw = {} if emb_proj_cache is None else {"emb_proj_cache": emb_proj_cache}
    result = vectorized_backtest(
        entry_tree=individual.entry_tree,
        exit_tree=individual.exit_tree,
        size_tree=individual.size_tree,
        data=data,
        template=template,
        terminal_data=terminal_data,
        delta_tree=getattr(individual, "delta_tree", None),
        width_tree=getattr(individual, "width_tree", None),
        shape=getattr(individual, "shape", None),
        shape_tree=getattr(individual, "shape_tree", None),
        entry_decision_mask=entry_decision_mask,
        entry_floor_within_day=entry_floor_within_day,
        **stop_kw,
        **warm_kw,
        **pca_kw,
        **cache_kw,
    )
    return result.trades


def _entry_dates(data, n_trades_hint=None):
    cols = getattr(data, "columns", ())
    if "date" in cols:
        return np.asarray(data["date"].values)
    return np.arange(len(data))


def genome_capture_objectives(
    individual,
    data,
    template,
    terminal_data,
    *,
    min_trades: int = DEFAULT_MIN_TRADES,
    cvar_frac: float = DEFAULT_CVAR_FRAC,
    cvar_tail_max: Optional[float] = None,
    entry_decision_mask=None,
    warmup_bars=None,
    pca_bases=None,
    emb_proj_cache=None,
    entry_floor_within_day=None,
    trim_frac: Optional[float] = None,
    min_coverage_frac: Optional[float] = None,
    min_days: Optional[int] = None,
    obj1_eligible_days: Optional[bool] = None,
) -> Objectives:
    trades = genome_trades(
        individual,
        data,
        template,
        terminal_data,
        entry_decision_mask=entry_decision_mask,
        warmup_bars=warmup_bars,
        pca_bases=pca_bases,
        emb_proj_cache=emb_proj_cache,
        entry_floor_within_day=entry_floor_within_day,
    )
    return capture_objectives_from_trades(
        trades,
        _entry_dates(data),
        individual.total_nodes(),
        min_trades=min_trades,
        cvar_frac=cvar_frac,
        cvar_tail_max=cvar_tail_max,
        trim_frac=trim_frac,
        min_coverage_frac=min_coverage_frac,
        min_days=min_days,
        obj1_eligible_days=obj1_eligible_days,
    )


DEFAULT_SAM_RADIUS: float = _env_float_or_none("GP_SAM_RADIUS") or 0.1
_SAM_SEED = 0
_sam_sigma_cache: Dict[int, Dict[str, float]] = {}


def sam_feature_sigma(terminal_data) -> Dict[str, float]:
    key = id(terminal_data)
    cached = _sam_sigma_cache.get(key)
    if cached is not None:
        return cached
    sig = {
        name: float(np.nanstd(arr))
        for (name, arr) in terminal_data.items()
        if not name.startswith("_") and getattr(arr, "ndim", 1) == 1
    }
    _sam_sigma_cache[key] = sig
    return sig


def _perturb_terminal_data(
    terminal_data, feature_sigma: Dict[str, float], radius: float, rng
) -> dict:
    out = {}
    for name, values in terminal_data.items():
        if name.startswith("_") or getattr(values, "ndim", 1) != 1:
            out[name] = values
            continue
        sigma = float(radius) * float(feature_sigma.get(name, 0.0))
        if sigma > 0.0:
            noise = rng.normal(0.0, sigma, size=values.shape).astype(values.dtype)
            out[name] = values + noise
        else:
            out[name] = values
    return out


def _sam_perturbed_neg_exp(o, trades_p, min_trades: int, base_neg_exp: float) -> float:
    neg_exp = float(o[0])
    if neg_exp < INFEASIBLE[0]:
        return neg_exp
    if len(trades_p) < int(min_trades):
        return float(INFEASIBLE[0])
    return float(base_neg_exp)


def genome_input_sharpness(
    individual,
    data,
    template,
    terminal_data,
    *,
    feature_sigma: Dict[str, float],
    base_neg_exp: float,
    n_pert: int,
    radius: float = DEFAULT_SAM_RADIUS,
    min_trades: int = DEFAULT_MIN_TRADES,
    cvar_frac: float = DEFAULT_CVAR_FRAC,
    entry_decision_mask=None,
    warmup_bars=None,
    pca_bases=None,
    emb_proj_cache=None,
    entry_floor_within_day=None,
    trim_frac=None,
    min_coverage_frac=None,
    min_days=None,
    obj1_eligible_days=None,
) -> float:
    if n_pert is None or int(n_pert) <= 0:
        return 0.0
    rng = np.random.RandomState(_SAM_SEED)
    worst = float(base_neg_exp)
    for _ in range(int(n_pert)):
        td_pert = _perturb_terminal_data(terminal_data, feature_sigma, radius, rng)
        trades_p = genome_trades(
            individual,
            data,
            template,
            td_pert,
            entry_decision_mask=entry_decision_mask,
            warmup_bars=warmup_bars,
            pca_bases=pca_bases,
            emb_proj_cache=emb_proj_cache,
            entry_floor_within_day=entry_floor_within_day,
        )
        o = capture_objectives_from_trades(
            trades_p,
            _entry_dates(data),
            individual.total_nodes(),
            min_trades=min_trades,
            cvar_frac=cvar_frac,
            trim_frac=trim_frac,
            min_coverage_frac=min_coverage_frac,
            min_days=min_days,
            obj1_eligible_days=obj1_eligible_days,
        )
        worst = max(
            worst, _sam_perturbed_neg_exp(o, trades_p, min_trades, base_neg_exp)
        )
    return max(worst - float(base_neg_exp), 0.0)


def per_eligible_day_vector(trades, dates) -> np.ndarray:
    day_pnl = per_day_sum_pnl(trades, dates)
    uniq = np.unique(np.asarray(dates))
    return np.array([day_pnl.get(d, 0.0) for d in uniq], dtype=np.float64)


def genome_capture_objectives_nested(
    individual,
    data,
    template,
    terminal_data,
    *,
    k_sub: int,
    lam: float,
    partition_seed: int = 0,
    min_trades: int = DEFAULT_MIN_TRADES,
    cvar_frac: float = DEFAULT_CVAR_FRAC,
    cvar_tail_max: Optional[float] = None,
    entry_decision_mask=None,
    warmup_bars=None,
    pca_bases=None,
    emb_proj_cache=None,
    entry_floor_within_day=None,
    trim_frac: Optional[float] = None,
    min_coverage_frac: Optional[float] = None,
    min_days: Optional[int] = None,
    min_subfold_days: Optional[int] = None,
) -> Objectives:
    trades = genome_trades(
        individual,
        data,
        template,
        terminal_data,
        entry_decision_mask=entry_decision_mask,
        warmup_bars=warmup_bars,
        pca_bases=pca_bases,
        emb_proj_cache=emb_proj_cache,
        entry_floor_within_day=entry_floor_within_day,
    )
    return capture_objectives_from_trades_nested(
        trades,
        _entry_dates(data),
        individual.total_nodes(),
        k_sub=k_sub,
        lam=lam,
        partition_seed=partition_seed,
        min_trades=min_trades,
        cvar_frac=cvar_frac,
        cvar_tail_max=cvar_tail_max,
        trim_frac=trim_frac,
        min_coverage_frac=min_coverage_frac,
        min_days=min_days,
        min_subfold_days=min_subfold_days,
    )
