"""Group observations by day and apply held-out strategy acceptance tests."""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .simulator import vectorized_backtest
from .grammar import GType, TermDef, TermNode
from .fitness import (
    cvar_left_tail,
    signal_gated_cvar_gain,
    per_day_sum_pnl,
    DEFAULT_CVAR_FRAC,
)

ALPHA = 0.05
DEFAULT_N_PERM = 1000
DEFAULT_DECISION_OFFSET = 30
BUCKETS = ("open", "late_morning", "midday", "power_hour", "pre_close")


def intraday_bucket(mtc) -> np.ndarray:
    mtc = np.asarray(mtc, float)
    out = np.full(len(mtc), "pre_close", dtype=object)
    out[mtc >= 360] = "open"
    out[(mtc < 360) & (mtc >= 270)] = "late_morning"
    out[(mtc < 270) & (mtc >= 150)] = "midday"
    out[(mtc < 150) & (mtc >= 60)] = "power_hour"
    return out


def unique_days(df) -> np.ndarray:
    return pd.unique(df["date"].values)


def _decision_mask(df, offset: int = DEFAULT_DECISION_OFFSET) -> np.ndarray:
    dates = df["date"].values
    day_start = np.r_[0, np.where(dates[1:] != dates[:-1])[0] + 1]
    day_end = np.r_[day_start[1:], len(df)]
    mask = np.zeros(len(df), dtype=np.float64)
    idx = day_start + offset
    mask[idx[idx < day_end]] = 1.0
    return mask


def kfold_day_split(df, fold_k: int, n_folds: int = 5, seed: int = 0):
    days = np.asarray(unique_days(df))
    n = len(days)
    if n < n_folds:
        raise ValueError(
            f"kfold_day_split: {n} days < n_folds={n_folds} -> empty folds; need >= n_folds days"
        )
    perm = np.random.default_rng(seed).permutation(n)
    test_positions = perm[np.arange(n) % n_folds == int(fold_k)]
    test_days = set(days[test_positions].tolist())
    dcol = df["date"].values
    test_mask = np.fromiter((d in test_days for d in dcol), dtype=bool, count=len(dcol))
    return (~test_mask, test_mask)


def slice_by_mask(df, td, mask):
    mask = np.asarray(mask, bool)
    sub = df.loc[mask].reset_index(drop=True)
    sub_td = {}
    for k, v in (td or {}).items():
        arr = np.asarray(v)
        sub_td[k] = arr[mask] if arr.ndim >= 1 and arr.shape[0] == mask.shape[0] else v
    return (sub, sub_td)


def _champion_trades(
    individual,
    df,
    template,
    td,
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
    floor_kw = (
        {}
        if entry_floor_within_day is None
        else {"entry_floor_within_day": int(entry_floor_within_day)}
    )
    return vectorized_backtest(
        entry_tree=individual.entry_tree,
        exit_tree=individual.exit_tree,
        size_tree=individual.size_tree,
        data=df,
        template=template,
        terminal_data=td,
        delta_tree=getattr(individual, "delta_tree", None),
        width_tree=getattr(individual, "width_tree", None),
        shape=getattr(individual, "shape", None),
        shape_tree=getattr(individual, "shape_tree", None),
        entry_decision_mask=entry_decision_mask,
        **stop_kw,
        **warm_kw,
        **pca_kw,
        **cache_kw,
        **floor_kw,
    ).trades


def _champion_everyday_trades(
    individual,
    df,
    template,
    td,
    offset: int = DEFAULT_DECISION_OFFSET,
    *,
    pca_bases=None,
    emb_proj_cache=None,
):
    stop = getattr(individual, "stop_mult", None)
    stop_kw = {} if stop is None else {"stop_loss_credit_multiple": float(stop)}
    pca_kw = {} if pca_bases is None else {"pca_bases": pca_bases}
    cache_kw = {} if emb_proj_cache is None else {"emb_proj_cache": emb_proj_cache}
    return vectorized_backtest(
        entry_tree=None,
        exit_tree=individual.exit_tree,
        size_tree=individual.size_tree,
        data=df,
        template=template,
        terminal_data=td,
        delta_tree=getattr(individual, "delta_tree", None),
        width_tree=getattr(individual, "width_tree", None),
        shape=getattr(individual, "shape", None),
        shape_tree=getattr(individual, "shape_tree", None),
        _override_entry_signals=_decision_mask(df, offset),
        warmup_bars=int(offset),
        **stop_kw,
        **pca_kw,
        **cache_kw,
    ).trades


def _always_enter_trades(
    df,
    template,
    td,
    offset: int = DEFAULT_DECISION_OFFSET,
    *,
    shape=None,
    shape_tree=None,
    delta_tree=None,
    width_tree=None,
    warmup_bars=None,
    pca_bases=None,
    emb_proj_cache=None,
    _sizing_log=None,
):
    size = TermNode(defn=TermDef("EphReal", GType.REAL), value=0.5)
    warm_kw = {} if warmup_bars is None else {"warmup_bars": int(warmup_bars)}
    pca_kw = {} if pca_bases is None else {"pca_bases": pca_bases}
    cache_kw = {} if emb_proj_cache is None else {"emb_proj_cache": emb_proj_cache}
    return vectorized_backtest(
        entry_tree=None,
        exit_tree=None,
        size_tree=size,
        data=df,
        template=template,
        terminal_data=td,
        shape=shape,
        shape_tree=shape_tree,
        delta_tree=delta_tree,
        width_tree=width_tree,
        _override_entry_signals=_decision_mask(df, offset),
        _override_exit_signals=np.zeros(len(df), dtype=np.float64),
        _sizing_log=_sizing_log,
        **warm_kw,
        **pca_kw,
        **cache_kw,
    ).trades


def _per_day_sum(trades, df) -> Dict:
    return per_day_sum_pnl(trades, df["date"].values)


def derive_cvar_ceiling(
    df,
    template,
    td,
    *,
    decision_offset: int = DEFAULT_DECISION_OFFSET,
    cvar_frac: float = DEFAULT_CVAR_FRAC,
) -> Optional[float]:
    trades = _always_enter_trades(df, template, td, decision_offset)
    if not trades:
        return None
    per_day = np.array(
        list(per_day_sum_pnl(trades, df["date"].values).values()), dtype=np.float64
    )
    ceiling = cvar_left_tail(per_day, cvar_frac)
    return ceiling if ceiling > 0.0 else None


def _per_day_pair(champion_trades, baseline_trades, df):
    cp = _per_day_sum(champion_trades, df)
    bp = _per_day_sum(baseline_trades, df)
    days = sorted(set(cp) | set(bp), key=lambda d: str(d))
    champ_day = np.array([cp.get(d, 0.0) for d in days], dtype=np.float64)
    base_day = np.array([bp.get(d, 0.0) for d in days], dtype=np.float64)
    traded_mask = np.array([d in cp for d in days], dtype=bool)
    return (champ_day, base_day, traded_mask, days)


def champ_and_base_trades(
    individual,
    train_df,
    template,
    td,
    *,
    decision_offset: int = DEFAULT_DECISION_OFFSET,
    entry_decision_mask=None,
    warmup_bars=None,
    pca_bases=None,
    emb_proj_cache=None,
    entry_floor_within_day=None,
    precomputed_champ=None,
):
    warm = int(decision_offset) if warmup_bars is None else int(warmup_bars)
    if precomputed_champ is not None:
        champ = precomputed_champ
    else:
        champ = _champion_trades(
            individual,
            train_df,
            template,
            td,
            entry_decision_mask=entry_decision_mask,
            warmup_bars=warm,
            pca_bases=pca_bases,
            emb_proj_cache=emb_proj_cache,
            entry_floor_within_day=entry_floor_within_day,
        )
    base = _always_enter_trades(
        train_df,
        template,
        td,
        decision_offset,
        shape=getattr(individual, "shape", None),
        shape_tree=getattr(individual, "shape_tree", None),
        delta_tree=getattr(individual, "delta_tree", None),
        width_tree=getattr(individual, "width_tree", None),
        warmup_bars=warm,
        pca_bases=pca_bases,
        emb_proj_cache=emb_proj_cache,
    )
    return (champ, base)


def genome_per_day_excess(
    individual,
    train_df,
    template,
    td,
    *,
    decision_offset: int = DEFAULT_DECISION_OFFSET,
    entry_decision_mask=None,
    warmup_bars=None,
    pca_bases=None,
    emb_proj_cache=None,
    entry_floor_within_day=None,
    precomputed_champ=None,
):
    cb = champ_and_base_trades(
        individual,
        train_df,
        template,
        td,
        decision_offset=decision_offset,
        entry_decision_mask=entry_decision_mask,
        warmup_bars=warmup_bars,
        pca_bases=pca_bases,
        emb_proj_cache=emb_proj_cache,
        entry_floor_within_day=entry_floor_within_day,
        precomputed_champ=precomputed_champ,
    )
    (champ, base) = cb
    (champ_day, base_day, traded_mask, _days) = _per_day_pair(champ, base, train_df)
    return (champ_day - base_day, traded_mask, cb)


def signal_gated_term(
    individual,
    train_df,
    template,
    td,
    *,
    decision_offset: int = DEFAULT_DECISION_OFFSET,
    cvar_frac: float = DEFAULT_CVAR_FRAC,
    min_count: int = 15,
    entry_decision_mask=None,
    warmup_bars=None,
    pca_bases=None,
    emb_proj_cache=None,
    entry_floor_within_day=None,
    precomputed=None,
):
    if precomputed is not None:
        (champ, base) = precomputed
    else:
        (champ, base) = champ_and_base_trades(
            individual,
            train_df,
            template,
            td,
            decision_offset=decision_offset,
            entry_decision_mask=entry_decision_mask,
            warmup_bars=warmup_bars,
            pca_bases=pca_bases,
            emb_proj_cache=emb_proj_cache,
            entry_floor_within_day=entry_floor_within_day,
        )
    base_pnl = _per_day_sum(base, train_df)
    champ_days = set(_per_day_sum(champ, train_df).keys())
    days = sorted(base_pnl.keys(), key=lambda d: str(d))
    per_day_pnl = np.array([base_pnl[d] for d in days], dtype=np.float64)
    fired_mask = np.array([d in champ_days for d in days], dtype=bool)
    finite = np.isfinite(per_day_pnl)
    per_day_pnl = per_day_pnl[finite]
    fired_mask = fired_mask[finite]
    n_fired = int(fired_mask.sum())
    n_quiet = int((~fired_mask).sum())
    thick = min(n_fired, n_quiet) >= int(min_count)
    gain = signal_gated_cvar_gain(per_day_pnl, fired_mask, cvar_frac)
    return (gain, n_fired, n_quiet, thick)


def day_tail_cvar_ok_traded(
    champ_day_pnls,
    base_day_pnls,
    traded_mask,
    cvar_frac: float = DEFAULT_CVAR_FRAC,
    *,
    min_traded_days: int = 0,
):
    cd = np.asarray(champ_day_pnls, dtype=np.float64)
    bd = np.asarray(base_day_pnls, dtype=np.float64)
    tm = np.asarray(traded_mask, dtype=bool)
    champ_traded = cd[tm]
    base_traded = bd[tm]
    c = cvar_left_tail(champ_traded, cvar_frac)
    b = cvar_left_tail(base_traded, cvar_frac)
    n_traded = int(champ_traded.size)
    if min_traded_days > 0 and n_traded < int(min_traded_days):
        return (False, c, b)
    return (bool(c <= b), c, b)


SIGNFLIP_TRIM_FRACTION = 0.2
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


def _trimmed_mean(y, frac: float, axis: int = -1):
    y = np.asarray(y, float)
    n = y.shape[axis]
    if _OBJ1_RAW_MEAN:
        return y.mean(axis=axis)
    k = int(np.floor(frac * n))
    if _ONESIDED_UPPER_TRIM:
        if k == 0 or n - k < 1:
            return y.mean(axis=axis)
        (lo, hi) = (0, n - k)
    else:
        if n - 2 * k < 1:
            return y.mean(axis=axis)
        (lo, hi) = (k, n - k)
    ys = np.sort(y, axis=axis)
    sl = [slice(None)] * ys.ndim
    sl[axis] = slice(lo, hi)
    return ys[tuple(sl)].mean(axis=axis)


def day_signflip_pvalue(
    excess,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = 0,
    trim: float = SIGNFLIP_TRIM_FRACTION,
) -> Tuple[float, float]:
    excess = np.asarray(excess, float)
    excess = excess[np.isfinite(excess)]
    if excess.size == 0:
        return (float("nan"), float("nan"))
    obs = float(_trimmed_mean(excess, trim))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, excess.size))
    null = _trimmed_mean(signs * excess, trim, axis=1)
    p = float((1 + int(np.sum(null >= obs))) / (1 + n_perm))
    return (obs, p)


def _cms_on() -> bool:
    return os.environ.get("GP_COVERAGE_MATCHED_TEST", "") not in (
        "",
        "0",
        "false",
        "False",
    )


def _cms_stratify() -> bool:
    return os.environ.get("GP_CMS_STRATIFY", "") not in ("", "0", "false", "False")


_CMS_REGIME_COLS = (
    "RealizedVol30m",
    "VIXSpot",
    "RealizedVol",
    "ATMImpliedVol",
    "atm_iv",
)


def _per_day_vol_strata(test_df, held_out_days, n_bins: int = 3):
    try:
        cols = getattr(test_df, "columns", [])
        col = next((c for c in _CMS_REGIME_COLS if c in cols), None)
        if col is None or "date" not in cols:
            return None
        by_day = test_df.groupby("date")[col].mean()
        vals = np.array(
            [float(by_day.get(d, np.nan)) for d in held_out_days], dtype=np.float64
        )
        finite = np.isfinite(vals)
        if finite.sum() < 2:
            return None
        labels = np.full(vals.shape, -1, dtype=int)
        edges = np.quantile(vals[finite], np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
        labels[finite] = np.digitize(vals[finite], edges)
        if len(np.unique(labels)) < 2:
            return None
        return labels
    except Exception:
        return None


def cms_resolution_ok(n_days: int, k: int, alpha: float) -> bool:
    import math

    if k <= 0 or n_days < 2 or k > n_days:
        return False
    return 1.0 / float(math.comb(int(n_days), int(k))) < float(alpha)


def _strata_groups(strata, n):
    if strata is None:
        return {0: np.arange(n)}
    s = np.asarray(strata)
    return {lbl: np.where(s == lbl)[0] for lbl in np.unique(s)}


def _stratified_fire_draw(rng, groups, k_per):
    drawn = [
        rng.choice(idxs, size=k_per[lbl], replace=False)
        for (lbl, idxs) in groups.items()
        if k_per[lbl] > 0
    ]
    return np.concatenate(drawn) if drawn else np.empty(0, dtype=int)


def coverage_matched_pvalue(
    day_pnl_full,
    fire_mask,
    *,
    strata=None,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = 0,
    trim: float = SIGNFLIP_TRIM_FRACTION,
) -> Tuple[float, float]:
    y = np.asarray(day_pnl_full, dtype=np.float64)
    if not np.isfinite(y).all():
        y = np.where(np.isfinite(y), y, 0.0)
    m = np.asarray(fire_mask, dtype=bool)
    N = int(y.shape[0])
    k = int(m.sum())
    if k == 0 or N < 2 or k > N or (m.shape[0] != N):
        return (float("nan"), float("nan"))
    obs = float(_trimmed_mean(y[m], trim))
    rng = np.random.default_rng(seed)
    if strata is None:
        idx = rng.random((n_perm, N)).argsort(axis=1)[:, :k]
        null = _trimmed_mean(y[idx], trim, axis=1)
    else:
        groups = _strata_groups(strata, N)
        k_per = {lbl: int(m[idxs].sum()) for (lbl, idxs) in groups.items()}
        null = np.empty(n_perm, dtype=np.float64)
        for i in range(n_perm):
            di = _stratified_fire_draw(rng, groups, k_per)
            null[i] = float(_trimmed_mean(y[di], trim)) if di.size else obs
    p = float((1 + int(np.sum(null >= obs))) / (1 + n_perm))
    return (obs, p)


def min_days_for_signflip(alpha: float) -> int:
    a = float(alpha)
    if not 0.0 < a < 1.0:
        return 1
    return int(np.floor(np.log2(1.0 / a))) + 1


def intraday_breakdown(trades, df) -> Dict[str, Dict[str, float]]:
    if "MinutesToClose" not in getattr(df, "columns", []):
        return {}
    mtc_all = df["MinutesToClose"].values
    bins: Dict[str, List[float]] = {b: [] for b in BUCKETS}
    for t in trades:
        b = str(intraday_bucket([mtc_all[t.entry_bar]])[0])
        bins[b].append(float(t.pnl))
    return {
        b: {"mean": float(np.mean(v)) if v else float("nan"), "n": len(v)}
        for (b, v) in bins.items()
    }


def _excess_decomposition(excess, traded_mask):
    excess = np.asarray(excess, dtype=np.float64)
    skipped = ~traded_mask
    edge = float(excess[traded_mask].mean()) if traded_mask.any() else float("nan")
    credit = float(excess[skipped].mean()) if skipped.any() else 0.0
    coverage = float(traded_mask.mean()) if traded_mask.size else float("nan")
    return (edge, credit, coverage)


def _as_float(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def classify_abstention(rec, *, alpha: float = ALPHA) -> str:
    if not rec or rec.get("champion_oos_mean_pnl") is None:
        return "no_champion"
    champ_mean = _as_float(rec.get("champion_oos_mean_pnl"))
    tp = _as_float(rec.get("traded_day_p"))
    if (
        champ_mean is not None
        and champ_mean > 0.0
        and (tp is not None)
        and (tp < alpha)
    ):
        return "real_traded_edge"
    excess = _as_float(rec.get("mean_per_day_excess"))
    edge = _as_float(rec.get("traded_day_edge"))
    credit = _as_float(rec.get("selection_credit"))
    cov = _as_float(rec.get("traded_coverage"))
    if excess is not None and excess > 0.0 and (None not in (edge, credit, cov)):
        return (
            "selective_low_coverage"
            if credit * (1.0 - cov) > edge * cov
            else "weak_traded_edge"
        )
    return "no_edge"


def _accept_decision(
    n_champ_trades,
    champ_mean,
    obs_excess,
    p,
    tail_ok,
    *,
    min_trades,
    alpha,
    traded_obs=None,
    traded_p=None,
    n_excess_days=None,
    n_traded_days=None,
    cms_obs=None,
    cms_p=None,
) -> bool:
    trade_floor = 1 if cms_obs is not None and cms_p is not None else max(1, min_trades)
    base = bool(
        n_champ_trades >= trade_floor
        and champ_mean > 0
        and np.isfinite(obs_excess)
        and (obs_excess > 0)
        and np.isfinite(p)
        and (p < alpha)
        and tail_ok
    )
    d_min = min_days_for_signflip(alpha)
    if n_excess_days is not None and n_excess_days < d_min:
        base = False
    if cms_obs is not None and cms_p is not None:
        base = base and bool(
            np.isfinite(cms_obs)
            and cms_obs > 0
            and np.isfinite(cms_p)
            and (cms_p < alpha)
        )
    elif traded_obs is not None and traded_p is not None:
        base = base and bool(
            np.isfinite(traded_obs)
            and traded_obs > 0
            and np.isfinite(traded_p)
            and (traded_p < alpha)
        )
        if n_traded_days is not None and n_traded_days < d_min:
            base = False
    return base


def judge_champion(
    individual,
    test_df,
    template,
    test_td,
    *,
    n_perm: int = DEFAULT_N_PERM,
    alpha: float = ALPHA,
    decision_offset: int = DEFAULT_DECISION_OFFSET,
    perm_seed: int = 0,
    gate_decisions: bool = False,
    min_trades: int = 0,
    cvar_frac: float = DEFAULT_CVAR_FRAC,
    pca_bases=None,
    emb_proj_cache=None,
) -> Dict:
    edm = _decision_mask(test_df, decision_offset) if gate_decisions else None
    entry_floor = None if gate_decisions else int(decision_offset)
    champ = _champion_trades(
        individual,
        test_df,
        template,
        test_td,
        entry_decision_mask=edm,
        warmup_bars=decision_offset,
        pca_bases=pca_bases,
        emb_proj_cache=emb_proj_cache,
        entry_floor_within_day=entry_floor,
    )
    base = _always_enter_trades(
        test_df,
        template,
        test_td,
        decision_offset,
        shape=getattr(individual, "shape", None),
        shape_tree=getattr(individual, "shape_tree", None),
        delta_tree=getattr(individual, "delta_tree", None),
        width_tree=getattr(individual, "width_tree", None),
        warmup_bars=decision_offset,
        pca_bases=pca_bases,
        emb_proj_cache=emb_proj_cache,
    )
    champ_pnls = np.array([float(t.pnl) for t in champ], dtype=np.float64)
    (champ_day, base_day, traded_mask, held_out_days) = _per_day_pair(
        champ, base, test_df
    )
    excess = champ_day - base_day
    (obs_excess, p) = day_signflip_pvalue(excess, n_perm, perm_seed)
    raw_mean_excess = float(np.mean(excess)) if excess.size else float("nan")
    champ_mean = float(champ_pnls.mean()) if champ_pnls.size else float("nan")
    d_min_tail = 2 if _cms_on() else min_days_for_signflip(alpha)
    (tail_ok, champ_day_cvar, base_day_cvar) = day_tail_cvar_ok_traded(
        champ_day, base_day, traded_mask, cvar_frac, min_traded_days=d_min_tail
    )
    (traded_day_edge, selection_credit, traded_coverage) = _excess_decomposition(
        excess, traded_mask
    )
    champ_traded_day = (
        champ_day[traded_mask] if excess.size else np.array([], dtype=np.float64)
    )
    traded_days = [d for (d, m) in zip(held_out_days, traded_mask) if m]
    (traded_obs, traded_p) = (
        day_signflip_pvalue(champ_traded_day, n_perm, perm_seed)
        if champ_traded_day.size
        else (float("nan"), float("nan"))
    )
    cms_obs = cms_p = None
    cms_strata = None
    cms_res_ok = None
    cms_everyday_day = None
    if _cms_on() and excess.size and (int(traded_mask.sum()) > 0):
        champ_everyday = _champion_everyday_trades(
            individual,
            test_df,
            template,
            test_td,
            decision_offset,
            pca_bases=pca_bases,
            emb_proj_cache=emb_proj_cache,
        )
        ed = _per_day_sum(champ_everyday, test_df)
        cms_everyday_day = np.array(
            [ed.get(d, 0.0) for d in held_out_days], dtype=np.float64
        )
        cms_strata = (
            _per_day_vol_strata(test_df, held_out_days) if _cms_stratify() else None
        )
        (cms_obs, cms_p) = coverage_matched_pvalue(
            cms_everyday_day,
            traded_mask,
            strata=cms_strata,
            n_perm=n_perm,
            seed=perm_seed,
        )
        cms_res_ok = cms_resolution_ok(
            int(cms_everyday_day.size), int(traded_mask.sum()), alpha
        )
        if not cms_res_ok:
            cms_obs = cms_p = None
    accepted = _accept_decision(
        champ_pnls.size,
        champ_mean,
        obs_excess,
        p,
        tail_ok,
        min_trades=min_trades,
        alpha=alpha,
        traded_obs=traded_obs,
        traded_p=traded_p,
        n_excess_days=int(excess.size),
        n_traded_days=int(champ_traded_day.size),
        cms_obs=cms_obs,
        cms_p=cms_p,
    )
    return {
        "accepted": accepted,
        "champion_oos_mean_pnl": champ_mean,
        "n_champion_trades": int(champ_pnls.size),
        "mean_per_day_excess": raw_mean_excess,
        "trimmed_mean_excess": obs_excess,
        "permutation_p": p,
        "n_test_days": int(excess.size),
        "held_out_day_excess": [float(x) for x in excess],
        "held_out_days": [str(d) for d in held_out_days],
        "n_baseline_trades": len(base),
        "champion_day_cvar": champ_day_cvar,
        "baseline_day_cvar": base_day_cvar,
        "tail_ok": tail_ok,
        "traded_day_edge": traded_day_edge,
        "selection_credit": selection_credit,
        "traded_coverage": traded_coverage,
        "held_out_traded_day_pnl": [float(x) for x in champ_traded_day],
        "held_out_traded_days": [str(d) for d in traded_days],
        "traded_day_p": float(traded_p),
        "traded_day_obs": float(traded_obs),
        "intraday": intraday_breakdown(champ, test_df),
        **(
            {
                "held_out_day_pnl_full": [
                    float(x)
                    for x in (
                        cms_everyday_day
                        if cms_everyday_day is not None
                        else np.zeros(len(held_out_days), dtype=np.float64)
                    )
                ],
                "held_out_fire_mask": [bool(m) for m in traded_mask],
                "coverage_matched_obs": (
                    float(cms_obs)
                    if cms_obs is not None and np.isfinite(cms_obs)
                    else None
                ),
                "coverage_matched_p": (
                    float(cms_p) if cms_p is not None and np.isfinite(cms_p) else None
                ),
                "cms_resolution_ok": (
                    bool(cms_res_ok) if cms_res_ok is not None else None
                ),
                "cms_stratified": bool(cms_strata is not None),
            }
            if _cms_on()
            else {}
        ),
    }
