"""Measure normalized entry-comparison margins on training days.

For each GT/LT comparison with exogenous operands, compute left minus right
at the decision bar and divide by that comparison's training scale. Small
absolute margins indicate entries close to a threshold. Summarize the
margin distribution without exposing entry counts or dates to agents.

Use the evolutionary-search evaluator and trades. The decision bar is
entry_bar - 1 because fills follow signals by one bar. Check signal parity
and training/held-out separation before returning measurements.
"""
from __future__ import annotations


import re
from typing import Dict, List, Optional, Sequence

import numpy as np

_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _daykey(k) -> str:
    """Normalize ISO date and datetime representations to YYYY-MM-DD.

    Leave keys without an ISO date prefix unchanged.
    """
    s = str(k)
    return s[:10] if _DATE_PREFIX.match(s) else s

from potofgold.evolutionary_search.search.grammar import FuncNode, Node, all_nodes_indexed, _POSITION_STATE_TERMINALS

# Trace GT/LT level margins; crossing events have different geometry.
_MARGIN_OPS = frozenset({"GT", "LT"})

# Report the share of entries within 0.1 training scale of a boundary.
KNIFE_EDGE_Z = 0.10
_SCALE_EPS = 1e-9


class ParityError(RuntimeError):
    """The reconstructed entry signal disagrees with the recorded trade entry."""


class LeakError(ValueError):
    """Training trace dates overlap held-out dates."""


# Tree evaluation and margin summaries.
def _nodes(tree: Optional[Node]) -> List[Node]:
    """Walk every tree node using the grammar's shared traversal."""
    if tree is None:
        return []
    return [n for n, _d, _p in all_nodes_indexed(tree)]


def terminal_names(node: Optional[Node]) -> set:
    """Collect named data terminals, excluding sampled constants.

    Walk nested operands so position-state terminals remain detectable.
    """
    from potofgold.evolutionary_search.search.grammar import TermNode
    return {n.name for n in _nodes(node)
            if isinstance(n, TermNode) and n.value is None and n.name}


def is_exogenous(cmp_node: FuncNode) -> bool:
    """Return whether the subtree contains no position-state or P&L terminals."""
    return not (terminal_names(cmp_node) & _POSITION_STATE_TERMINALS)


def classify_comparison_nodes(tree: Optional[Node]):
    """Separate GT/LT nodes with exogenous operands from those using position state.

    Crossing operators are events and are not included in level-margin tracing.
    """
    kept: List[FuncNode] = []
    excluded: List[FuncNode] = []
    for n in _nodes(tree):
        if isinstance(n, FuncNode) and n.name in _MARGIN_OPS:
            (kept if is_exogenous(n) else excluded).append(n)
    return kept, excluded


def _eval_1d(evaluator, node) -> np.ndarray:
    """Evaluate a numeric operand as a one-dimensional float array.

    Reject vector or categorical outputs.
    """
    arr = np.asarray(evaluator.evaluate(node))
    if arr.ndim != 1:
        raise ValueError(f"operand '{getattr(node, 'name', '?')}' evaluated to ndim={arr.ndim} "
                         f"(expected 1-D REAL) — not a numeric comparison operand")
    if arr.dtype == object:
        raise ValueError(f"operand '{getattr(node, 'name', '?')}' is object-dtype (enum?), not numeric")
    return arr.astype(np.float64)


def margin_series(evaluator, cmp_node: FuncNode) -> np.ndarray:
    """Evaluate the signed difference, left minus right, on every bar."""
    return _eval_1d(evaluator, cmp_node.children[0]) - _eval_1d(evaluator, cmp_node.children[1])


def robust_scale(x: Sequence[float]) -> Optional[float]:
    """Use standard deviation, then 1.4826 times MAD, to scale a margin series.

    Return None when the margin is effectively constant.
    """
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return None
    s = float(np.std(a))
    if s > _SCALE_EPS:
        return s
    mad = float(np.median(np.abs(a - np.median(a)))) * 1.4826
    return mad if mad > _SCALE_EPS else None


def normalized_margins_at_bars(evaluator, cmp_nodes: Sequence[FuncNode], decision_bars: Sequence[int]):
    """Normalize each comparison's margins and select its decision-bar values.

    Return the margin matrix and number of constant-scale comparisons omitted.
    """
    db = np.asarray(decision_bars, dtype=np.int64)
    cols: List[np.ndarray] = []
    n_degen = 0
    for node in cmp_nodes:
        raw = margin_series(evaluator, node)           # Estimate scale from all training bars.
        sc = robust_scale(raw)
        if sc is None:
            n_degen += 1
            continue
        cols.append(raw[db] / sc)
    margins = np.column_stack(cols) if cols else np.empty((len(db), 0), dtype=np.float64)
    return margins, n_degen


def binding_margin(margins: np.ndarray) -> np.ndarray:
    """Select the signed margin closest to zero on each entry.

    Ignore nonfinite operands; leave rows without a finite margin as NaN.
    """
    if margins.size == 0:
        return np.empty(0, dtype=np.float64)
    absm = np.abs(margins)
    safe = np.where(np.isfinite(absm), absm, np.inf)   # Nonfinite operands cannot determine the closest margin.
    j = np.argmin(safe, axis=1)
    return margins[np.arange(margins.shape[0]), j]


def summarize(signed: Sequence[float]) -> Dict:
    """Summarize normalized margins with quantiles, threshold proximity and sign balance.

    Return availability as a Boolean rather than a raw entry count.
    """
    s = np.asarray(signed, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {"n_available": False}
    a = np.abs(s)
    q = np.quantile(s, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "n_available": True,
        "p10": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
        "p75": float(q[3]), "p90": float(q[4]),
        "mean": float(s.mean()), "std": float(s.std()),
        "abs_median": float(np.median(a)),
        "knife_edge_frac": float(np.mean(a < KNIFE_EDGE_Z)),   # Fraction within 0.1 scale of the threshold.
        "positive_frac": float(np.mean(s > 0)),
    }


def check_parity(evaluator, entry_tree: Node, decision_bars: Sequence[int], *, thresh: float = 0.5) -> bool:
    """Require a true entry signal at every actual trade's decision bar.

    This checks executed entries; it does not imply that every signal fills.
    """
    db = np.asarray(decision_bars, dtype=np.int64)
    if len(db) == 0:
        return True
    sig = _eval_1d(evaluator, entry_tree)
    ok = sig[db] > thresh
    if not np.all(ok):
        bad = db[~ok]
        raise ParityError(
            f"entry-signal parity failed at {int((~ok).sum())} decision bar(s) (e.g. {bad[:5].tolist()}): "
            f"my evaluator disagrees with _champion_trades' entries — eval-path drift")
    return True


def assert_disjoint_days(train_days: Sequence, heldout_days: Sequence) -> bool:
    """Reject overlap between training and held-out dates after normalizing date keys."""
    inter = {str(d) for d in train_days} & {str(d) for d in heldout_days}
    if inter:
        raise LeakError(
            f"train ∩ held-out non-empty ({len(inter)} days, e.g. {sorted(inter)[:3]}) — "
            f"margins would be computed on labeled days")
    return True


def trace_with_evaluator(evaluator, individual, decision_bars: Sequence[int], *,
                         train_days: Optional[Sequence] = None,
                         heldout_days: Optional[Sequence] = None,
                         bar_dates: Optional[Sequence] = None) -> Dict:
    """Check split separation and signal parity, then summarize entry margins.

    traceable distinguishes a lack of eligible comparisons from a lack of entries.
    """
    if train_days is not None and heldout_days is not None:
        assert_disjoint_days(train_days, heldout_days)
    entry_tree = individual.entry_tree
    kept, excluded = classify_comparison_nodes(entry_tree)
    check_parity(evaluator, entry_tree, decision_bars)
    margins, n_degen = normalized_margins_at_bars(evaluator, kept, decision_bars)
    from potofgold.multi_agent.diagnostics.fire_temporal import fire_temporal_metrics
    bm = binding_margin(margins)
    return {
        "n_cmp_nodes": len(kept),
        "n_traced": int(margins.shape[1]),
        "n_excluded_position_state": len(excluded),
        "n_degenerate_scale": n_degen,
        "traceable": len(kept) > 0,
        "binding": summarize(bm),
        "per_node": [summarize(margins[:, j]) for j in range(margins.shape[1])] if margins.size else [],
        "fire_temporal": (fire_temporal_metrics(decision_bars, bm, bar_dates) if bar_dates is not None
                          else {"fire_day_gap_gini": None, "behavior_half_consistency": None}),
        "parity_ok": True,
    }


# Training preparation and replay.
def decision_bars_from_trades(trades) -> np.ndarray:
    """Map fills to decision bars using max(entry_bar - 1, 0)."""
    return np.asarray([max(int(t.entry_bar) - 1, 0) for t in trades], dtype=np.int64)


def build_evaluator(train_df, td_tr, *, bases_k=None, emb_cache_tr=None, default_iv=0.20):
    """Construct the shared evaluator over the training slice.

    Use the simulator's static-array preparation and default IV. Fold recentering
    is omitted because these day-grouped runs recorded it as dropped.
    """
    from potofgold.evolutionary_search.search.simulator import VectorizedTreeEvaluator, prepare_static_arrays
    sa = prepare_static_arrays(train_df, default_iv)
    return VectorizedTreeEvaluator(td_tr, len(train_df),
                                   day_boundary_mask=sa["day_mask"], within_day_pos=sa["within_day_pos"],
                                   pca_bases=bases_k, emb_proj_cache=emb_cache_tr)


def _assert_basis_flags_match(rc, effective_flags):
    """Require import-time target residualization settings to match recorded provenance.

    Set the compatible GP_D1_* environment variables before importing embedding_data.
    """
    ef = effective_flags or {}
    want = {"_D1_RESIDUALIZE": str(ef.get("GP_D1_RESIDUALIZE", "0")) == "1",
            "_D1_SCOPE": str(ef.get("GP_D1_RESIDUALIZE_SCOPE", "all")).strip().lower(),
            "_D1_ESTIMATOR": str(ef.get("GP_D1_ESTIMATOR", "ridge")).strip().lower()}
    for attr, wv in want.items():
        gv = getattr(rc, attr, None)
        if gv != wv:
            raise RuntimeError(f"basis-config flag mismatch: embedding_data.{attr}={gv!r} but the champion "
                               f"needs {wv!r}. Set GP_D1_* in os.environ BEFORE importing embedding_data.")


def build_embedding_train_prep(*, data, targets, basis, grammar_mode, template, fold, n_folds,
                               gate=False, decision_offset_ungated=2, gated_decision_offset=None,
                               effective_flags=None):
    """Prepare training inputs for embedding or combined-input strategies.

    Fit the basis on training rows, preserve split dates and use the recorded
    decision offsets and flags. Keep the aligned corpus for subsequent basis
    refits. Indicator-only strategies use build_scalar_train_prep.
    """
    import os
    import potofgold.multi_agent.input_tools.embedding_data as rc

    if grammar_mode not in ("emb", "combined"):
        raise ValueError(f"build_embedding_train_prep handles only embedding/combined arms; got "
                         f"grammar_mode={grammar_mode!r} (scalar-A has a separate prep)")
    # Residualization flags must match the settings read at module import.
    _assert_basis_flags_match(rc, effective_flags)
    for _k, _v in (effective_flags or {}).items():
        if _v is not None:
            os.environ[str(_k)] = str(_v)
    os.environ["GP_CVAR_TAIL_MAX_AUTO"] = "0" if gate else "1"
    df, td_full, n = rc._load_and_prepare(data)
    if n_folds is None:
        n_folds = n
    tr, te = rc.kfold_day_split(df, fold, n_folds, seed=rc.FOLD_PARTITION_SEED)
    rc.assert_basis_fit_is_causal(tr, te, df)
    train_df, td_tr = rc.slice_by_mask(df, td_full, tr)
    test_df = df.loc[te]
    targets_, valid, pidx = rc._load_minute_targets(targets, df)
    # Clear the object-ID cache before a discarded dataframe ID can be reused.
    surface_cache = getattr(rc, "_D1_SURFACE_CACHE", None)
    if surface_cache is not None:
        surface_cache.pop(id(df), None)
    bases_k, info = rc._fit_fold_basis(df, tr, basis, targets_, valid, primary_target_idx=pidx)
    from potofgold.multi_agent.diagnostics.basis_refit import make_basis_refit_fn
    # Keep the aligned corpus so refits use the same scalar history.
    frozen_flags = dict(effective_flags or {})
    def check_refit_config():
        _assert_basis_flags_match(rc, frozen_flags)
        for key, value in frozen_flags.items():
            if value is not None and os.environ.get(str(key)) != str(value):
                raise RuntimeError(f"basis replay environment changed: {key}")
    basis_refit_fn = make_basis_refit_fn(
        df=df, train_mask=tr, basis_method=basis, targets=targets_, targets_valid=valid,
        primary_target_idx=pidx, fit_fn=rc._fit_fold_basis, config_check=check_refit_config)
    if surface_cache is not None:
        import weakref
        weakref.finalize(basis_refit_fn, surface_cache.pop, id(df), None)
    # Compute only projections used by this strategy instead of a population-wide cache.
    emb_cache_tr = None
    tmpl = rc.base_template_by_name(template)
    gated_off = rc.DECISION_OFFSET_REAL if gated_decision_offset is None else int(gated_decision_offset)
    off = gated_off if gate else int(decision_offset_ungated)
    return {
        "train_df": train_df, "td_tr": td_tr, "tmpl": tmpl, "bases_k": bases_k,
        "emb_cache_tr": emb_cache_tr, "off": off,
        "train_days": [str(d) for d in sorted(train_df["date"].unique())],
        "heldout_days": [str(d) for d in sorted(test_df["date"].unique())],
        "basis_info": info,
        "basis_refit_fn": basis_refit_fn,
    }


def build_scalar_train_prep(*, template, fold, n_folds=None, data=None, decision_offset_ungated=2,
                            gated_decision_offset=None, gate=False, effective_flags=None):
    """Prepare indicator-only training inputs with the shared evaluator helpers.

    Compute calendar, gamma and terminal data on the ordered corpus before
    slicing the fold. No embedding basis is fitted.
    """
    import os
    from potofgold.evolutionary_search.search.data import load_minute_parquet
    from potofgold.multi_agent.diagnostics.calendar_features import add_calendar_columns
    from potofgold.multi_agent.input_tools.scalar_data import assert_gamma_present
    from potofgold.evolutionary_search.search.simulator import prepare_terminal_data
    from potofgold.evolutionary_search.search.validation import kfold_day_split, slice_by_mask
    from potofgold.evolutionary_search.search.templates import base_template_by_name
    import potofgold.multi_agent.input_tools.scalar_data as rca                 # Import lazily to avoid the input-module cycle.

    # Restore recorded flags before preparing scalar surfaces.
    for _k, _v in (effective_flags or {}).items():
        if _v is not None:
            os.environ[str(_k)] = str(_v)
    os.environ["GP_CVAR_TAIL_MAX_AUTO"] = "0" if gate else "1"

    data = data or os.environ.get("CAT_RUNNER_SCALAR_DATA")
    if not data:
        raise ValueError("Supply scalar market inputs through data or CAT_RUNNER_SCALAR_DATA")
    df, _ = load_minute_parquet(data)
    if hasattr(rca, "assert_grid_present"):
        rca.assert_grid_present(df)                       # Pricing credit factors require the option-IV grid.
    if n_folds is None:
        n_folds = rca.N_FOLDS
    tr, te = kfold_day_split(df, fold, n_folds, seed=rca.FOLD_PARTITION_SEED)
    add_calendar_columns(df)
    assert_gamma_present(df)
    td_full = prepare_terminal_data(df)                   # Compute terminals on the ordered corpus before slicing training rows.
    train_df, td_tr = slice_by_mask(df, td_full, tr)
    test_df = df.loc[te]
    tmpl = base_template_by_name(template)
    gated_off = int(gated_decision_offset) if gated_decision_offset is not None else int(decision_offset_ungated)
    off = gated_off if gate else int(decision_offset_ungated)
    return {
        "train_df": train_df, "td_tr": td_tr, "tmpl": tmpl, "bases_k": None,
        "emb_cache_tr": None, "off": off,
        "train_days": [str(d) for d in sorted(train_df["date"].unique())],
        "heldout_days": [str(d) for d in sorted(test_df["date"].unique())],
        "basis_info": {},
    }


def reconstructed_day_pnl(trades, dates) -> Dict:
    """Sum trade P&L by entry day using the shared scorer aggregation."""
    from potofgold.evolutionary_search.search.fitness import per_day_sum_pnl
    return dict(per_day_sum_pnl(trades, dates))


def fidelity_vs_stored(recon_day_pnl: Dict, stored_day_pnl, tol: float = 1e-4) -> Dict:
    """Compare reconstructed daily P&L with the saved training values.

    Require the same number of entries in the daily series and sorted values
    within tol. The saved reference lacks date labels, so this check cannot
    verify which day produced each value.
    """
    recon = sorted(float(v) for v in recon_day_pnl.values()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v)
    stored = sorted(float(x) for x in (stored_day_pnl or [])
                    if isinstance(x, (int, float)) and not isinstance(x, bool) and x == x)
    if len(recon) != len(stored):
        return {"ok": False, "maxdelta": float("inf"), "n_recon": len(recon), "n_stored": len(stored)}
    maxd = max((abs(a - b) for a, b in zip(recon, stored)), default=0.0)
    return {"ok": bool(maxd <= tol), "maxdelta": float(maxd), "n_recon": len(recon), "n_stored": len(stored)}


def trace_champion(genome: dict, prep: dict, *, decision_offset: Optional[int] = None,
                   stored_day_pnl=None, fidelity_tol: float = 1e-4) -> Dict:
    """Reconstruct training trades and summarize their decision margins.

    When saved daily P&L is supplied, check reconstruction against it and
    record fidelity_ok. Split and signal-parity checks also apply.
    """
    import numpy as _np
    from potofgold.evolutionary_search.search.genetics import individual_from_genome_dict
    from potofgold.evolutionary_search.search.validation import _champion_trades

    ind = individual_from_genome_dict(genome)
    off = prep["off"] if decision_offset is None else decision_offset
    trades = _champion_trades(
        ind, prep["train_df"], prep["tmpl"], prep["td_tr"],
        entry_decision_mask=None, warmup_bars=off,
        pca_bases=prep.get("bases_k"), emb_proj_cache=prep.get("emb_cache_tr"),
        entry_floor_within_day=int(off))
    dates = _np.asarray(prep["train_df"]["date"].values)
    fid = None
    if stored_day_pnl is not None:
        fid = fidelity_vs_stored(reconstructed_day_pnl(trades, dates), stored_day_pnl, fidelity_tol)
    decision_bars = decision_bars_from_trades(trades)
    ev = build_evaluator(prep["train_df"], prep["td_tr"],
                         bases_k=prep.get("bases_k"), emb_cache_tr=prep.get("emb_cache_tr"))
    out = trace_with_evaluator(ev, ind, decision_bars,
                               train_days=prep.get("train_days"), heldout_days=prep.get("heldout_days"),
                               bar_dates=dates)
    out["n_trades"] = int(len(decision_bars))   # Case rendering omits this internal replay diagnostic.
    if fid is not None:
        out["fidelity_ok"] = fid["ok"]
        out["_fidelity_maxdelta"] = fid["maxdelta"]
    return out
