"""Measure entry-day changes under training threshold and operand perturbations.

Move evolved comparison constants by fixed fractions of training margin
scale and replay the strategy. Optional signal interventions cross-fit
conditional means and exchange whole-day residual blocks. Withhold
unsupported interventions.

Report agreement ratios and structural counts after baseline reconstruction
and entry-parity checks. Entry dates, counts and outcomes stay internal.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from potofgold.evolutionary_search.search.grammar import FuncNode, TermNode, Node, all_nodes_indexed, _node_at_path, _replace_at_path
from potofgold.multi_agent.diagnostics.decision_tracer import _MARGIN_OPS, _eval_1d, margin_series, robust_scale

DEFAULT_EPS_GRID: Tuple[float, ...] = (0.1, 0.25, 0.5)   # Fractions of each comparison's training margin scale.
_SCALE_EPS = 1e-9


# Identify eligible threshold perturbations.
def _is_perturbable_const(node: Node) -> bool:
    """Identify a sampled floating-point REAL threshold; exclude integers and Booleans."""
    return (isinstance(node, TermNode) and isinstance(node.value, float)
            and not isinstance(node.value, bool))


def _is_int_const(node: Node) -> bool:
    """Identify an integer structural constant, excluding Booleans."""
    return (isinstance(node, TermNode) and isinstance(node.value, int)
            and not isinstance(node.value, bool))


def _enclosing_comparison_path(tree: Node, const_path: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """Find the nearest enclosing GT/LT node, or return None."""
    for k in range(len(const_path) - 1, -1, -1):
        anc = _node_at_path(tree, const_path[:k])
        if isinstance(anc, FuncNode) and anc.name in _MARGIN_OPS:
            return const_path[:k]
    return None


def perturbation_targets(entry_tree: Optional[Node]) -> Dict:
    """Find floating-point constants that are direct GT/LT operands.

    For direct operands, moving the constant changes the margin by the same
    magnitude. Count but omit constants nested inside other operations, integer
    constants and constants outside comparisons.
    """
    if entry_tree is None:
        return {"targets": [], "n_int_skipped": 0, "n_no_comparison": 0, "n_nested_skipped": 0}
    targets: List[Dict] = []
    n_int = n_no_cmp = n_nested = 0
    for node, _d, path in all_nodes_indexed(entry_tree):
        if _is_int_const(node):
            n_int += 1
            continue
        if not _is_perturbable_const(node):
            continue
        parent = _node_at_path(entry_tree, path[:-1]) if path else None
        if isinstance(parent, FuncNode) and parent.name in _MARGIN_OPS:
            targets.append({"const_path": path, "cmp_path": path[:-1]})
        elif _enclosing_comparison_path(entry_tree, path) is not None:
            n_nested += 1
        else:
            n_no_cmp += 1
    return {"targets": targets, "n_int_skipped": n_int, "n_no_comparison": n_no_cmp,
            "n_nested_skipped": n_nested}


def comparison_operands(entry_tree: Optional[Node]) -> List[Dict]:
    """Collect nonconstant GT/LT operands with their structural expression labels."""
    if entry_tree is None:
        return []
    from potofgold.evolutionary_search.search.grammar import to_str
    out: List[Dict] = []
    for node, _d, path in all_nodes_indexed(entry_tree):
        if isinstance(node, FuncNode) and node.name in _MARGIN_OPS:
            for j, ch in enumerate(node.children):
                if not (_is_perturbable_const(ch) or _is_int_const(ch)):
                    out.append({"path": path + (j,), "label": to_str(ch)})
    return out


def jaccard(a: set, b: set) -> float:
    """Return intersection over union for entry-day sets; two empty sets have agreement 1."""
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def phi_hat(sets: Sequence[set], d: int) -> float:
    """Compute chance-corrected set stability from Nogueira, Sechidis and Brown (2018).

    For M subsets of a universe of size d, use
    Phi = 1 - mean(s_f^2) / ((k_mean/d) * (1 - k_mean/d)),
    where s_f^2 = M/(M-1) * p_f * (1-p_f). One denotes identical selection;
    zero denotes chance-level agreement. Return NaN for insufficient subsets,
    an empty universe or a zero denominator.
    """
    m = len(sets)
    if m < 2 or d <= 0:
        return float("nan")
    from collections import Counter
    cnt: Counter = Counter()
    total = 0
    for s in sets:
        cnt.update(s)
        total += len(s)
    kbar = total / m
    if kbar <= 0 or kbar >= d:
        return float("nan")
    # Unselected universe items contribute zero to the variance sum.
    ssum = sum((m / (m - 1)) * (c / m) * (1.0 - c / m) for c in cnt.values())
    denom = (kbar / d) * (1.0 - kbar / d)
    return float(1.0 - (ssum / d) / denom)


def _agg(values: Sequence[float]) -> Dict:
    v = [x for x in values if x == x]
    if not v:
        return {"min": float("nan"), "mean": float("nan"), "n": 0}
    return {"min": float(min(v)), "mean": float(sum(v) / len(v)), "n": len(v)}


def perturbation_summary(baseline: set, perturbed_by_eps: Dict[float, List[set]], d: int) -> Dict:
    """Summarize minimum and mean agreement across perturbation directions at each epsilon."""
    curve: Dict[str, Dict] = {}
    for eps in sorted(perturbed_by_eps):
        sets = perturbed_by_eps[eps]
        phi = _agg([phi_hat([baseline, s], d) for s in sets])
        jac = _agg([jaccard(baseline, s) for s in sets])
        curve[f"{eps:g}"] = {"phi_min": phi["min"], "phi_mean": phi["mean"],
                             "jaccard_min": jac["min"], "jaccard_mean": jac["mean"],
                             "n_perturbations": len(sets)}
    return curve


# Correlation between entry operands.
def terminal_correlations(evaluator, entry_tree: Optional[Node]) -> Dict:
    """Absolute Pearson correlation among distinct comparison operands on training bars.
    This is descriptive context, not sufficient validation for a conditional intervention.
    Returns maximum off-diagonal correlation overall and per operand, using the real evaluator."""
    ops = comparison_operands(entry_tree)
    seen: Dict[str, np.ndarray] = {}
    labels: List[str] = []
    for op in ops:
        if op["label"] in seen:
            continue
        try:
            series = _eval_1d(evaluator, _node_at_path(entry_tree, op["path"]))
        except Exception:                                  # noqa: BLE001
            continue
        fin = series[np.isfinite(series)]
        if fin.size < 2 or float(np.std(fin)) <= _SCALE_EPS:
            continue
        seen[op["label"]] = series
        labels.append(op["label"])
    n = len(labels)
    if n < 2:
        return {"n_operands": n, "max_abs_corr": None, "per_operand": []}
    mat = np.vstack([seen[l] for l in labels])
    finmask = np.all(np.isfinite(mat), axis=0)            # Use a common finite-row subset for all operands.
    if int(finmask.sum()) > 2:
        with np.errstate(invalid="ignore", divide="ignore"):   # A constant column on the common subset has undefined correlation.
            corr = np.corrcoef(mat[:, finmask])
    else:
        corr = np.full((n, n), np.nan)
    np.fill_diagonal(corr, np.nan)
    per = [{"label": labels[i],
            "max_abs_corr": (float(np.nanmax(np.abs(corr[i]))) if np.isfinite(corr[i]).any() else None)}
           for i in range(n)]
    overall = max((p["max_abs_corr"] for p in per if p["max_abs_corr"] is not None), default=None)
    return {"n_operands": n, "max_abs_corr": overall, "per_operand": per}


# Create modified trees without mutating the baseline.


def perturbed_entry_tree(entry_tree: Node, const_path: Tuple[int, ...], new_value: float) -> Node:
    """Return a new tree with one threshold replaced, preserving its constant definition."""
    orig = _node_at_path(entry_tree, const_path)
    return _replace_at_path(entry_tree, const_path, TermNode(defn=orig.defn, value=float(new_value)))


# Training preparation and replay.
def _fired_days(trades, dates) -> set:
    """Return the entry days of executed trades using the shared daily aggregator."""
    from potofgold.multi_agent.diagnostics.decision_tracer import reconstructed_day_pnl
    return set(reconstructed_day_pnl(trades, dates).keys())


def _run_trades(individual, prep):
    """Replay the training strategy with its recorded decision offset."""
    from potofgold.evolutionary_search.search.validation import _champion_trades
    off = prep["off"]
    return _champion_trades(
        individual, prep["train_df"], prep["tmpl"], prep["td_tr"],
        entry_decision_mask=None, warmup_bars=off,
        pca_bases=prep.get("bases_k"), emb_proj_cache=prep.get("emb_cache_tr"),
        entry_floor_within_day=int(off))


def probe_champion(genome: dict, prep: dict, *, eps_grid: Sequence[float] = DEFAULT_EPS_GRID,
                   ablate: bool = False, stored_day_pnl=None, fidelity_tol: float = 1e-4) -> Dict:
    """Replay threshold perturbations and summarize changes in entry days.

    Validate baseline reconstruction, split separation and signal parity. Move
    each eligible threshold by fixed fractions of its comparison's training
    scale. Optionally run validated conditional operand interventions.
    traceable distinguishes absent thresholds from absent entries.
    """
    import dataclasses
    import numpy as _np
    from potofgold.evolutionary_search.search.genetics import individual_from_genome_dict
    from potofgold.multi_agent.diagnostics.decision_tracer import build_evaluator, reconstructed_day_pnl, fidelity_vs_stored, check_parity, decision_bars_from_trades, assert_disjoint_days

    # This path checks split separation independently of the margin tracer.
    if prep.get("train_days") is not None and prep.get("heldout_days") is not None:
        assert_disjoint_days(prep["train_days"], prep["heldout_days"])

    base_ind = individual_from_genome_dict(genome)
    entry_tree = base_ind.entry_tree
    dates = _np.asarray(prep["train_df"]["date"].values)
    # Use the same date representation for the universe and entry-day sets.
    d_universe = int(_np.unique(dates).size)

    base_trades = _run_trades(base_ind, prep)
    baseline = _fired_days(base_trades, dates)
    fid = None
    if stored_day_pnl is not None:
        fid = fidelity_vs_stored(reconstructed_day_pnl(base_trades, dates), stored_day_pnl, fidelity_tol)

    ev = build_evaluator(prep["train_df"], prep["td_tr"],
                         bases_k=prep.get("bases_k"), emb_cache_tr=prep.get("emb_cache_tr"))
    check_parity(ev, entry_tree, decision_bars_from_trades(base_trades))
    corr = terminal_correlations(ev, entry_tree)

    # Threshold stability requires an eligible comparison and baseline entries.
    tinfo = perturbation_targets(entry_tree)
    traceable = bool(tinfo["targets"]) and bool(baseline)
    scale_cache: Dict[Tuple[int, ...], Optional[float]] = {}
    perturbed_by_eps: Dict[float, List[set]] = {float(e): [] for e in eps_grid}
    per_node: List[Dict] = []
    n_degenerate = 0
    for tgt in (tinfo["targets"] if traceable else []):
        cp = tgt["cmp_path"]
        if cp not in scale_cache:
            scale_cache[cp] = robust_scale(margin_series(ev, _node_at_path(entry_tree, cp)))
        scale = scale_cache[cp]
        base_val = float(_node_at_path(entry_tree, tgt["const_path"]).value)
        if scale is None or scale <= _SCALE_EPS:          # A constant margin supplies no usable perturbation scale.
            n_degenerate += 1
            continue
        node_phis: List[float] = []
        for e in eps_grid:
            for sign in (+1.0, -1.0):
                tree = perturbed_entry_tree(entry_tree, tgt["const_path"], base_val + sign * float(e) * scale)
                pert = dataclasses.replace(base_ind, entry_tree=tree)
                fired = _fired_days(_run_trades(pert, prep), dates)
                perturbed_by_eps[float(e)].append(fired)
                node_phis.append(phi_hat([baseline, fired], d_universe))
        finite = [x for x in node_phis if x == x]
        per_node.append({"cmp": _node_at_path(entry_tree, cp).name,
                         "worst_phi": float(min(finite)) if finite else float("nan")})

    curve = perturbation_summary(baseline, perturbed_by_eps, d_universe) if traceable else {}

    # Conditional interventions require training-data support and dependence checks.
    ablation = None
    if ablate:
        from potofgold.multi_agent.diagnostics.conditional_intervention import conditional_signal_profile
        ablation = conditional_signal_profile(base_ind, prep, ev, baseline)

    out = {
        "n_targets": len(tinfo["targets"]),
        "n_int_skipped": tinfo["n_int_skipped"],
        "n_no_comparison": tinfo["n_no_comparison"],
        "n_nested_skipped": tinfo["n_nested_skipped"],
        "n_degenerate_scale": n_degenerate,
        "traceable": traceable,
        "curve": curve,
        "per_node": per_node,
        "correlation": corr,
        "parity_ok": True,
        "n_trades": int(len(base_trades)),
    }
    if ablation is not None:
        out["ablation"] = ablation
    if fid is not None:
        out["fidelity_ok"] = fid["ok"]
        out["_fidelity_maxdelta"] = fid["maxdelta"]
    return out
