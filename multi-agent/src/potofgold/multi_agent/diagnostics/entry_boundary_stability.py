"""Measure training entry sensitivity to fixed GT/LT boundary offsets.

Keep other trees, embedding coordinates and the fitted basis fixed. Report
signal changes separately from trade changes: unchanged entry days can
still contain different decisions or execution times.
"""
from __future__ import annotations

from dataclasses import replace
import math
from numbers import Integral
from typing import Any

import numpy as np


METHOD = "fixed_basis_entry_margin_offsets_v1"
EPS_GRID = (0.05, 0.10)
MAX_BOUNDARIES = 8
_SCALE_EPS = 1e-9


class BoundaryValidationError(ValueError):
    """The proposed measurement cannot be interpreted reliably."""


def boundary_targets(tree) -> list[dict[str, Any]]:
    """Group identical GT/LT comparisons; move all identical occurrences together.

    Composite REAL operands are supported: the offset is applied outside the
    computed RHS, not inside its transformation. Nested comparisons (for example
    a comparison inside a numeric conditional used by another comparison) are
    withheld to avoid overlapping interventions. Cross events are not targets.
    """
    from potofgold.evolutionary_search.search.grammar import FuncNode, all_nodes_indexed, to_str

    if tree is None:
        return []
    groups = {}
    paths = []
    for node, _, path in all_nodes_indexed(tree):
        if isinstance(node, FuncNode) and node.name in {"GT", "LT"}:
            groups.setdefault(to_str(node), []).append(path)
            paths.append(path)
    if any(p != q and len(p) < len(q) and q[:len(p)] == p for p in paths for q in paths):
        raise BoundaryValidationError("nested_comparison_boundaries")
    if len(groups) > MAX_BOUNDARIES:
        raise BoundaryValidationError("boundary_budget_exceeded")
    return [{"label": label, "paths": paths} for label, paths in groups.items()]


def offset_boundary(tree, paths, offset: float):
    """Functional rewrite: lhs OP rhs becomes lhs OP Add(rhs, offset)."""
    from potofgold.evolutionary_search.search.grammar import ADD, FuncNode, GType, TermDef, TermNode, _node_at_path, _replace_at_path

    if not math.isfinite(offset):
        raise BoundaryValidationError("nonfinite_boundary_offset")
    for path in paths:
        comparison = _node_at_path(tree, path)
        if not isinstance(comparison, FuncNode) or comparison.name not in {"GT", "LT"}:
            raise BoundaryValidationError("unsupported_comparison_boundary")
        rhs = FuncNode(ADD, [comparison.children[1],
                            TermNode(TermDef("EphReal", GType.REAL), value=float(offset))])
        tree = _replace_at_path(tree, path, FuncNode(comparison.defn, [comparison.children[0], rhs]))
    return tree


def eligible_entry_mask(prep: dict) -> np.ndarray:
    """Find signal bars with valid same-session, next-bar fill timing.

    Exclude warm-up, cutoff and final-session bars, and invalid spot or time
    data. Capital, position state and trade caps are checked by trade replay.
    """
    from potofgold.evolutionary_search.search.simulator import ENTRY_CUTOFF_MTC, prepare_static_arrays

    off = prep.get("off")
    if isinstance(off, bool) or not isinstance(off, Integral) or off < 0:
        raise BoundaryValidationError("invalid_entry_floor")
    if prep.get("entry_decision_mask") is not None or prep.get("gate") is True:
        raise BoundaryValidationError("gated_entry_replay_not_supported")
    df = prep["train_df"]
    if "MinutesToClose" not in df or not any(k in df for k in ("SPXClose", "spx_close", "close")):
        raise BoundaryValidationError("entry_timing_data_required")
    arrays = prepare_static_arrays(df)
    spot, mtc, dates = arrays["spot_arr"], arrays["mtc_arr"], arrays["date_arr"]
    valid = np.isfinite(spot) & (spot > 0) & np.isfinite(mtc) & (mtc > ENTRY_CUTOFF_MTC)
    following = np.zeros(len(df), dtype=bool)
    following[:-1] = valid[1:] & (dates[1:] == dates[:-1])
    return valid & following & (arrays["within_day_pos"] >= int(off)) & (np.arange(len(df)) >= int(off))


def _margin_scale(values: np.ndarray) -> float:
    """Training MAD scale, with an IQR fallback; no outcome-dependent fitting."""
    if len(values) < 3 or not np.all(np.isfinite(values)):
        raise BoundaryValidationError("insufficient_finite_boundary_support")
    scale = 1.4826 * float(np.median(np.abs(values - np.median(values))))
    if scale <= _SCALE_EPS:
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
    if not math.isfinite(scale) or scale <= _SCALE_EPS:
        raise BoundaryValidationError("degenerate_margin_scale")
    return scale


def _require_entry_terminal_data(tree, prep):
    """Require raw inputs and the fitted prototype bank used by the evaluator.

    PROTO terminals resolve from the bank and use their group's raw embedding
    to mask cold bars. Both sources must be available.
    """
    from potofgold.evolutionary_search.search.grammar import TermNode, all_nodes_indexed

    data = prep["td_tr"]
    for node, _, _ in all_nodes_indexed(tree):
        if not isinstance(node, TermNode) or node.value is not None:
            continue
        if not node.name.startswith("PROTO_"):
            if node.name not in data or data[node.name] is None:
                raise BoundaryValidationError("missing_entry_terminal_data")
            continue
        try:
            group, index = node.name[len("PROTO_"):].rsplit("_", 1)
            index = int(index)
        except (ValueError, TypeError):
            raise BoundaryValidationError("invalid_fixed_prototype_basis") from None
        if group != node.ret_type.name or not group.startswith("EMB_") or index < 0:
            raise BoundaryValidationError("invalid_fixed_prototype_basis")
        if group not in data or data[group] is None:
            raise BoundaryValidationError("missing_entry_terminal_data")
        basis = (prep.get("bases_k") or {}).get(group)
        if not isinstance(basis, dict) or basis.get("proto_bank_") is None:
            raise BoundaryValidationError("fixed_prototype_basis_required")
        try:
            bank = np.asarray(basis["proto_bank_"], dtype=float)
            raw = np.asarray(data[group], dtype=float)
        except (ValueError, TypeError):
            raise BoundaryValidationError("invalid_fixed_prototype_basis") from None
        if (bank.ndim != 2 or not bank.shape[1] or not 0 <= index < bank.shape[0]
                or not np.all(np.isfinite(bank))):
            raise BoundaryValidationError("invalid_fixed_prototype_basis")
        if (raw.shape != (len(prep["train_df"]), bank.shape[1])
                or not np.all(np.isfinite(raw))):
            raise BoundaryValidationError("invalid_prototype_terminal_data")


def _require_fixed_projection_bases(tree, prep):
    """Require the recorded fold basis for each projected operand."""
    from potofgold.evolutionary_search.search.grammar import FuncNode, all_nodes_indexed

    for node, _, _ in all_nodes_indexed(tree):
        if not isinstance(node, FuncNode):
            continue
        if node.name.startswith("EmbProj_"):
            group, index = node.name[len("EmbProj_"):].rsplit("_", 1)
            keys = ("components", "mean_", "pc_std_")
        elif node.name.startswith("EmbDir_"):
            group, index = node.name[len("EmbDir_"):], None
            keys = ("dir_bank_", "mean_", "dir_scale_")
        else:
            continue
        basis = (prep.get("bases_k") or {}).get(group)
        if not isinstance(basis, dict) or any(basis.get(key) is None for key in keys):
            raise BoundaryValidationError("fixed_projection_basis_required")
        matrix, center, scales = (np.asarray(basis[key]) for key in keys)
        if (matrix.ndim != 2 or not matrix.shape[0] or center.shape != (matrix.shape[1],)
                or scales.shape != (matrix.shape[0],)
                or not all(np.all(np.isfinite(v)) for v in (matrix, center, scales))
                or (index is not None and not 0 <= int(index) < matrix.shape[0])):
            raise BoundaryValidationError("invalid_fixed_projection_basis")


def entry_boundary_stability(genome: dict, prep: dict) -> dict:
    """Sweep every eligible comparison by +/-0.05 and +/-0.10 training margin scales.

    Higher statistics indicate more signal movement. Withhold the result when
    a boundary or scenario cannot be validated. P&L is used for baseline replay
    validation and is not returned.
    """
    from potofgold.evolutionary_search.search.genetics import individual_from_genome_dict
    from potofgold.evolutionary_search.search.grammar import GType, _node_at_path
    from potofgold.multi_agent.diagnostics.decision_tracer import build_evaluator, _eval_1d, is_exogenous, margin_series
    from potofgold.multi_agent.diagnostics.perturbation_prober import _run_trades, phi_hat, jaccard
    from potofgold.multi_agent.diagnostics.transaction_cost_stress import _calendar, _validated_trades, StressValidationError

    details = {"method": METHOD, "scope": "training_only_fixed_basis_entry_boundary",
               "eps_grid": list(EPS_GRID), "measurement_usable": False,
               "baseline_parity_ok": None, "baseline_parity_status": "not_tested", "reliability_ok": False,
               "timing_mask": "per_day_floor_and_cutoff_with_valid_same_day_next_bar",
               "interpretation": "Numerical entry-boundary sensitivity, not a feature-distribution intervention, "
                                 "memorization test, economic loss measurement or generalization forecast. "
                                 "Eligible bars are time-eligible opportunities, not necessarily flat or funded."}

    def fail(reason):
        from potofgold.multi_agent.diagnostics.boundary_status import PARITY_FAILURES
        if reason in PARITY_FAILURES:
            details.update(baseline_parity_ok=False, baseline_parity_status="failed")
        return {"applicable": False, "reason": reason, "detail": details}

    try:
        days, bar_days = _calendar(prep)
        individual = individual_from_genome_dict(genome)
        tree = individual.entry_tree
        if tree is None or tree.ret_type != GType.BOOL:
            return fail("boolean_entry_tree_required")
        if not is_exogenous(tree):
            return fail("outcome_dependent_entry_operand")
        targets = boundary_targets(tree)
        if not targets:
            return fail("no_level_comparison_boundary")
        _require_entry_terminal_data(tree, prep)
        _require_fixed_projection_bases(tree, prep)
        eligible = eligible_entry_mask(prep)
        if int(eligible.sum()) < 3:
            return fail("insufficient_eligible_entry_bars")
        evaluator = build_evaluator(prep["train_df"], prep["td_tr"],
                                    bases_k=prep.get("bases_k"), emb_cache_tr=prep.get("emb_cache_tr"))
        original = _eval_1d(evaluator, tree)
        if len(original) != len(eligible) or not np.all(np.isfinite(original)):
            return fail("invalid_entry_signal_array")
        scales = [_margin_scale(margin_series(evaluator, _node_at_path(tree, target["paths"][0]))[eligible])
                  for target in targets]
        details.update(n_boundaries=len(targets), n_comparison_occurrences=sum(len(t["paths"]) for t in targets))

        baseline = _validated_trades(_run_trades(individual, prep), bar_days)
        if not baseline:
            return fail("no_baseline_trades")
        baseline_fires = {trade.day for trade in baseline}
        baseline_entries = {trade.entry_bar for trade in baseline}
        if any(t.entry_bar < 1 or not eligible[t.entry_bar - 1] or original[t.entry_bar - 1] <= .5
               for t in baseline):
            return fail("baseline_entry_signal_timing_mismatch")

        zero_tree = tree
        for target in targets:
            zero_tree = offset_boundary(zero_tree, target["paths"], 0.0)
        zero_signals = _eval_1d(evaluator, zero_tree)
        if not np.array_equal(original, zero_signals):
            return fail("zero_offset_entry_signal_mismatch")
        zero_trades = _validated_trades(_run_trades(replace(individual, entry_tree=zero_tree), prep), bar_days)
        if zero_trades != baseline:
            return fail("zero_offset_trade_replay_mismatch")
        details["baseline_parity_ok"] = True
        details["baseline_parity_status"] = "passed"

        rows = []
        for index, (target, scale) in enumerate(zip(targets, scales)):
            for eps in EPS_GRID:
                for direction in (-1, 1):
                    changed_tree = offset_boundary(tree, target["paths"], direction * eps * scale)
                    signals = _eval_1d(evaluator, changed_tree)
                    if len(signals) != len(original) or not np.all(np.isfinite(signals)):
                        return fail("invalid_perturbed_entry_signal_array")
                    trades = _validated_trades(_run_trades(replace(individual, entry_tree=changed_tree), prep), bar_days)
                    if any(t.entry_bar < 1 or not eligible[t.entry_bar - 1] or signals[t.entry_bar - 1] <= .5
                           for t in trades):
                        return fail("perturbed_entry_signal_timing_mismatch")
                    fires, entries = {t.day for t in trades}, {t.entry_bar for t in trades}
                    phi = phi_hat([baseline_fires, fires], len(days))
                    rows.append({"boundary_index": index, "offset_margin_scales": direction * eps,
                        "eligible_entry_disagreement": float(np.mean((original[eligible] > .5) != (signals[eligible] > .5))),
                        "fired_day_phi": float(phi) if math.isfinite(phi) else None,
                        "fired_day_jaccard": jaccard(baseline_fires, fires),
                        "realized_entry_jaccard": jaccard(baseline_entries, entries),
                        "trade_count_fractional_change": (len(trades) - len(baseline)) / len(baseline)})
        phis = [row["fired_day_phi"] for row in rows if row["fired_day_phi"] is not None]
        statistic = max(row["eligible_entry_disagreement"] for row in rows)
        details.update(measurement_usable=True, reliability_ok=True,
            n_perturbations=len(rows), scenarios=rows,
            boundary_entry_disagreement_max=statistic,
            boundary_fired_day_phi_min=min(phis) if phis else None,
            boundary_fired_day_phi_defined_fraction=len(phis) / len(rows),
            boundary_fired_day_jaccard_min=min(row["fired_day_jaccard"] for row in rows),
            boundary_realized_entry_jaccard_min=min(row["realized_entry_jaccard"] for row in rows),
            boundary_trade_count_change_max=max(abs(row["trade_count_fractional_change"]) for row in rows))
        return {"applicable": True, "statistic": statistic, "detail": details}
    except (BoundaryValidationError, StressValidationError) as exc:
        return fail(str(exc))
