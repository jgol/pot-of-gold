"""Measure training P&L sensitivity to fixed entry-boundary offsets.

Reuse entry_boundary_stability's perturbations at +/-0.05 and +/-0.10
training margin scales, keeping other rules and the basis fixed. Retain
net P&L and daily 5% CVaR from the same trade replays. Model-facing summaries
and dated offline reports share one cached replay.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np

from potofgold.multi_agent.diagnostics.economic_fragility import MIN_ABS_BASELINE_NET
from potofgold.multi_agent.diagnostics.entry_boundary_stability import BoundaryValidationError, EPS_GRID, boundary_targets, eligible_entry_mask, offset_boundary, _margin_scale, _require_entry_terminal_data, _require_fixed_projection_bases

METHOD = "fixed_basis_entry_boundary_economics_v1"
OFFLINE_SCHEMA = "boundary-economics-offline-v1"

# Distinguish structural inapplicability from replay or validation failure.
# Enumerate structural reasons; classify other unavailable results as failures.
STRUCTURAL_INAPPLICABLE_REASONS = frozenset({
    # Unsupported entry-tree structure.
    "boolean_entry_tree_required", "outcome_dependent_entry_operand", "no_level_comparison_boundary",
    "nested_comparison_boundaries", "boundary_budget_exceeded", "unsupported_comparison_boundary",
    "gated_entry_replay_not_supported", "entry_timing_data_required", "invalid_entry_floor",
    "insufficient_finite_boundary_support", "degenerate_margin_scale",
    # Missing inputs or projection basis.
    "missing_entry_terminal_data", "fixed_prototype_basis_required", "invalid_fixed_prototype_basis",
    "invalid_prototype_terminal_data", "fixed_projection_basis_required", "invalid_fixed_projection_basis",
    # Insufficient eligible bars or entries.
    "insufficient_eligible_entry_bars", "no_baseline_trades", "no_baseline_exposure",
})

# Subset of failures specifically indicating replay disagreement.
PARITY_FAILURE_REASONS = frozenset({
    "invalid_entry_signal_array",
    "baseline_entry_signal_timing_mismatch",
    "zero_offset_entry_signal_mismatch",
    "zero_offset_trade_replay_mismatch",
    "invalid_perturbed_entry_signal_array",
    "perturbed_entry_signal_timing_mismatch",
    "nonfinite_boundary_offset",
})


def _economics(trade_rows, days):
    """Aggregate net P&L, daily 5% CVaR and a dated ledger on the training calendar.

    Only the scalar summaries are exposed to agents.
    """
    from potofgold.evolutionary_search.search.fitness import per_day_sum_pnl
    from potofgold.multi_agent.analysis.court_statistics import metrics

    local = [SimpleNamespace(entry_bar=i, pnl=row.pnl) for i, row in enumerate(trade_rows)]
    sparse = per_day_sum_pnl(local, [row.day for row in trade_rows])
    daily = [float(sparse.get(day, 0.0)) for day in days]
    return {"net": float(sum(daily)), "cvar5": float(metrics(daily)["cvar5"]),
            "day_pnl": {str(day): float(sparse.get(day, 0.0)) for day in days}}


@dataclass
class BoundaryReplay:
    """Cache one boundary replay for scalar exhibits and dated offline reports."""
    _public: Dict[str, Any] = field(repr=False)
    _days: Tuple[Any, ...] = field(default=(), repr=False)
    _baseline: Tuple[Any, ...] = field(default=(), repr=False)
    _scenarios: Tuple[Tuple[int, float, Tuple[Any, ...]], ...] = field(default=(), repr=False)   # Each scenario stores (boundary index, offset, trades).

    def exhibit(self) -> Dict[str, Any]:
        return deepcopy(self._public)

    def offline_report(self) -> Dict[str, Any]:
        """Build a dated report from cached trades without repeating the replay."""
        if not self._public.get("applicable"):
            return {"schema": OFFLINE_SCHEMA, "applicable": False, "reason": self._public.get("reason"),
                    "measurement": self.exhibit()}
        from potofgold.multi_agent.diagnostics.economic_fragility import forced_exit_profit_attribution
        base = _economics(self._baseline, self._days)
        scen = []
        for index, offset, trades in self._scenarios:
            e = _economics(trades, self._days)
            scen.append({"boundary_index": index, "offset_margin_scales": offset, "net": e["net"], "cvar5": e["cvar5"],
                         "net_change": e["net"] - base["net"], "cvar5_change": e["cvar5"] - base["cvar5"],
                         "trade_count": len(trades), "day_pnl": e["day_pnl"]})
        return {"schema": OFFLINE_SCHEMA, "applicable": True, "measurement": self.exhibit(),
                "scope": "training_only_offline", "days": list(self._days),
                "pnl_units": "engine_points_times_strategy_units",
                "baseline_net": base["net"], "baseline_cvar5": base["cvar5"],
                "baseline_trade_count": len(self._baseline), "baseline_day_pnl": base["day_pnl"],
                "scenarios": scen,
                "forced_exit_attribution": forced_exit_profit_attribution(self._baseline),
                "portfolio_convention": "calendar_zero_filled; use court_statistics.portfolio_series for cohorts"}


def run_boundary_replay(genome: dict, prep: dict) -> BoundaryReplay:
    """Run and cache the baseline and fixed entry-boundary perturbations.

    Require the baseline to reproduce canonical trades. Report the worst net
    deterioration over the +/-0.05 and +/-0.10 offsets.
    """
    from potofgold.evolutionary_search.search.genetics import individual_from_genome_dict
    from potofgold.evolutionary_search.search.grammar import GType, _node_at_path
    from potofgold.multi_agent.diagnostics.decision_tracer import build_evaluator, _eval_1d, is_exogenous, margin_series
    from potofgold.multi_agent.diagnostics.perturbation_prober import _run_trades
    from potofgold.multi_agent.diagnostics.transaction_cost_stress import _calendar, _validated_trades, StressValidationError

    details = {"method": METHOD, "scope": "training_only_fixed_basis_entry_boundary_economics",
               "eps_grid": list(EPS_GRID), "pnl_units": "engine_points_times_strategy_units",
               "measurement_usable": False, "baseline_parity_ok": None,
               "baseline_parity_status": "not_tested", "reliability_ok": False,
               "interpretation": "Training-only net and 5% tail (CVaR) deterioration under fixed small entry-boundary "
                                 "offsets, with other rules and the fitted basis held fixed. Economic fragility of the "
                                 "recorded strategy, not a sampled market scenario or a future-loss forecast."}

    def fail(reason):
        from potofgold.multi_agent.diagnostics.boundary_status import PARITY_FAILURES
        if reason in PARITY_FAILURES:
            details.update(baseline_parity_ok=False, baseline_parity_status="failed")
        return BoundaryReplay({"applicable": False, "reason": reason, "detail": details})

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

        baseline = _validated_trades(_run_trades(individual, prep), bar_days)
        if not baseline:
            return fail("no_baseline_trades")
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

        base_econ = _economics(baseline, days)
        rows = []
        scenario_trades = []
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
                    econ = _economics(trades, days)
                    rows.append({"boundary_index": index, "offset_margin_scales": direction * eps,
                                 "net": econ["net"], "cvar5": econ["cvar5"],
                                 "net_change": econ["net"] - base_econ["net"],
                                 "cvar5_change": econ["cvar5"] - base_econ["cvar5"],
                                 "trade_count": len(trades),
                                 "trade_count_change": len(trades) - len(baseline),
                                 "trade_count_fractional_change": (len(trades) - len(baseline)) / len(baseline)})
                    scenario_trades.append((index, direction * eps, tuple(trades)))
        worst_net = min(row["net_change"] for row in rows)
        worst_cvar = min(row["cvar5_change"] for row in rows)
        base_net_abs = abs(base_econ["net"])
        # A near-zero baseline leaves the deterioration ratio undefined.
        worst_net_frac = worst_net / base_net_abs if base_net_abs >= MIN_ABS_BASELINE_NET else None
        # Derive exit attribution from these cached trades.
        from potofgold.multi_agent.diagnostics.economic_fragility import forced_exit_profit_attribution
        exit_attr = forced_exit_profit_attribution(baseline)
        details.update(measurement_usable=True, reliability_ok=True,
                       n_boundaries=len(targets), n_perturbations=len(rows), n_days=len(days),
                       baseline_net=base_econ["net"], baseline_cvar5=base_econ["cvar5"],
                       baseline_trade_count=len(baseline),
                       scenarios=rows,
                       forced_exit_attribution=exit_attr,
                       boundary_worst_net_change=worst_net, boundary_worst_cvar5_change=worst_cvar,
                       boundary_worst_net_change_frac=worst_net_frac)
        public = {"applicable": True, "statistic": worst_net, "detail": details}
        return BoundaryReplay(public, tuple(days), tuple(baseline), tuple(scenario_trades))
    except (BoundaryValidationError, StressValidationError) as exc:
        return fail(str(exc))


def entry_boundary_economics(genome: dict, prep: dict) -> dict:
    """Return the scalar boundary-economics exhibit from a cached replay."""
    return run_boundary_replay(genome, prep).exhibit()
