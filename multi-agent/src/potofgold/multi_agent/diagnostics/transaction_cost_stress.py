"""Replay fixed transaction-cost scenarios on training data.

Run the sequential engine at 1.0, 1.10 and 1.25 times its explicit cost
terms, reusing the fold basis. The behavioral statistic is the largest
calendar-day mean of abs(stressed exposure - baseline exposure) divided
by their sum, with 0/0 defined as zero. Exposure is strategy-unit bars
(entry contracts times bars held).

Keep dated P&L ledgers in the offline report. The training_economics
module validates the scalar economic summaries supplied to agents.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime
import math
from numbers import Integral, Real
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple


COST_MULTIPLIERS = (1.0, 1.10, 1.25)
EXIT_REASONS = ("signal", "max_hold", "eod", "session", "end_of_data", "stop_loss", "time_decay")
_LIMITATION = (
    "Sensitivity describes response to explicit engine transaction costs, not future loss. "
    "Reduced exposure may reduce risk; stable exposure may still incur larger costs. "
    "This does not stress every price haircut, marking model, market impact or execution delay."
)


class StressValidationError(ValueError):
    """The replay cannot supply a valid diagnostic measurement."""


def _day(value: Any) -> str:
    """Canonical ISO day keys prevent date/datetime representation overlap gaps."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value)
    try:
        parsed = date.fromisoformat(text[:10])
    except (ValueError, TypeError):
        raise StressValidationError("invalid_calendar_day") from None
    if text != parsed.isoformat() and not (
        len(text) > 10 and text[10] in ("T", " ")
    ):
        raise StressValidationError("invalid_calendar_day")
    return parsed.isoformat()


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StressValidationError("invalid_trade_" + field_name)
    value = float(value)
    if not math.isfinite(value):
        raise StressValidationError("nonfinite_trade_" + field_name)
    return value


def _calendar(prep: dict) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if prep.get("replay_fidelity") != 1.0 or isinstance(prep.get("replay_fidelity"), bool):
        raise StressValidationError("verified_training_replay_required")
    if prep.get("train_days") is None or prep.get("heldout_days") is None:
        raise StressValidationError("split_calendar_required")
    train = tuple(_day(d) for d in prep["train_days"])
    heldout = tuple(_day(d) for d in prep["heldout_days"])
    if not train or not heldout or len(set(train)) != len(train) or len(set(heldout)) != len(heldout):
        raise StressValidationError("invalid_split_calendar")
    if set(train) & set(heldout):
        raise StressValidationError("train_heldout_overlap")
    try:
        bars = tuple(_day(d) for d in prep["train_df"]["date"].values)
    except (KeyError, AttributeError, TypeError):
        raise StressValidationError("training_dates_required") from None
    if not bars or set(bars) != set(train):
        raise StressValidationError("training_calendar_mismatch")
    if tuple(sorted(bars)) != bars:
        raise StressValidationError("training_dates_not_ordered")
    return tuple(sorted(train)), bars


@dataclass(frozen=True)
class _TradeRow:
    entry_bar: int
    exit_bar: int
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    bars_held: int
    exit_reason: str
    shape: Optional[str]
    entry_n_contracts: float
    day: str


def _validated_trades(trades: Sequence[Any], bar_days: Tuple[str, ...]) -> Tuple[_TradeRow, ...]:
    """Validate engine Trade values and ordered position identity before aggregation."""
    rows = []
    previous_exit = -1
    for trade in trades:
        try:
            entry, exit_, hold = trade.entry_bar, trade.exit_bar, trade.bars_held
            if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (entry, exit_, hold)):
                raise StressValidationError("invalid_trade_bar_index")
            entry, exit_, hold = int(entry), int(exit_), int(hold)
            if not (0 <= entry <= exit_ < len(bar_days)) or hold != exit_ - entry:
                raise StressValidationError("invalid_trade_holding_interval")
            if entry <= previous_exit:
                raise StressValidationError("unordered_or_overlapping_trades")
            if bar_days[entry] != bar_days[exit_]:
                raise StressValidationError("cross_session_trade")
            contracts = _number(trade.entry_n_contracts, "contracts")
            if contracts <= 0 or not contracts.is_integer():
                raise StressValidationError("invalid_trade_contracts")
            prices = [_number(getattr(trade, name), name) for name in ("entry_price", "exit_price")]
            # These evaluator fields contain underlying spot prices.
            if any(p <= 0 for p in prices):
                raise StressValidationError("invalid_trade_spot_price")
            pnl = _number(trade.pnl, "pnl")
            if trade.exit_reason not in EXIT_REASONS:
                raise StressValidationError("unknown_exit_reason")
            side = getattr(trade.side, "value", trade.side)
            if not isinstance(side, (str, int)) or isinstance(side, bool):
                raise StressValidationError("invalid_trade_side")
            if trade.shape is not None and not isinstance(trade.shape, str):
                raise StressValidationError("invalid_trade_shape")
            row = _TradeRow(entry, exit_, str(side), prices[0], prices[1], pnl, hold,
                            trade.exit_reason, trade.shape, contracts, bar_days[entry])
        except AttributeError:
            raise StressValidationError("missing_trade_field") from None
        rows.append(row)
        previous_exit = exit_
    return tuple(rows)


def _cost_trades(individual: Any, prep: dict, multiplier: float):
    """Mirror _champion_trades options; baseline parity catches later wiring drift."""
    from potofgold.evolutionary_search.search.simulator import vectorized_backtest

    kwargs = dict(entry_tree=individual.entry_tree, exit_tree=individual.exit_tree,
                  size_tree=individual.size_tree, data=prep["train_df"], template=prep["tmpl"],
                  terminal_data=prep["td_tr"], delta_tree=getattr(individual, "delta_tree", None),
                  width_tree=getattr(individual, "width_tree", None), shape=getattr(individual, "shape", None),
                  shape_tree=getattr(individual, "shape_tree", None), entry_decision_mask=None,
                  warmup_bars=int(prep["off"]), entry_floor_within_day=int(prep["off"]),
                  cost_multiplier=multiplier)
    if getattr(individual, "stop_mult", None) is not None:
        kwargs["stop_loss_credit_multiple"] = float(individual.stop_mult)
    if prep.get("bases_k") is not None:
        kwargs["pca_bases"] = prep["bases_k"]
    if prep.get("emb_cache_tr") is not None:
        kwargs["emb_proj_cache"] = prep["emb_cache_tr"]
    return vectorized_backtest(**kwargs).trades


def _exposure(rows: Tuple[_TradeRow, ...], days: Tuple[str, ...]) -> Dict[str, float]:
    result = dict.fromkeys(days, 0.0)
    for row in rows:
        result[row.day] += row.entry_n_contracts * row.bars_held
    if any(not math.isfinite(v) for v in result.values()):
        raise StressValidationError("nonfinite_exposure")
    return result


def _behaviour(rows: Tuple[_TradeRow, ...], baseline: Tuple[_TradeRow, ...],
               days: Tuple[str, ...]) -> Dict[str, Any]:
    base = _exposure(baseline, days)
    current = _exposure(rows, days)
    fractions = Counter(r.exit_reason for r in rows)
    base_fires, fires = {r.day for r in baseline}, {r.day for r in rows}
    changes = [abs(current[d] - base[d]) / (current[d] + base[d])
               if current[d] + base[d] else 0.0 for d in days]
    base_total, total = sum(base.values()), sum(current.values())
    if not math.isfinite(base_total + total):
        raise StressValidationError("nonfinite_exposure")
    return {
        "mean_daily_exposure_change": sum(changes) / len(days),
        "exposure_ratio": total / base_total,
        "trade_count_ratio": len(rows) / len(baseline),
        "fired_day_ratio": len(fires) / len(base_fires),
        "fired_day_jaccard": len(fires & base_fires) / len(fires | base_fires),
        "trade_count": len(rows), "fired_day_count": len(fires),
        "active_day_fraction": len(fires) / len(days),
        "entry_bar_sequence_unchanged": [r.entry_bar for r in rows] == [r.entry_bar for r in baseline],
        "exit_bar_sequence_unchanged": [r.exit_bar for r in rows] == [r.exit_bar for r in baseline],
        "exit_reason_fractions": {k: fractions[k] / len(rows) if rows else 0.0 for k in EXIT_REASONS},
        "strategy_unit_bars": total,
    }


def _public_behaviour(behaviour: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return behavior ratios without raw activity or sample sizes."""
    result = {"baseline_cost_multiplier": 1.0}
    for multiplier, prefix in ((1.0, "baseline"), (1.1, "cost_110"), (1.25, "cost_125")):
        current = behaviour[f"{multiplier:g}"]
        fractions = current["exit_reason_fractions"]
        result.update({f"{prefix}_exit_signal_frac": fractions["signal"],
                       f"{prefix}_exit_stop_frac": fractions["stop_loss"],
                       f"{prefix}_exit_forced_frac": sum(fractions[k] for k in
                           ("session", "eod", "end_of_data", "max_hold"))})
        if multiplier == 1.0:
            continue
        result.update({f"{prefix}_multiplier": multiplier,
                       f"{prefix}_mean_daily_exposure_change": current["mean_daily_exposure_change"],
                       f"{prefix}_exposure_ratio": current["exposure_ratio"],
                       f"{prefix}_trade_ratio": current["trade_count_ratio"],
                       f"{prefix}_active_day_ratio": current["fired_day_ratio"],
                       f"{prefix}_day_set_jaccard": current["fired_day_jaccard"],
                       f"{prefix}_entry_sequence_unchanged": current["entry_bar_sequence_unchanged"],
                       f"{prefix}_exit_sequence_unchanged": current["exit_bar_sequence_unchanged"]})
    return result


@dataclass(frozen=True)
class StressReplay:
    """Cache one cost replay for scalar exhibits and dated offline reports."""
    _public: Dict[str, Any] = field(repr=False)
    _days: Tuple[str, ...] = field(default=(), repr=False)
    _scenarios: Dict[float, Tuple[_TradeRow, ...]] = field(default_factory=dict, repr=False)

    def cause_only(self) -> Dict[str, Any]:
        return deepcopy(self._public)

    def offline_report(self) -> Dict[str, Any]:
        """Build a dated training report using the scorer's calendar and tail convention.

        P&L is measured in engine points times strategy units. Cohort portfolio
        aggregation must also apply the scorer's responsible-fold weights.
        """
        from potofgold.evolutionary_search.search.fitness import per_day_sum_pnl
        from potofgold.multi_agent.analysis.court_statistics import metrics

        if not self._public.get("applicable"):
            return {"schema": "transaction-cost-stress-v1", "applicable": False,
                    "reason": self._public.get("reason"), "measurement": self.cause_only()}
        out = {}
        for multiplier, rows in self._scenarios.items():
            # Aggregate cached trades without retaining the original dataframe.
            local_rows = [SimpleNamespace(entry_bar=i, pnl=row.pnl)
                          for i, row in enumerate(rows)]
            sparse = per_day_sum_pnl(local_rows, [row.day for row in rows])
            daily = {day: float(sparse.get(day, 0.0)) for day in self._days}
            out[f"{multiplier:g}"] = {"day_pnl": daily, "metrics": metrics(list(daily.values())),
                                     "net": sum(daily.values()),
                                     "activity": _behaviour(rows, self._scenarios[1.0], self._days)}
        base = out["1"]
        for key, item in out.items():
            item["net_change"] = item["net"] - base["net"]
            item["cvar5_change"] = item["metrics"]["cvar5"] - base["metrics"]["cvar5"]
        return {"schema": "transaction-cost-stress-v1", "applicable": True,
                "measurement": self.cause_only(), "scope": "training_only_offline", "days": list(self._days),
                "pnl_units": "engine_points_times_strategy_units", "scenarios": out,
                "portfolio_convention": "calendar_zero_filled; use court_statistics.portfolio_series for cohorts"}


def run_stress_replay(genome: dict, prep: dict, *, canonical_fn: Optional[Callable] = None,
                      replay_fn: Optional[Callable] = None,
                      individual_factory: Optional[Callable] = None) -> StressReplay:
    """Replay the fixed cost scenarios and validate the baseline.

    Optional Python callables support testing the replay and canonical-trade
    comparison. Model requests cannot change scenarios, dates or flags.
    """
    try:
        days, bar_days = _calendar(prep)
        if not getattr(prep.get("tmpl"), "legs", None):
            raise StressValidationError("option_transaction_costs_not_applicable")
        if individual_factory is None:
            from potofgold.evolutionary_search.search.genetics import individual_from_genome_dict
            individual_factory = individual_from_genome_dict
        if canonical_fn is None:
            from potofgold.multi_agent.diagnostics.perturbation_prober import _run_trades
            canonical_fn = _run_trades
        individual = individual_factory(genome)
        canonical = _validated_trades(list(canonical_fn(individual, prep)), bar_days)
        baseline = _validated_trades(list((replay_fn or _cost_trades)(individual, prep, 1.0)), bar_days)
        # Validate ordered trade identity, including prices, size, holding time and P&L.
        if canonical != baseline:
            raise StressValidationError("baseline_trade_sequence_mismatch")
        if not baseline:
            raise StressValidationError("no_baseline_trades")
        if sum(_exposure(baseline, days).values()) <= 0:
            raise StressValidationError("no_baseline_exposure")
        scenarios = {1.0: baseline}
        for multiplier in COST_MULTIPLIERS[1:]:
            scenarios[multiplier] = _validated_trades(
                list((replay_fn or _cost_trades)(individual, prep, multiplier)), bar_days)
        behaviour = {f"{m:g}": _behaviour(rows, baseline, days) for m, rows in scenarios.items()}
        statistic = max(behaviour[f"{m:g}"]["mean_daily_exposure_change"] for m in COST_MULTIPLIERS[1:])
        public = {"applicable": True, "statistic": statistic,
                  "detail": {"baseline_parity_ok": True, "baseline_trade_sequence_parity": True,
                             "training_replay_verified": True,
                             "split_disjoint": True,
                             "exposure_unit": "strategy_units_times_held_bars; not option-leg contracts or dollars",
                             "day_universe": "all training-calendar days, including days with zero exposure",
                             **_public_behaviour(behaviour), "interpretation": _LIMITATION}}
        return StressReplay(public, days, scenarios)
    except StressValidationError as exc:
        return StressReplay({"applicable": False, "reason": str(exc),
                             "detail": {"measurement_usable": False, "interpretation": "inconclusive measurement"}})
    except Exception as exc:
        # Keep exception text and outcome values out of the model-facing failure record.
        return StressReplay({"applicable": False, "reason": "stress_replay_failed",
                             "detail": {"measurement_usable": False, "error_type": type(exc).__name__}})


def transaction_cost_sensitivity(genome: dict, prep: dict, **params) -> Dict[str, Any]:
    """Return the behavioral cost-sensitivity exhibit without dated outcomes."""
    if params:
        return {"applicable": False, "reason": "fixed_scenarios_do_not_accept_parameters"}
    return run_stress_replay(genome, prep).cause_only()
