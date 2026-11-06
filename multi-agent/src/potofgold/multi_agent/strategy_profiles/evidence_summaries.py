"""Summarize validated training entry-boundary scenarios for assessment."""
from copy import deepcopy
import math

SUMMARY_VERSION = "boundary-context-v1"
MAX_SCENARIOS = 128
SCENARIO_FIELDS = {"boundary_index", "offset_margin_scales", "net", "cvar5", "net_change",
                   "cvar5_change", "trade_count", "trade_count_change", "trade_count_fractional_change"}
ATTRIBUTION_FIELDS = {"baseline_top3_positive_day_net", "scenario_top3_positive_day_net",
                      "top3_positive_day_net_change", "other_day_net_change", "top3_days_still_positive"}
EXIT_REASONS = {"signal", "max_hold", "eod", "session", "end_of_data", "stop_loss", "time_decay"}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _count(value):
    return number(value) and value >= 0 and int(value) == value


def _close(a, b):
    return number(a) and number(b) and math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-7)


def approved_boundary_context(record):
    """Validate the scenario schema and expose approved numeric fields as citation aliases."""
    if record.get("test_id") != "entry_boundary_economics" or record.get("test_version") != "v1":
        return {}, {}
    control = record.get("control_results") or {}
    measured = record.get("measured_statistics") or {}
    rows = control.get("scenarios")
    approved, scalars = {}, {}
    required = {"boundary_index", "offset_margin_scales", "net", "cvar5", "net_change", "cvar5_change", "trade_count"}
    if (isinstance(rows, list) and 0 < len(rows) <= MAX_SCENARIOS
            and all(number(measured.get(k)) for k in ("baseline_net", "baseline_cvar5",
                    "boundary_worst_net_change", "boundary_worst_cvar5_change"))
            and _count(control.get("n_boundaries")) and 0 < control["n_boundaries"] <= MAX_SCENARIOS // 4
            and len(rows) == control.get("n_perturbations")
            and all(isinstance(r, dict) and required <= set(r) <= SCENARIO_FIELDS | ATTRIBUTION_FIELDS
                    and all(number(v) for v in r.values()) and _count(r["boundary_index"])
                    and _count(r["trade_count"])
                    and r["offset_margin_scales"] in (-.1, -.05, .05, .1)
                    and _close(r["net"] - measured.get("baseline_net", math.nan), r["net_change"])
                    and _close(r["cvar5"] - measured.get("baseline_cvar5", math.nan), r["cvar5_change"])
                    and (not ATTRIBUTION_FIELDS.intersection(r) or (
                        ATTRIBUTION_FIELDS <= set(r)
                        and r["baseline_top3_positive_day_net"] > 0
                        and _count(r["top3_days_still_positive"]) and r["top3_days_still_positive"] <= 3
                        and _close(r["scenario_top3_positive_day_net"] - r["baseline_top3_positive_day_net"],
                                   r["top3_positive_day_net_change"])
                        and _close(r["top3_positive_day_net_change"] + r["other_day_net_change"], r["net_change"])))
                    for r in rows)):
        identities = {(r["boundary_index"], r["offset_margin_scales"]) for r in rows}
        expected = {(i, offset) for i in range(int(control.get("n_boundaries", 0)))
                    for offset in (-.1, -.05, .05, .1)}
        if (len(identities) == len(rows) and identities == expected
                and _close(min(r["net_change"] for r in rows), measured.get("boundary_worst_net_change"))
                and _close(min(r["cvar5_change"] for r in rows), measured.get("boundary_worst_cvar5_change"))):
            approved["scenarios"] = sorted(deepcopy(rows), key=lambda r: (r["boundary_index"], r["offset_margin_scales"]))
            for i, row in enumerate(approved["scenarios"]):
                for key, value in row.items():
                    scalars[f"boundary_scenario_{i}_{key}"] = value
            scalars.update(boundary_scenarios_net_improved=sum(r["net_change"] > 0 for r in rows),
                           boundary_scenarios_tail_improved=sum(r["cvar5_change"] > 0 for r in rows),
                           boundary_scenarios_evaluated=len(rows))
    exit_summary = control.get("forced_exit_attribution")
    if isinstance(exit_summary, dict) and exit_summary.get("applicable") is True:
        shares = exit_summary.get("positive_net_share_by_exit_reason")
        fields = ("engine_forced_profit_share", "strategy_driven_profit_share", "n_trades",
                  "n_positive_trades", "total_positive_net")
        if (isinstance(shares, dict) and set(shares) <= EXIT_REASONS and shares
                and all(number(v) and 0 <= v <= 1 for v in shares.values())
                and _close(sum(shares.values()), 1)
                and all(number(exit_summary.get(k)) for k in fields)
                and all(0 <= exit_summary[k] <= 1 for k in fields[:2])
                and _close(sum(exit_summary[k] for k in fields[:2]), 1)
                and _close(sum(shares.get(k, 0) for k in ("eod", "session", "end_of_data", "max_hold")),
                           exit_summary["engine_forced_profit_share"])
                and _count(exit_summary["n_trades"]) and _count(exit_summary["n_positive_trades"])
                and 0 < exit_summary["n_positive_trades"] <= exit_summary["n_trades"]
                and exit_summary["n_trades"] == control.get("baseline_trade_count")
                and exit_summary["total_positive_net"] > 0):
            approved["forced_exit_attribution"] = {k: exit_summary[k] for k in fields}
            approved["forced_exit_attribution"]["applicable"] = True
            approved["forced_exit_attribution"]["positive_net_share_by_exit_reason"] = dict(shares)
            approved["forced_exit_attribution"]["interpretation"] = (
                "Positive-trade profit attribution, not positive-day attribution. Engine-forced exits are "
                "not automatically marking artifacts; attribution does not certify exit robustness.")
            scalars.update({f"exit_attribution_{k}": exit_summary[k] for k in fields})
            scalars.update({f"exit_profit_share_{k}": v for k, v in shares.items()})
    counts = control.get("profit_day_context")
    if (isinstance(counts, dict) and set(counts) == {"n_positive_days", "n_days", "top3_positive_share"}
            and _count(counts["n_positive_days"]) and _count(counts["n_days"])
            and 0 < counts["n_positive_days"] <= counts["n_days"]
            and number(counts["top3_positive_share"]) and 0 <= counts["top3_positive_share"] <= 1):
        approved["profit_day_context"] = dict(counts)
        scalars.update({f"boundary_profit_{k}": v for k, v in counts.items()})
    return approved, scalars
