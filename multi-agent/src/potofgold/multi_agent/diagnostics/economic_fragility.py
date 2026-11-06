"""Derive cost sensitivity and profit concentration from training replay reports.

Cost erosion compares the 1.0x, 1.10x and 1.25x explicit-cost scenarios.
Profit concentration uses the baseline daily ledger. Near-zero baseline
ratios remain undefined. Exit attribution reports where positive P&L was
realized, without assigning an adverse interpretation to forced closes.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional

# Numerical denominator floor, in engine points times strategy units.
MIN_ABS_BASELINE_NET = 1e-6
MIN_POSITIVE_NET = 1e-6


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def cost_erosion(stats: Mapping[str, float]) -> dict:
    """Calculate net and 5% tail erosion under a 25% explicit-cost increase.

    Compare erosion per unit of cost increment to detect acceleration across
    the 1.0x, 1.10x and 1.25x scenarios.
    """
    needed = ("train_baseline_net", "train_cost_110_net", "train_cost_125_net",
              "train_baseline_cvar5", "train_cost_125_cvar5")
    if any(not _finite(stats.get(k)) for k in needed):
        return {"applicable": False, "reason": "missing_rung_statistics"}
    base_net = float(stats["train_baseline_net"])
    net_110 = float(stats["train_cost_110_net"])
    net_125 = float(stats["train_cost_125_net"])
    base_cvar = float(stats["train_baseline_cvar5"])
    cvar_125 = float(stats["train_cost_125_cvar5"])
    if abs(base_net) < MIN_ABS_BASELINE_NET:
        return {"applicable": False, "reason": "near_zero_baseline_net",
                "sign_crossed_net": bool(base_net > 0 >= net_125)}
    # Normalize erosion by the unequal cost increments, 0.10 and 0.15.
    step_low_rate = (base_net - net_110) / 0.10
    step_high_rate = (net_110 - net_125) / 0.15
    return {"applicable": True,
            "net_erosion_ratio": (base_net - net_125) / abs(base_net),
            "sign_crossed_net": bool(base_net > 0 >= net_125),
            "cvar_erosion": base_cvar - cvar_125,   # Falling CVaR means a more severe loss tail.
            "superlinear_erosion": bool(step_high_rate > step_low_rate),
            "baseline_net": base_net, "cost_125_net": net_125}


def day_profit_concentration(day_pnl: Mapping[str, float]) -> dict:
    """Measure how positive P&L is distributed across training days.

    Return largest-day and top-three shares and a Gini coefficient. Leave
    concentration undefined when there is no positive P&L to attribute.
    """
    if not isinstance(day_pnl, Mapping) or not day_pnl:
        return {"applicable": False, "reason": "empty_day_ledger"}
    values = list(day_pnl.values())
    if any(not _finite(v) for v in values):
        return {"applicable": False, "reason": "nonfinite_day_pnl"}
    positives = sorted((float(v) for v in values if v > 0), reverse=True)
    total_positive = sum(positives)
    if total_positive < MIN_POSITIVE_NET:
        return {"applicable": False, "reason": "no_positive_net", "n_days": len(values)}
    top1 = positives[0] / total_positive
    top3 = sum(positives[:3]) / total_positive
    # Include every calendar day; clamp losses and flat days to zero for this Gini.
    clamped = [max(float(v), 0.0) for v in values]
    return {"applicable": True, "top1_positive_share": top1, "top3_positive_share": top3,
            "profit_gini": _gini(clamped), "n_positive_days": len(positives),
            "n_days": len(values), "total_positive_net": total_positive}


# Engine-forced closes exclude the evolved time_decay timeout.
ENGINE_FORCED_EXITS = frozenset({"session", "eod", "end_of_data", "max_hold"})


def forced_exit_profit_attribution(trades) -> dict:
    """Attribute positive training P&L to exit reasons.

    Separate engine-forced closes from strategy exits, counting time_decay as
    a strategy timeout. Read the baseline trades because saved daily summaries
    do not retain the exit-reason/P&L association.
    """
    from potofgold.multi_agent.diagnostics.transaction_cost_stress import EXIT_REASONS
    rows = list(trades)
    if not rows:
        return {"applicable": False, "reason": "no_trades"}
    reasons, pnls = [], []
    for t in rows:
        reason, pnl = getattr(t, "exit_reason", None), getattr(t, "pnl", None)
        if reason not in EXIT_REASONS or not _finite(pnl):
            return {"applicable": False, "reason": "invalid_trade_row"}
        reasons.append(reason)
        pnls.append(float(pnl))
    total_positive = sum(p for p in pnls if p > 0)
    if total_positive < MIN_POSITIVE_NET:
        return {"applicable": False, "reason": "no_positive_net", "n_trades": len(rows)}
    per_reason = {r: sum(p for rr, p in zip(reasons, pnls) if p > 0 and rr == r) / total_positive
                  for r in EXIT_REASONS}
    engine_forced = sum(v for r, v in per_reason.items() if r in ENGINE_FORCED_EXITS)
    return {"applicable": True, "positive_net_share_by_exit_reason": per_reason,
            "engine_forced_profit_share": engine_forced, "strategy_driven_profit_share": 1.0 - engine_forced,
            "n_trades": len(rows), "n_positive_trades": sum(1 for p in pnls if p > 0),
            "total_positive_net": total_positive,
            "interpretation": ("Descriptive profit attribution by exit reason. Engine-forced exits are active market "
                               "closes, not a marking artifact, so a high forced share is not itself adverse; "
                               "time_decay is an engineered timeout counted as strategy-driven.")}


def _gini(sorted_desc: list) -> Optional[float]:
    """Gini coefficient of a non-negative distribution (0 = even, ->1 = concentrated)."""
    n = len(sorted_desc)
    if n == 0:
        return None
    if n == 1:
        return 0.0
    ascending = sorted(sorted_desc)
    total = sum(ascending)
    if total <= 0:
        return None
    cum = sum((i + 1) * x for i, x in enumerate(ascending))
    return (2.0 * cum) / (n * total) - (n + 1.0) / n
