"""Compare a strategy's admitted training scenarios with its own baseline."""
from __future__ import annotations

from collections.abc import Mapping
import math
import re

from potofgold.multi_agent.strategy_profiles.training_economics import CLAIM_SCOPE, TEST_ID


_SCENARIOS = (("110", 1.1), ("125", 1.25))
_METRICS = ("net", "cvar5")
_NAMES = {f"train_{prefix}_{metric}"
          for prefix in ("baseline", "cost_110", "cost_125") for metric in _METRICS}
_UNITS = "engine_points_times_strategy_units"
_INTERPRETATION = (
    "Descriptive arithmetic against this candidate's own training baseline, on the admitted shared "
    "training replay. Both signed metrics are higher-is-better: a negative change is deterioration "
    "under this tested scenario, not established future loss. Relative change divides the signed "
    "change by the absolute baseline; it is dimensionless, not an account-return percentage, "
    "profit-retention rate, cohort percentile, or calibrated small/large cutoff. It establishes "
    "neither predictive support nor an automatic verdict. Cite the original baseline and scenario "
    "aliases for the source values; this context creates no new measured evidence or citation ID."
)


def _finite(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def economic_comparison_context(case_file: dict, citation_info: dict) -> list[dict]:
    """Validate scenario identity and measurements, then calculate baseline differences.

    Relative changes are undefined when the baseline is zero.
    """
    if not isinstance(case_file, Mapping) or not isinstance(citation_info, Mapping):
        return []
    groups: dict[str, dict[str, dict]] = {}
    for alias, meta in citation_info.items():
        if not isinstance(alias, str) or not isinstance(meta, Mapping):
            continue
        rid, name = meta.get("result_id"), meta.get("statistic_name")
        if (not isinstance(rid, str) or not rid or "::" in rid
                or not isinstance(name, str) or name not in _NAMES or alias != f"{rid}::{name}"
                or meta.get("test_id") != TEST_ID
                or meta.get("claim_scope") != CLAIM_SCOPE
                or meta.get("evidence_kind") != CLAIM_SCOPE
                or any(meta.get(flag) is not True for flag in ("valid", "citeable", "research_admissible"))):
            continue
        value = case_file.get(alias)
        if not _finite(value) or not _finite(meta.get("statistic")) or value != meta["statistic"]:
            continue
        groups.setdefault(rid, {})[name] = {"exhibit_id": alias, "value": value}

    output = []
    for rid, values in sorted(groups.items()):
        for scenario, multiplier in _SCENARIOS:
            for metric in _METRICS:
                baseline = values.get(f"train_baseline_{metric}")
                stressed = values.get(f"train_cost_{scenario}_{metric}")
                if baseline is None or stressed is None:
                    continue
                change = stressed["value"] - baseline["value"]
                change_reason = None
                if not _finite(change):
                    change, change_reason = None, "nonfinite_difference"
                denominator = abs(baseline["value"])
                relative, relative_reason = None, None
                if denominator == 0:
                    relative_reason = "zero_baseline"
                elif change is None:
                    relative_reason = "nonfinite_difference"
                else:
                    relative = change / denominator
                    if not _finite(relative):
                        relative, relative_reason = None, "nonfinite_relative_change"
                output.append({
                    "result_id": rid, "metric": metric, "scenario": f"cost_{scenario}",
                    "cost_multiplier": multiplier, "baseline_cost_multiplier": 1.0,
                    "reference_kind": "candidate_training_baseline", "claim_scope": CLAIM_SCOPE,
                    "baseline": dict(baseline), "scenario_measurement": dict(stressed),
                    "signed_change": change, "signed_change_reason": change_reason,
                    "absolute_baseline_denominator": denominator,
                    "change_relative_to_absolute_baseline": relative,
                    "relative_change_reason": relative_reason,
                    "units": _UNITS, "relative_change_units": "dimensionless",
                    "context_only": True, "is_cohort_percentile": False,
                    "predictive_support_established": False, "calibrated_magnitude_cutoff": False,
                    "interpretation": _INTERPRETATION,
                })
    return output


def economic_scenario_context(case_file: dict, citation_info: dict) -> list[dict]:
    """Collect cost and entry-boundary scenarios with their source references."""
    if not isinstance(case_file, Mapping) or not isinstance(citation_info, Mapping):
        return []
    groups = {}
    for alias, meta in citation_info.items():
        if not isinstance(alias, str) or not isinstance(meta, Mapping):
            continue
        rid, name, test = meta.get("result_id"), meta.get("statistic_name"), meta.get("test_id")
        if (test not in (TEST_ID, "entry_boundary_economics")
                or not isinstance(rid, str) or not rid or "::" in rid
                or not isinstance(name, str) or alias != f"{rid}::{name}"
                or meta.get("claim_scope") != CLAIM_SCOPE
                or meta.get("evidence_kind") != CLAIM_SCOPE
                or any(meta.get(flag) is not True for flag in ("valid", "citeable", "research_admissible"))):
            continue
        value = case_file.get(alias)
        if not _finite(value) or not _finite(meta.get("statistic")) or value != meta["statistic"]:
            continue
        pattern = (r"train_(baseline|cost_110|cost_125)_(net|cvar5)" if test == TEST_ID else
                   r"(baseline|boundary_scenario_\d+)_(net|cvar5)")
        match = re.fullmatch(pattern, name)
        if not match:
            continue
        scenario, metric = match.groups()
        groups.setdefault((rid, test), {}).setdefault(scenario, {})[metric] = {
            "value": value, "exhibit_id": alias}
    output = []
    for (rid, test), values in sorted(groups.items()):
        baseline = values.get("baseline", {})
        scenarios = [{"scenario": name, **metrics} for name, metrics in sorted(values.items())
                     if name != "baseline"]
        if not scenarios:
            continue
        nets = [s["net"]["value"] for s in scenarios if "net" in s]
        tails = [(s["cvar5"]["value"] > baseline["cvar5"]["value"]) -
                 (s["cvar5"]["value"] < baseline["cvar5"]["value"]) for s in scenarios
                 if "cvar5" in s and "cvar5" in baseline]
        output.append({
            "result_id": rid, "family": "cost" if test == TEST_ID else "boundary",
            "intervention": ("cost_multiplier_with_selected_entry_rule" if test == TEST_ID
                             else "entry_boundary_offset_with_cost_settings_fixed"),
            "baseline": baseline, "scenarios": scenarios,
            "net_sign_counts": {"positive": sum(x > 0 for x in nets),
                                "zero": sum(x == 0 for x in nets), "negative": sum(x < 0 for x in nets)},
            "tail_change_counts": {"improved": tails.count(1), "unchanged": tails.count(0),
                                   "worsened": tails.count(-1)},
            "context_only": True,
            "interpretation": "Counts exclude the baseline and cover only supplied measured scenarios. "
                "Cost and boundary tests are different interventions on shared training data. Favorable boundary "
                "offsets concern neighboring rules, not permission to substitute or deploy those rules. "
                "Keep profit sign, profit erosion and tail direction separate. Cite the original aliases. "
                "These summaries set no action, severity threshold or predictive-support requirement.",
        })
    return output
