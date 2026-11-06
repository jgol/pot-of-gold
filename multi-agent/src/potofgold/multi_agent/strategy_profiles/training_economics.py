"""Extract and validate scalar training results from a replay report.

The report must identify the strategy, training split and source measurement.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping


TEST_ID = "training_economics"
VERSION = "v1"
CLAIM_SCOPE = "TRAINING_ECONOMICS"
# This reference applies to stressed net P&L at 1.25x cost, not other statistics.
PREDICTOR = {"diagnostic": TEST_ID, "statistic": "train_cost_125_net", "reject_when": "below", "version": VERSION,
             "units": "engine_points_times_strategy_units",
             "definition": "signed net PnL of the training portfolio at 1.25x explicit engine cost; adverse = below"}
_SOURCE_TEST = "transaction_cost_sensitivity"
_PREFIXES = {"1": "train_baseline", "1.1": "train_cost_110", "1.25": "train_cost_125"}
_MULTIPLIERS = {"1": 1.0, "1.1": 1.1, "1.25": 1.25}
_GUARDS = ("baseline_parity_ok", "baseline_trade_sequence_parity", "training_replay_verified", "split_disjoint")
_UNITS = "engine_points_times_strategy_units"
_CONVENTION = "calendar_zero_filled; use court_statistics.portfolio_series for cohorts"
_INTERPRETATION = (
    "Observed economics on the training calendar at fixed explicit engine cost multipliers. "
    "Signed net and CVaR5 are higher-is-better, in engine points times strategy units, not dollars or percent. "
    "A negative stressed net establishes training loss under that scenario, not future loss. "
    "Positive stressed net establishes training profitability under that scenario, not future profitability. "
    "These cost scenarios do not cover marking, market impact or every execution assumption. "
    "No diagnostic development support is inherited from the behavioral cost statistic."
)
_TOP_KEYS = frozenset({"result_id", "test_id", "test_version", "parameters", "data_split_hash",
                       "applicability", "replay_fidelity", "measured_statistics", "control_results",
                       "calibration_id", "decision"})
_SERVER_ANNOTATIONS = frozenset({"citation_admission", "measurement_interpretation", "preloaded",
                                "preload_provenance", "development_evidence", "requested_interpretation"})
_CONTROL_CONSTANTS = {
    "claim_scope": CLAIM_SCOPE, "evidence_kind": CLAIM_SCOPE, "evidence_mode": "training_economics",
    "scope": "training_only", "training_replay_verified": True,
    "baseline_trade_sequence_parity": True, "split_disjoint": True,
    "pnl_units": _UNITS, "portfolio_convention": _CONVENTION, "interpretation": _INTERPRETATION,
    "predictive_support_established": False,
}
PRIVATE_CONTROL_FIELDS = frozenset({"champion_binding", "source_measurement_fingerprint", "source_result_id",
                                     "source_test_version"})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _numeric(value: Any, name: str) -> float:
    if not _finite(value):
        raise ValueError("Training economics requires finite numeric " + name)
    return float(value)


def _champion_binding(champion_id: str) -> str:
    if not isinstance(champion_id, str) or not champion_id.strip():
        raise ValueError("Training economics requires a champion identity")
    return _digest([TEST_ID, champion_id])


def _stats_keys() -> set[str]:
    keys = set()
    for scenario, prefix in _PREFIXES.items():
        keys.update({f"{prefix}_net", f"{prefix}_cvar5", f"{prefix}_multiplier"})
        if scenario != "1":
            keys.update({f"{prefix}_net_change", f"{prefix}_cvar5_change"})
    return keys


STATISTIC_NAMES = frozenset(_stats_keys())


def training_economics_record(report: Mapping[str, Any], *, champion_id: str,
                              source_measurement_fingerprint: str) -> dict:
    """Verify the report schema and arithmetic, then select its training measurements.

    This function validates a saved report; it does not rerun the strategy.
    """
    binding = _champion_binding(champion_id)
    if not _hex(source_measurement_fingerprint, 64):
        raise ValueError("Training economics requires a source measurement fingerprint")
    if (not isinstance(report, Mapping) or report.get("schema") != "transaction-cost-stress-v1"
            or report.get("scope") != "training_only_offline" or report.get("applicable") is not True
            or report.get("pnl_units") != _UNITS or report.get("portfolio_convention") != _CONVENTION):
        raise ValueError("Training economics requires a verified training-only stress report")
    from potofgold.multi_agent.strategy_profiles.tool_evidence import measurement_admission_reason
    from potofgold.multi_agent.diagnostics.vulnerability_catalogue import REGISTRY
    source = report.get("measurement_record")
    if (not isinstance(source, Mapping) or source.get("test_id") != _SOURCE_TEST
            or measurement_admission_reason(source) is not None
            or not _hex(source.get("result_id"), 16) or not _hex(source.get("data_split_hash"), 16)
            or not _finite(source.get("replay_fidelity")) or source["replay_fidelity"] != 1.0):
        raise ValueError("Training economics requires its valid source measurement record")
    measurement = report.get("measurement")
    if not isinstance(measurement, Mapping) or measurement.get("applicable") is not True:
        raise ValueError("Training economics lacks a usable stress measurement")
    detail, source_control = measurement.get("detail"), source.get("control_results")
    if (not isinstance(detail, Mapping) or not isinstance(source_control, Mapping)
            or any(detail.get(key) is not True or source_control.get(key) is not True for key in _GUARDS)
            or measurement.get("statistic") != source["measured_statistics"].get(REGISTRY[_SOURCE_TEST].statistic_name)
            or not _finite(measurement.get("statistic"))):
        raise ValueError("Training economics source parity or training scope is unverified")
    days, scenarios = report.get("days"), report.get("scenarios")
    if (not isinstance(days, list) or not days or any(not isinstance(day, str) for day in days)
            or len(set(days)) != len(days) or not isinstance(scenarios, Mapping)
            or set(scenarios) != set(_PREFIXES)):
        raise ValueError("Training economics requires the fixed scenarios and training calendar")
    from potofgold.multi_agent.analysis.court_statistics import metrics
    values = {}
    for scenario, prefix in _PREFIXES.items():
        item = scenarios[scenario]
        if not isinstance(item, Mapping) or not isinstance(item.get("day_pnl"), Mapping):
            raise ValueError("Training economics lacks a scenario calendar")
        daily = item["day_pnl"]
        if set(daily) != set(days):
            raise ValueError("Training economics scenario calendars differ")
        sequence = [_numeric(daily[day], "daily outcome") for day in days]
        net = _numeric(item.get("net"), "net")
        metric_values = item.get("metrics")
        if not isinstance(metric_values, Mapping):
            raise ValueError("Training economics lacks scenario metrics")
        cvar = _numeric(metric_values.get("cvar5"), "CVaR5")
        if (not math.isclose(net, sum(sequence), rel_tol=1e-10, abs_tol=1e-10)
                or not math.isclose(cvar, metrics(sequence)["cvar5"], rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError("Training economics summaries disagree with the source scenario")
        values.update({f"{prefix}_net": net, f"{prefix}_cvar5": cvar,
                       f"{prefix}_multiplier": _MULTIPLIERS[scenario]})
        for name, observed in (("net", net), ("cvar5", cvar)):
            change = _numeric(item.get(name + "_change"), name + " change")
            expected = observed - values[f"train_baseline_{name}"]
            if not math.isclose(change, expected, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("Training economics changes disagree with the baseline")
            if scenario != "1":
                values[f"{prefix}_{name}_change"] = change
    record = {
        "test_id": TEST_ID, "test_version": VERSION, "parameters": {},
        "data_split_hash": source["data_split_hash"], "applicability": "applicable", "replay_fidelity": 1.0,
        "measured_statistics": values, "calibration_id": None, "decision": "review",
        "control_results": {**_CONTROL_CONSTANTS, "champion_binding": binding,
                            "source_measurement_fingerprint": source_measurement_fingerprint,
                            "source_result_id": source["result_id"], "source_test_version": source["test_version"]},
    }
    record["result_id"] = _digest(record)[:16]
    return validate_training_economics_record(record, champion_id=champion_id,
                                               source_measurement_fingerprint=source_measurement_fingerprint)


def validate_training_economics_record(record: Any, *, champion_id: str | None = None,
                                        source_measurement_fingerprint: str | None = None) -> dict:
    """Check schema, content hash and strategy binding for a training-economics record."""
    from potofgold.multi_agent.diagnostics.vulnerability_catalogue import REGISTRY
    if (not isinstance(record, Mapping) or not _TOP_KEYS.issubset(record)
            or set(record) - _TOP_KEYS - _SERVER_ANNOTATIONS):
        raise ValueError("Invalid training economics record fields")
    # Remove display annotations before validating the source record.
    record = {key: record[key] for key in _TOP_KEYS}
    control, values = record.get("control_results"), record.get("measured_statistics")
    if (record.get("test_id") != TEST_ID or record.get("test_version") != VERSION
            or record.get("parameters") != {} or record.get("applicability") != "applicable"
            or record.get("calibration_id") is not None or record.get("decision") != "review"
            or not _finite(record.get("replay_fidelity")) or record["replay_fidelity"] != 1.0
            or not _hex(record.get("data_split_hash"), 16)
            or not isinstance(control, Mapping) or set(control) != set(_CONTROL_CONSTANTS) | PRIVATE_CONTROL_FIELDS
            or not isinstance(values, Mapping) or set(values) != STATISTIC_NAMES
            or any(not _finite(value) for value in values.values())):
        raise ValueError("Invalid training economics record contract")
    for key, expected in _CONTROL_CONSTANTS.items():
        if control[key] != expected or type(control[key]) is not type(expected):
            raise ValueError("Invalid training economics scope or interpretation")
    if (not _hex(control["champion_binding"], 64) or not _hex(control["source_measurement_fingerprint"], 64)
            or not _hex(control["source_result_id"], 16)
            or control["source_test_version"] != REGISTRY[_SOURCE_TEST].test_version):
        raise ValueError("Invalid training economics provenance")
    if champion_id is not None and control["champion_binding"] != _champion_binding(champion_id):
        raise ValueError("Training economics belongs to another champion")
    if (source_measurement_fingerprint is not None
            and control["source_measurement_fingerprint"] != source_measurement_fingerprint):
        raise ValueError("Training economics belongs to another measurement collection")
    for scenario, prefix in _PREFIXES.items():
        if values[f"{prefix}_multiplier"] != _MULTIPLIERS[scenario]:
            raise ValueError("Training economics scenario multiplier changed")
        if scenario != "1":
            for name in ("net", "cvar5"):
                if not math.isclose(values[f"{prefix}_{name}_change"],
                                    values[f"{prefix}_{name}"] - values[f"train_baseline_{name}"],
                                    rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError("Training economics baseline difference changed")
    if record.get("result_id") != _digest({key: value for key, value in record.items() if key != "result_id"})[:16]:
        raise ValueError("Training economics record integrity check failed")
    return deepcopy(dict(record))


def training_economics_admission_reason(record: Any, *, champion_id: str | None = None,
                                         source_measurement_fingerprint: str | None = None) -> str | None:
    try:
        validate_training_economics_record(record, champion_id=champion_id,
                                          source_measurement_fingerprint=source_measurement_fingerprint)
    except (ValueError, TypeError, KeyError, OverflowError):
        return "invalid_training_economics_record"
    return None
