"""Date-free descriptive economics, derived only from verified training ledgers.

The three chronological calendar blocks are fixed before seeing outcomes. They
are descriptions of one training sample, not independent validation folds.
"""
from copy import deepcopy
from datetime import date

from potofgold.multi_agent.strategy_profiles.training_economics import _digest, _finite, _hex, _SERVER_ANNOTATIONS, _TOP_KEYS, training_economics_record

TEST_ID = "training_profit_recurrence"
VERSION = "v1"
PREFIXES = {"1": "baseline", "1.1": "cost_110", "1.25": "cost_125"}
CONTROL = {
    "claim_scope": "TRAINING_ECONOMICS", "scope": "training_only",
    "pnl_units": "engine_points_times_strategy_units",
    "partition": "three_chronological_equal_calendar_count_blocks_remainder_first",
    "interpretation": "Descriptive training recurrence; blocks are not independent validation folds. "
        "Zero-PnL days may include active breakeven days: nonzero days are not traded-day counts. "
        "Positive-PnL concentration divides by gross positive day PnL, not signed net. "
        "This does not measure signal-margin stability, joint stresses, or altered exit/sizing rules.",
    "predictive_support_established": False,
}
PRIVATE = {"champion_binding", "source_measurement_fingerprint", "source_result_id"}
BASE_NAMES = {"calendar_days", "nonzero_pnl_days", "positive_pnl_days", "negative_pnl_days",
              "zero_pnl_days", "gross_positive_day_pnl", "gross_negative_day_pnl", "net",
              "top1_positive_day_pnl", "top3_positive_day_pnl", "positive_blocks", "nonempty_blocks"}
BASE_NAMES |= {f"block_{i}_{k}" for i in (1, 2, 3)
               for k in ("calendar_days", "positive_pnl_days", "nonzero_pnl_days", "net")}
NAMES = {p + "_" + k for p in PREFIXES.values() for k in BASE_NAMES}
OPTIONAL_NAMES = {p + "_" + k for p in PREFIXES.values()
                  for k in ("fired_day_count", "trade_count", "top3_positive_share")}


def recurrence_record(report, *, champion_id, source_measurement_fingerprint):
    source = training_economics_record(report, champion_id=champion_id,
        source_measurement_fingerprint=source_measurement_fingerprint)
    days = sorted(report["days"])
    if any(date.fromisoformat(d).isoformat() != d for d in days):
        raise ValueError("Recurrence requires an unambiguous training calendar")
    values = {}
    for scenario, prefix in PREFIXES.items():
        sequence = [float(report["scenarios"][scenario]["day_pnl"][d]) for d in days]
        positive = sorted((v for v in sequence if v > 0), reverse=True)
        stats = {"calendar_days": len(days), "nonzero_pnl_days": sum(v != 0 for v in sequence),
                 "positive_pnl_days": len(positive), "negative_pnl_days": sum(v < 0 for v in sequence),
                 "zero_pnl_days": sum(v == 0 for v in sequence), "gross_positive_day_pnl": sum(positive),
                 "gross_negative_day_pnl": sum(v for v in sequence if v < 0), "net": sum(sequence),
                 "top1_positive_day_pnl": sum(positive[:1]), "top3_positive_day_pnl": sum(positive[:3]),
                 "positive_blocks": 0, "nonempty_blocks": 0}
        activity = report["scenarios"][scenario].get("activity") or {}
        for key in ("fired_day_count", "trade_count"):
            count = activity.get(key)
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                if key == "fired_day_count" and count > len(days):
                    raise ValueError("Fired-day count exceeds the training calendar")
                stats[key] = count
        if positive:
            stats["top3_positive_share"] = sum(positive[:3]) / sum(positive)
        start = 0
        for i in (1, 2, 3):
            count = len(days) // 3 + (i <= len(days) % 3)
            block = sequence[start:start + count]
            start += count
            stats.update({f"block_{i}_calendar_days": count, f"block_{i}_net": sum(block),
                          f"block_{i}_positive_pnl_days": sum(v > 0 for v in block),
                          f"block_{i}_nonzero_pnl_days": sum(v != 0 for v in block)})
            stats["positive_blocks"] += sum(block) > 0
            stats["nonempty_blocks"] += bool(block)
        values.update({prefix + "_" + k: v for k, v in stats.items()})
    record = {"test_id": TEST_ID, "test_version": VERSION, "parameters": {},
              "data_split_hash": source["data_split_hash"], "applicability": "applicable",
              "replay_fidelity": 1.0, "measured_statistics": values,
              "control_results": {**CONTROL, "champion_binding": _digest([TEST_ID, champion_id]),
                  "source_measurement_fingerprint": source_measurement_fingerprint,
                  "source_result_id": source["result_id"]}, "calibration_id": None, "decision": "review"}
    record["result_id"] = _digest(record)[:16]
    return validate_recurrence_record(record, champion_id=champion_id,
        source_measurement_fingerprint=source_measurement_fingerprint)


def validate_recurrence_record(record, *, champion_id=None, source_measurement_fingerprint=None):
    if (not isinstance(record, dict) or not _TOP_KEYS <= record.keys()
            or record.keys() - _TOP_KEYS - _SERVER_ANNOTATIONS):
        raise ValueError("Invalid recurrence fields")
    rec = {k: deepcopy(record[k]) for k in _TOP_KEYS}
    c, v = rec["control_results"], rec["measured_statistics"]
    if (rec["test_id"] != TEST_ID or rec["test_version"] != VERSION or rec["parameters"] != {}
            or rec["applicability"] != "applicable" or rec["decision"] != "review"
            or not _finite(rec["replay_fidelity"]) or rec["replay_fidelity"] != 1
            or rec["calibration_id"] is not None or not _hex(rec["data_split_hash"], 16)
            or not isinstance(c, dict) or set(c) != set(CONTROL) | PRIVATE
            or any(c[k] != val or type(c[k]) is not type(val) for k, val in CONTROL.items())
            or not isinstance(v, dict) or not NAMES <= v.keys() or v.keys() - NAMES - OPTIONAL_NAMES
            or not all(_finite(x) for x in v.values())
            or not _hex(c["champion_binding"], 64) or not _hex(c["source_measurement_fingerprint"], 64)
            or not _hex(c["source_result_id"], 16)):
        raise ValueError("Invalid recurrence contract")
    if champion_id is not None and c["champion_binding"] != _digest([TEST_ID, champion_id]):
        raise ValueError("Recurrence belongs to another champion")
    if source_measurement_fingerprint is not None and c["source_measurement_fingerprint"] != source_measurement_fingerprint:
        raise ValueError("Recurrence belongs to another measurement collection")
    if rec["result_id"] != _digest({k: val for k, val in rec.items() if k != "result_id"})[:16]:
        raise ValueError("Recurrence integrity check failed")
    return rec


def recurrence_admission_reason(record):
    try:
        validate_recurrence_record(record)
    except (ValueError, TypeError, KeyError, OverflowError):
        return "invalid_training_recurrence"
    return None
