"""Derive cost erosion and profit concentration from training economics."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Optional

import potofgold.multi_agent.strategy_profiles.training_economics as _te
from potofgold.multi_agent.strategy_profiles.training_economics import PREDICTOR as _TRAINING_ECONOMICS, TEST_ID as _TE_ID

CLAIM_SCOPE = "TRAINING_ECONOMICS"

COST_EROSION = {"diagnostic": "cost_erosion", "statistic": "net_erosion_ratio", "reject_when": "above",
                "version": "v1", "units": "fraction_of_baseline_net",
                "definition": "fraction of baseline net PnL eroded by a 25% explicit-cost bump; adverse = above"}
DAY_CONCENTRATION = {"diagnostic": "day_profit_concentration", "statistic": "top3_positive_share",
                     "reject_when": "above", "version": "v1", "units": "share_of_positive_net",
                     "definition": "share of positive training net from the top three days; adverse = above"}

# Primary economic measurements and derived predictor schemas.
ECONOMIC_PREDICTORS = {_TE_ID: _TRAINING_ECONOMICS, COST_EROSION["diagnostic"]: COST_EROSION,
                       DAY_CONCENTRATION["diagnostic"]: DAY_CONCENTRATION}
DERIVED_PREDICTORS = {COST_EROSION["diagnostic"]: COST_EROSION, DAY_CONCENTRATION["diagnostic"]: DAY_CONCENTRATION}
_DERIVED_TOP_KEYS = frozenset({"result_id", "test_id", "test_version", "parameters", "data_split_hash",
                               "applicability", "replay_fidelity", "measured_statistics", "control_results",
                               "calibration_id", "decision"})


def is_economic_predictor(test_id: Any) -> bool:
    return test_id in ECONOMIC_PREDICTORS


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _hex(v: Any, n: int) -> bool:
    return isinstance(v, str) and len(v) == n and all(c in "0123456789abcdef" for c in v)


def economic_admission_reason(record: Any) -> Optional[str]:
    """Validate the economic schema, finite primary statistic and claim scope."""
    rec = dict(record) if isinstance(record, Mapping) else {}
    tid = rec.get("test_id")
    if tid == _TE_ID:
        return _te.training_economics_admission_reason(rec)
    pred = DERIVED_PREDICTORS.get(tid)
    if pred is None:
        return "not_an_economic_predictor"
    if set(rec) - _DERIVED_TOP_KEYS - _te._SERVER_ANNOTATIONS:
        return "derived_economic_record_has_extra_fields"
    control = rec.get("control_results")
    values = rec.get("measured_statistics")
    if (rec.get("test_version") != pred["version"] or rec.get("parameters") != {}
            or rec.get("applicability") != "applicable" or rec.get("calibration_id") is not None
            or rec.get("decision") != "review" or rec.get("replay_fidelity") != 1.0
            or not _hex(rec.get("data_split_hash"), 16) or not _hex(rec.get("result_id"), 16)):
        return "invalid_derived_economic_record_contract"
    if (not isinstance(control, Mapping) or control.get("claim_scope") != CLAIM_SCOPE
            or control.get("evidence_kind") != CLAIM_SCOPE or control.get("predictive_support_established") is not False
            or not control.get("champion_binding")):
        return "invalid_derived_economic_control"
    if not isinstance(values, Mapping) or set(values) != {pred["statistic"]} or not _finite(values[pred["statistic"]]):
        return "missing_finite_primary_statistic"
    return None


def economic_binding_reason(record: Any, *, champion_id: str, source_fingerprint: Optional[str] = None) -> Optional[str]:
    """Check that a measurement belongs to the strategy and run being assessed."""
    rec = dict(record) if isinstance(record, Mapping) else {}
    control = rec.get("control_results") or {}
    if control.get("champion_binding") != _digest([rec.get("test_id"), champion_id], 64):
        return "champion_binding_mismatch"
    if source_fingerprint is not None and control.get("source_measurement_fingerprint") != source_fingerprint:
        return "source_fingerprint_mismatch"
    return None


def _digest(value: Any, length: int) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, default=str).encode()).hexdigest()[:length]


def _derived_record(pred: dict, value: float, *, champion_id: str, data_split_hash: str,
                    source_fingerprint: str) -> dict:
    """Create a scalar measurement with its source identity and content hash."""
    if not isinstance(champion_id, str) or not champion_id.strip():
        raise ValueError("A derived economic record requires a champion identity")
    if not _hex(data_split_hash, 16) or not _hex(source_fingerprint, 64):
        raise ValueError("A derived economic record requires a 16-hex split hash and a 64-hex source fingerprint")
    control = {"claim_scope": CLAIM_SCOPE, "evidence_kind": CLAIM_SCOPE, "scope": "training_only",
               "predictive_support_established": False,
               "champion_binding": _digest([pred["diagnostic"], champion_id], 64),
               "source_measurement_fingerprint": source_fingerprint,
               "interpretation": f"Training-only {pred['diagnostic']} ({pred['definition']}); not a future-loss forecast."}
    rec = {"test_id": pred["diagnostic"], "test_version": pred["version"], "parameters": {},
           "data_split_hash": data_split_hash, "applicability": "applicable", "replay_fidelity": 1.0,
           "measured_statistics": {pred["statistic"]: float(value)}, "calibration_id": None,
           "decision": "review", "control_results": control}
    rec["result_id"] = _digest(rec, 16)
    problem = economic_admission_reason(rec)
    if problem:                                             # Validate the derived record before exposing it to the agents.
        raise ValueError(f"Built an inadmissible derived economic record: {problem}")
    return rec


def cost_erosion_candidate_record(training_economics_stats: Mapping[str, float], *, champion_id: str,
                                  data_split_hash: str, source_fingerprint: str) -> Optional[dict]:
    """Derive cost erosion from an admitted training-economics record."""
    from potofgold.multi_agent.diagnostics.economic_fragility import cost_erosion
    ce = cost_erosion(training_economics_stats or {})
    if not ce.get("applicable") or not _finite(ce.get("net_erosion_ratio")):
        return None
    return _derived_record(COST_EROSION, ce["net_erosion_ratio"], champion_id=champion_id,
                           data_split_hash=data_split_hash, source_fingerprint=source_fingerprint)


def day_concentration_candidate_record(train_traded_day_pnl, *, champion_id: str, data_split_hash: str,
                                       source_fingerprint: str) -> Optional[dict]:
    """Derive profit concentration from an admitted training-economics record."""
    from potofgold.multi_agent.diagnostics.economic_fragility import day_profit_concentration
    if not isinstance(train_traded_day_pnl, (list, tuple)) or not train_traded_day_pnl:
        return None
    conc = day_profit_concentration({i: v for i, v in enumerate(train_traded_day_pnl)})
    if not conc.get("applicable") or not _finite(conc.get("top3_positive_share")):
        return None
    return _derived_record(DAY_CONCENTRATION, conc["top3_positive_share"], champion_id=champion_id,
                           data_split_hash=data_split_hash, source_fingerprint=source_fingerprint)
