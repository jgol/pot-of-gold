"""Format verified measurements and reference findings for both assessment methods."""
from potofgold.multi_agent.assessment.prompts import INTERPRETATION
from copy import deepcopy
import json

CARD_VERSION = "shared-research-cards-v1"


def measurement_card(record):
    admissions = record.get("citation_admission") or {}
    groups = {}
    values = {}
    rid = record.get("result_id")
    for name, value in (record.get("measured_statistics") or {}).items():
        alias = f"{rid}::{name}"
        meta = admissions.get(alias, {})
        # Preserve unavailable measurements as unavailable.
        usable = meta.get("citeable") is True
        values[alias] = value if usable else None
        meaning = {"measurement_valid": usable,
                   "research_admissible": meta.get("research_admissible", False),
                   "claim_scope": meta.get("claim_scope", "OBSERVED_BEHAVIOR"),
                   "predictive_support": meta.get("dev_status", "not_evaluated"),
                   "adverse_direction": meta.get("direction"),
                   "limitation": meta.get("research_admission_reason")}
        key = json.dumps(meaning, sort_keys=True)
        groups.setdefault(key, {**meaning, "statistics": []})["statistics"].append(name)
    control = record.get("control_results") or {}
    # Omit private provenance and scenario arrays already represented by summaries.
    drop = {"champion_binding", "source_measurement_fingerprint", "source_result_id", "source_test_version",
            "calibration_context_id", "threshold", "threshold_calibrated", "threshold_status",
            "uniform_threshold_used", "scenarios",
            "forced_exit_attribution", "profit_day_context", "predictive_support_established"}
    context = {k: deepcopy(v) for k, v in control.items() if k not in drop}
    if context.get("reason") == "uncalibrated_threshold":
        context.pop("reason")
    if record.get("test_id") in ("entry_boundary_economics", "entry_boundary_stability"):
        from potofgold.multi_agent.diagnostics.boundary_status import parity_presentation
        context = parity_presentation(context)
    validity = {"status": "valid_observation" if any(m.get("citeable") for m in admissions.values())
                else "unavailable", "replay_fidelity": record.get("replay_fidelity")}
    if validity["status"] == "unavailable":
        validity["reason"] = control.get("reason", record.get("applicability"))
    return {"test_id": record.get("test_id"), "result_id": rid, "test_version": record.get("test_version"),
            "measurement_validity": validity, "values": values, "interpretation_groups": list(groups.values()),
            "context": context}


def model_case(case):
    """Include shared numeric measurements once in the model context."""
    return {k: deepcopy(v) for k, v in case.items() if "::" not in k}


def reference_bundle(cards):
    """Collect reference findings with shared interpretation instructions."""
    if not cards:
        return {"shared": {}, "cards": []}
    common = {k: deepcopy(v) for k, v in cards[0].items()
              if all(k in c and c[k] == v for c in cards[1:])
              and k not in {"diagnostic", "test_id", "evidence_id", "measurement_id"}}
    return {"shared": common, "cards": [{k: deepcopy(v) for k, v in c.items() if k not in common} for c in cards]}


def validate_model_packet(packet):
    """Check that model-visible numbers match their source measurements."""
    expected = {k: m["statistic"] for k, m in packet["citation_info"].items() if m.get("citeable")}
    shown = {}
    for card in packet["visible_records"]:
        for key, value in card["values"].items():
            if value is not None:
                if key in shown:
                    raise ValueError("Duplicate numeric fact on the evidence cards")
                shown[key] = value
    if shown != expected:
        raise ValueError("Compact evidence cards lost or changed an admitted measurement")
    if packet["model_case_file"] != model_case(packet["case_file"]):
        raise ValueError("Static model context differs from the validated case")
    recurrence = [r for r in packet["records"] if r.get("test_id") == "training_profit_recurrence"]
    return {"evidence_card_version": CARD_VERSION, "n_numeric_facts": len(shown),
            "training_recurrence_available": bool(recurrence),
            "model_measurement_characters": len(json.dumps(packet["visible_records"], sort_keys=True)),
            "audit_measurement_characters": len(json.dumps(packet["records"], sort_keys=True))}
