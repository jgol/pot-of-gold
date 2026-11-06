"""Prepare the shared strategy evidence and run the single-LLM assessment.

The system prompt comes from prompts.build_single_llm_prompt. Strategy data,
measurements and reference findings are supplied as JSON user context.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from potofgold.multi_agent.assessment import court
from potofgold.multi_agent.assessment.prompts import build_single_llm_prompt

DEFAULT_MODEL = "gpt-6-astra"


def assess_strategy(cf, runner, reference_context, *, run_agent, model, evidence=None,
                    cohort_stats=None, enable_compute=False, catalogue_allow=None, allocation_policy=None):
    """Assess the strategy using the same case and training evidence as the court."""
    from potofgold.multi_agent.assessment.allocation_policy import PERCENTAGE, check_policy
    allocation_policy = check_policy(allocation_policy or getattr(runner, "allocation_policy", PERCENTAGE))
    packet = court.build_measurement_context(cf, runner, evidence=evidence, catalogue_allow=catalogue_allow,
        evidence_mode=getattr(runner, "evidence_mode", "training_economics"))
    from potofgold.multi_agent.assessment.reasoning_diagnostics import DiagnosticSession, peer_context, ACCESS_VERSION
    from potofgold.multi_agent.assessment.allocation_reasoning import investigation_schema
    session = DiagnosticSession(cf, runner, packet["records"], evidence=evidence,
                                allowed=catalogue_allow, max_requests=court._MAX_CATALOGUE_REQUESTS)
    peer = peer_context(packet["case_file"], cohort_stats) if enable_compute else {}
    reference = court.validate_exploratory_reference_context(reference_context)
    reference = reference_context if reference is None else reference
    dev_cards = court.development_reference_cards(
        cf, evidence, catalogue_allow=catalogue_allow, run_catalogue_test=runner)
    from potofgold.multi_agent.strategy_profiles.evidence_cards import INTERPRETATION, reference_bundle
    context = {"case_file": packet["model_case_file"], "measurements": packet["visible_records"],
               "training_economic_comparisons": packet["training_economic_comparisons"],
               "economic_scenario_context": packet["economic_scenario_context"],
               "exploratory_reference_only": reference,
               "development_reference_evidence": reference_bundle(dev_cards),
               "peer_percentiles": peer, "diagnostic_access": session.instructions()}
    context["evidence_card_guide"] = INTERPRETATION
    risk_fields = court.exploratory_risk_fields(reference, case_file=packet["case_file"],
        citation_info=packet["citation_info"])
    alloc_props = {
        "allocation_rationale": deepcopy(court._SUBMIT_VERDICT["input_schema"]["properties"]["allocation_rationale"]),
        "allocation_exhibit_ids": {"type": "array", "items": {"type": "string"}},
        "allocation_metric_values": deepcopy(court._SUBMIT_VERDICT["input_schema"]["properties"]["allocation_metric_values"]),
        "allocation_competing_explanation": {"type": "string"},
        "allocation_limitations": {"type": "string"},
        "allocation_investigations": investigation_schema(allocation_policy),
        "allocation_percent": deepcopy(court._SUBMIT_VERDICT["input_schema"]["properties"]["allocation_percent"]),
    }
    tool = {"name": "report_result", "description": "Submit a grounded comparative risk assessment + exposure action.",
            "input_schema": {"type": "object", "properties": {**deepcopy(risk_fields), **alloc_props},
                             "required": list(risk_fields) + ["allocation_rationale", "allocation_percent"], "additionalProperties": False}}
    prompt = build_single_llm_prompt(allocation_policy=allocation_policy,
        peer_compare_available=bool(peer), reference_cards=dev_cards)
    _alloc_state = {"retries_left": 1, "citation_retries_left": 1}
    def submit_assessment(value):
        return value
    def validate_assessment(value):
        session.refresh()
        for key in [k for k in packet["case_file"] if "::" in k]:
            packet["case_file"].pop(key)
        packet["case_file"].update(session.measured)
        packet["citation_info"] = session.citation_info
        court.validate_exploratory_risk(value, packet["case_file"], reference_context=reference,
            citation_info=packet["citation_info"])
        court._require_allocation_action(value, _alloc_state, allocation_policy)
        court._require_allocation_citations(value, packet["case_file"], _alloc_state)
        return value
    submit_assessment.validate_submission = validate_assessment
    diagnostic_tools, diagnostic_handlers = session.tools_and_handlers()
    result, metadata = run_agent("one_llm", model=model, max_turns=8, system_prompt=prompt,
        user_message="Assess this case and submit report_result.", tools=diagnostic_tools + [tool],
        tool_handlers={**diagnostic_handlers, "report_result": submit_assessment}, cache_prefix=json.dumps(context, sort_keys=True))
    session.refresh()
    for key in [k for k in packet["case_file"] if "::" in k]:
        packet["case_file"].pop(key)
    packet["case_file"].update(session.measured)
    packet["citation_info"] = session.citation_info
    submitted = deepcopy(result)
    try:
        court.validate_exploratory_risk(result, packet["case_file"], reference_context=reference,
            citation_info=packet["citation_info"])
    except Exception as exc:
        # Preserve the full rejected response; previews can omit the failing fields.
        exc._ranking_audit = {
            "submitted_assessment": submitted, "metadata": deepcopy(metadata),
            "validation_error": {"error_type": type(exc).__name__, "message": str(exc),
                                 **deepcopy(getattr(exc, "_validation_details", {}))},
            "evidence_digest": hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest(), "tool_records": deepcopy(session.records),
            "diagnostic_access": {"version": ACCESS_VERSION, "new_measurements": session.n_requests,
                                  "max_requests": session.max_requests}}
        raise
    court._normalize_allocation_vote(result, allocation_policy)
    court._verify_allocation_cites(result, packet["case_file"])
    _status = result.get("allocation_status")
    _a = result.get("allocation_action")
    if _status == "explicit" and _a in ("FULL", "CASH", "PARTIAL"):
        alloc_action, _alloc_fallback = _a, None
    else:
        # Record why a missing usable recommendation left the default allocation in place.
        alloc_action = "FULL"
        _alloc_fallback = {"deliberate_abstention": "baseline_deliberate_abstention",
                           "ungrounded_downgrade": "baseline_ungrounded_downgrade",
                           "technical_failure": "baseline_technical_failure"}.get(_status, "baseline_no_action")
    percent = result["allocation_percent"] if _alloc_fallback is None else 100.0
    return {"risk_score": float(result["risk_score"]), "assessment": result, "metadata": metadata,
            "submitted_assessment": submitted, "allocation_action": alloc_action,
            "allocation_multiplier": percent / 100,
            "allocation_percent": percent, "allocation_policy": allocation_policy,
            "allocation_status": _status, "allocation_fallback_reason": _alloc_fallback,
            "tool_records": deepcopy(session.records),
            "diagnostic_access": {"version": ACCESS_VERSION, "new_measurements": session.n_requests,
                                  "max_requests": session.max_requests},
            "evidence_digest": hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest()}
