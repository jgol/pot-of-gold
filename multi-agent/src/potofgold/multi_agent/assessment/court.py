"""Assess strategy risk with a prosecutor, defender and a panel of judges.

Agents receive training profiles, development references and diagnostic tools.
Judges supply risk scores and allocation reasoning; the preserved veto label
remains available for the published allocation rule. Held-out outcomes enter
only subsequent evaluation."""
from __future__ import annotations

from potofgold.multi_agent.assessment.prompts import (
    DEFENDER_SYSTEM,
    DEFENSE_CLAIM_GUIDE,
    EXPLORATORY_RISK_NOTE,
    JUDGE_SYSTEM,
    PROSECUTOR_SYSTEM,
    _COST_MEASUREMENT_GUIDE,
    _ECONOMICS_ABSENT_STATUS,
    _capability_note,
    _verdict_evidence_policy,
)

import hashlib
import json
import logging
from copy import deepcopy
import math
import re
from statistics import median
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

_log = logging.getLogger(__name__)

from potofgold.multi_agent.assessment.court_citations import MalformedVerdict, _verify_cites
from potofgold.multi_agent.assessment.court_claims import bind_defense_claims
from potofgold.multi_agent.assessment.allocation_policy import (
    PERCENTAGE,
    MATERIAL_SPREAD_PP,
    check_policy,
    valid_percent,
    percentage_vote_issue,
    normalize_percentage_vote,
    aggregate_percentages,
)
from potofgold.multi_agent.strategy_profiles.economic_comparisons import (
    economic_comparison_context,
    economic_scenario_context,
)
from potofgold.multi_agent.assessment.reasoning_claims import validate_scenario_claims
from potofgold.multi_agent.strategy_profiles.generalization_case_file import (
    GeneralizationCaseFile,
    _LEAKY_PATTERNS,
    _opaque_alias,
)
from potofgold.multi_agent.strategy_profiles.tool_evidence import citation_metadata, development_reference_context


REFERENCE_POLICY_VERSION = "development-risk-ranking-v2"
REFERENCE_GRADE_VERSION = "excluded-fold-predictive-reference-v1"
REFERENCE_SCOPE = "same_arm_original_reference_folds_only"
REFERENCE_LIMITATIONS = (
    "Reference-rule validation does not predict this candidate or validate a combined ranking.",
    "The return-change interval resamples days; it does not cover threshold-selection or seed uncertainty.",
    "CVaR improvement is a point estimate; no CVaR confidence interval is supplied.",
    "No positive-net reference winners: winner-rejection risk cannot be estimated.",
    "No negative-net reference losers: loss discrimination cannot be established.",
    "This diagnostic has no supportive reference-validation relationship with loss.",
    "Support is descriptive; the return-change interval does not establish an improvement.",
    "Some applicable reference cases lack an excluded-fold fitted decision.",
)

_DECISIONS = ("WILL_GENERALIZE", "WONT_GENERALIZE", "UNSURE")
_EVIDENCE_HEADER = "CASE FILE (approved training exhibits including economics; held-out outcomes hidden):"
_MAX_CATALOGUE_REQUESTS = 3    # Cache repeated requests and limit distinct new measurements per strategy.
COURT_POLICY_VERSION = "training-informed-research-v12-no-analyst"
EVIDENCE_MODES = ("training_economics",)
DECISION_POLICIES = ("research_prediction",)


_INCONCLUSIVE_CONDITIONS = (
    "measurement_failed", "control_failed", "baseline_or_projection_parity_failed",
    "nonfinite_statistic", "not_applicable",
)


def _model_inventory(inventory: dict) -> dict:
    """Select inventory fields for the model context."""
    return {"n_records": inventory["n_records"], "n_usable_records": inventory["n_usable_records"],
            "development_support_role": "evidence_strength_only",
            "interpretation": "Reliable observations can support an explicitly uncertain research prediction. "
                "Development support affects weight; its absence is not a permission gate."}


def _case_structural_applicability(cf: GeneralizationCaseFile) -> dict:
    """Exclude known structural mismatches; defer missing information to the executor."""
    from potofgold.multi_agent.diagnostics.vulnerability_catalogue import structural_applicability

    thresholds = getattr(cf, "n_evolved_thresholds", None)
    signals = getattr(cf, "n_signals", None)
    applicable = structural_applicability(
        n_perturbable_targets=1 if thresholds is None else int(thresholds),
        uses_embedding=bool(getattr(cf, "uses_embedding", False)),
        n_signals=2 if signals is None else int(signals))
    if thresholds is None:
        applicable["threshold_sensitivity"] = {
            "applicable": True, "reason": "unknown threshold count; executor checks applicability"}
    if signals is None:
        for diagnostic in ("signal_reliance", "dependence_preserving_control"):
            applicable[diagnostic] = {
                "applicable": True, "reason": "unknown signal count; executor checks applicability"}
    return applicable


def _add_economics_reference(reference_ids: Sequence[str]) -> list:
    """Include the training-economics reference when the rule book supplies it."""
    from potofgold.multi_agent.strategy_profiles.training_economics import TEST_ID as _ECON_ID
    ids = list(reference_ids)
    if _ECON_ID not in ids:
        ids.append(_ECON_ID)
    return ids


def _add_derived_economic_references(reference_ids: Sequence[str]) -> list:
    """Include cost-erosion and profit-concentration references from the rule book."""
    from potofgold.multi_agent.strategy_profiles.economic_predictors import DERIVED_PREDICTORS
    ids = list(reference_ids)
    for d in DERIVED_PREDICTORS:
        if d not in ids:
            ids.append(d)
    return ids


def _drop_absent_economics(reference_context: Sequence[dict]) -> list:
    """Omit reference cards that the rule book could not resolve."""
    from potofgold.multi_agent.strategy_profiles.training_economics import TEST_ID as _ECON_ID
    from potofgold.multi_agent.strategy_profiles.economic_predictors import DERIVED_PREDICTORS
    store_refs = {_ECON_ID, *DERIVED_PREDICTORS}
    return [r for r in reference_context
            if not (r.get("diagnostic") in store_refs and r.get("dev_status") in _ECONOMICS_ABSENT_STATUS)]


def development_reference_cards(cf, evidence, *, catalogue_allow=None, run_catalogue_test=None) -> list:
    """Build the reference cards shared by the court and single LLM."""
    if evidence is None:
        return []
    applicable = _case_structural_applicability(cf)
    reference_ids = [tid for tid, info in applicable.items() if info["applicable"]
                     and (catalogue_allow is None or tid in catalogue_allow)]
    if run_catalogue_test is not None and hasattr(run_catalogue_test, "available_diagnostics"):
        reference_ids = [tid for tid in reference_ids if tid in run_catalogue_test.available_diagnostics]
    reference_ids = _add_economics_reference(reference_ids)
    reference_ids = _add_derived_economic_references(reference_ids)
    return _drop_absent_economics(development_reference_context(
        reference_ids, evidence=evidence, arm=getattr(cf, "arm", None), scope=evidence.get("scope", "all")))


def _safe_precommit(request: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the requested interpretation in the audit for comparison with the result."""
    return {
        "resolves_uncertainty": request.get("resolves_uncertainty"),
        "keep_if": "A reliable applicable observation weakens this specific allegation; this does not establish WILL_GENERALIZE.",
        "reject_if": "A reliable adverse observation may inform an explicitly uncertain prediction under the research policy; development support affects its weight, and the active verdict policy still applies.",
        "inconclusive_if": list(_INCONCLUSIVE_CONDITIONS),
        "failure_interpretation": "INCONCLUSIVE",
    }


def _agent_record_view(record: Dict[str, Any]) -> Dict[str, Any]:
    """Present the verified measurement card shared by all H3 agents."""
    from potofgold.multi_agent.strategy_profiles.evidence_cards import measurement_card
    return measurement_card(record)


def _measurement_key(test_id: Any) -> str:
    return "signal_reliance" if test_id in ("signal_reliance", "dependence_preserving_control") else str(test_id)


def _saved_measurements(runner: Any, *, allowed: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    """Read the runner's verified snapshot without requesting another measurement."""
    fetch = getattr(runner, "get_preloaded_measurements", None)
    if not callable(fetch):
        return []
    from potofgold.multi_agent.assessment.run_readiness import assert_runner_ready
    from potofgold.multi_agent.diagnostics.vulnerability_catalogue import REGISTRY
    assert_runner_ready(runner)
    readiness = getattr(runner, "readiness", {})
    fingerprint = readiness.get("source_measurement_fingerprint")
    if not fingerprint:
        raise ValueError("Preloaded measurements require verified saved-run provenance")
    raw_records = fetch()
    if not isinstance(raw_records, (list, tuple)) or len(raw_records) > len(REGISTRY):
        raise ValueError("Preloaded measurement snapshot must be a bounded record sequence")
    fields = ("result_id", "test_id", "test_version", "parameters", "data_split_hash", "replay_fidelity",
              "measured_statistics", "control_results", "applicability", "decision", "calibration_id")
    accepted, identities = [], {}
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("Preloaded measurement snapshot contains a malformed record")
        tid = raw.get("test_id")
        if allowed is not None and tid not in allowed:
            continue
        if (tid not in REGISTRY or raw.get("test_version") != REGISTRY[tid].test_version
                or raw.get("parameters") != {} or raw.get("replay_fidelity") != 1.0
                or isinstance(raw.get("replay_fidelity"), bool) or not raw.get("result_id")):
            raise ValueError("Preloaded measurement identity/version/parameters/replay is incompatible")
        # Exclude labels, held-out outcomes and arbitrary top-level fields.
        ex = _scrub_leaky_exhibit(_record_exhibit({key: deepcopy(raw[key]) for key in fields if key in raw}, tid))
        rid = str(ex["result_id"])
        identity = json.dumps(ex, sort_keys=True, default=str)
        if rid in identities:
            if identities[rid] != identity:
                raise ValueError("Conflicting preloaded result identity")
            continue
        identities[rid] = identity
        ex["preloaded"] = True
        source = (readiness.get("measurement_source_fingerprints") or {}).get(rid, fingerprint)
        if not isinstance(source, str) or not source:
            raise ValueError("Preloaded measurement has no bound source fingerprint")
        ex["preload_provenance"] = {"source_measurement_fingerprint": source}
        accepted.append(ex)
    return accepted


def _record_exhibit(res: Any, tid: str) -> Dict[str, Any]:
    """Convert a diagnostic result into the common measurement-record format."""
    if hasattr(res, "to_exhibit"):
        ex = dict(res.to_exhibit())
    elif isinstance(res, dict) and res.get("result_id"):
        ex = dict(res)
    else:
        d = dict(res) if isinstance(res, dict) else {}
        stat = d.get("statistic")
        from potofgold.multi_agent.diagnostics.vulnerability_catalogue import REGISTRY
        entry = REGISTRY.get(tid)
        statistic_name = entry.statistic_name if entry else f"{tid}_statistic"
        measured = {statistic_name: stat} if isinstance(stat, (int, float)) and not isinstance(stat, bool) else {}
        rid = hashlib.sha256(json.dumps([tid, d], sort_keys=True, default=str).encode()).hexdigest()[:16]
        ctrl = {k: v for k, v in d.items() if k not in ("statistic", "test_id")}
        if not d.get("decision"):                   # Missing runner output cannot supply a usable measurement.
            ctrl.setdefault("reason", "runner_returned_no_decision")
        ex = {"result_id": rid, "test_id": d.get("test_id", tid), "measured_statistics": measured,
              "control_results": ctrl, "applicability": ("applicable" if d.get("applicable") else "not_applicable"),
              "decision": d.get("decision") or "review", "calibration_id": d.get("calibration_id")}
    if (ex.get("control_results") or {}).get("uniform_threshold_used"):
        ex["self_calibrating"] = True
    return ex


def _scrub_leaky_exhibit(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Remove fields outside the permitted training-measurement schema."""
    from potofgold.multi_agent.diagnostics.vulnerability_catalogue import REGISTRY
    from potofgold.multi_agent.strategy_profiles.evidence_summaries import approved_boundary_context
    summaries, summary_statistics = approved_boundary_context(ex)
    entry = REGISTRY.get(ex.get("test_id"))
    registered = ({entry.statistic_name, *entry.secondary_statistics}
                  if entry is not None and ex.get("test_version") == entry.test_version else set())

    def _clean(d, *, measured=False):
        return {k: v for k, v in (d or {}).items()
                if (not any(p in str(k).lower() for p in _LEAKY_PATTERNS)
                    or (measured and k in registered and isinstance(v, (int, float))
                        and not isinstance(v, bool) and math.isfinite(v)))
                and str(k).lower() not in {"arm", "fold", "seed", "champion_id", "uses_embedding", "code_sha"}
                and not isinstance(v, (list, dict, tuple, set))}
    ex = dict(ex)
    ex["measured_statistics"] = _clean(ex.get("measured_statistics"), measured=True)
    if ex.get("test_id") == "entry_boundary_economics":
        # Rebuild citation aliases from the validated records.
        ex["measured_statistics"] = {k: v for k, v in ex["measured_statistics"].items()
            if not k.startswith(("boundary_scenario_", "boundary_scenarios_", "boundary_profit_",
                                 "exit_attribution_", "exit_profit_share_"))}
    ex["control_results"] = _clean(ex.get("control_results"))
    ex["control_results"].update(summaries)
    ex["measured_statistics"].update(summary_statistics)
    return ex


def _matched_verdict_mapping(verdict: str) -> Dict[str, Any]:
    """Map matched-control findings to diagnostic status.

    An exceptional result describes unusual behavior relative to its control;
    it does not by itself establish a relationship with future loss.
    """
    holds = True if verdict == "EXCEPTIONAL" else False if verdict == "NOT_EXCEPTIONAL" else None
    return {"applicability": "applicable" if verdict in ("EXCEPTIONAL", "NOT_EXCEPTIONAL") else "not_applicable",
            "allegation_holds": holds,
            "decision": "keep" if verdict == "NOT_EXCEPTIONAL" else "review"}


def _measured_exhibits(records: List[Dict[str, Any]], *, citation_info: Optional[dict] = None) -> Dict[str, Any]:
    """Expose reliable observations with their citation metadata."""
    info = citation_metadata(records) if citation_info is None else citation_info
    return {key: meta["statistic"] for key, meta in info.items() if meta.get("citeable")}


def _annotate_measurements(records, *, evidence, arm, scope):
    """Validate measurements and attach citation information for both assessments."""
    info = citation_metadata(records, evidence=evidence, arm=arm, scope=scope)
    fields = ("family", "direction", "citeable", "dev_status", "dev_supported", "qualified_support",
              "guarded_admissible", "claim_scope", "evidence_kind", "research_admissible",
              "research_admission_reason", "adverse", "threshold_source", "evidence_id", "reason")
    for record in records:
        record["citation_admission"] = {
            key: {field: meta.get(field) for field in fields} for key, meta in info.items()
            if meta.get("result_id") == record.get("result_id")}
        valid = any(meta.get("citeable") for meta in record["citation_admission"].values())
        record["measurement_interpretation"] = {
            "status": "VALID_OBSERVATION" if valid else "INCONCLUSIVE",
            "failure_interpretation": "INCONCLUSIVE",
            "supports_strategy_rejection": any(meta.get("qualified_support")
                for meta in record["citation_admission"].values()),
            "support_field_meaning": "Validated development rejection support only; not decision permission under research_prediction.",
        }
        if record.get("test_id") == "transaction_cost_sensitivity":
            # Presentation changes do not alter stored measurements or diagnostic versions.
            record["measurement_interpretation"].update(
                observation_scope="trading_behaviour_under_explicit_cost_multipliers",
                profitability_assessment="UNRESOLVED",
                interpretation=_COST_MEASUREMENT_GUIDE)
    return info


def build_measurement_context(cf: GeneralizationCaseFile, runner: Any, *,
                              evidence: Optional[dict] = None,
                              catalogue_allow: Optional[Sequence[str]] = None,
                              evidence_mode: str = "training_economics") -> Dict[str, Any]:
    """Prepare the shared strategy case, measurements and reference information.

    Reuse saved records or obtain the runner's cached training calculations.
    Keep provenance in records and expose the selected fields in visible_records.
    Recompute reference support from the supplied rule book.
    """
    if evidence_mode not in EVIDENCE_MODES:
        raise ValueError("Unknown court evidence mode")
    records = _saved_measurements(runner, allowed=catalogue_allow)
    from potofgold.multi_agent.assessment.run_readiness import assert_runner_ready
    from potofgold.multi_agent.strategy_profiles.training_economics import validate_training_economics_record
    assert_runner_ready(runner)
    fetch_economics = getattr(runner, "get_training_economics", None)
    if not callable(fetch_economics):
        raise ValueError("Training-informed court requires approved training economics before model calls")
    record = validate_training_economics_record(fetch_economics(), champion_id=cf.champion_id,
        source_measurement_fingerprint=getattr(runner, "readiness", {}).get("source_measurement_fingerprint"))
    record["preloaded"] = True
    record["preload_provenance"] = {"source_measurement_fingerprint":
        record["control_results"]["source_measurement_fingerprint"]}
    records.append(record)
    fetch_recurrence = getattr(runner, "get_training_recurrence", None)
    if callable(fetch_recurrence):
        from potofgold.multi_agent.diagnostics.training_recurrence import validate_recurrence_record
        recurrence = validate_recurrence_record(fetch_recurrence(), champion_id=cf.champion_id,
            source_measurement_fingerprint=getattr(runner, "readiness", {}).get("source_measurement_fingerprint"))
        if recurrence["control_results"]["source_result_id"] != record["result_id"]:
            raise ValueError("Recurrence and training economics have different sources")
        recurrence["preloaded"] = True
        recurrence["preload_provenance"] = deepcopy(record["preload_provenance"])
        records.append(recurrence)
    # Derive cost erosion and profit concentration from the cached training report.
    fetch_derived = getattr(runner, "get_derived_economics", None)
    if callable(fetch_derived) and getattr(runner, "readiness", {}).get("ready"):
        from potofgold.multi_agent.assessment.run_readiness import assert_runner_ready
        from potofgold.multi_agent.strategy_profiles.economic_predictors import economic_admission_reason, economic_binding_reason
        assert_runner_ready(runner)
        _fp = getattr(runner, "readiness", {}).get("source_measurement_fingerprint")
        for drec in fetch_derived():
            problem = economic_admission_reason(drec) or economic_binding_reason(
                drec, champion_id=cf.champion_id, source_fingerprint=_fp)
            if problem:
                raise ValueError(f"Inadmissible/unbound derived economic record ({drec.get('test_id')!r}): {problem}")
            if any(r.get("result_id") == drec.get("result_id") for r in records):
                raise ValueError("Derived economic record duplicates a preloaded result identity")
            drec = dict(drec)
            drec["preloaded"] = True
            drec["preload_provenance"] = {"source_measurement_fingerprint":
                                          drec["control_results"]["source_measurement_fingerprint"]}
            records.append(drec)
    info = _annotate_measurements(records, evidence=evidence, arm=getattr(cf, "arm", None),
                                  scope=(evidence or {}).get("scope", "all"))
    measured = _measured_exhibits(records, citation_info=info)
    qualifying = {key: meta for key, meta in info.items()
                  if (meta.get("qualified_support") or meta.get("guarded_admissible"))
                  and meta.get("family") not in (None, "other")}
    families = sorted({meta["family"] for meta in qualifying.values()})
    inventory = {
        "n_records": len(records),
        "n_usable_records": len({meta["result_id"] for meta in info.values() if meta.get("citeable")}),
        "qualifying_adverse_families": families,
        "qualified_measurement_ids": sorted(qualifying),
        "rejection_evidence_available": len(families) >= 2,
        "interpretation": "Development support describes evidence strength; it does not gate an H3 prediction.",
    }
    public_case = {**cf.to_llm_dict(), **measured}
    from potofgold.multi_agent.strategy_profiles.evidence_cards import model_case
    return {"case_file": public_case, "model_case_file": model_case(public_case),
            "records": records,
            "training_economic_comparisons": economic_comparison_context(public_case, info),
            "economic_scenario_context": economic_scenario_context(public_case, info),
            "visible_records": [_agent_record_view(record) for record in records],
            "citation_info": info, "preloaded_evidence": inventory, "evidence_mode": evidence_mode}


def _sanitize_defense(defense: Dict[str, Any], case: Dict[str, Any],
                      citation_info: Dict[str, Any], *, prosecution: Optional[dict] = None) -> Dict[str, Any]:
    """Check the scope and citations of factual defense assessments.

    Keep the submitted response separately in the audit.
    """
    # Build the judging summary from admitted assessments; retain the submission in the audit.
    out = {}
    binding_errors = []
    if prosecution is not None:
        defense, binding_errors = bind_defense_claims(defense, prosecution)
        out.update({key: deepcopy(defense[key]) for key in ("source_position", "prosecution_rationale")})
    assessments = defense.get("allegation_assessments")
    corrections = ["Unbound prosecution quotation: " + item["reason"] for item in binding_errors]
    if not isinstance(assessments, list) or not assessments:
        assessments = []
        corrections.append("No scoped factual allegations were supplied; no profitability affirmation is established.")
    cleaned = []
    for raw in assessments:
        assessment = ({key: deepcopy(raw[key]) for key in (
            "allegation", "prosecutor_quote", "claim_binding", "claim_scope", "classification",
            "decisive_exhibit_ids", "context_exhibit_ids", "basis") if key in raw}
            if isinstance(raw, dict) else {})
        scope = assessment.get("claim_scope")
        classification = assessment.get("classification")
        reason = None
        if scope in ("PROFITABILITY", "GENERALIZATION") and classification != "UNRESOLVED":
            reason = ("A factual rebuttal cannot settle unbounded profitability or future generalization. "
                      "Use TRAINING_ECONOMICS only for an explicitly measured training scenario. "
                      "Unchanged behaviour does not refute unprofitability; judges separately assess prediction.")
        elif scope not in ("OBSERVED_BEHAVIOR", "TRAINING_ECONOMICS", "PROFITABILITY", "GENERALIZATION"):
            reason = "Missing or unknown claim scope; an unscoped rebuttal cannot settle the allegation."
        elif classification not in ("SUPPORTED", "OVERSTATED", "CONTRADICTED", "UNRESOLVED"):
            reason = "Missing or invalid factual assessment classification."
        elif classification != "UNRESOLVED":
            ids = assessment.get("decisive_exhibit_ids")

            def admissible(key):
                if not isinstance(key, str):
                    return False
                value = case.get(key)
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value)):
                    return False
                if "::" in key:
                    meta = citation_info.get(key, {})
                    # A citation's declared scope must match its measurement type.
                    return bool(meta.get("citeable")
                        and (scope == "TRAINING_ECONOMICS") == (meta.get("claim_scope") == "TRAINING_ECONOMICS"))
                if scope == "TRAINING_ECONOMICS":
                    return False
                if case.get("fidelity_ok") is False:
                    return False
                return not (case.get("ablation_confounded") and
                            any(token in key for token in ("ablation_phi", "worst_ablation")))

            admitted_ids = [key for key in ids if admissible(key)] if isinstance(ids, list) else []
            contextual_ids = [key for key in ids if isinstance(key, str) and not admissible(key)
                              and key in case] if isinstance(ids, list) else []
            unknown_ids = [key for key in ids if not isinstance(key, str) or key not in case] if isinstance(ids, list) else []
            assessment["decisive_exhibit_ids"] = admitted_ids
            if contextual_ids:
                # Citing a measurement's limitation does not establish a claim in another scope.
                assessment["decisive_exhibit_ids"] = admitted_ids
                assessment["context_exhibit_ids"] = list(dict.fromkeys(
                    [key for key in assessment.get("context_exhibit_ids", []) if isinstance(key, str) and key in case]
                    + contextual_ids))
                assessment["admission_notes"] = [
                    "Context citations do not establish this claim: they have a different scope or are "
                    "unreliable for decisive inference. They may explain limitations. "
                    "Assess the authored reasoning separately from citation admission."]
            if unknown_ids:
                reason = "Unknown decisive citations were excluded; this classification is not admitted as a factual finding."
            elif not admitted_ids:
                reason = "No reliable numeric exhibit of the required scope supports this factual rebuttal."
            elif scope == "TRAINING_ECONOMICS" and all(key.endswith("_multiplier") for key in admitted_ids):
                reason = "A cost multiplier specifies the scenario; cite its measured economic response."
        if reason:
            assessment["classification"] = "UNRESOLVED"
            if scope in ("OBSERVED_BEHAVIOR", "TRAINING_ECONOMICS"):
                assessment.setdefault("admission_notes", []).append(reason)
            else:
                assessment["basis"] = reason
            corrections.append(reason)
        cleaned.append(assessment)
    out["allegation_assessments"] = cleaned
    retention = defense.get("retention_case")
    if isinstance(retention, dict):
        out["retention_case"] = {k: deepcopy(retention[k]) for k in
            ("rationale", "supporting_exhibit_ids", "limitations") if k in retention}
        out["retention_case"]["scope"] = "UNCERTAIN_ALLOCATION_ARGUMENT"
        out["retention_case"]["interpretation"] = (
            "Independent case for retaining exposure; assess its cited facts and reasoning, not as proof of future profit.")
    classes = [item.get("classification") for item in cleaned]
    out["verdict_on_claim"] = (
        "UNRESOLVED" if not classes or "UNRESOLVED" in classes else
        "QUOTED_CLAIMS_ASSESSED" if prosecution is not None else
        "OBSERVED_CONCERNS_REMAIN" if any(c in ("SUPPORTED", "OVERSTATED") for c in classes) else
        "OBSERVED_ALLEGATIONS_CONTRADICTED")
    out["rebuttal_rationale"] = (
        "Scoped factual assessments: " + ", ".join(
            f"{label}={classes.count(label)}" for label in
            ("SUPPORTED", "OVERSTATED", "CONTRADICTED", "UNRESOLVED"))
        + ". These classify observed claims only. Approved economics settle only tested training scenarios, never future generalization.")
    if corrections:
        out["scope_corrections"] = corrections
    return out


def _validate_defense_submission(defense: Any, prosecution: dict, *, case=None, require_retention=False) -> dict:
    """Check that the defender addresses claim identities from the prosecution."""
    if not isinstance(defense, dict) or not isinstance(defense.get("allegation_assessments"), list):
        raise ValueError("INVALID_PROSECUTOR_QUOTE: supply allegation_assessments as an array; it may be empty")
    if any(not isinstance(defense.get(key), str) or not defense[key].strip()
           for key in ("verdict_on_claim", "rebuttal_rationale")):
        raise ValueError("INVALID_PROSECUTOR_QUOTE: provide a nonempty verdict_on_claim and rebuttal_rationale")
    if len(defense["allegation_assessments"]) > 3:
        raise ValueError("INVALID_DEFENSE_SUBMISSION: assess at most three material prosecutor statements; "
                         "select the most relevant ones and resubmit the complete defense object")
    retention = defense.get("retention_case")
    if require_retention or retention is not None:
        if (not isinstance(retention, dict) or any(not isinstance(retention.get(k), str)
                or not retention[k].strip() for k in ("rationale", "limitations"))
                or not isinstance(retention.get("supporting_exhibit_ids"), list)):
            raise ValueError("INVALID_RETENTION_CASE: supply independent rationale, supporting_exhibit_ids and limitations; "
                             "you may explain that the case for retaining exposure is weak")
        if case is not None and any(not isinstance(k, str) or k not in case
                                   for k in retention["supporting_exhibit_ids"]):
            raise ValueError("INVALID_RETENTION_CITATION: use exact existing exhibit IDs")
    _, corrections = bind_defense_claims(defense, prosecution)
    if corrections:
        errors = [{key: item[key] for key in ("assessment_index", "reason", "suggested_prosecutor_quote") if key in item}
                  for item in corrections]
        raise ValueError("INVALID_PROSECUTOR_QUOTE: copy complete sentences verbatim from this prosecutor's "
                         "rationale, preserving negations and qualifications, or remove an invented allegation: "
                         + json.dumps(errors))
    return deepcopy(defense)


_SUBMIT_PROSECUTION = {"name": "submit_prosecution", "description": "Submit the prosecution brief.",
    "input_schema": {"type": "object", "properties": {
        "alleged_class": {"type": "string"}, "confidence": {"type": "number"},
        "decisive_exhibit_ids": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"}}, "required": ["alleged_class", "rationale"]}}
_SUBMIT_DEFENSE = {"name": "submit_defense", "description": "Submit the defense brief.",
    "input_schema": {"type": "object", "properties": {
        "verdict_on_claim": {"type": "string"}, "confidence": {"type": "number"},
        "allegation_assessments": {"type": "array", "maxItems": 3,
            "description": "At most three material statements from the actual prosecution. Empty is valid; prioritize relevance over exhaustive coverage.",
            "items": {"type": "object", "properties": {
            "allegation": {"type": "string"},
            "prosecutor_quote": {"type": "string", "description": "Verbatim complete sentence(s) from the actual prosecutor rationale; preserve uncertainty and negation. The server binds this assessment to that text."},
            "claim_scope": {"type": "string", "enum": ["OBSERVED_BEHAVIOR", "TRAINING_ECONOMICS", "PROFITABILITY", "GENERALIZATION", "UNKNOWN"]},
            "decisive_exhibit_ids": {"type": "array", "items": {"type": "string"}},
            "context_exhibit_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional citations used only to discuss limitations or a different measurement scope; not decisive support."},
            "classification": {"type": "string", "enum": ["SUPPORTED", "OVERSTATED", "CONTRADICTED", "UNRESOLVED"]},
            "basis": {"type": "string"}},
            "required": ["allegation", "prosecutor_quote", "claim_scope", "decisive_exhibit_ids", "classification", "basis"]}},
        "retention_case": {"type": "object", "properties": {
            "rationale": {"type": "string"},
            "supporting_exhibit_ids": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "string"}},
            "required": ["rationale", "supporting_exhibit_ids", "limitations"], "additionalProperties": False},
        "rebuttal_rationale": {"type": "string"}},
        "required": ["verdict_on_claim", "allegation_assessments", "rebuttal_rationale", "retention_case"]}}
_SUBMIT_VERDICT = {'name': 'submit_verdict',
 'description': 'Submit the generalization verdict.',
 'input_schema': {'type': 'object',
                  'properties': {'decision': {'type': 'string',
                                              'enum': ['WILL_GENERALIZE', 'WONT_GENERALIZE', 'UNSURE']},
                                 'stability_assessment': {'type': 'string',
                                                          'enum': ['ROBUST',
                                                                   'FRAGILE',
                                                                   'INSUFFICIENT_EVIDENCE']},
                                 'confidence': {'type': 'number'},
                                 'uncertainty_reason': {'type': 'string'},
                                 'decisive_exhibit_ids': {'type': 'array', 'items': {'type': 'string'}},
                                 'decisive_metric_values': {'type': 'object'},
                                 'unresolved_concern': {'type': 'boolean'},
                                 'verdict_rationale': {'type': 'string'},
                                 'allocation_rationale': {'type': 'string',
                                                          'description': 'Explain why this percentage is '
                                                                         'justified relative to higher and '
                                                                         'lower exposure, including the '
                                                                         'measured concern, counterevidence '
                                                                         'and sacrificed opportunity. '
                                                                         'Required for every numeric '
                                                                         'recommendation, including 0 and '
                                                                         '100.'},
                                 'allocation_exhibit_ids': {'type': 'array', 'items': {'type': 'string'}},
                                 'allocation_metric_values': {'type': 'object',
                                                              'additionalProperties': {'type': 'number'},
                                                              'description': 'Copy exact full aliases from '
                                                                             'allocation_exhibit_ids as keys '
                                                                             'and their numeric values. For '
                                                                             'abc::train_baseline_net use '
                                                                             '{"abc::train_baseline_net": '
                                                                             '5}, never shortened keys or '
                                                                             'arrays.'},
                                 'allocation_competing_explanation': {'type': 'string'},
                                 'allocation_limitations': {'type': 'string'},
                                 'allocation_percent': {'type': ['number', 'null'],
                                                        'minimum': 0,
                                                        'maximum': 100,
                                                        'description': "Percentage of this strategy's "
                                                                       'original allocation to retain: 0=No '
                                                                       'position, 100=Full, any intermediate '
                                                                       'percentage. Null is deliberate '
                                                                       'abstention, never 0. Always supply '
                                                                       'this field and justify a numeric '
                                                                       'recommendation.'},
                                 'allocation_investigations': {'type': 'array',
                                                               'maxItems': 3,
                                                               'items': {'type': 'object',
                                                                         'properties': {'question': {'type': 'string'},
                                                                                        'test_id': {'type': 'string'},
                                                                                        'requested_observation': {'type': 'string',
                                                                                                                  'description': 'Exact '
                                                                                                                                 'observation '
                                                                                                                                 'capability '
                                                                                                                                 'needed; '
                                                                                                                                 'use '
                                                                                                                                 'an '
                                                                                                                                 'unavailable '
                                                                                                                                 'experiment '
                                                                                                                                 'name '
                                                                                                                                 'when '
                                                                                                                                 'no '
                                                                                                                                 'offered '
                                                                                                                                 'tool '
                                                                                                                                 'can '
                                                                                                                                 'supply '
                                                                                                                                 'it.'},
                                                                                        'existing_evidence_ids': {'type': 'array',
                                                                                                                  'items': {'type': 'string'}},
                                                                                        'missing_observation': {'type': 'string'},
                                                                                        'source_quote': {'type': 'string'},
                                                                                        'interpretation_error': {'type': 'string'},
                                                                                        'allocation_percent_if_concern_confirmed': {'type': 'number',
                                                                                                                                    'minimum': 0,
                                                                                                                                    'maximum': 100},
                                                                                        'allocation_percent_if_concern_weakened': {'type': 'number',
                                                                                                                                   'minimum': 0,
                                                                                                                                   'maximum': 100}},
                                                                         'required': ['question',
                                                                                      'test_id',
                                                                                      'requested_observation',
                                                                                      'existing_evidence_ids',
                                                                                      'missing_observation',
                                                                                      'allocation_percent_if_concern_confirmed',
                                                                                      'allocation_percent_if_concern_weakened'],
                                                                         'additionalProperties': False}}},
                  'required': ['decision',
                               'stability_assessment',
                               'verdict_rationale',
                               'allocation_rationale',
                               'allocation_percent']}}

_REQUEST_CATALOGUE_TEST = {"name": "request_catalogue_test",
    "description": "Request ONE bounded catalogue test to be RUN on this champion; returns the MEASURED result "
                   "(decision keep/reject/review/not_applicable + statistic). You MUST state which judge uncertainty "
                   "it resolves and which result points which way — logged, not optional.",
    "input_schema": {"type": "object", "properties": {
        "test_id": {"type": "string"},
        "resolves_uncertainty": {"type": "string"},
        "keep_if": {"type": "string"},
        "reject_if": {"type": "string"},
        "inconclusive_if": {"type": "array", "items": {"type": "string", "enum": list(_INCONCLUSIVE_CONDITIONS)},
                            "description": "Include every failed-validity condition. The server always enforces all of them."}},
        "required": ["test_id", "resolves_uncertainty", "keep_if", "reject_if", "inconclusive_if"]}}


EXPLORATORY_RISK_FIELDS = {
    "risk_score": {"type": "number", "minimum": 0, "maximum": 100},
    "score_rationale": {"type": "string"},
    "risk_citations": {"type": "array", "minItems": 1,
        "description": "Numeric observations grounding the score. Copy each exact exhibit id and its observed value once.",
        "items": {"type": "object", "additionalProperties": False,
            "properties": {"exhibit_id": {"type": "string"}, "value": {"type": "number"}},
            "required": ["exhibit_id", "value"]}},
}


def _risk_citation_values(case_file, *, citation_info=None):
    """Eligible observations, using the same reliability restrictions as ranking validation."""
    return {key: value for key, value in case_file.items()
            if isinstance(key, str) and isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value)
            and not key.endswith("_available")
            and not (case_file.get("ablation_confounded") is True and key.startswith("ablation_phi"))
            and not (case_file.get("fidelity_ok") is False and "::" not in key)
            and not ("::" in key and
                     not (citation_info or {}).get(key, {}).get("research_admissible"))}


def exploratory_risk_fields(reference_context=None, *, case_file=None,
                            citation_info=None):
    """Use one citation representation, constrained to this panel's current observations."""
    result = deepcopy(EXPLORATORY_RISK_FIELDS)
    if case_file is not None:
        keys = sorted(_risk_citation_values(case_file, citation_info=citation_info))
        if not keys:
            raise ValueError("Exploratory ranking requires at least one eligible numeric observation")
        result["risk_citations"]["items"]["properties"]["exhibit_id"]["enum"] = keys
    if citation_info:
        result["risk_citations"]["description"] += (
            " Copy approved training economic values without sign-changing rounding. A fixed cost multiplier "
            "only defines a scenario; at least one citation must describe an observed response or other measurement.")
    if reference_context is not None:
        reference = validate_exploratory_reference_context(reference_context)
        rows = reference["diagnostics"]
        properties = {"diagnostic": {"type": "string"}, "grade": {"type": "string"},
                      "limitation": {"type": "string"}}
        if rows:
            properties["diagnostic"]["enum"] = sorted(row["diagnostic"] for row in rows)
            properties["grade"]["enum"] = sorted({row["predictive_evidence"]["grade"] for row in rows})
            properties["limitation"]["enum"] = sorted({limitation for row in rows
                for limitation in row["predictive_evidence"]["limitations"]})
        result["reference_evidence_assessment"] = {
            "type": "array", "minItems": len(rows), "maxItems": len(rows),
            "description": "Exactly one assessment per supplied diagnostic; copy its grade and one of its exact limitations.",
            "items": {"type": "object", "additionalProperties": False,
                "properties": properties,
                "required": ["diagnostic", "grade", "limitation"]}}
    return result


def _validate_training_economic_citations(ids, values, case, citation_info):
    """Check numerical training-economics citations against their declared scope."""
    info = citation_info or {}
    for key in ids:
        meta = info.get(key, {})
        if meta.get("claim_scope") != "TRAINING_ECONOMICS":
            continue
        observed, cited = case.get(key), values.get(key)
        if (not meta.get("citeable") or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) for value in (observed, cited))
                or not math.isclose(observed, cited, rel_tol=1e-8, abs_tol=1e-8)
                or (observed > 0) - (observed < 0) != (cited > 0) - (cited < 0)):
            raise MalformedVerdict("Training economic citation does not match the measured value and sign")
    if ids and all(info.get(key, {}).get("claim_scope") == "TRAINING_ECONOMICS" and
                   str(info[key].get("statistic_name", "")).endswith("_multiplier") for key in ids):
        raise MalformedVerdict("A fixed cost multiplier alone is not a measured economic response")


def _reject_positive_margin_only_cites(ids):
    """Sign positivity can follow from the entry predicate; it is not standalone risk evidence."""
    if ids and all(key == "fire_positive_frac" or re.fullmatch(r"per_node_\d+_positive_frac", key)
                   for key in ids):
        raise MalformedVerdict("Positive entry-margin fractions alone do not establish robustness, fragility or relative risk")


def validate_exploratory_risk(output: Dict[str, Any], case_file: Dict[str, Any], *,
                              reference_context=None, citation_info=None) -> None:
    """Validate the risk score and supporting citations from a court or single-LLM response."""
    def reject(path, message, **details):
        error = MalformedVerdict(f"{path}: {message}")
        error._validation_details = {"path": path, "message": message, **details}
        raise error

    if not isinstance(output, dict):
        reject("assessment", "Expected a structured ranking object", received_type=type(output).__name__)
    try:
        validate_scenario_claims(output, economic_scenario_context(case_file, citation_info or {}))
    except ValueError as exc:
        reject("reasoning", str(exc))
    score = output.get("risk_score")
    if (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
            or not 0 <= score <= 100):
        reject("risk_score", "must be a finite number between 0 and 100", received=score)
    if not isinstance(output.get("score_rationale"), str) or not output["score_rationale"].strip():
        reject("score_rationale", "Exploratory risk score needs a rationale")
    if "risk_citations" in output:
        rows = output["risk_citations"]
        if not isinstance(rows, list) or not rows:
            reject("risk_citations", "Expected a nonempty list of exhibit_id/value objects")
        values = {}
        for index, row in enumerate(rows):
            path = f"risk_citations[{index}]"
            if not isinstance(row, dict) or set(row) != {"exhibit_id", "value"}:
                reject(path, "Expected exactly exhibit_id and value", received=row)
            key = row["exhibit_id"]
            if not isinstance(key, str):
                reject(path + ".exhibit_id", "Expected a string", received=key)
            if key in values:
                reject(path + ".exhibit_id", "Duplicate exhibit identity", exhibit_id=key)
            values[key] = row["value"]
        ids = list(values)
        # Reject conflicting structured and legacy citation representations.
        for field, expected in (("risk_exhibit_ids", ids), ("risk_metric_values", values)):
            if field not in output:
                continue
            legacy = output[field]
            if field == "risk_exhibit_ids":
                if not isinstance(legacy, list) or any(not isinstance(key, str) for key in legacy):
                    reject(field, "Expected a string list", received=legacy)
            elif (not isinstance(legacy, dict) or any(
                    not isinstance(key, str) or isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for key, value in legacy.items())):
                reject(field, "Expected a finite numeric value map without booleans", received=legacy)
            if legacy != expected:
                reject(field, "Disagrees with risk_citations", received=legacy, expected=expected)
    else:
        ids, values = output.get("risk_exhibit_ids"), output.get("risk_metric_values")
        if not isinstance(ids, list) or not ids or any(not isinstance(key, str) for key in ids):
            reject("risk_exhibit_ids", "Legacy citations require a nonempty string list")
        if len(set(ids)) != len(ids):
            reject("risk_exhibit_ids", "Duplicate exhibit identity", received=ids)
        if not isinstance(values, dict) or not values:
            reject("risk_metric_values", "Legacy citations require a nonempty numeric value map")
        if set(ids) != set(values):
            reject("risk_metric_values", "Legacy exhibit identities and values disagree",
                   missing_values=sorted(set(ids) - set(values)),
                   unlisted_values=sorted(set(values) - set(ids), key=str))
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            reject("risk_citations", "Citations must be finite numeric observations", exhibit_id=key, received=value)
    if case_file.get("ablation_confounded") is True and any(key.startswith("ablation_phi") for key in ids):
        raise MalformedVerdict("Exploratory risk cannot rely on confounded legacy ablation measurements")
    if case_file.get("fidelity_ok") is False and any("::" not in key for key in ids):
        raise MalformedVerdict("Exploratory risk cannot rely on static measurements with failed replay fidelity")
    citations = {"decisive_exhibit_ids": ids, "decisive_metric_values": values}
    try:
        _verify_cites(citations, case_file, {})
        _reject_flag_cites(citations, case_file)
        _reject_positive_margin_only_cites(ids)
        _validate_training_economic_citations(ids, values, case_file, citation_info)
    except MalformedVerdict as error:
        reject("risk_citations", str(error))
    for key in ids:
        if key not in _risk_citation_values(case_file, citation_info=citation_info):
            reject("risk_citations", "Exhibit is not an eligible finite numeric observation", exhibit_id=key)
    if reference_context is not None:
        context = validate_exploratory_reference_context(reference_context)
        expected = {row["diagnostic"]: row["predictive_evidence"] for row in context["diagnostics"]}
        assessments = output.get("reference_evidence_assessment")
        if not isinstance(assessments, list) or len(assessments) != len(expected):
            raise MalformedVerdict("Ranking must acknowledge each diagnostic's predictive-evidence limitations")
        seen = set()
        for assessment in assessments:
            if (not isinstance(assessment, dict) or set(assessment) != {"diagnostic", "grade", "limitation"}
                    or assessment.get("diagnostic") not in expected or assessment["diagnostic"] in seen):
                raise MalformedVerdict("Ranking evidence assessments must identify each diagnostic exactly once")
            seen.add(assessment["diagnostic"])
            reference = expected[assessment["diagnostic"]]
            if (assessment["grade"] != reference["grade"]
                    or assessment["limitation"] not in reference["limitations"]):
                raise MalformedVerdict("Ranking must report the actual reference grade and a supplied limitation")
    # Populate compatibility fields only after validating the response.
    output["risk_citations"] = [{"exhibit_id": key, "value": values[key]} for key in ids]
    output["risk_exhibit_ids"] = list(ids)
    output["risk_metric_values"] = {key: values[key] for key in ids}


def validate_exploratory_reference_context(context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate aggregate reference information before formatting model context.

    The caller is responsible for producing it from the development partition.
    """
    if context is None:
        return None
    from potofgold.multi_agent.diagnostics.vulnerability_catalogue import REGISTRY
    keys = {"schema", "status", "claim", "outcome", "n_reference", "n_reference_folds",
            "evaluated_outcomes_used", "diagnostics"}
    if (not isinstance(context, dict) or set(context) != keys
            or context.get("schema") != REFERENCE_POLICY_VERSION
            or context.get("status") != "exploratory_reference_only"
            or context.get("evaluated_outcomes_used") is not False):
        raise ValueError("Exploratory reference must be the aggregate-only, excluded-outcome schema")

    def count(value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Exploratory reference counts must be nonnegative integers")
        return value

    def number(value):
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)

    n_reference = count(context["n_reference"])
    count(context["n_reference_folds"])
    diagnostics = context["diagnostics"]
    if not isinstance(diagnostics, list) or len(diagnostics) > len(REGISTRY):
        raise ValueError("Exploratory reference diagnostics must be a bounded list")
    seen = set()
    for row in diagnostics:
        if (not isinstance(row, dict) or set(row) != {"diagnostic", "n_measured", "reference_measurement_fraction",
                                                     "reference_groups", "reference_variation", "predictive_evidence"}
                or row.get("diagnostic") not in REGISTRY or row["diagnostic"] in seen):
            raise ValueError("Malformed or duplicated exploratory reference diagnostic")
        seen.add(row["diagnostic"])
        n_measured = count(row["n_measured"])
        fraction = row["reference_measurement_fraction"]
        if (n_measured > n_reference or not number(fraction)
                or fraction != (n_measured / n_reference if n_reference else 0.0)
                or not isinstance(row["reference_variation"], bool)):
            raise ValueError("Invalid exploratory reference diagnostic counts or fraction")
        groups = row["reference_groups"]
        if not isinstance(groups, dict) or set(groups) != {"negative_net", "positive_net", "zero_net"}:
            raise ValueError("Exploratory reference requires aggregate outcome groups only")
        total = 0
        for group in groups.values():
            if not isinstance(group, dict) or set(group) != {"n", "median"}:
                raise ValueError("Exploratory reference group contains unsupported fields")
            total += count(group["n"])
            if group["median"] is not None and not number(group["median"]):
                raise ValueError("Exploratory reference medians must be finite or null")
        if total != n_measured:
            raise ValueError("Exploratory reference groups exceed measured population")
        _validate_predictive_reference(row, n_reference, context["n_reference_folds"])
    safe = deepcopy(context)
    safe["claim"] = "Exploratory reference-fold associations; not qualified development support or validated causal mechanisms."
    safe["outcome"] = "Sign of net return on reference members' held-out development days."
    return safe


def _validate_predictive_reference(row, n_reference, n_folds):
    """Bound the aggregate grade to the actual registered diagnostic and scope."""
    from potofgold.multi_agent.strategy_profiles.diagnostic_evidence import STATUSES, _status
    from potofgold.multi_agent.strategy_profiles.tool_evidence import TOOL_FAMILIES
    from potofgold.multi_agent.diagnostics.vulnerability_catalogue import REGISTRY
    evidence = row.get("predictive_evidence")
    keys = {"schema", "scope", "diagnostic_version", "family", "statistic_name", "reject_when",
            "grade", "reference_status", "reference_id", "n_applicable", "n_folds_evaluated",
            "n_positive_net_reference", "n_negative_net_reference", "validation_fraction", "confusion",
            "portfolio", "uncertainty", "limitations"}
    entry = REGISTRY[row["diagnostic"]]
    if (not isinstance(evidence, dict) or set(evidence) != keys
            or evidence["schema"] != REFERENCE_GRADE_VERSION or evidence["scope"] != REFERENCE_SCOPE
            or evidence["diagnostic_version"] != entry.test_version
            or evidence["family"] != TOOL_FAMILIES[row["diagnostic"]]
            or evidence["statistic_name"] != entry.statistic_name or evidence["reject_when"] != entry.reject_when
            or evidence["reference_status"] not in STATUSES
            or evidence["grade"] != {"supported_on_dev_validation": "supported", "contrary_evidence": "contrary"}.get(
                evidence["reference_status"], "insufficient")
            or not isinstance(evidence["reference_id"], str)
            or not re.fullmatch(r"reference_only_[0-9a-f]{20}", evidence["reference_id"])):
        raise ValueError("Predictive reference diagnostic/version/family/direction/scope identity mismatch")

    def count(value):
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    for key in ("n_applicable", "n_folds_evaluated", "n_positive_net_reference", "n_negative_net_reference"):
        if not count(evidence[key]):
            raise ValueError("Predictive reference counts must be nonnegative integers")
    if (evidence["n_applicable"] != row["n_measured"] or evidence["n_folds_evaluated"] > n_folds
            or evidence["n_positive_net_reference"] != row["reference_groups"]["positive_net"]["n"]
            or evidence["n_negative_net_reference"] != row["reference_groups"]["negative_net"]["n"]):
        raise ValueError("Predictive reference coverage does not match reference measurements")
    conf = evidence["confusion"]
    count_keys = {"n_validated", "n_removed", "n_dev_winners", "n_dev_losers", "n_dev_neutral",
                  "winners_removed", "losers_removed", "neutral_removed"}
    rate_keys = {"winner_rejection_rate", "loser_removal_rate"}
    if not isinstance(conf, dict) or set(conf) != count_keys | rate_keys or any(not count(conf[k]) for k in count_keys):
        raise ValueError("Malformed predictive reference confusion")
    if (conf["n_validated"] > row["n_measured"] or conf["n_removed"] > conf["n_validated"]
            or conf["n_dev_winners"] + conf["n_dev_losers"] + conf["n_dev_neutral"] > conf["n_validated"]
            or conf["winners_removed"] + conf["losers_removed"] + conf["neutral_removed"] > conf["n_removed"]
            or conf["neutral_removed"] > conf["n_dev_neutral"]):
        raise ValueError("Predictive reference confusion counts are inconsistent")
    for rate, removed, denominator in (("winner_rejection_rate", "winners_removed", "n_dev_winners"),
                                       ("loser_removal_rate", "losers_removed", "n_dev_losers")):
        expected = conf[removed] / conf[denominator] if conf[denominator] else None
        if conf[removed] > conf[denominator] or conf[rate] != expected:
            raise ValueError("Predictive reference rejection rates lack the stated class support")
    expected_fraction = conf["n_validated"] / n_reference if n_reference else 0.0
    if not finite(evidence["validation_fraction"]) or evidence["validation_fraction"] != expected_fraction:
        raise ValueError("Predictive reference validation coverage mismatch")
    portfolio = evidence["portfolio"]
    if not isinstance(portfolio, dict) or set(portfolio) != {
            "cvar_improvement", "return_change", "return_retained", "return_change_ci90",
            "return_change_ci_positive"}:
        raise ValueError("Malformed predictive reference portfolio summary")
    if any(portfolio[k] is not None and not finite(portfolio[k]) for k in
           ("cvar_improvement", "return_change", "return_retained")):
        raise ValueError("Predictive reference portfolio values must be finite or null")
    interval = portfolio["return_change_ci90"]
    if interval is not None and (not isinstance(interval, list) or len(interval) != 2
                                or not all(finite(x) for x in interval) or interval[0] > interval[1]):
        raise ValueError("Malformed predictive reference return interval")
    if evidence["uncertainty"] != {
            "return_interval_level": .90, "return_interval_method": "paired_day_bootstrap_of_daily_return_change",
            "cvar_interval_available": False, "winner_rejection_rate_estimable": bool(conf["n_dev_winners"])}:
        raise ValueError("Predictive reference uncertainty must distinguish return and CVaR evidence")
    if evidence["reference_status"] in {"supported_on_dev_validation", "contrary_evidence", "no_relationship_detected"}:
        if evidence["n_folds_evaluated"] < 2 or _status(conf, portfolio, min_removed=1) != evidence["reference_status"]:
            raise ValueError("Predictive reference grade is inconsistent with validation metrics")
    limits = evidence["limitations"]
    if (not isinstance(limits, list) or len(limits) != len(set(limits))
            or any(limit not in REFERENCE_LIMITATIONS for limit in limits)
            or not set(REFERENCE_LIMITATIONS[:3]).issubset(limits)
            or (not evidence["n_positive_net_reference"] and REFERENCE_LIMITATIONS[3] not in limits)
            or (evidence["grade"] != "supported" and REFERENCE_LIMITATIONS[5] not in limits)):
        raise ValueError("Predictive reference must disclose uncertainty and absent class support")


def format_exploratory_reference_context(context: Optional[Dict[str, Any]]) -> str:
    """Canonical model-facing reference text; no caller prose or row-level values are interpolated."""
    safe = validate_exploratory_reference_context(context)
    if safe is None:
        return ""
    return ("\n\nEXPLORATORY DEVELOPMENT REFERENCE (aggregate reference-only context; evaluated candidate "
            "outcomes excluded by the caller; never qualified_support and never a candidate numeric citation):\n"
            + json.dumps(safe, sort_keys=True, allow_nan=False))


@dataclass
class GeneralizationVerdict:
    champion_id: str
    decision: str
    confidence: float                # Fraction of judges supporting the final decision; not a probability or vote weight.
    predicted_label: str
    stability_assessment: str = "INSUFFICIENT_EVIDENCE"
    uncertainty_reason: Optional[str] = None
    decisive_exhibit_ids: List[str] = field(default_factory=list)
    decisive_metric_values: Dict[str, Any] = field(default_factory=dict)
    disagreement: Dict[str, Any] = field(default_factory=dict)
    prosecution: Optional[Dict[str, Any]] = None
    defense: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None    # Retained for reading historical records.
    panel: List[Dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    tokens: Optional[Dict[str, int]] = None
    agent_transcripts: List[Dict[str, Any]] = field(default_factory=list)
    tool_records: List[Dict[str, Any]] = field(default_factory=list)
    evidence_rejections: List[Dict[str, Any]] = field(default_factory=list)
    allocation_action: str = "FULL"
    allocation_percent: Optional[float] = None   # Model-recommended percentage; report allocations are computed separately from risk scores.
    allocation_policy: str = PERCENTAGE
    allocation_multiplier: float = 1.0
    allocation_fallback_reason: Optional[str] = None   # Retain 100% when fewer than two usable percentage votes remain.
    allocation_votes: Dict[str, int] = field(default_factory=dict)
    allocation_status_counts: Dict[str, int] = field(default_factory=dict)


_STABILITY = frozenset({"ROBUST", "FRAGILE", "INSUFFICIENT_EVIDENCE"})
# Resolve stability ties in the order insufficient evidence, fragile, robust.
_STAB_ORDER = {"INSUFFICIENT_EVIDENCE": 0, "FRAGILE": 1, "ROBUST": 2}
_UNSURE_REASONS = frozenset({"MISSING_EVIDENCE", "CONFLICTING_EVIDENCE", "UNVALIDATED_DIAGNOSTIC"})

ALLOCATION_POLICY_VERSION = PERCENTAGE


def _coerce_uncertainty_reason(v: dict) -> None:
    """Separate a valid uncertainty code from an appended explanation.

    Retain the explanation in verdict_rationale.
    """
    r = v.get("uncertainty_reason")
    if not isinstance(r, str) or r in _UNSURE_REASONS:
        return
    rs = r.strip()
    for enum in _UNSURE_REASONS:
        if rs.startswith(enum) and (len(rs) == len(enum) or not rs[len(enum)].isalnum() and rs[len(enum)] != "_"):
            tail = rs[len(enum):].lstrip(" :;-–—.\n\t").strip()
            v["uncertainty_reason"] = enum
            if tail:
                rat = (v.get("verdict_rationale") or "").strip()
                v["verdict_rationale"] = f"{rat} {tail}".strip() if rat else tail
            return


def _salvage_cites(jdg: dict, cf_dict: Dict[str, Any]) -> None:
    """Recover missing structured citations from existing case fields.

    Resolve listed IDs directly, or match numbers in the rationale to case
    values within the citation tolerance. Subsequent validation still applies.
    """
    if not isinstance(jdg, dict) or jdg.get("decision") not in ("WILL_GENERALIZE", "WONT_GENERALIZE"):
        return
    ids = [i for i in (jdg.get("decisive_exhibit_ids") or []) if isinstance(i, str)]
    vals = dict(jdg.get("decisive_metric_values") or {})
    if ids and not vals:                                     # Recover values for explicitly listed case-field IDs.
        for i in ids:
            cv = cf_dict.get(i)
            if isinstance(cv, (int, float)) and not isinstance(cv, bool):
                vals[i] = cv
        if vals:
            jdg["decisive_metric_values"] = vals
            jdg["decisive_exhibit_ids"] = [i for i in ids if i in vals] or ids
        return
    if not ids and not vals:                                 # Recover prose citations only when both the ID and value match the case.
        import re
        rat = jdg.get("verdict_rationale") or ""
        fids, fvals = [], {}
        for key, cval in cf_dict.items():
            if not isinstance(cval, (int, float)) or isinstance(cval, bool):
                continue
            m = re.search(re.escape(key) + r"[^0-9\-]{0,12}(-?\d+\.?\d*)", rat)
            if not m:
                continue
            try:
                num = float(m.group(1))
            except ValueError:
                continue
            if abs(num - float(cval)) <= max(1e-3, abs(float(cval)) * 0.05):
                fids.append(key)
                fvals[key] = cval
        if fids:
            jdg["decisive_exhibit_ids"] = fids
            jdg["decisive_metric_values"] = fvals


def _require_allocation_action(v: dict, alloc_state: Optional[dict], allocation_policy=PERCENTAGE) -> None:
    """Allow one correction of an invalid percentage without discarding the prediction."""
    check_policy(allocation_policy)
    issue = percentage_vote_issue(v)
    v.pop("_allocation_reasoning_missing", None)
    if issue is None:
        return
    if alloc_state and alloc_state.get("retries_left", 0) > 0:
        alloc_state["retries_left"] -= 1
        err = MalformedVerdict(f"{issue[0]}: {issue[1]} Resubmit using the same evidence; preserve the prediction.")
        err._validation_details = {"path": issue[0], "message": issue[1], "allocation_retry": True}
        raise err
    if issue[0] == "allocation_rationale":
        v["_allocation_reasoning_missing"] = True
    return

    # Preserve the prediction; the missing allocation is recorded as a technical failure.


def _normalize_allocation_vote(v: dict, allocation_policy=PERCENTAGE) -> None:
    """Distinguish a valid percentage, deliberate abstention and technical failure."""
    check_policy(allocation_policy)
    normalize_percentage_vote(v)


def _allocation_citation_error(jdg: dict, cf_dict: Dict[str, Any]) -> Optional[str]:
    if (str(jdg.get("allocation_action", "")).upper() not in ("FULL", "CASH", "PARTIAL")
            and not valid_percent(jdg.get("allocation_percent"))):
        return None
    ids, vals = jdg.get("allocation_exhibit_ids") or [], jdg.get("allocation_metric_values") or {}
    if not ids and not vals:
        return None
    if (not isinstance(ids, list) or any(not isinstance(k, str) for k in ids)
            or not isinstance(vals, dict) or set(ids) != set(vals) or len(ids) != len(set(ids))):
        return "allocation_metric_values keys must exactly match allocation_exhibit_ids (full result_id::statistic aliases)"
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vals.values()):
        return "allocation_metric_values must contain one finite number per full alias, never arrays or summaries"
    for key, value in vals.items():
        observed = cf_dict.get(key)
        if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(observed):
            return f"{key}: cite an available numeric measurement, not a flag or missing value"
        if (value > 0) - (value < 0) != (observed > 0) - (observed < 0):
            return f"{key}: preserve the measured sign; do not round across zero"
    try:
        _verify_cites({"decisive_exhibit_ids": ids, "decisive_metric_values": vals}, cf_dict, {})
    except MalformedVerdict as exc:
        return str(exc)
    return None


def _require_allocation_citations(v: dict, case: dict, state: Optional[dict]) -> None:
    """Allow correction of malformed allocation citations within the current turn budget."""
    problem = _allocation_citation_error(v, case)
    if problem and state and state.get("citation_retries_left", 0) > 0:
        state["citation_retries_left"] -= 1
        err = MalformedVerdict("INVALID_ALLOCATION_CITATION: " + problem +
            ". Resubmit the COMPLETE object, copying exact full IDs and numeric values from the evidence cards. "
            "Correct the citation format using the same evidence; this does not require changing your action.")
        err._validation_details = {"path": "allocation_metric_values", "allocation_citation_retry": True}
        raise err


def _verify_allocation_cites(jdg: dict, cf_dict: Dict[str, Any]) -> None:
    """Validate any numerical citations supplied for the allocation recommendation.

    An invalid citation removes the allocation vote while preserving the
    separately validated prediction.
    """
    problem = _allocation_citation_error(jdg, cf_dict)
    if problem:
        jdg["allocation_action"] = "ABSTAIN"
        jdg["allocation_status"] = "ungrounded_downgrade"
        jdg["allocation_downgraded_reason"] = "unverified_allocation_cite"
        jdg["allocation_citation_error"] = problem


def _validate_gen_verdict(v: Any, allocation_policy=PERCENTAGE) -> None:
    if not isinstance(v, dict):
        raise MalformedVerdict(f"verdict is not a dict: {type(v)}")
    if v.get("decision") not in _DECISIONS:
        raise MalformedVerdict(f"decision must be one of {_DECISIONS}, got {v.get('decision')!r}")
    # Stability, confidence and allocation are recorded separately from the prediction.
    if v.get("decision") in ("WILL_GENERALIZE", "WONT_GENERALIZE"):
        if not v.get("decisive_exhibit_ids") or not v.get("decisive_metric_values"):
            raise MalformedVerdict("a WILL/WONT verdict needs decisive_exhibit_ids + decisive_metric_values "
                                   "(numbers-from-exhibits guardrail)")
    else:
        _coerce_uncertainty_reason(v)
        reason = v.get("uncertainty_reason")
        if reason not in _UNSURE_REASONS:
            raise MalformedVerdict(f"UNSURE needs an uncertainty_reason in {sorted(_UNSURE_REASONS)}, got {reason!r}")
        # Abstention requires a rationale; any supplied citations are still checked.
        if not (v.get("verdict_rationale") or "").strip():
            raise MalformedVerdict(f"UNSURE({reason}) must state its reasoning in verdict_rationale")
    _normalize_allocation_vote(v, allocation_policy)


_PRED = {"WILL_GENERALIZE": "win", "WONT_GENERALIZE": "loss", "UNSURE": "abstain"}


def _safe_float(x, default: float = 0.0) -> float:
    """Parse a finite number from a model field, returning None when invalid."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if v == v else default


def _conf01(x) -> Optional[float]:
    """Return a finite confidence value in [0, 1], or None.

    Confidence is recorded but does not weight judge votes.
    """
    v = _safe_float(x, default=float("nan"))
    return v if (v == v and 0.0 <= v <= 1.0) else None


def _reject_flag_cites(verdict: dict, cf_dict: dict) -> None:
    """Reject Boolean or availability flags used as decisive numerical evidence."""
    for k in (verdict.get("decisive_metric_values") or {}):
        if isinstance(cf_dict.get(k), bool) or str(k).endswith("_available"):
            raise MalformedVerdict(f"decisive cite '{k}' is an availability/boolean flag, not a cause exhibit")


def _reject_confounded_ablation_cite(verdict: dict, cf_dict: dict) -> None:
    """Reject a verdict supported only by unreliable ablation measurements."""
    if not cf_dict.get("ablation_confounded"):
        return
    cites = list(verdict.get("decisive_metric_values") or {})
    if cites and all(str(k).startswith("ablation_phi") for k in cites):
        raise MalformedVerdict("verdict rests solely on ablation_phi while ablation_confounded=true "
                               "(off-manifold, unreliable) — cite a non-ablation cause exhibit")


def _validate_research_prediction(verdict: dict, case: dict, citation_info: dict) -> None:
    """Check the scope, limitations and citations of a directional prediction."""
    if verdict.get("decision") not in ("WILL_GENERALIZE", "WONT_GENERALIZE"):
        if (verdict.get("decision") == "UNSURE" and verdict.get("prediction_scope") is not None
                and verdict["prediction_scope"] != "ABSTENTION"):
            raise MalformedVerdict("UNSURE requires prediction_scope=ABSTENTION, not a directional prediction scope")
        return
    if verdict.get("prediction_scope") != "UNCERTAIN_OUT_OF_SAMPLE_PREDICTION":
        raise MalformedVerdict("A research prediction must explicitly declare its uncertain out-of-sample scope")
    limits = verdict.get("prediction_limitations")
    if not isinstance(limits, list) or not limits or not all(isinstance(x, str) and x.strip() for x in limits):
        raise MalformedVerdict("A research prediction requires explicit prediction_limitations")
    ids, values = verdict.get("decisive_exhibit_ids"), verdict.get("decisive_metric_values")
    if (not isinstance(ids, list) or not ids or not all(isinstance(k, str) for k in ids)
            or len(set(ids)) != len(ids) or not isinstance(values, dict) or set(ids) != set(values)):
        raise MalformedVerdict("Research prediction citations must name the same unique measurements and values")
    for key in ids:
        observed, cited = case.get(key), values.get(key)
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
               for v in (observed, cited)):
            raise MalformedVerdict("Research predictions require finite numeric measurement citations")
        if "::" in key:
            if not citation_info.get(key, {}).get("research_admissible"):
                raise MalformedVerdict("Research prediction cites an unusable or incorrectly scoped executed measurement")
        elif (case.get("fidelity_ok") is False or (case.get("ablation_confounded") and
                any(token in key for token in ("ablation_phi", "worst_ablation")))):
            raise MalformedVerdict("Research prediction cites an unreliable static measurement")
    if not str(verdict.get("verdict_rationale") or "").strip():
        raise MalformedVerdict("Research prediction must explain its evidence-to-prediction inference")
    _reject_positive_margin_only_cites(ids)
    _validate_training_economic_citations(ids, values, case, citation_info)


def _bad_direction(key: str, citation_info: Optional[dict] = None) -> Optional[str]:
    """Return the registered adverse direction for a measurement, if defined.

    Tool aliases use executor metadata; static profile fields use their recorded
    name conventions. Context-only measurements have no fixed direction.
    """
    k = str(key)
    if "::" in k:
        meta = (citation_info or {}).get(k, {})
        return meta.get("direction") if meta.get("citeable") else None
    if k.endswith("knife_edge_frac") or k in ("dead_comparison_frac", "dow_tv_distance", "intraday_margin_range"):
        return "high"
    if k.endswith("_abs_median") or k in ("fire_margin_abs_median", "ablation_phi_min", "ablation_phi_mean",
                                          "nudge_phi_min", "nudge_phi_mean", "intraday_sign_consistency"):
        return "low"
    return None


def validate_generalization(cf: GeneralizationCaseFile, *, run_agent: Callable = None,
                            model: Optional[str] = None, pros_model: Optional[str] = None,
                            def_model: Optional[str] = None,
                            judge_model: Optional[str] = None,
                            judge_models: Optional[List[str]] = None,
                            n_judges: int = 3, rounds: int = 1,
                            cohort_stats: Optional[Dict[str, Any]] = None, enable_compute: bool = False,
                            max_turns: int = 8,
                            run_catalogue_test: Optional[Callable] = None,
                            evidence: Optional[Dict[str, Any]] = None,
                            catalogue_allow: Optional[Sequence[str]] = None,
                            evidence_mode: str = "training_economics",
                            decision_policy: str = "research_prediction",
                            exploratory_ranking: bool = True,
                            allocation_policy: str = PERCENTAGE,
                            exploratory_reference_context: Optional[Dict[str, Any]] = None) -> GeneralizationVerdict:
    """Run the prosecution, defense and judge panel for one strategy.

    Provide the shared case and diagnostic access, validate each response, and
    aggregate the judges' risk scores, percentage recommendations and decisions.
    A directional decision needs two agreeing judges; confidence does not weight votes.
    """
    if rounds != 1:
        raise ValueError("H3 uses one prosecution/defense round")
    if run_agent is None:
        from potofgold.multi_agent.assessment.agents import task_sync as run_agent
    if decision_policy not in DECISION_POLICIES:
        raise ValueError("Unknown court decision policy")
    if not exploratory_ranking:
        raise ValueError("H3 assessment requires judge risk scores")
    check_policy(allocation_policy)
    _exploratory_context_text = (format_exploratory_reference_context(exploratory_reference_context)
                                 if exploratory_reference_context is not None else "")
    pros_model = pros_model or model
    def_model = def_model or model
    # An explicit model list assigns one model or fallback chain to each judge seat.
    judge_models = judge_models or [judge_model or model] * max(1, n_judges)
    _packet = build_measurement_context(cf, run_catalogue_test, evidence=evidence,
        catalogue_allow=catalogue_allow, evidence_mode=evidence_mode)
    cf_dict = _packet["case_file"]
    _preloaded_records = _packet["records"]
    _preloaded_inventory = _packet["preloaded_evidence"]
    _citation_info: Dict[str, Any] = dict(_packet["citation_info"])
    _reasoning_case = dict(cf_dict)
    _verdict_policy = _verdict_evidence_policy()
    _verdict_tool = deepcopy(_SUBMIT_VERDICT)
    from potofgold.multi_agent.assessment.allocation_reasoning import investigation_schema
    _verdict_tool["input_schema"]["properties"]["allocation_investigations"] = investigation_schema(allocation_policy)
    _verdict_tool["input_schema"]["properties"].update({
        "prediction_scope": {"type": "string", "enum": ["UNCERTAIN_OUT_OF_SAMPLE_PREDICTION", "ABSTENTION"]},
        "prediction_limitations": {"type": "array", "items": {"type": "string"}, "minItems": 1}})
    _verdict_tool["input_schema"]["required"] += ["prediction_scope", "prediction_limitations"]
    _risk_fields = (exploratory_risk_fields(exploratory_reference_context, case_file=cf_dict,
                                          citation_info=_citation_info))
    _verdict_tool["input_schema"]["properties"].update(deepcopy(_risk_fields))
    _verdict_tool["input_schema"]["required"] += list(_risk_fields)

    cf_json = json.dumps(_packet["model_case_file"], default=str)
    alias = _opaque_alias(cf.champion_id)          # Raw strategy IDs encode condition and fold, so prompts use an opaque alias.

    # Include peer statistics as context; decisive citations must resolve to case measurements.
    from potofgold.multi_agent.assessment.reasoning_diagnostics import peer_context, DiagnosticSession
    _peer = peer_context(cf_dict, cohort_stats) if enable_compute else {}
    _diagnostic_access = DiagnosticSession(cf, run_catalogue_test, _preloaded_records,
        evidence=evidence, allowed=catalogue_allow, max_requests=_MAX_CATALOGUE_REQUESTS)
    _evidence = (_EVIDENCE_HEADER + "\n" + cf_json
                 + (("\n\nCOHORT PERCENTILES (descriptive standing in the frozen reference population; "
                     "adverse direction depends on the metric, not low percentile alone):\n"
                     + json.dumps(_peer, default=str)) if _peer else ""))
    from potofgold.multi_agent.strategy_profiles.evidence_cards import INTERPRETATION, reference_bundle
    _evidence += "\nSHARED EVIDENCE CARD GUIDE: " + INTERPRETATION
    _evidence += ("\n\nTRAINING ECONOMIC COMPARISONS (candidate baseline arithmetic; "
                  "not calibrated severity or predictive evidence):\n"
                  + json.dumps(_packet["training_economic_comparisons"], allow_nan=False))
    _evidence += ("\n\nSCENARIO COVERAGE (separate measured interventions; context, not an action rule):\n"
                  + json.dumps(_packet["economic_scenario_context"], allow_nan=False))
    _evidence += (f"\n\nEVIDENCE MODE: {evidence_mode}; DECISION POLICY: "
                  f"{decision_policy}; POLICY VERSION: {COURT_POLICY_VERSION}.\n")
    _evidence += "\n\n" + EXPLORATORY_RISK_NOTE + _exploratory_context_text
    if _preloaded_records:
        _evidence += ("\n\nPRELOADED VERIFIED MEASUREMENTS (already executed for this exact champion; "
                      "shared by every role; no new replay):\n"
                      + json.dumps(_packet["visible_records"], default=str)
                      + "\nCite reliable values by their exact <result_id>::<statistic> aliases in the case file. "
                        "Validity, direction and development support are recomputed server-side. "
                        "Requesting the same measurement again does not create new evidence. "
                      + ("\nAVAILABLE MEASUREMENT INVENTORY:\n")
                      + json.dumps(_model_inventory(_preloaded_inventory), default=str))
    reference_context = []
    if evidence is not None:
        reference_context = development_reference_cards(cf, evidence, catalogue_allow=catalogue_allow,
                                                        run_catalogue_test=run_catalogue_test)
        _evidence += (
            "\n\nDEVELOPMENT REFERENCE EVIDENCE (applicable diagnostic definitions; shared by every role):\n"
            + json.dumps(reference_bundle(reference_context), default=str)
            + "\nThese are aggregate results from development strategies, not measurements or outcomes of this "
              "champion. When dimensions_present is true, assess discrimination, return_effect, tail_effect and "
              "winner_preservation separately using their confidence intervals, coverage, denominators, comparators "
              "and limitations. Mixed findings must remain distinct: improved portfolio tails do not establish "
              "loser discrimination or preservation of winners. An unresolved dimension does not establish safety "
              "or an inverse relationship. dev_status summarizes discrimination, not the other dimensions; "
              "Unresolved means uncertain, not an established absence of relationship. Historical composite "
              "labels remain in the store audit. Use the explicit winner and loser denominators and the stated "
              "handling of folds without a fitted threshold. Reference return and CVaR effects use the stated "
              "reference accounting; they do not estimate the court's Full/Half/No-position policy, which leaves "
              "released capital in cash. Only when dimensions are absent use the legacy summary with its stated limitations. Supported "
              "evidence does not establish that this champion is adverse: a reliable executed measurement is "
              "still required. Interpret evidence strength under the active verdict policy. "
              "Do not cite these aggregates as champion exhibits or transfer support to a related static metric.")
    _evidence += "\nDIAGNOSTIC CAPABILITIES AND GAPS: " + json.dumps(_diagnostic_access.capability_context())
    _evidence = _evidence.replace("Full/Half/No-position policy", "percentage-allocation policy")
    _cap = _capability_note(bool(_peer), reference_cards=reference_context)
    _cache_ctx = f"Champion {alias}.\n{_evidence}"      # Share the same case text across roles and subsequent turns.

    def _handlers(final_name, final_fn):
        return {final_name: final_fn}

    _tok = {"input": 0, "output": 0, "calls": 0,
            "cache_read": 0, "cache_write": 0}
    _agent_log: List[Dict[str, Any]] = []
    _cb = {"provider_fallbacks": 0}                    # Count provider-driven fallbacks separately from validation retries.

    def _record_attempt(role, model, meta, error=None):
        """Count each returned or failed agent attempt once, including known partial usage."""
        meta = meta or {}
        _in = int(meta.get("input_tokens", 0) or 0)
        _out = int(meta.get("output_tokens", 0) or 0)
        _cr = int(meta.get("cache_read_input_tokens", meta.get("cached_input_tokens", 0)) or 0)
        _cw = int(meta.get("cache_creation_input_tokens", 0) or 0)
        _tok["input"] += _in
        _tok["output"] += _out
        _tok["calls"] += int(error is None or bool(meta))
        _tok["cache_read"] += _cr
        _tok["cache_write"] += _cw
        rec = {"role": role, "model": meta.get("model", model), "input_tokens": _in, "output_tokens": _out,
               "cache_read_input_tokens": _cr, "cache_creation_input_tokens": _cw,
               "transcript": meta.get("transcript", [])}
        if error is not None:
            rec.update(error=f"{type(error).__name__}: {error}", usage_incomplete=True)
        _agent_log.append(rec)
        return rec

    def _call(role, *, model_chain=None, validate=None, alloc_state=None, **kw):
        """Try a role's configured model chain until one response passes validation.

        Record usage from every attempt and distinguish provider failures from
        invalid submissions. Propagate unexpected validation errors.
        """
        # Return factual contradictions and citation errors as correction feedback.
        handlers = dict(kw.get("tool_handlers") or {})
        for name, original in list(handlers.items()):
            if name not in {"submit_prosecution", "submit_defense", "submit_verdict"}:
                continue
            prior = getattr(original, "validate_submission", None)
            def guarded(args, _original=original):
                return _original(args)
            def validate_submission(args, _prior=prior, _name=name):
                validate_scenario_claims(args, economic_scenario_context(_reasoning_case, _citation_info))
                if _name == "submit_verdict" and alloc_state is not None:
                    # Allow one correction of a missing or invalid allocation percentage.
                    _require_allocation_action(args, alloc_state, allocation_policy)
                    _require_allocation_citations(args, _reasoning_case, alloc_state)
                return _prior(args) if _prior is not None else args
            guarded.validate_submission = validate_submission
            handlers[name] = guarded
        kw["tool_handlers"] = handlers
        chain = [m.strip() for m in str(model_chain or "").split(">") if m.strip()] or [None]
        prov_err = val_err = None                            # Preserve provider failures for the run-level stop conditions.
        primary_provider_error = False
        for mi, m in enumerate(chain):
            nxt = (lambda: f" — falling back to {chain[mi + 1]}" if mi + 1 < len(chain) else " — chain exhausted")
            try:
                out, meta = run_agent(role=role, **({"model": m} if m else {}), **kw)
            except Exception as e:                          # noqa: BLE001
                prov_err = e
                if mi == 0:
                    primary_provider_error = True
                _log.warning("run_agent[%s] model=%s failed (%s: %s)%s", role, m, type(e).__name__, e, nxt())
                _record_attempt(role, m, getattr(e, "_agent_metadata", None), error=e)
                continue
            rec = _record_attempt(role, m, meta)
            # Keep the original structured response before citation recovery or normalization.
            rec["submitted_output"] = deepcopy(out)
            rec["metadata"] = deepcopy(meta)
            # Custom agent callables must pass the same response checks.
            try:
                validate_scenario_claims(out, economic_scenario_context(_reasoning_case, _citation_info))
            except ValueError as exc:
                error = MalformedVerdict(str(exc))
                rec["rejected"] = str(error)
                error._court_transcripts = deepcopy(_agent_log)
                error._court_tokens = dict(_tok)
                raise error from exc
            _ran = rec["model"]
            if validate is not None:
                try:
                    validate(out)
                except MalformedVerdict as ve:              # Only MalformedVerdict advances to a backup; unexpected errors propagate.
                    val_err = ve
                    rec["rejected"] = f"{type(ve).__name__}: {ve}"
                    rec["validation_error"] = {
                        "error_type": type(ve).__name__, "message": str(ve),
                        **deepcopy(getattr(ve, "_validation_details", {}))}
                    _log.warning("run_agent[%s] model=%s verdict REJECTED (%s: %s)%s",
                                 role, m, type(ve).__name__, ve, nxt())
                    continue
                if isinstance(out, dict):
                    out.setdefault("_ran_model", _ran)
            if primary_provider_error:
                _cb["provider_fallbacks"] += 1
            return out, None
        return None, (prov_err or val_err)

    pros = {"alleged_class": "UNAVAILABLE", "brief_status": "TECHNICAL_FAILURE", "confidence": 0.0,
            "rationale": "Prosecutor produced no brief; no evidential conclusion follows."}
    dfn = {"verdict_on_claim": "UNRESOLVED", "brief_status": "TECHNICAL_FAILURE", "allegation_assessments": [],
           "rebuttal_rationale": "No defense brief; absence of a response is not a concession."}
    pros_msg = ("Using the CASE FILE above, submit_prosecution. "
                "Name one principal failure concern, explaining relevant interacting observations and the "
                "strongest competing explanation; cite the exhibit id + number.")
    r_pros, _ = _call("prosecutor", user_message=pros_msg, cache_prefix=_cache_ctx,
        system_prompt=PROSECUTOR_SYSTEM + _cap + _verdict_policy, tools=[_SUBMIT_PROSECUTION],
        tool_handlers=_handlers("submit_prosecution", lambda inp: inp), max_turns=min(max_turns, 3), model_chain=pros_model)
    pros = r_pros or pros
    _defense_submissions = []

    def _validate_defense(inp):
        audit = {"submitted_defense": deepcopy(inp)}
        _defense_submissions.append(audit)
        try:
            validated = _validate_defense_submission(inp, pros, case=_reasoning_case,
                                                     require_retention=True)
        except ValueError as exc:
            audit.update(status=("invalid_economic_magnitude" if
                str(exc).startswith("UNBENCHMARKED_ECONOMIC_MAGNITUDE") else "invalid_claim_binding"),
                validation_error=str(exc))
            raise
        audit["status"] = "accepted"
        return validated

    def _defense_handler(inp):
        return inp

    _defense_handler.validate_submission = _validate_defense
    r_dfn, _ = _call("defender", cache_prefix=_cache_ctx,
        user_message=f"PROSECUTION:\n{json.dumps(pros, default=str)}\n"
                     f"Assess at most three material statements actually made above. Quote complete source sentences in prosecutor_quote, "
                     f"preserving whether each concern was asserted, considered or withdrawn. Agree when warranted; "
                     f"an empty assessment list is valid. Submit the complete defense object with verdict_on_claim, "
                     f"allegation_assessments, rebuttal_rationale and your independent retention_case, including after any correction request.",
        system_prompt=DEFENDER_SYSTEM + DEFENSE_CLAIM_GUIDE + _cap + _verdict_policy, tools=[_SUBMIT_DEFENSE],
        tool_handlers=_handlers("submit_defense", _defense_handler), max_turns=min(max_turns, 3), model_chain=def_model)
    if isinstance(r_dfn, dict):
        dfn = _sanitize_defense(r_dfn, cf_dict, _citation_info, prosecution=pros)
    for attempt in reversed(_agent_log):
        if attempt.get("role") == "defender":
            attempt["submitted_defense"] = deepcopy(r_dfn)
            attempt["scope_corrections"] = deepcopy(dfn.get("scope_corrections", []))
            if _defense_submissions:
                attempt["defense_submissions"] = deepcopy(_defense_submissions)
            break

    def _ordered_briefs(prosecution, defense, *, prefix="", separator="\n\n"):
        # Avoid repeating the prosecution text inside the defense shown to judges.
        defense_view = {k: deepcopy(v) for k, v in defense.items() if k not in ("prosecution_rationale", "source_position")}
        for assessment in defense_view.get("allegation_assessments", []):
            if assessment.get("allegation") == assessment.get("prosecutor_quote"):
                assessment.pop("allegation", None)
            assessment.pop("claim_binding", None)
        briefs = [("PROSECUTION", prosecution), ("DEFENSE", defense_view)]
        return separator.join(f"{prefix}{name}:\n{json.dumps(brief, default=str)}" for name, brief in briefs)

    _base_judge_msg = _ordered_briefs(pros, dfn)
    _JUDGE_SUFFIX = ("\n\nUsing the CASE FILE above, submit_verdict with the primary allocation_percent and its "
                     "rationale, plus the separate stability assessment and prediction with its required cites.")

    def _validate_judge(jdg, cf_for_cites):
        # Allocation errors affect that vote, independently of prediction validation.
        _salvage_cites(jdg, cf_for_cites)
        _validate_gen_verdict(jdg, allocation_policy)
        _verify_allocation_cites(jdg, cf_for_cites)
        if jdg.get("decisive_metric_values"):
            _verify_cites(jdg, cf_for_cites, {})
            _reject_flag_cites(jdg, cf_for_cites)
            _reject_confounded_ablation_cite(jdg, cf_for_cites)
        _validate_research_prediction(jdg, cf_for_cites, _citation_info)
        validate_exploratory_risk(jdg, cf_for_cites, reference_context=exploratory_reference_context,
                                 citation_info=_citation_info)

    def _judge_panel(judge_msg, cf_for_cites=None):
        """Run each judge seat, including its configured fallback models.

        Validate responses against the shared case and retain the failure reasons
        for any seat that produces no usable assessment.
        """
        _cf = cf_dict if cf_for_cites is None else cf_for_cites
        _panel_tool = deepcopy(_verdict_tool)
        _panel_tool["input_schema"]["properties"].update(
            exploratory_risk_fields(exploratory_reference_context, case_file=_cf,
                                   citation_info=_citation_info))
        panel, provider_err, malformed_err, n_failed = [], None, None, 0
        for seat, _jm in enumerate(judge_models):
            # Each judge seat has its own one-correction allocation budget.
            _alloc_state = {"retries_left": 1 if max_turns >= 2 else 0,
                            "citation_retries_left": 1 if max_turns >= 2 else 0}
            message = judge_msg
            verdict_state = {"retries_left": 1 if max_turns >= 2 else 0}
            def submit_judge(inp):
                return inp
            def check_submission(inp):
                try:
                    _validate_judge(deepcopy(inp), _cf)
                except MalformedVerdict as exc:
                    if verdict_state["retries_left"]:
                        verdict_state["retries_left"] -= 1
                        raise MalformedVerdict("CORRECT_VERDICT_SUBMISSION: " + str(exc) +
                            ". Resubmit the complete object using the same evidence and exact supplied aliases/values. "
                            "A citation correction does not require changing your allocation.") from exc
                return inp
            submit_judge.validate_submission = check_submission
            def _validate(jdg):
                _validate_judge(jdg, _cf)
            jdg, _jerr = _call("judge", user_message=message, cache_prefix=_cache_ctx,
                system_prompt=JUDGE_SYSTEM + _cap + _verdict_policy,
                tools=[_panel_tool], validate=_validate, alloc_state=_alloc_state,
                tool_handlers=_handlers("submit_verdict", submit_judge), max_turns=max_turns, model_chain=_jm)
            if _jerr is not None:
                n_failed += 1
                if isinstance(_jerr, MalformedVerdict):
                    malformed_err = _jerr
                else:
                    provider_err = _jerr
                _log.warning("judge %s seat=%s: whole backup chain failed (%s: %s) -> seat dropped",
                             alias, _jm, type(_jerr).__name__, _jerr)
                continue
            jdg["_judge_model"] = jdg.get("_ran_model", _jm)
            jdg["_judge_seat"] = seat
            panel.append(jdg)
        if n_failed and panel:
            _log.warning("champion %s: %d/%d judge seat(s) failed their whole backup chain; panel formed from %d — "
                         "cross-provider diversity degraded", alias, n_failed, len(judge_models), len(panel))
        return panel, provider_err, malformed_err

    def _aggregate(panel, provider_err, malformed_err, *, analysis, n_passes):
        """Aggregate judge assessments, retaining failures and usage in the audit.

        A directional decision requires two agreeing votes even when a seat fails.
        Risk scores and valid percentage recommendations are aggregated separately.
        """
        if not panel:
            err = provider_err or malformed_err or MalformedVerdict("no judge produced a valid verdict")
            err._court_tokens = dict(_tok)     # Preserve usage and transcripts even when no judge response survives.
            err._court_transcripts = list(_agent_log)
            err._court_provider_fallbacks = _cb["provider_fallbacks"]
            raise err
        n_will = sum(1 for j in panel if j["decision"] == "WILL_GENERALIZE")
        n_wont = sum(1 for j in panel if j["decision"] == "WONT_GENERALIZE")
        n_uns = sum(1 for j in panel if j["decision"] == "UNSURE")
        degraded = len(panel) < len(judge_models)
        # The two-vote requirement does not shrink when a judge fails.
        if n_will >= 2 and n_wont >= 2:
            decision = "UNSURE"
        elif n_will >= 2:
            decision = "WILL_GENERALIZE"
        elif n_wont >= 2:
            decision = "WONT_GENERALIZE"
        else:
            decision = "UNSURE"
        # Abstain when at least half the panel reports an unresolved opposing concern.
        _concern = sum(1 for j in panel if j.get("unresolved_concern") is True)
        _capped = decision in ("WILL_GENERALIZE", "WONT_GENERALIZE") and 2 * _concern >= len(panel)
        if _capped:
            decision = "UNSURE"
        allegation = str(pros.get("alleged_class") or "").strip().upper()
        _no_fault_blocked = False
        if decision == "UNSURE":
            # Distinguish disagreement, missing evidence and incomplete panels.
            if _capped or (n_will >= 1 and n_wont >= 1):
                unsure_reason = "CONFLICTING_EVIDENCE"
            else:
                _r = [j.get("uncertainty_reason") for j in panel
                      if j["decision"] == "UNSURE" and j.get("uncertainty_reason") in _UNSURE_REASONS]
                if _r:
                    unsure_reason = max(sorted(set(_r)), key=_r.count)
                elif degraded:
                    unsure_reason = "DEGRADED_PANEL"
                else:
                    unsure_reason = "DEGRADED_PANEL"
        else:
            unsure_reason = None
        winners = [j for j in panel if j["decision"] == decision]
        agreement = len(winners) / len(panel) if panel and not _no_fault_blocked else 0.0
        # Agreement describes the surviving votes.
        disagreement = {"WILL_GENERALIZE": n_will, "WONT_GENERALIZE": n_wont, "UNSURE": n_uns,
                        "panel_size": len(panel), "configured": len(judge_models), "degraded": degraded,
                        "unresolved_concern": _concern, "capped": _capped,
                        "no_fault_affirmation_blocked": _no_fault_blocked,

                        "distinct_models": len({j.get("_judge_model") for j in panel}),
                        "provider_fallbacks": _cb["provider_fallbacks"],
                        "passes": n_passes}
        disagreement.update(evidence_mode=evidence_mode, decision_policy=decision_policy,
                            policy_version=COURT_POLICY_VERSION)
        if callable(getattr(run_catalogue_test, "get_preloaded_measurements", None)):
            disagreement["preloaded_evidence"] = deepcopy(_preloaded_inventory)
        disagreement["exploratory_ranking"] = {
            "median_risk_score": float(median(j["risk_score"] for j in panel)),
            "n_judges": len(panel), "higher_means": "greater_comparative_downside_concern",
            "not_a_probability": True, "affects_verdict_admission": False,
        }
        # Aggregate the separate stability category with the recorded tie order.
        _stab = {s: sum(1 for j in panel if j.get("stability_assessment") == s) for s in _STABILITY}
        stability = min(_STABILITY, key=lambda s: (-_stab[s], _STAB_ORDER[s]))
        if winners and decision != "UNSURE":
            lead = winners[0]
            rationale = lead.get("verdict_rationale", "")
            cite_ids = list(lead.get("decisive_exhibit_ids") or [])
            cite_vals = dict(lead.get("decisive_metric_values") or {})
        else:
            rationale = (f"{n_uns}/{len(panel)} judges abstained from a directional forecast; "
                         f"UNSURE ({unsure_reason}). Allocation is voted separately.")
            cite_ids, cite_vals = [], {}
        aggregate = aggregate_percentages(panel)
        _final_action, _final_percent, _final_mult = aggregate["action"], aggregate["percent"], aggregate["multiplier"]
        _alloc_fallback = aggregate["fallback_reason"]
        _action_votes = {a: sum(j.get("allocation_status") == "explicit" and j.get("allocation_action") == a
                                for j in panel) for a in ("FULL", "PARTIAL", "CASH")}
        _percentage_details = {"allocation_percent": _final_percent, "percentage_votes": aggregate["percent_votes"],
            "aggregation": "median_of_at_least_two_valid_percentages", "spread_pp": aggregate["spread_pp"],
            "material_review_spread_pp": MATERIAL_SPREAD_PP}
        # Keep deliberate abstentions, invalid citations and technical failures distinct.
        _status_counts = {s: sum(1 for j in panel if j.get("allocation_status") == s)
                          for s in ("explicit", "deliberate_abstention", "ungrounded_downgrade", "technical_failure")}
        disagreement["allocation"] = {
            "votes": _action_votes, "status_counts": _status_counts,
            "final_action": _final_action, "multiplier": _final_mult,
            "fallback_reason": _alloc_fallback, "policy": allocation_policy, **_percentage_details}
        return GeneralizationVerdict(
            champion_id=cf.champion_id, decision=decision, confidence=agreement,
            allocation_action=_final_action, allocation_multiplier=_final_mult,
            allocation_percent=_final_percent, allocation_policy=allocation_policy,
            allocation_fallback_reason=_alloc_fallback, allocation_votes=_action_votes,
            allocation_status_counts=_status_counts,
            predicted_label=_PRED[decision], stability_assessment=stability, uncertainty_reason=unsure_reason,
            decisive_exhibit_ids=cite_ids, decisive_metric_values=cite_vals, disagreement=disagreement,
            prosecution=pros, defense=dfn, analysis=analysis,
            panel=[{"decision": j["decision"], "stability_assessment": j.get("stability_assessment"),
                    "confidence": _conf01(j.get("confidence")), "model": j.get("_judge_model"),
                    "judge_seat": j.get("_judge_seat"),
                    "reconsideration": deepcopy(j.get("reconsideration")),
                    # Retain each judge's reasoning and citations for inspection.
                    "verdict_rationale": j.get("verdict_rationale"),
                    "decisive_exhibit_ids": list(j.get("decisive_exhibit_ids") or []),
                    "decisive_metric_values": dict(j.get("decisive_metric_values") or {}),
                    "uncertainty_reason": j.get("uncertainty_reason"),
                    "allocation_action": j.get("allocation_action"),
                    **({"allocation_percent": j.get("allocation_percent"),
                        "allocation_multiplier": j.get("allocation_multiplier")}),
                    "allocation_status": j.get("allocation_status"),
                    "allocation_investigations": deepcopy(j.get("allocation_investigations") or []),
                    "allocation_rationale": j.get("allocation_rationale"),
                    "allocation_competing_explanation": j.get("allocation_competing_explanation"),
                    "allocation_limitations": j.get("allocation_limitations"),
                    "allocation_exhibit_ids": deepcopy(j.get("allocation_exhibit_ids"))
                        if isinstance(j.get("allocation_exhibit_ids"), list) else [],
                    "allocation_metric_values": deepcopy(j.get("allocation_metric_values"))
                        if isinstance(j.get("allocation_metric_values"), dict) else {},
                    "prediction_limitations": deepcopy(j.get("prediction_limitations"))
                        if isinstance(j.get("prediction_limitations"), list) else [],
                    **({key: deepcopy(j[key]) for key in (*_risk_fields, "risk_exhibit_ids", "risk_metric_values")})}
                   for j in panel],
            rationale=rationale, tokens=dict(_tok), agent_transcripts=list(_agent_log),
            tool_records=deepcopy(_preloaded_records))


    panel1, perr1, merr1 = _judge_panel(_base_judge_msg + _JUDGE_SUFFIX)
    v1 = _aggregate(panel1, perr1, merr1, analysis=None, n_passes=1)
    from potofgold.multi_agent.assessment.allocation_reasoning import allocation_investigations
    v1.disagreement["allocation_investigations"] = allocation_investigations(
        panel1, _diagnostic_access, cf_dict,
        [j.get(field, "") for j in panel1 for field in ("verdict_rationale", "allocation_rationale")],
        allocation_policy,
    )
    from potofgold.multi_agent.assessment.allocation_reasoning import finish_investigations
    finish_investigations(v1, _preloaded_records)
    return v1
