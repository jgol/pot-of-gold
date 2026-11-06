"""Check strategy inputs and citation validation before model calls.

Each strategy needs readable rules, permitted profile fields and at least one
usable measurement. Report missing inputs and validation failures.
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from potofgold.multi_agent.strategy_profiles import terminal_registry as _treg
from potofgold.multi_agent.strategy_profiles.generalization_case_file import GeneralizationCaseFile, assert_cause_only, _LLM_WITHHELD
from potofgold.multi_agent.assessment.court import _validate_gen_verdict, _reject_flag_cites, _scrub_leaky_exhibit, _measured_exhibits
from potofgold.multi_agent.assessment.court_citations import _verify_cites, MalformedVerdict


def _citable(llm_dict: Dict[str, Any]) -> List[str]:
    """Return finite numeric fields that can be cited, excluding Boolean flags."""
    return [k for k, v in llm_dict.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            and not str(k).endswith("_available")]


def _validated_measurement_view(cf: GeneralizationCaseFile, context: Dict[str, Any]) -> Dict[str, Any]:
    """Recheck measurement records before adding their citation aliases to the case."""
    from potofgold.multi_agent.strategy_profiles.tool_evidence import citation_metadata
    from potofgold.multi_agent.diagnostics.vulnerability_catalogue import REGISTRY
    from potofgold.multi_agent.strategy_profiles.economic_predictors import DERIVED_PREDICTORS, economic_admission_reason, economic_binding_reason
    if not isinstance(context, dict) or not isinstance(context.get("records"), list):
        raise ValueError("measurement context must contain verified saved records")
    records = []
    for record in context["records"]:
        if not isinstance(record, dict):
            raise ValueError("malformed saved measurement record")
        tid = record.get("test_id")
        provenance = record.get("preload_provenance")
        if tid in ("training_economics", "training_profit_recurrence"):
            from potofgold.multi_agent.strategy_profiles.training_economics import validate_training_economics_record
            from potofgold.multi_agent.diagnostics.training_recurrence import validate_recurrence_record
            if (context.get("evidence_mode") != "training_economics" or not isinstance(provenance, dict)
                    or record.get("preloaded") is not True):
                raise ValueError("Training economics requires an explicitly approved evidence mode and provenance")
            validator = validate_training_economics_record if tid == "training_economics" else validate_recurrence_record
            records.append(validator(record, champion_id=cf.champion_id,
                source_measurement_fingerprint=provenance.get("source_measurement_fingerprint")))
            continue
        if (not record.get("result_id") or record.get("replay_fidelity") != 1.0
                or isinstance(record.get("replay_fidelity"), bool)
                or record.get("preloaded") is not True or not isinstance(provenance, dict)
                or not isinstance(record.get("measured_statistics"), dict)
                or not isinstance(record.get("control_results"), dict)
                or not isinstance(provenance.get("source_measurement_fingerprint"), str)
                or not provenance["source_measurement_fingerprint"]):
            raise ValueError("saved measurement identity, provenance or replay is unverified")
        if tid in DERIVED_PREDICTORS:
            # Derived economic measurements have their own record validators.
            problem = economic_admission_reason(record) or economic_binding_reason(
                record, champion_id=cf.champion_id,
                source_fingerprint=provenance.get("source_measurement_fingerprint"))
            if problem:
                raise ValueError(f"derived economic record invalid: {problem}")
        elif (tid not in REGISTRY or record.get("test_version") != REGISTRY[tid].test_version
                or record.get("parameters") != {}):
            raise ValueError("saved measurement identity, provenance or replay is unverified")
        records.append(_scrub_leaky_exhibit(record))
    economics_ids = {r["result_id"] for r in records if r.get("test_id") == "training_economics"}
    if any(r["control_results"]["source_result_id"] not in economics_ids
           for r in records if r.get("test_id") == "training_profit_recurrence"):
        raise ValueError("Recurrence lacks its bound training economics source")
    info = citation_metadata(records)
    expected = {**cf.to_llm_dict(), **_measured_exhibits(records, citation_info=info)}
    if context.get("case_file") != expected:
        raise ValueError("measurement case view differs from the case file or its reliable executed aliases")
    return expected


def check_champion(cf: GeneralizationCaseFile, rule_trees: Sequence[str],
                   basis_targets: Sequence[str] = (), *,
                   measurement_context: Optional[Dict[str, Any]] = None,
                   require_measurements: bool = True) -> Dict[str, Any]:
    """Check one strategy's raw rule trees, profile fields and measurements.

    Use raw genome trees for glossary coverage. A structural check with
    require_measurements=False must be followed by a full measurement check.
    """
    if rule_trees is None:
        raise ValueError("check_champion: rule_trees is required (the genome tree strings); the rendered "
                         "rule-text is not a valid token source (false-red + misses shape_tree)")
    issues: List[str] = []
    llm = cf.to_llm_dict()
    if not (cf.entry_rule or cf.structure_rule):
        issues.append("no rule (empty genome) — policy: a rule-reasoning panel should not court a rule-less champion")
    toks = _treg.identifiers(*[t or "" for t in rule_trees])
    unglossed = sorted(t for t in toks if _treg.describe(t, basis_targets) is None)
    if unglossed:
        issues.append(f"unglossed rule tokens {unglossed} — add glosses to terminal_registry")
    try:
        assert_cause_only(cf)
    except ValueError as e:
        issues.append(f"leaky case-file field: {e}")
    leaked = [k for k in _LLM_WITHHELD if k in llm]
    if leaked:
        issues.append(f"withheld identity field(s) reached the LLM view: {leaked}")
    if measurement_context is not None:
        try:
            llm = _validated_measurement_view(cf, measurement_context)
        except ValueError as e:
            issues.append(f"invalid verified measurement context: {e}")
    citable = [key for key in _citable(llm)
               if not (llm.get("fidelity_ok") is False and "::" not in key)
               and not (llm.get("ablation_confounded") is True and key.startswith("ablation_phi"))]
    if require_measurements and not citable:
        issues.append("no citable numeric exhibit — the judge cannot produce a valid decisive cite; "
                      "supply reliable static exhibits or a verified executed measurement context for this champion")
    return {"champion_id": cf.champion_id, "ready": not issues, "issues": issues, "n_citable": len(citable)}


def _rejected(fn) -> bool:
    """Return whether a validation call raises MalformedVerdict; propagate other errors."""
    try:
        fn()
        return False
    except MalformedVerdict:
        return True


def check_cite_enforcement() -> Dict[str, Any]:
    """Check acceptance of valid citations and rejection of flags or fabricated values."""
    cf_dict = {"champion_id": "champion_x", "fire_margin_abs_median": 0.44, "margins_available": True}
    ok = {"decision": "WILL_GENERALIZE", "decisive_exhibit_ids": ["fire_margin_abs_median"],
          "decisive_metric_values": {"fire_margin_abs_median": 0.44}, "verdict_rationale": "room to spare"}
    bad_flag = {**ok, "decisive_exhibit_ids": ["margins_available"],
                "decisive_metric_values": {"margins_available": True}}
    bad_val = {**ok, "decisive_metric_values": {"fire_margin_abs_median": 9.99}}   # Existing field with an incorrect value.
    bad_id = {**ok, "decisive_exhibit_ids": ["not_a_real_exhibit"],
              "decisive_metric_values": {"not_a_real_exhibit": 0.44}}              # Field absent from the case.
    valid_passes = not _rejected(lambda: (_validate_gen_verdict(ok), _verify_cites(ok, cf_dict, {}),
                                          _reject_flag_cites(ok, cf_dict)))
    bool_rejected = _rejected(lambda: _reject_flag_cites(bad_flag, cf_dict))
    value_mismatch_rejected = _rejected(lambda: _verify_cites(bad_val, cf_dict, {}))
    fabricated_id_rejected = _rejected(lambda: _verify_cites(bad_id, cf_dict, {}))
    ok_ = valid_passes and bool_rejected and value_mismatch_rejected and fabricated_id_rejected
    return {"valid_passes": valid_passes, "bool_cite_rejected": bool_rejected,
            "value_mismatch_rejected": value_mismatch_rejected,
            "fabricated_id_rejected": fabricated_id_rejected, "ok": ok_}


def readiness_report(cfs: Sequence[Tuple[GeneralizationCaseFile, Sequence[str], Sequence[str]]], *,
                     require_measurements: bool = True) -> Dict[str, Any]:
    """Summarize per-strategy readiness and citation checks for a nonempty cohort."""
    per = [check_champion(cf, trees, basis, require_measurements=require_measurements) for cf, trees, basis in cfs]
    enforce = check_cite_enforcement()
    n_ready = sum(1 for r in per if r["ready"])
    fix_list = [{"champion_id": r["champion_id"], "issues": r["issues"]} for r in per if not r["ready"]]
    green = len(per) > 0 and n_ready == len(per) and enforce["ok"]
    return {"n": len(per), "n_ready": n_ready, "green": green,
            "cite_enforcement": enforce, "fix_list": fix_list}


def assert_court_ready(cfs: Sequence[Tuple[GeneralizationCaseFile, Sequence[str], Sequence[str]]], *,
                       require_measurements: bool = True) -> Dict[str, Any]:
    """Raise unless the requested checks pass. Deferring measurements requires a later full packet check."""
    r = readiness_report(cfs, require_measurements=require_measurements)
    if not r["green"]:
        raise RuntimeError(f"court NOT ready: {r['n_ready']}/{r['n']} champions ready, "
                           f"cite_enforcement_ok={r['cite_enforcement']['ok']}; fix_list={r['fix_list'][:5]}")
    return r
