"""Validate diagnostic records and their development references for agent citations.

The catalogue defines measurement direction and diagnostic families.
Reference checks determine whether an observation also supports an
adverse finding.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Optional, Sequence

from potofgold.multi_agent.diagnostics.vulnerability_catalogue import REGISTRY


TOOL_FAMILIES = {
    "entry_boundary_stability": "margin",
    # These tests share an intervention and count as one corroborating family.
    "entry_boundary_economics": "margin",
    "threshold_sensitivity": "margin",
    "signal_reliance": "single_signal",
    "dependence_preserving_control": "single_signal",
    # This diagnostic measures temporal occupancy, not regime coverage.
    "within_regime_repeatability": "temporal",
    "regime_concentration": "regime",
    "exit_behavior_profile": "exit",
    "basis_projection_stability": "basis",
    "transaction_cost_sensitivity": "execution_cost",
    # Cost erosion shares the cost-stress family; profit concentration has its own family.
    "cost_erosion": "execution_cost",
    "day_profit_concentration": "concentration",
    "training_profit_recurrence": "concentration",
    "matched_allegation_control:boundary_brittleness": "margin",
    "matched_allegation_control:regime_concentration": "regime",
}
# Match the claim scope in training_economics.py.
ECONOMIC_CLAIM_SCOPE = "TRAINING_ECONOMICS"
# Economic diagnostics retain their training-economics claim scope.
ECONOMIC_DIAGNOSTICS = frozenset({"entry_boundary_economics"})
ECONOMIC_SOURCES = frozenset({"training_economics", "cost_erosion", "day_profit_concentration",
                            "training_profit_recurrence"}) | ECONOMIC_DIAGNOSTICS
_MATCHED_PRIMARY = {
    "matched_allegation_control:boundary_brittleness": "rank",
    "matched_allegation_control:regime_concentration": "p_value",
}
_RESEARCH_MATCHED_STATISTICS = {
    "matched_allegation_control:boundary_brittleness": {"rank", "center_local_phi"},
    "matched_allegation_control:regime_concentration": {"p_value", "t_observed"},
}
LIVE_MATCHED_VERSION = "verified-live-matched-control-v1"
_UNRELIABLE_REASONS = {
    "control_unreliable", "confounded_by_correlation", "train_replay_fidelity_failed",
    "replay_fidelity_failed", "fidelity_failed", "not_applicable", "statistic_undefined",
}
_MEASUREMENT_SCOPE_REASONS = {
    "no_registered_development_diagnostic", "measurement_implementation_version_mismatch",
    "measurement_parameters_not_bound_by_development_evidence", "missing_result_id",
}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _record(value: Any) -> dict:
    if hasattr(value, "to_exhibit"):
        value = value.to_exhibit()
    return dict(value) if isinstance(value, Mapping) else {}


def calibration_context_id(*, test_id: str, impl_version: str, arm: str, scope: str,
                           dev_manifest_id: str, reject_when: str) -> Optional[str]:
    """Hash the measurement definition and development population for a threshold.

    Records need this provenance to support rejection. The hash preserves the
    binding without exposing the strategy's input condition to agents.
    """
    values = (test_id, impl_version, arm, scope, dev_manifest_id, reject_when)
    if (any(not isinstance(value, str) or not value.strip() for value in values)
            or reject_when not in ("above", "below")):
        return None
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()[:24]


def _threshold_source(record: dict, entry: Any, *, evidence: Optional[dict],
                      arm: Optional[str], scope: str) -> Optional[str]:
    """Recognize only runner-stamped calibrated or exact registered uniform cutoffs."""
    control = record.get("control_results") or {}
    if entry is None or not isinstance(control, Mapping) or not _number(control.get("threshold")):
        return None
    if control.get("uniform_threshold_used") is True:
        if _number(entry.uniform_threshold) and control["threshold"] == entry.uniform_threshold:
            return "uniform"
        return None
    calibration_id = record.get("calibration_id")
    if (control.get("threshold_calibrated") is True
            and isinstance(calibration_id, str) and bool(calibration_id.strip())
            and calibration_id != "uniform_fixed"):
        expected = calibration_context_id(test_id=entry.id, impl_version=entry.test_version, arm=arm,
                                           scope=scope, dev_manifest_id=(evidence or {}).get("dev_manifest_id"),
                                           reject_when=entry.reject_when)
        if expected is not None and control.get("calibration_context_id") == expected:
            return "calibrated"
    return None


def _reliability_reason(record: dict) -> Optional[str]:
    control = record.get("control_results")
    if control is None:
        control = {}
    if not isinstance(control, Mapping):
        return "malformed_control_results"
    applicability = record.get("applicability")
    if applicability != "applicable" and not (
            applicability == "review" and control.get("reason") == "uncalibrated_threshold"):
        return "measurement_not_applicable_or_unreliable"
    fidelity = record.get("replay_fidelity")
    if fidelity is not None and fidelity is not True and (not _number(fidelity) or fidelity != 1.0):
        return "replay_fidelity_failed"
    flags = (control.get("fidelity_ok"), control.get("parity_ok"))
    if any(flag is False or (_number(flag) and flag == 0) for flag in flags):
        return "replay_fidelity_failed"
    if control.get("reason") in _UNRELIABLE_REASONS:
        return str(control["reason"])
    if control.get("confounded") or control.get("confounded_by_correlation") or control.get("control_unreliable"):
        return "control_unreliable"
    entry = REGISTRY.get(record.get("test_id"))
    if entry is not None and entry.guard is not None:
        try:
            if entry.guard(dict(control)):
                return "control_unreliable"
        except (KeyError, TypeError, ValueError):
            return "control_unreliable"
    return None


def measurement_admission_reason(record: Any) -> Optional[str]:
    """Validate a current primary measurement without requiring an existing development study.

    Calibration producers and court consumers share these checks. Reliable observations from an
    uncalibrated diagnostic remain admissible; the relationship with loss is evaluated separately.
    """
    rec = _record(record)
    from potofgold.multi_agent.strategy_profiles.economic_predictors import economic_admission_reason, is_economic_predictor
    if rec.get("test_id") == "training_profit_recurrence":
        from potofgold.multi_agent.diagnostics.training_recurrence import recurrence_admission_reason
        return recurrence_admission_reason(rec)
    if is_economic_predictor(rec.get("test_id")):
        return economic_admission_reason(rec)
    problem = _reliability_reason(rec)
    if problem:
        return problem
    entry = REGISTRY.get(rec.get("test_id"))
    if entry is None:
        return "no_registered_development_diagnostic"
    if rec.get("test_version") != entry.test_version:
        return "measurement_implementation_version_mismatch"
    if rec.get("replay_fidelity") is None:
        return "missing_replay_fidelity"
    # Custom diagnostic parameters cannot inherit support for default measurements.
    parameters = rec.get("parameters")
    if parameters is not None and (not isinstance(parameters, Mapping) or parameters):
        return "measurement_parameters_not_bound_by_development_evidence"
    values = rec.get("measured_statistics") or {}
    if not isinstance(values, Mapping) or not _number(values.get(entry.statistic_name)):
        return "missing_finite_primary_measurement"
    if not rec.get("result_id"):
        return "missing_result_id"
    return None


def record_evidence_lookup(record: Any, *, evidence: Optional[dict], arm: Optional[str],
                           scope: str = "all") -> dict:
    """Look up a reliable measurement against compatible development evidence for its strategy arm.

    An unusable execution is different from an incompatible development study:
    reliability/nonfinite failures return ``measurement_unusable``; diagnostic,
    implementation, parameter and study identity mismatches remain ``out_of_scope``.
    Neither status supplies development support or relaxes citation admission.
    """
    rec = _record(record)
    rid = rec.get("result_id")

    def refused(reason: str, *, status: str = "out_of_scope") -> dict:
        return {"measurement_id": rid, "diagnostic": rec.get("test_id"),
                "status": status, "reason": reason}

    if rec.get("test_id") == "training_profit_recurrence":
        problem = measurement_admission_reason(rec)
        return refused(problem or "descriptive_training_summary_no_fitted_predictive_reference",
                       status="measurement_unusable" if problem else "not_evaluated")

    from potofgold.multi_agent.strategy_profiles.economic_predictors import ECONOMIC_PREDICTORS, is_economic_predictor
    if is_economic_predictor(rec.get("test_id")):
        pred = ECONOMIC_PREDICTORS[rec.get("test_id")]
        problem = measurement_admission_reason(rec)
        if problem:
            return refused(problem, status="measurement_unusable")
        if not evidence or evidence.get("store") is None:
            return refused("no_development_evidence")
        if arm is None:
            return refused("missing_strategy_arm")
        if not evidence.get("version") or not evidence.get("dev_manifest_id"):
            return refused("missing_development_evidence_identity")
        from potofgold.multi_agent.strategy_profiles.diagnostic_evidence import diagnostic_evidence_lookup, make_evidence_resolver
        # Resolve each derived economic predictor against its own reference.
        resolver = make_evidence_resolver(
            {str(rid): rec}, arm=arm, version=evidence["version"],
            dev_manifest_id=evidence["dev_manifest_id"], scope=scope,
            statistic_name=pred["statistic"], reject_when=pred["reject_when"])
        try:
            return diagnostic_evidence_lookup(str(rid), evidence["store"], resolver, require_compatible=True)
        except Exception as exc:                             # noqa: BLE001
            return {"measurement_id": rid, "diagnostic": rec.get("test_id"),
                    "status": "store_integrity_error", "reason": f"corrupt_dev_record:{type(exc).__name__}"}

    # Preserve an unknown or obsolete executor identity even when replay failed.
    entry = REGISTRY.get(rec.get("test_id"))
    if entry is None:
        return refused("no_registered_development_diagnostic")
    control = rec.get("control_results") or {}
    if (rec.get("test_version") == "unknown" and isinstance(control, Mapping)
            and control.get("reason") == "train_replay_fidelity_failed"):
        # A diagnostic that was never executed has no implementation to compare.
        return refused("train_replay_fidelity_failed", status="measurement_unusable")
    if rec.get("test_version") != entry.test_version:
        return refused("measurement_implementation_version_mismatch")

    problem = measurement_admission_reason(rec)
    if problem:
        return refused(problem, status=("out_of_scope" if problem in _MEASUREMENT_SCOPE_REASONS
                                        else "measurement_unusable"))
    if not evidence or evidence.get("store") is None:
        return refused("no_development_evidence")
    if arm is None:
        return refused("missing_strategy_arm")
    if not evidence.get("version") or not evidence.get("dev_manifest_id"):
        return refused("missing_development_evidence_identity")
    from potofgold.multi_agent.strategy_profiles.diagnostic_evidence import diagnostic_evidence_lookup, make_evidence_resolver
    resolver = make_evidence_resolver(
        {str(rid): rec}, arm=arm, version=evidence["version"],
        dev_manifest_id=evidence["dev_manifest_id"], scope=scope,
        statistic_name=entry.statistic_name, reject_when=entry.reject_when)
    # Report malformed references as integrity failures.
    try:
        return diagnostic_evidence_lookup(str(rid), evidence["store"], resolver, require_compatible=True)
    except (KeyError, TypeError, ValueError) as exc:
        return {"measurement_id": rid, "diagnostic": rec.get("test_id"),
                "status": "store_integrity_error", "reason": f"corrupt_dev_record:{type(exc).__name__}"}


DEVELOPMENT_CARD_VERSION = "development-cards-v2"


def development_evidence_view(lookup: Mapping[str, Any], *, diagnostic: Optional[str] = None) -> dict:
    """Expose reference aggregates without strategy identities or raw outcomes.

    Present discrimination, return, tail risk and winner preservation separately,
    alongside the candidate rule and the relevant denominators.
    """
    reference = lookup.get("evidence") or {}
    block = lookup.get("block") or {}
    confusion, portfolio = block.get("confusion") or {}, block.get("portfolio") or {}
    dims = block.get("dimensions") or {}
    disc = dims.get("discrimination") or {}
    ret = dims.get("return_effect") or {}
    tail = dims.get("tail_effect") or {}
    wp = dims.get("winner_preservation") or {}
    n_winners, rejected = wp.get("n_dev_winners"), wp.get("winners_rejected")
    rejection_rate = wp.get("winner_rejection_rate")
    winner_counts_valid = (
        isinstance(n_winners, int) and not isinstance(n_winners, bool) and n_winners > 0
        and isinstance(rejected, int) and not isinstance(rejected, bool)
        and 0 <= rejected <= n_winners and _number(rejection_rate)
        and math.isclose(rejection_rate, rejected / n_winners, rel_tol=0, abs_tol=1e-12))
    evidence_id = reference.get("evidence_id")
    # Secondary reference cards supply context without validated support.
    dev_supported = bool(disc.get("classification") == "supported" and evidence_id
                         and not _is_secondary_reference(lookup))
    # Use the same dimensional interpretation for cards and citations.
    if dims:
        dev_status = _composite_dev_status(disc.get("classification"), dev_supported, lookup.get("status"))
    else:
        _raw = lookup.get("status")
        dev_status = ("insufficient_evidence" if (_is_secondary_reference(lookup)
                      and _raw == "supported_on_dev_validation") else _composite_dev_status(None, False, _raw))

    # Keep winner and loser counts on the same dimensional population.
    n_losers, losers_rejected = disc.get("n_dev_losers"), None
    if (winner_counts_valid and disc.get("n_dev_winners") == n_winners
            and disc.get("statistic") == "loser_reject_rate_minus_winner_reject_rate"
            and _number(disc.get("D")) and isinstance(n_losers, int)
            and not isinstance(n_losers, bool) and n_losers > 0):
        count = (disc["D"] + rejection_rate) * n_losers
        if (math.isfinite(count) and 0 <= round(count) <= n_losers
                and math.isclose(count, round(count), rel_tol=0, abs_tol=1e-8)):
            losers_rejected = round(count)
    fit_mode = ((reference.get("config") or {}).get("fit_policy") or {}).get("portfolio_mode")
    mean_book = fit_mode == "mean" or (fit_mode is None and dims.get("schema") == "dimensional-evidence-v1")
    multiplicity = {key: value for key, value in (dims.get("multiplicity") or {}).items() if key != "families"}
    if multiplicity:
        # Use numeric alpha when historical family descriptions differ.
        multiplicity["interpretation"] = ("Numeric alpha values are those of the frozen reference. They describe "
                                         "the reported intervals; this card makes no additional simultaneous-coverage claim.")
    card = {
        "presentation_version": DEVELOPMENT_CARD_VERSION,
        "diagnostic": diagnostic or lookup.get("diagnostic"),
        "reference_role": lookup.get("reference_role"),
        "deterioration_benchmarks": (lookup.get("evidence") or {}).get("deterioration_benchmarks"),
        "value_beyond_statistical_filter": (lookup.get("evidence") or {}).get("value_beyond_statistical_filter"),
        "pooled_result": (lookup.get("evidence") or {}).get("promising_result"),
        "reference_limitations": (lookup.get("evidence") or {}).get("limitations"),
        "winner_definition": reference.get("winner_definition"),
        "loser_definition": reference.get("loser_definition"),
        "reference_accounting": {
            "portfolio_mode": "mean_over_surviving_responsible_champions" if mean_book else fit_mode or "not_reported",
            "weighting": ("Each day averages the surviving champions responsible for that day. Removing a champion "
                          "changes the averaging denominator; surviving weights are renormalized." if mean_book
                          else "The averaging convention is not identified by this card; consult the reference provenance."),
            "court_allocation": "Original allocations are scaled by Full/Half/No position; released capital remains in cash without redistribution.",
            "interpretation": "Return and CVaR effects below evaluate the fitted reference rule, not the court's graduated allocation policy.",
        },
        # Include denominators and comparators for each dimension.
        "discrimination": {"classification": disc.get("classification"), "reason": disc.get("reason"),
                           "D": disc.get("D"), "ci_composition": disc.get("ci_composition"),
                           "ci_market": disc.get("ci_market"), "coverage_ok": disc.get("coverage_ok"),
                           "alpha": disc.get("alpha"), "n_dev_winners": disc.get("n_dev_winners"),
                           "n_dev_losers": disc.get("n_dev_losers"),
                           "losers_rejected": losers_rejected,
                           "loser_rejection_rate": losers_rejected / n_losers if losers_rejected is not None else None,
                           "rate_denominator": "n_dev_losers",
                           "no_threshold_handling": disc.get("no_threshold_fallback"),
                           "n_independent_clusters": disc.get("n_independent_clusters"),
                           "n_folds_evaluated": disc.get("n_folds_evaluated"),
                           "n_folds_total": disc.get("n_folds_total"),
                           "valid_replicates": disc.get("valid_replicates")},
        "return_effect": {"classification": ret.get("classification"), "delta": ret.get("delta"),
                          "ci": ret.get("ci"), "alpha": ret.get("alpha"),
                          "conditional_on_fitted_rule": ret.get("conditional_on_fitted_rule"),
                          "comparator": (ret.get("method") or {}).get("comparator"),
                          "limitation": (ret.get("method") or {}).get("limitation")},
        "tail_effect": {"classification": tail.get("classification"), "delta": tail.get("delta"),
                        "ci": tail.get("ci"), "alpha": tail.get("alpha"),
                        "conditional_on_fitted_rule": tail.get("conditional_on_fitted_rule"),
                        "comparator": (tail.get("method") or {}).get("comparator"),
                        "limitation": (tail.get("method") or {}).get("limitation")},
        "winner_preservation": {"rate": rejection_rate,
                                "rate_definition": "legacy alias for winner_rejection_rate, NOT winner retention",
                                "winner_rejection_rate": rejection_rate,
                                "winners_retained": n_winners - rejected if winner_counts_valid else None,
                                "winner_retention_rate": (n_winners - rejected) / n_winners if winner_counts_valid else None,
                                "rate_denominator": "n_dev_winners",
                                "population": ("Frozen dimensional winner-preservation population; both rates use the same denominator."
                                               if wp else "Winner-preservation estimate unavailable in this reference."),
                                "no_threshold_handling": disc.get("no_threshold_fallback"),
                                "retention_derivation_available": winner_counts_valid,
                                "interpretation": "Lower rejection preserves more winners. Zero rejection among N>0 winners means all N were retained; it does not mean zero preservation.",
                                "wilson_95ci": wp.get("wilson_95ci"),
                                "wilson_95ci_quantity": "winner_rejection_rate",
                                "n_dev_winners": wp.get("n_dev_winners"), "winners_rejected": wp.get("winners_rejected"),
                                "preservation_constraint": wp.get("preservation_constraint"),
                                "constraint_breached": wp.get("constraint_breached"),
                                "interval_limitation": wp.get("interval_limitation")},
        "adverse_point_estimate": dims.get("adverse_point_estimate"),
        "final_rule": dims.get("final_rule"), "dev_distribution": dims.get("dev_distribution"),
        "multiplicity": multiplicity or None, "dimensions_present": bool(dims),
        "authoritative_interpretation": ("the four dimensions above are authoritative; dev_status summarizes "
                                         "discrimination only. Return, tail and winner preservation remain separate. "
                                         "Unresolved means the relationship is uncertain, not established absent. "
                                         "Counts include the stated no-threshold handling; historical composites and "
                                         "evaluated-fold-only counts are retained in the store audit." if dims else
                                         "Legacy reference only; dimensional estimates are unavailable. Unresolved means uncertain, not absent."),
        "reason": lookup.get("reason"), "evidence_id": evidence_id,
        "reference_rule": lookup.get("reference_rule"), "dev_status": dev_status,
    }
    if not dims:
        # Legacy references retain their available summary and population counts.
        card.update(support_basis=lookup.get("support_basis"),
            dev_confusion={**{k: confusion.get(k) for k in (
                "winner_rejection_rate", "loser_removal_rate", "n_validated", "n_removed",
                "n_dev_winners", "n_dev_losers", "winners_removed", "losers_removed")},
                "population": "Legacy evaluated-fold population only; folds without a fitted threshold are omitted."},
            dev_tail={k: portfolio.get(k) for k in (
                "cvar_improvement", "return_change", "return_change_ci_excludes_0")})
    return card


def development_reference_context(test_ids: Sequence[str], *, evidence: Optional[dict],
                                  arm: Optional[str], scope: str = "all") -> list[dict]:
    """Give first-pass agents compatible reference evidence without fabricating an executed result."""
    if evidence is None:
        return []
    from potofgold.multi_agent.strategy_profiles.diagnostic_evidence import diagnostic_reference_lookup
    from potofgold.multi_agent.strategy_profiles.training_economics import PREDICTOR as _ECON
    from potofgold.multi_agent.strategy_profiles.economic_predictors import DERIVED_PREDICTORS as _DERIVED
    context = []
    for test_id in sorted(set(test_ids)):
        entry = REGISTRY.get(test_id)
        if entry is not None:
            statistic_name, reject_when, impl_version = entry.statistic_name, entry.reject_when, entry.test_version
            family = TOOL_FAMILIES.get(test_id, "other")
        elif test_id == _ECON["diagnostic"]:
            statistic_name, reject_when, impl_version = _ECON["statistic"], _ECON["reject_when"], _ECON["version"]
            family = "execution_cost"
        elif test_id in _DERIVED:
            pred = _DERIVED[test_id]
            statistic_name, reject_when, impl_version = pred["statistic"], pred["reject_when"], pred["version"]
            family = TOOL_FAMILIES.get(test_id, "other")
        else:
            continue
        if evidence.get("store") is None:
            lookup = {"status": "out_of_scope", "reason": "no_development_evidence"}
        else:
            try:
                lookup = diagnostic_reference_lookup(
                    test_id, evidence["store"], scope=scope, version=evidence.get("version"),
                    dev_manifest_id=evidence.get("dev_manifest_id"), arm=arm,
                    impl_version=impl_version, reject_when=reject_when)
            except (KeyError, TypeError, ValueError) as exc:
                lookup = {"status": "store_integrity_error", "reason": f"corrupt_dev_record:{type(exc).__name__}"}
        primary_card = {**development_evidence_view(lookup, diagnostic=test_id),
                        "measurement_required": True, "statistic_name": statistic_name, "family": family}
        # Add secondary economics context once, without treating it as primary support.
        if not (test_id == _ECON["diagnostic"] and primary_card.get("reference_role") == "secondary_context"):
            context.append(primary_card)
        if test_id == _ECON["diagnostic"] and evidence.get("store") is not None:
            try:
                sec = diagnostic_reference_lookup(
                    test_id, evidence["store"], scope=scope, version=evidence.get("version"),
                    dev_manifest_id=evidence.get("dev_manifest_id"), arm=arm,
                    impl_version=impl_version, reject_when=reject_when, reference_role="secondary_context")
            except (KeyError, TypeError, ValueError) as exc:
                sec = {"status": "store_integrity_error", "reason": f"corrupt_dev_record:{type(exc).__name__}"}
            if sec.get("reference_role") == "secondary_context" and sec.get("status") not in (
                    "out_of_scope", "no_development_evidence", "store_integrity_error", None):
                context.append({**development_evidence_view(sec, diagnostic=test_id),
                                "measurement_required": True, "statistic_name": statistic_name, "family": family})
    return context


def research_measurement_admission_reason(record: Any, *, expected_data_split_hash: Optional[str] = None) -> Optional[str]:
    """Require current measurement identity and a verified replay for new predictions.

    The caller also checks that the measurement belongs to the strategy and
    source being assessed. Citation of legacy observations has separate checks.
    """
    rec = _record(record)
    if rec.get("test_id") in ("training_economics", "training_profit_recurrence"):
        return measurement_admission_reason(rec)
    problem = _reliability_reason(rec)
    if problem:
        return problem
    if not _number(rec.get("replay_fidelity")) or rec["replay_fidelity"] != 1.0:
        return "missing_verified_training_replay"
    if not isinstance(rec.get("result_id"), str) or not rec["result_id"].strip():
        return "missing_result_id"
    if not isinstance(rec.get("data_split_hash"), str) or not rec["data_split_hash"].strip():
        return "missing_data_split_identity"
    if rec.get("test_id") in _MATCHED_PRIMARY:
        if rec.get("test_version") == LIVE_MATCHED_VERSION:
            if rec.get("parameters") != {}:
                return "nondefault_matched_control_parameters"
            if expected_data_split_hash is not None and rec["data_split_hash"] != expected_data_split_hash:
                return "measurement_split_mismatch"
            values, control = rec.get("measured_statistics") or {}, rec.get("control_results") or {}
            if control.get("training_replay_verified") is not True:
                return "missing_verified_training_replay"
            verdict = control.get("verdict")
            if verdict not in ("EXCEPTIONAL", "NOT_EXCEPTIONAL") or control.get("reason") is not None:
                return "unusable_matched_control"
            if rec.get("decision") != ("keep" if verdict == "NOT_EXCEPTIONAL" else "review"):
                return "invalid_matched_control_interpretation"
            if rec.get("test_id") == "matched_allegation_control:regime_concentration":
                from potofgold.multi_agent.diagnostics.matched_controls import NULL_K_SMOKE, REGIME_ALPHA
                if (not all(_number(values.get(key)) for key in ("p_value", "t_observed", "k_null"))
                        or values["k_null"] != NULL_K_SMOKE or control.get("alpha") != REGIME_ALPHA
                        or not 1 / (NULL_K_SMOKE + 1) <= values["p_value"] <= 1
                        or not 0 <= values["t_observed"] <= 1
                        or (values["p_value"] <= REGIME_ALPHA) != (verdict == "EXCEPTIONAL")):
                    return "invalid_matched_control_measurement"
            else:
                from potofgold.multi_agent.diagnostics.matched_controls import MIN_VALID_NEIGHBOURS, NEIGHBOURHOOD_OFFSETS
                if (not all(_number(values.get(key)) for key in ("rank", "center_local_phi", "n_valid_neighbours"))
                        or not MIN_VALID_NEIGHBOURS <= values["n_valid_neighbours"] <= len(NEIGHBOURHOOD_OFFSETS)
                        or values["n_valid_neighbours"] != int(values["n_valid_neighbours"])
                        or not 1 <= values["rank"] <= values["n_valid_neighbours"] + 1
                        or values["rank"] != int(values["rank"])
                        or (values["rank"] == 1) != (verdict == "EXCEPTIONAL")
                        or control.get("method") != "descriptive_rank"):
                    return "invalid_matched_control_measurement"
            return None
        return "unverified_matched_control_implementation"
    problem = measurement_admission_reason(rec)
    if problem:
        return problem
    if rec.get("parameters") != {}:
        return "measurement_parameters_not_explicitly_default"
    if expected_data_split_hash is not None and rec["data_split_hash"] != expected_data_split_hash:
        return "measurement_split_mismatch"
    return None


_DEV_STATUS_LABELS = {"supported": "supported_on_dev_validation", "contrary": "contrary_evidence",
                      "unresolved": "unresolved", "insufficient": "insufficient_evidence"}


def _composite_dev_status(classification: Optional[str], dev_supported: bool, fallback: Optional[str]) -> Optional[str]:
    """Give cards and citations a consistent reference status.

    A supported classification without an admissible evidence ID is reported
    as insufficient evidence.
    """
    if classification == "supported" and not dev_supported:
        return "insufficient_evidence"
    return _DEV_STATUS_LABELS.get(classification, "unresolved" if fallback == "no_relationship_detected" else fallback)


def _valid_final_rule(threshold: Any, direction: Any, expected_direction: Optional[str] = None) -> bool:
    """Require a finite threshold and a recognized adverse direction.

    When an expected direction is supplied, the stored direction must match it.
    """
    if not (_number(threshold) and direction in ("below", "above")):
        return False
    return expected_direction is None or direction == expected_direction


def _is_secondary_reference(lookup: Optional[Mapping[str, Any]]) -> bool:
    """Identify references available as context but excluded from development support."""
    lk = lookup or {}
    role = lk.get("reference_role") or (lk.get("evidence") or {}).get("reference_role")
    return role == "secondary_context"


def citation_metadata(records: Sequence[Any], *, evidence: Optional[dict] = None,
                      arm: Optional[str] = None, scope: str = "all",
                      expected_data_split_hash: Optional[str] = None) -> dict[str, dict]:
    """Resolve result_id::statistic citations against executed measurement records.

    valid/citeable identifies a reliable numeric observation; direction records
    the diagnostic's adverse direction. dev_supported requires compatible
    development evidence. qualified_support additionally requires an adverse
    reading under an admissible threshold.

    Unknown diagnostics and secondary measurements can supply context without
    rejection corroboration. Reject conflicting records with the same result ID.
    """
    from potofgold.multi_agent.strategy_profiles.economic_predictors import is_economic_predictor
    normalized = [_record(value) for value in records or []]
    # Retain the source split for training-economics citations.
    economics_splits = {rec.get("data_split_hash") for rec in normalized
                        if rec.get("test_id") == "training_economics"
                        and measurement_admission_reason(rec) is None}
    split_conflict = len(economics_splits) > 1 or (
        expected_data_split_hash is not None and bool(economics_splits - {expected_data_split_hash}))
    if expected_data_split_hash is None and len(economics_splits) == 1:
        expected_data_split_hash = next(iter(economics_splits))
    fingerprints: dict[str, set[str]] = {}
    for rec in normalized:
        rid = rec.get("result_id")
        if rid:
            fingerprints.setdefault(str(rid), set()).add(json.dumps(rec, sort_keys=True, default=str))
    result: dict[str, dict] = {}
    for rec in normalized:
        rid = rec.get("result_id")
        if not rid:
            continue
        tid = rec.get("test_id")
        if tid == "training_profit_recurrence":
            from potofgold.multi_agent.diagnostics.training_recurrence import NAMES, OPTIONAL_NAMES
            problem = measurement_admission_reason(rec)
            if len(fingerprints[str(rid)]) != 1:
                problem = "conflicting_result_id"
            if split_conflict or (expected_data_split_hash is not None
                                  and rec.get("data_split_hash") != expected_data_split_hash):
                problem = "conflicting_training_split_identity"
            for name, value in (rec.get("measured_statistics") or {}).items():
                if name not in NAMES | OPTIONAL_NAMES:
                    continue
                valid = problem is None and _number(value)
                result[f"{rid}::{name}"] = {
                    "family": "concentration", "direction": None, "valid": valid, "citeable": valid,
                    "primary": False, "research_admissible": valid, "research_admission_reason": problem,
                    "claim_scope": ECONOMIC_CLAIM_SCOPE, "evidence_kind": ECONOMIC_CLAIM_SCOPE,
                    "dev_supported": False, "qualified_support": False, "guarded_admissible": False,
                    "adverse": False, "evidence_id": None, "dev_status": "not_evaluated",
                    "reason": problem or "descriptive_training_recurrence_not_a_fitted_predictor",
                    "test_id": tid, "result_id": str(rid), "statistic": value, "statistic_name": name}
            continue
        if is_economic_predictor(tid):
            from potofgold.multi_agent.strategy_profiles.economic_predictors import CLAIM_SCOPE, ECONOMIC_PREDICTORS, economic_admission_reason
            from potofgold.multi_agent.strategy_profiles.training_economics import STATISTIC_NAMES, TEST_ID as _TE_ID
            pred = ECONOMIC_PREDICTORS[tid]
            stat_names = STATISTIC_NAMES if tid == _TE_ID else {pred["statistic"]}
            reject_when = pred["reject_when"]
            problem = economic_admission_reason(rec)
            if len(fingerprints[str(rid)]) != 1:
                problem = "conflicting_result_id"
            # Development support applies to this predictor and its primary statistic.
            econ_lk = (record_evidence_lookup(rec, evidence=evidence, arm=arm, scope=scope)
                       if (problem is None and evidence and evidence.get("store") is not None) else {})
            econ_dims = (econ_lk.get("block") or {}).get("dimensions") or {}
            econ_class = (econ_dims.get("discrimination") or {}).get("classification")
            econ_evid = (econ_lk.get("evidence") or {}).get("evidence_id")
            econ_final = econ_dims.get("final_rule") or {}
            guarded_ok = bool(((econ_lk.get("evidence") or {}).get("guarded_admissibility") or {}).get("admissible"))
            values = rec.get("measured_statistics") or {}
            if not isinstance(values, Mapping):
                continue
            for name, value in values.items():
                if name not in stat_names:
                    continue
                valid = problem is None and _number(value)
                economic = not name.endswith("_multiplier")
                predictor = name == pred["statistic"]
                dev_supported = bool(valid and predictor and econ_class == "supported" and econ_evid
                                     and not _is_secondary_reference(econ_lk))
                thr, ddir = econ_final.get("final_all_dev_threshold"), econ_final.get("direction")
                # The stored adverse direction must match the predictor definition.
                valid_rule = _valid_final_rule(thr, ddir, reject_when)
                adverse = bool(valid and economic and (value < 0 if reject_when == "below" else value > 0))
                store_adverse = bool(valid and predictor and valid_rule and
                                     (value < thr if ddir == "below" else value > thr))
                qualified = bool(dev_supported and store_adverse)
                # Require an adverse reading and an admissible primary reference for corroboration.
                guarded_admissible = bool(predictor and store_adverse and guarded_ok and econ_evid)
                # Derive the explanation and support flags from the same reference.
                if problem:
                    econ_reason = problem
                elif not economic:
                    econ_reason = "context_only_measurement"
                elif not predictor:
                    econ_reason = "descriptive_economic_not_the_calibrated_predictor"
                elif qualified:
                    econ_reason = "supported_adverse_economic_predictor"
                elif guarded_admissible:
                    econ_reason = "guarded_admissible_economic_predictor"
                elif dev_supported:
                    econ_reason = "supported_economic_predictor_without_adverse_finding"
                else:
                    econ_reason = "observed_economic_predictive_relationship_unvalidated"
                result[f"{rid}::{name}"] = {
                    "family": (TOOL_FAMILIES.get(tid, "execution_cost") if economic else "other"),
                    "direction": (("low" if reject_when == "below" else "high") if valid and economic else None),
                    "valid": valid, "citeable": valid, "primary": False,
                    "research_admissible": valid and not split_conflict,
                    "research_admission_reason": problem or ("conflicting_training_split_identity" if split_conflict else None),
                    "claim_scope": CLAIM_SCOPE, "evidence_kind": CLAIM_SCOPE,
                    "dev_supported": dev_supported, "qualified_support": qualified,
                    "guarded_admissible": guarded_admissible,
                    "adverse": adverse, "store_adverse": store_adverse,
                    "threshold_source": ("calibrated" if predictor and econ_evid else None),
                    "store_threshold": thr if (predictor and valid_rule) else None,
                    "evidence_id": econ_evid if (dev_supported or guarded_admissible) else None,
                    "dev_status": (_composite_dev_status(econ_class, dev_supported, econ_lk.get("status"))
                                   if (predictor and econ_dims)
                                   else ("not_evaluated" if valid else "measurement_unusable")),
                    "reason": econ_reason,
                    "test_id": tid, "result_id": str(rid), "statistic": value,
                    "statistic_name": name,
                }
            continue
        entry = REGISTRY.get(tid)
        registered = entry is not None and rec.get("test_version") == entry.test_version
        # Missing versions may describe current observations; explicit stale versions cannot.
        directional = entry is not None and rec.get("test_version") in (None, "", entry.test_version)
        primary_name = entry.statistic_name if directional else _MATCHED_PRIMARY.get(tid)
        direction = ("low" if entry.reject_when == "below" else "high") if directional else (
            "low" if tid in _MATCHED_PRIMARY else None)
        problem = _reliability_reason(rec)
        if len(fingerprints[str(rid)]) != 1:
            problem = "conflicting_result_id"
        lookup = (record_evidence_lookup(rec, evidence=evidence, arm=arm, scope=scope)
                  if problem != "conflicting_result_id" else
                  {"status": "measurement_unusable", "reason": problem})
        _dims = (lookup.get("block") or {}).get("dimensions") or {}
        # Prefer dimensional discrimination; use the composite only for legacy records.
        supported = (((_dims.get("discrimination") or {}).get("classification") == "supported") if _dims
                     else lookup.get("status") == "supported_on_dev_validation")
        evidence_id = (lookup.get("evidence") or {}).get("evidence_id")
        _final_rule = _dims.get("final_rule") or {}
        _store_threshold = _final_rule.get("final_all_dev_threshold")
        _store_direction = _final_rule.get("direction")
        _has_dims = bool(_dims)
        _valid_store_rule = _valid_final_rule(_store_threshold, _store_direction,
                                              entry.reject_when if entry is not None else None)
        values = rec.get("measured_statistics") or {}
        if not isinstance(values, Mapping):
            continue
        control = rec.get("control_results") or {}
        runner_threshold = control.get("threshold") if isinstance(control, Mapping) else None
        threshold_source = _threshold_source(rec, entry, evidence=evidence, arm=arm, scope=scope)
        research_problem = ("conflicting_training_split_identity" if split_conflict else
            research_measurement_admission_reason(rec, expected_data_split_hash=expected_data_split_hash))
        from potofgold.multi_agent.strategy_profiles.evidence_summaries import approved_boundary_context
        _, summary_values = approved_boundary_context(rec)
        for name, value in values.items():
            primary = name == primary_name
            secondary = bool(registered and name in entry.secondary_statistics)
            summary = name in summary_values and value == summary_values[name]
            research_statistic = (primary or secondary or summary) if tid not in _MATCHED_PRIMARY else name in _RESEARCH_MATCHED_STATISTICS[tid]
            valid = problem is None and _number(value)
            # Use the stored final rule for candidate standing and retain the runner cutoff separately.
            if _valid_store_rule:
                adverse = bool(valid and primary and
                               (value < _store_threshold if _store_direction == "below" else value > _store_threshold))
            elif _has_dims:
                # Only legacy records without dimensions may fall back to the runner cutoff.
                adverse = False
            else:
                adverse = bool(valid and primary and registered and threshold_source and rec.get("decision") == "reject"
                               and runner_threshold is not None
                               and (value < runner_threshold if entry.reject_when == "below" else value > runner_threshold))
            dev_supported = bool(valid and primary and supported and evidence_id
                                 and not _is_secondary_reference(lookup))
            # Legacy qualification additionally requires a calibrated threshold.
            qualified = dev_supported and adverse and (_valid_store_rule or (not _has_dims and threshold_source == "calibrated"))
            reason = problem or ("nonfinite_or_non_numeric_measurement" if not _number(value) else None)
            if reason is None:
                if not primary:
                    reason = "context_only_measurement"
                elif qualified:
                    reason = "supported_adverse_measurement"
                elif dev_supported:
                    reason = ("design_cutoff_not_loss_validated" if threshold_source == "uniform" else
                              "calibration_context_missing_or_mismatched"
                              if control.get("threshold_calibrated") is True and threshold_source is None
                              else "supported_relationship_without_adverse_thresholded_finding")
                else:
                    reason = lookup.get("reason") or _composite_dev_status(
                        (_dims.get("discrimination") or {}).get("classification"), dev_supported,
                        lookup.get("status")) or "measurement_only"
            # Corroboration requires an adverse primary measurement and an admissible reference.
            _guarded_admissible = bool(primary and adverse and evidence_id
                and ((lookup.get("evidence") or {}).get("guarded_admissibility") or {}).get("admissible"))
            result[f"{rid}::{name}"] = {
                "family": TOOL_FAMILIES.get(tid, "other") if primary or secondary else "other",
                "direction": direction if (valid and primary and
                    (not getattr(entry, "descriptive_only", False) or dev_supported)) else
                    entry.secondary_statistics[name] if valid and secondary else None,
                "valid": valid, "citeable": valid, "primary": primary,
                "research_admissible": bool(valid and research_problem is None and
                    research_statistic),
                "research_admission_reason": problem or research_problem or
                    ("nonfinite_or_non_numeric_measurement" if not _number(value) else
                     "unregistered_secondary_statistic" if not research_statistic else None),
                "dev_supported": dev_supported, "qualified_support": qualified,
                "adverse": adverse, "threshold_source": threshold_source,
                "store_threshold": _store_threshold if _valid_store_rule else None,
                "runner_threshold": runner_threshold,
                "evidence_id": evidence_id if (dev_supported or _guarded_admissible) else None,
                # Keep the public status consistent with the support flags.
                "dev_status": (_composite_dev_status((_dims.get("discrimination") or {}).get("classification"),
                                                     dev_supported, lookup.get("status"))
                               if _dims else _composite_dev_status(None, dev_supported, lookup.get("status"))),
                "reason": reason,
                "test_id": tid, "result_id": str(rid), "statistic": value,
                "statistic_name": name,
                # Economic diagnostic citations retain their training-economics scope.
                **({"claim_scope": ECONOMIC_CLAIM_SCOPE, "evidence_kind": ECONOMIC_CLAIM_SCOPE}
                   if tid in ECONOMIC_DIAGNOSTICS else {}),
                "guarded_admissible": _guarded_admissible,
            }
    return result
