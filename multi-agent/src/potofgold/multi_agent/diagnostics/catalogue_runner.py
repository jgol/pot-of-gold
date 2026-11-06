"""Bind diagnostic tools to a strategy and its prepared training fold.

The runner supplies the genome, data and recorded diagnostic parameters.
Agents select a test and cite its result ID. Measurements and provenance
come from the executor.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable, Dict, Optional, Tuple

from potofgold.multi_agent.diagnostics.vulnerability_catalogue import TestResultRecord, _data_split_hash, run_test_record

# Include all executable fields consumed by individual_from_genome_dict.
_GENOME_KEYS = ("entry_tree", "exit_tree", "size_tree", "template_name",
                "delta_tree", "width_tree", "shape", "shape_tree", "stop_mult")


def genome_of(champion: dict) -> Dict[str, Any]:
    """Extract the executable genome from member.genome or a top-level genome.

    Keep every field used by individual_from_genome_dict, including optional
    structure and risk parameters. Exclude cohort identity and outcome fields.
    """
    g = (champion.get("member") or {}).get("genome") or champion.get("genome") or {}
    return {k: g[k] for k in _GENOME_KEYS if k in g}


def _default_train_fidelity(genome: Dict[str, Any], prep: dict, stored_day_pnl: Any,
                            fidelity_tol: float) -> Tuple[bool, float]:
    """Compare reconstructed training daily P&L with the saved values.

    Use the shared evaluator and daily aggregation convention. Return
    (ok, max_delta); saved references without dates permit only a multiset check.
    """
    import numpy as _np
    from potofgold.evolutionary_search.search.genetics import individual_from_genome_dict
    from potofgold.multi_agent.diagnostics.decision_tracer import fidelity_vs_stored, reconstructed_day_pnl
    from potofgold.multi_agent.diagnostics.perturbation_prober import _run_trades
    individual = individual_from_genome_dict(genome)
    dates = _np.asarray(prep["train_df"]["date"].values)
    trades = _run_trades(individual, prep)
    fid = fidelity_vs_stored(reconstructed_day_pnl(trades, dates), stored_day_pnl, fidelity_tol)
    return bool(fid["ok"]), float(fid["maxdelta"])


def _fidelity_failed_record(test_id: str, prep: dict, replay_fidelity: Optional[float],
                            maxdelta: float) -> TestResultRecord:
    """Return an unavailable diagnostic record when training replay validation fails."""
    return TestResultRecord(
        result_id=_data_split_hash(prep) + ":fidfail",
        test_id=test_id, test_version="unknown", parameters={},
        data_split_hash=_data_split_hash(prep), applicability="not_applicable",
        replay_fidelity=replay_fidelity, measured_statistics={},
        control_results={"reason": "train_replay_fidelity_failed", "fidelity_maxdelta": maxdelta,
                         "threshold_calibrated": False},
        calibration_id=None, decision="not_applicable")


def make_run_catalogue_test(
    champion: dict, prep: dict, *,
    stored_day_pnl: Any = None, fidelity_tol: float = 1e-4,
    replay_fn: Optional[Callable[[Dict[str, Any], dict, Any, float], Tuple[bool, float]]] = None,
    saved_stress_report: Optional[dict] = None,
) -> Callable[[str, Optional[dict]], TestResultRecord]:
    """Bind diagnostic execution to one strategy and training preparation.

    If stored_day_pnl is supplied, verify reconstruction once before allowing
    measurements. A failed check produces NOT_APPLICABLE records. Without a
    stored reference, preserve any validation status supplied by the executor.

    Record the agent's requested interpretation separately; it does not choose
    the measurement parameters or evaluation dates.
    """
    genome = genome_of(champion)
    fold = champion.get("fold")

    # Validate the training reconstruction once before exposing diagnostic tools.
    # Saved training P&L lacks dates; this check compares counts and sorted values.
    gate_ok = True
    gate_maxdelta = 0.0
    prep = dict(prep)
    if stored_day_pnl is not None or replay_fn is not None:
        gate_ok, gate_maxdelta = (replay_fn or _default_train_fidelity)(genome, prep, stored_day_pnl, fidelity_tol)
        prep["replay_fidelity"] = (1.0 if gate_ok else 0.0)
    stress_cache = {}

    def _cached_stress_replay():
        # Reuse one replay for behavior and approved economic summaries.
        if "replay" not in stress_cache:
            from potofgold.multi_agent.diagnostics.transaction_cost_stress import run_stress_replay
            stress_cache["replay"] = run_stress_replay(genome, prep)
        return stress_cache["replay"]

    boundary_cache = {}

    def _cached_boundary_replay():
        # Boundary exhibits and dated reports share the same cached trade replays.
        if "replay" not in boundary_cache:
            from potofgold.multi_agent.diagnostics.entry_boundary_economics import run_boundary_replay
            boundary_cache["replay"] = run_boundary_replay(genome, prep)
        return boundary_cache["replay"]

    def run_catalogue_test(test_id: str, request: Optional[dict] = None) -> TestResultRecord:
        if not gate_ok:
            return _fidelity_failed_record(test_id, prep, prep.get("replay_fidelity"), gate_maxdelta)
        thresholds, calibration_id = (None, None)
        if test_id == "transaction_cost_sensitivity":
            from potofgold.multi_agent.diagnostics.vulnerability_catalogue import record_from_output
            result = record_from_output(test_id, genome, prep, _cached_stress_replay().cause_only(),
                                        parameters={}, thresholds=thresholds, calibration_id=calibration_id)
        elif test_id == "entry_boundary_economics":
            from potofgold.multi_agent.diagnostics.vulnerability_catalogue import record_from_output
            result = record_from_output(test_id, genome, prep, _cached_boundary_replay().exhibit(),
                                        parameters={}, thresholds=thresholds, calibration_id=calibration_id)
        else:
            result = run_test_record(test_id, genome, prep, parameters={},
                                     thresholds=thresholds, calibration_id=calibration_id)
        # H3 uses the recorded measurement schema without fitting new fold cutoffs.
        return replace(result, control_results={**result.control_results, "calibration_context_id": None})

    def run_matched_control(allegation: str, **params) -> Dict[str, Any]:
        """Run the requested allegation control with the same training replay checks.

        The agent selects the allegation; the runner supplies the genome and data.
        """
        from potofgold.multi_agent.diagnostics.matched_controls import matched_allegation_control, matched_allegation_applicability, Verdict
        if not gate_ok:
            return {"allegation": allegation, "verdict": Verdict.INCONCLUSIVE.value,
                    "reason": "train_replay_fidelity_failed", "replay_fidelity": prep.get("replay_fidelity"),
                    "maxdelta": gate_maxdelta}
        applicable = matched_allegation_applicability(allegation, genome)
        if not applicable["applicable"]:
            return {"allegation": allegation, "verdict": Verdict.INCONCLUSIVE.value,
                    "reason": applicable["reason"]}
        output = matched_allegation_control(allegation, genome, prep, **params)
        if run_catalogue_test.readiness["ready"]:
            from potofgold.multi_agent.strategy_profiles.tool_evidence import LIVE_MATCHED_VERSION
            # Stamp verified execution on the server.
            output = {**output, "test_version": LIVE_MATCHED_VERSION,
                      "parameters": {} if not params else {"nondefault": True},
                      "replay_fidelity": prep.get("replay_fidelity"),
                      "data_split_hash": _data_split_hash(prep), "training_replay_verified": True}
        return output

    run_catalogue_test.run_matched_control = run_matched_control
    from potofgold.multi_agent.diagnostics.vulnerability_catalogue import REGISTRY
    from potofgold.multi_agent.diagnostics.matched_controls import available_matched_allegations
    run_catalogue_test.available_diagnostics = tuple(REGISTRY)
    run_catalogue_test.available_matched_allegations = available_matched_allegations(genome)

    def stress_report():
        """Export the development training report outside the agent tool interface."""
        from potofgold.multi_agent.input_tools.seed_firewall import load_manifest, firewall_role, _load_canonical_ids
        manifest = load_manifest()
        cid = champion.get("champion_id")
        canonical = _load_canonical_ids(manifest)
        if (not cid or firewall_role(cid, manifest) != "dev" or not canonical or cid not in canonical):
            raise ValueError("Stress outcome reports are restricted to canonical development champions")
        if not run_catalogue_test.readiness["ready"]:
            raise ValueError("Stress report requires a verified training replay")
        return _training_report()

    def _training_report():
        """Return the cached training report or validate a supplied saved report.

        Check that the report belongs to this strategy and split before reusing it.
        """
        from potofgold.multi_agent.diagnostics.vulnerability_catalogue import record_from_output, expected_result_id
        if saved_stress_report is not None:
            saved_id = (saved_stress_report.get("measurement_record") or {}).get("result_id")
            want_id = expected_result_id("transaction_cost_sensitivity", genome, prep)
            if saved_id != want_id:
                raise ValueError(f"saved stress report result_id {saved_id!r} != expected {want_id!r} for champion "
                                 f"{champion.get('champion_id')!r} (wrong champion/split — refusing to reuse)")
            return deepcopy(saved_stress_report)               # Return a fresh mapping so callers cannot mutate the cached report.

        replay = _cached_stress_replay()
        report = replay.offline_report()
        report["measurement_record"] = record_from_output(
            "transaction_cost_sensitivity", genome, prep, replay.cause_only()).to_exhibit()
        return report

    def get_training_economics():
        """Return approved scalar training economics from the verified report.

        Run the cost replay if needed, then reuse it for subsequent requests.
        """
        import hashlib
        import json
        from potofgold.multi_agent.strategy_profiles.training_economics import training_economics_record
        cid = champion.get("champion_id")
        if not isinstance(cid, str) or not cid:
            raise ValueError("Training economics requires a champion identity")
        if not run_catalogue_test.readiness["ready"]:
            raise ValueError("Training economics requires a verified training replay")
        report = _training_report()
        fingerprint = hashlib.sha256(json.dumps(
            ["live-training-cost-replay-v1", cid, report["measurement_record"]],
            sort_keys=True, allow_nan=False).encode()).hexdigest()
        result = training_economics_record(report, champion_id=cid,
                                            source_measurement_fingerprint=fingerprint)
        run_catalogue_test.readiness["source_measurement_fingerprint"] = fingerprint
        return result

    def get_derived_economics():
        """Return cost erosion and profit concentration from the training report.

        Reuse a cached or verified saved report and omit inapplicable predictors.
        """
        from potofgold.multi_agent.strategy_profiles.economic_predictors import cost_erosion_candidate_record, day_concentration_candidate_record
        te = get_training_economics()
        fingerprint = run_catalogue_test.readiness["source_measurement_fingerprint"]
        split = te["data_split_hash"]
        cid = champion.get("champion_id")
        baseline_ledger = list((_training_report()["scenarios"]["1"]["day_pnl"] or {}).values())
        built = [cost_erosion_candidate_record(te["measured_statistics"], champion_id=cid,
                                               data_split_hash=split, source_fingerprint=fingerprint),
                 day_concentration_candidate_record(baseline_ledger, champion_id=cid,
                                                    data_split_hash=split, source_fingerprint=fingerprint)]
        return [rec for rec in built if rec is not None]

    run_catalogue_test.stress_report = stress_report
    def get_training_offline_report():
        """Export the dated training ledger outside the agent tool interface.

        Use the same verified report as get_training_economics, including when
        the strategy belongs to the evaluation cohort.
        """
        if not run_catalogue_test.readiness["ready"]:
            raise ValueError("Training report requires a verified training replay")
        return _training_report()

    run_catalogue_test.get_training_offline_report = get_training_offline_report
    run_catalogue_test.get_training_economics = get_training_economics
    def get_training_recurrence():
        from potofgold.multi_agent.diagnostics.training_recurrence import recurrence_record
        get_training_economics()
        return recurrence_record(_training_report(), champion_id=champion["champion_id"],
            source_measurement_fingerprint=run_catalogue_test.readiness["source_measurement_fingerprint"])
    run_catalogue_test.get_training_recurrence = get_training_recurrence
    run_catalogue_test.get_derived_economics = get_derived_economics
    # Dated boundary reports are exported outside model context.
    run_catalogue_test.get_boundary_offline_report = lambda: _cached_boundary_replay().offline_report()
    def get_training_case_exhibits(names):
        if not run_catalogue_test.readiness["ready"] or stored_day_pnl is None:
            raise ValueError("Case exhibits require a verified training replay and stored training reference")
        from potofgold.multi_agent.strategy_profiles.training_case_exhibits import measure_case_exhibits
        return measure_case_exhibits(champion, prep, stored_day_pnl=stored_day_pnl, names=names)

    run_catalogue_test.get_training_case_exhibits = get_training_case_exhibits
    run_catalogue_test.readiness = {
        "ready": bool(gate_ok and (stored_day_pnl is not None or replay_fn is not None)),
        "replay_checked": stored_day_pnl is not None or replay_fn is not None,
        "replay_fidelity": prep.get("replay_fidelity"), "fidelity_maxdelta": gate_maxdelta,
        "data_split_hash": _data_split_hash(prep),
    }
    return run_catalogue_test


_D1_PROV = {"d1_residualize": ("GP_D1_RESIDUALIZE", lambda v: "1" if v else "0"),
            "d1_estimator": ("GP_D1_ESTIMATOR", str),
            "d1_residualize_scope": ("GP_D1_RESIDUALIZE_SCOPE", str)}

def _flags_from_prov(pr):
    f = {k: v for k, v in (pr.get("effective_flags") or {}).items() if v is not None}
    for pk, (env, conv) in _D1_PROV.items():
        if pr.get(pk) is not None:
            f[env] = conv(pr[pk])
    # Restore the nonlinear-prototype flag from lean provenance when needed.
    if "GP_LEAN_PROTO_NONLINEAR" not in f and pr.get("lean_proto_nonlinear") is not None:
        f["GP_LEAN_PROTO_NONLINEAR"] = str(pr["lean_proto_nonlinear"])
    return f

def _champion_replay_config(champion_record: dict) -> Tuple[bool, int, Optional[int], Dict[str, Any], Any]:
    """Read decision offsets, gate, flags and saved training P&L from provenance."""
    prov = champion_record.get("provenance") or {}
    gate = bool(champion_record.get("gate", prov.get("gate", False)))
    off_ungated = int(prov.get("decision_offset_ungated", 2))
    gated_off = prov.get("decision_offset")
    gated_off = int(gated_off) if gated_off is not None else None
    # Merge residualization settings stored outside effective_flags.
    effective_flags = _flags_from_prov(prov)
    # Cohort records nest the training reference under member.
    stored_day_pnl = (champion_record.get("train_traded_day_pnl")
                      or (champion_record.get("member") or {}).get("train_traded_day_pnl"))
    return gate, off_ungated, gated_off, effective_flags, stored_day_pnl


def _stamp_flags_env_first(effective_flags: Dict[str, Any]) -> None:
    """Set recorded environment flags before importing the input preparation module.

    Target residualization settings are read at module import.
    """
    import os
    for _k, _v in (effective_flags or {}).items():
        if _v is not None:
            os.environ[str(_k)] = str(_v)


def make_box_prep_factory(*, arm: str, template: Any, n_folds: int, data: Any = None, targets: Any = None,
                          basis: Any = None, grammar_mode: Any = None, fidelity_tol: float = 1e-4,
                          saved_report_for: Optional[Callable[[dict], Optional[dict]]] = None
                          ) -> Callable[..., Any]:
    """Build a diagnostic runner for each strategy using its recorded preparation settings.

    Reuse one fold preparation at a time, then bind the strategy genome and
    check its saved training P&L. Embedding preparation also supplies the
    training-day basis-refit function.
    """
    from potofgold.multi_agent.diagnostics import decision_tracer as dt

    _prep_cache: Dict[Any, Any] = {}   # Cache one fold preparation to limit memory use across strategies.

    def _build_prep(fold, off_ungated, gated_off, gate, effective_flags):
        if str(arm).upper() == "A":
            return dt.build_scalar_train_prep(
                template=template, fold=fold, n_folds=n_folds, data=data,
                decision_offset_ungated=off_ungated, gated_decision_offset=gated_off,
                gate=gate, effective_flags=effective_flags)
        return dt.build_embedding_train_prep(
            data=data, targets=targets, basis=basis, grammar_mode=grammar_mode, template=template,
            fold=fold, n_folds=n_folds, decision_offset_ungated=off_ungated,
            gated_decision_offset=gated_off, gate=gate, effective_flags=effective_flags)

    def factory(champion_record: dict):
        if not champion_record:
            return None
        prov = champion_record.get("provenance") or {}
        if str(arm).upper() != "A":
            recorded_basis = prov.get("basis_method") or prov.get("basis")
            if recorded_basis is not None and str(recorded_basis) != str(basis):
                raise ValueError("basis method differs from champion provenance")
            recorded_grammar = prov.get("grammar")
            if recorded_grammar is not None and recorded_grammar != grammar_mode:
                raise ValueError("embedding grammar differs from champion provenance")
        fold = int(champion_record.get("fold"))
        gate, off_ungated, gated_off, effective_flags, stored_day_pnl = _champion_replay_config(champion_record)
        _stamp_flags_env_first(effective_flags)   # Restore flags even when the fold preparation is cached.
        key = (fold, str(template), off_ungated, gated_off, gate,
               tuple(sorted((str(k), str(v)) for k, v in (effective_flags or {}).items())))
        prep = _prep_cache.get(key)
        if prep is None:
            _prep_cache.clear()
            prep = _build_prep(fold, off_ungated, gated_off, gate, effective_flags)
            _prep_cache[key] = prep
        # Dated training references belong to a strategy, not the shared fold.
        prep = dict(prep)
        member = champion_record.get("member") or champion_record
        recorded_fires = member.get("train_traded_days")
        if recorded_fires is not None:
            prep["recorded_train_fired_days"] = tuple(recorded_fires)
        # Verify saved report identity before reusing its stress replay.
        saved = saved_report_for(champion_record) if callable(saved_report_for) else None
        return make_run_catalogue_test(champion_record, prep,
                                       stored_day_pnl=stored_day_pnl, fidelity_tol=fidelity_tol,
                                       saved_stress_report=saved)

    return factory


def saved_stress_resolver(records: Sequence[dict]) -> Callable[[dict], Optional[dict]]:
    """Index saved training reports by strategy for reuse during preparation.

    Accept records with the expected schema and measurement record. The
    runner checks strategy and split identity before using each report.
    """
    index: Dict[Any, dict] = {}
    for rec in records or ():
        cid = (rec or {}).get("champion_id")
        report = ((rec or {}).get("economics") or {}).get("stress_report")
        if (cid and isinstance(report, dict) and report.get("schema") == "transaction-cost-stress-v1"
                and isinstance(report.get("measurement_record"), dict)):
            index[cid] = report

    def resolve(champion_record: dict) -> Optional[dict]:
        return index.get((champion_record or {}).get("champion_id"))

    return resolve
