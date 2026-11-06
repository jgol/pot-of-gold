"""The same bounded diagnostic access for the court agents and single-model control."""
from copy import deepcopy
import hashlib
import json

ACCESS_VERSION = "shared-bounded-diagnostics-v2"
OBSERVATIONS = {
    "entry_boundary_economics": ["entry_boundary_pnl"],
    "entry_boundary_stability": ["entry_boundary_behavior"],
    "transaction_cost_sensitivity": ["cost_behavior"],
    "training_economics": ["cost_pnl"],
    "training_profit_recurrence": ["profit_recurrence", "profit_concentration_denominators"],
    "day_profit_concentration": ["positive_day_profit_concentration"],
    "cost_erosion": ["cost_erosion"],
    "exit_behavior_profile": ["observed_exit_profile"],
    "regime_concentration": ["fired_regime_concentration"],
    "signal_reliance": ["single_signal_ablation"],
    "dependence_preserving_control": ["single_signal_ablation"],
    "within_regime_repeatability": ["temporal_firing_occupancy"],
    "threshold_sensitivity": ["threshold_perturbation"],
    "basis_projection_stability": ["basis_perturbation"],
    "matched_allegation_control:regime_concentration": ["matched_regime_concentration"],
    "matched_allegation_control:boundary_brittleness": ["matched_boundary_brittleness"],
}
UNAVAILABLE_EXPERIMENTS = ["joint_cost_boundary_stress", "exit_rule_stress", "sizing_rule_stress"]


def peer_context(case, cohort_stats):
    from potofgold.multi_agent.assessment.court_compute import peer_compare
    result = {}
    for key, value in case.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and (cohort_stats or {}).get(key):
            item = peer_compare(key, value, cohort_stats)
            if item.get("percentile") is not None:
                result[key] = {"value": value, "cohort_pctile": round(item["percentile"], 3),
                               **{k: item.get(k) for k in ("cohort_median", "cohort_p10", "cohort_p90")}}
    return result


class DiagnosticSession:
    def __init__(self, cf, runner, records, *, evidence=None, allowed=None, max_requests=3):
        from potofgold.multi_agent.assessment import court as court
        from potofgold.multi_agent.diagnostics.vulnerability_catalogue import REGISTRY
        self.cf, self.runner, self.evidence = cf, runner, evidence
        self.records, self.cache = deepcopy(records), {}
        self.max_requests, self.n_requests = max_requests, 0
        structural = {k for k, v in court._case_structural_applicability(cf).items() if v["applicable"]}
        structural |= {r["test_id"] for r in records}
        available = set(getattr(runner, "available_diagnostics", REGISTRY))
        self.allowed = structural & available & (set(allowed) if allowed is not None else set(REGISTRY))
        if not callable(runner):
            self.allowed = set()
        default_matched = {"regime_concentration"}
        if int(getattr(cf, "n_evolved_thresholds", 0) or 0) > 0:
            default_matched.add("boundary_brittleness")
        self.matched = (set(getattr(runner, "available_matched_allegations", default_matched))
                        & {"regime_concentration", "boundary_brittleness"}
                        if callable(getattr(runner, "run_matched_control", None)) else set())
        for record in self.records:
            self.cache[court._measurement_key(record["test_id"])] = record
        self.refresh()

    def refresh(self):
        from potofgold.multi_agent.assessment import court as court
        self.citation_info = court._annotate_measurements(self.records, evidence=self.evidence,
            arm=getattr(self.cf, "arm", None), scope=(self.evidence or {}).get("scope", "all"))
        self.measured = court._measured_exhibits(self.records, citation_info=self.citation_info)

    @property
    def request_ids(self):
        return sorted(self.allowed | {"matched_allegation_control:" + k for k in self.matched})

    @property
    def unmeasured_ids(self):
        from potofgold.multi_agent.assessment.court import _measurement_key
        return [k for k in self.request_ids if _measurement_key(k) not in self.cache]

    def _budget(self, key):
        if self.n_requests >= self.max_requests:
            return {"test_id": key, "decision": "review", "reason": "request_budget_exhausted"}
        self.n_requests += 1
        return None

    def question_status(self, test_id, observation):
        """A cached test name does not establish that a different requested experiment was run."""
        if observation and observation not in OBSERVATIONS.get(test_id, []):
            return "observation_unavailable_from_named_test"
        cached = self.cache.get(test_id)
        if cached is not None:
            if not observation:
                return "requested_observation_not_established_by_cached_test"
            if not any(m.get("citeable") and m.get("result_id") == cached.get("result_id")
                       for m in self.citation_info.values()):
                return "supplied_measurement_unusable"
            return "observation_available_in_supplied_evidence"
        return None if test_id in self.unmeasured_ids else "measurement_unavailable"

    def capability_context(self):
        return {"tests": {k: {"observations": OBSERVATIONS.get(k, []),
                              "state": "supplied" if k in self.cache else "available_new_measurement"}
                          for k in sorted(set(self.request_ids) | set(self.cache))},
                "unavailable_experiments": UNAVAILABLE_EXPERIMENTS,
                "limitations": ["Signal-margin half-consistency cannot be reconstructed from a daily PnL ledger.",
                    "Regime firing concentration is not profit recurrence; exit profile is not exit/sizing stress.",
                    "Separate cost and boundary scenarios do not measure joint stress."]}

    def request(self, inp):
        from potofgold.multi_agent.assessment import court as court
        inp = inp or {}
        tid = inp.get("test_id")
        observation = inp.get("requested_observation")
        if observation and observation not in OBSERVATIONS.get(tid, []):
            return json.dumps({"test_id": tid, "reason": "observation_unavailable_from_named_test",
                               "observations_this_test_supplies": OBSERVATIONS.get(tid, [])})
        if tid not in self.allowed or not callable(self.runner):
            return json.dumps({"test_id": tid, "decision": "review", "reason": "test_not_in_allowed_set"})
        key = court._measurement_key(tid)
        if key in self.cache:
            return json.dumps({**court._agent_record_view(self.cache[key]), "cached": True}, default=str)
        refused = self._budget(key)
        if refused:
            return json.dumps(refused)
        try:
            raw = self.runner(tid, inp)
            record = court._scrub_leaky_exhibit(court._record_exhibit(raw, tid))
        except Exception as exc:
            record = {"test_id": tid, "decision": "review", "reason": f"runner_error:{type(exc).__name__}"}
            self.cache[key] = record
            return json.dumps(record)
        record["committed"] = court._safe_precommit(inp)
        record["requested_interpretation"] = {k: inp.get(k) for k in
            ("resolves_uncertainty", "keep_if", "reject_if", "inconclusive_if") if k in inp}
        self.cache[key] = record
        self.records.append(record)
        self.refresh()
        return json.dumps(court._agent_record_view(record), default=str)

    def lookup(self, inp):
        from potofgold.multi_agent.strategy_profiles.tool_evidence import record_evidence_lookup, development_evidence_view
        rid = str((inp or {}).get("result_id", "")).split("::")[0]
        matches = [r for r in self.records if r.get("result_id") == rid]
        if len(matches) != 1:
            return json.dumps({"result_id": rid, "dev_status": "out_of_scope", "reason": "unknown or ambiguous result_id"})
        lookup = record_evidence_lookup(matches[0], evidence=self.evidence, arm=getattr(self.cf, "arm", None),
                                       scope=(self.evidence or {}).get("scope", "all"))
        result = {"result_id": rid, **development_evidence_view(lookup)}
        matches[0]["development_evidence"] = result
        self.refresh()
        return json.dumps(result, default=str)

    def matched_control(self, inp):
        from potofgold.multi_agent.assessment import court as court
        allegation = str((inp or {}).get("allegation", "")).strip()
        key = "matched_allegation_control:" + allegation
        if allegation not in self.matched:
            return json.dumps({"verdict": "INCONCLUSIVE", "reason": "allegation_not_applicable_or_unavailable"})
        if key in self.cache:
            return json.dumps({**court._agent_record_view(self.cache[key]), "cached": True}, default=str)
        refused = self._budget(key)
        if refused:
            return json.dumps(refused)
        try:
            out = self.runner.run_matched_control(allegation)
        except Exception as exc:
            out = {"verdict": "INCONCLUSIVE", "reason": f"runner_error:{type(exc).__name__}"}
        numeric = {k: v for k, v in out.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                   and k in ("p_value", "rank", "center_local_phi", "t_observed", "n_valid_neighbours", "k_null")}
        rid = hashlib.sha256(json.dumps(["matched", allegation, out], sort_keys=True, default=str).encode()).hexdigest()[:16]
        verdict = str(out.get("verdict", "INCONCLUSIVE"))
        record = court._scrub_leaky_exhibit({"result_id": rid, "test_id": key,
            **{k: deepcopy(out[k]) for k in ("test_version", "parameters", "replay_fidelity", "data_split_hash") if k in out},
            "measured_statistics": numeric, "control_results": {k: v for k, v in out.items() if k not in numeric},
            **court._matched_verdict_mapping(verdict), "calibration_id": None,
            "self_calibrating": verdict in ("EXCEPTIONAL", "NOT_EXCEPTIONAL"), "verdict": verdict})
        self.cache[key] = record
        self.records.append(record)
        self.refresh()
        return json.dumps(court._agent_record_view(record), default=str)

    def tools_and_handlers(self):
        from potofgold.multi_agent.assessment.court import _REQUEST_CATALOGUE_TEST
        from potofgold.multi_agent.strategy_profiles.diagnostic_evidence import EVIDENCE_LOOKUP_TOOL
        from potofgold.multi_agent.diagnostics.matched_controls import MATCHED_CONTROL_TOOL
        tools, handlers = [], {}
        if callable(self.runner) and self.allowed:
            tool = deepcopy(_REQUEST_CATALOGUE_TEST)
            tool["input_schema"]["properties"]["test_id"]["enum"] = sorted(self.allowed)
            tool["input_schema"]["properties"]["requested_observation"] = {"type": "string",
                "description": "One exact observation capability from the supplied menu; a test cannot answer a different experiment."}
            tools.append(tool)
            handlers["request_catalogue_test"] = self.request
        if self.evidence is not None and (self.records or callable(self.runner)):
            tools.append(deepcopy(EVIDENCE_LOOKUP_TOOL))
            handlers["diagnostic_evidence_lookup"] = self.lookup
        if self.matched:
            tool = deepcopy(MATCHED_CONTROL_TOOL)
            tool["input_schema"]["properties"]["allegation"]["enum"] = sorted(self.matched)
            tools.append(tool)
            handlers["matched_allegation_control"] = self.matched_control
        return tools, handlers

    def instructions(self):
        from potofgold.multi_agent.diagnostics.vulnerability_catalogue import list_catalogue
        menu = [k for k in sorted(self.allowed) if k != "dependence_preserving_control" or "signal_reliance" not in self.allowed]
        return ("Shared bounded diagnostic access. Read existing measurements first. At most "
                f"{self.max_requests} distinct new catalogue/matched measurements combined; repeats are cached. "
                "Request only a factual observation that could change allocation. An unavailable observation about "
                "future performance remains a limitation. No requirement to spend the budget. "
                "Use exact result_id::statistic aliases returned by tools; failed measurements are inconclusive. "
                "Development discrimination, return, tail and winner preservation are separate dimensions; "
                "predictive validation is not permission to reason. No matched-null result automatically dictates an action. "
                "Catalogue: " + json.dumps(list_catalogue(allow=menu), default=str)
                + "; matched allegations: " + json.dumps(sorted(self.matched))
                + "; exact observation capabilities (test_id -> requested_observation): "
                + json.dumps({k: OBSERVATIONS.get(k, []) for k in sorted(set(self.request_ids) | set(self.cache))})
                + "; experiments not implemented by these tools: " + json.dumps(UNAVAILABLE_EXPERIMENTS)
                + ". Regime firing concentration does not measure profit recurrence. Exit profiles do not test changed exits or sizing. "
                  "Separate cost and boundary grids are not joint stresses. training_profit_recurrence, when supplied, "
                  "answers chronological training-profit questions but not signal-margin half-consistency. "
                  "Name missing experiments honestly; do not route them to a similarly named cached test.")
