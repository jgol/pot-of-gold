"""Run court and single-LLM assessments from training profiles.

Check inputs before model calls, record decisions and transcripts, and resume
only when the source, inputs and settings match the previous run.
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import math
import signal
import sys

import yaml
from copy import deepcopy
from pathlib import Path

_ALL_DROP_TRIPWIRE = 5   # Stop after a run of strategies produces no usable judge responses.
_PROVIDER_FALLBACK_TRIPWIRE = 3   # Stop repeated provider failures that would otherwise keep using backups.

# Estimated USD per million tokens, used for the run's spending checks.
_PRICE_PER_MTOK = {
    "anthropic": {"input": 15.0, "cache_read": 1.50, "cache_write": 18.75, "output": 75.0},
    "openai":    {"input":  1.25, "cache_read": 0.125, "cache_write": 0.0,  "output": 10.0},
    "gemini":    {"input":  1.25, "cache_read": 0.3125, "cache_write": 0.0, "output": 10.0},
}


def _est_cost_usd(spent_by_provider: dict) -> float:
    """Estimate cost from token usage and the configured per-provider rates."""
    total = 0.0
    for prov, tok in (spent_by_provider or {}).items():
        rates = _PRICE_PER_MTOK.get(prov, _PRICE_PER_MTOK["anthropic"])
        for k in ("input", "cache_read", "cache_write", "output"):
            total += (int(tok.get(k, 0) or 0) / 1_000_000.0) * rates.get(k, 0.0)
    return total


def _spend_from_audit(audit_records):
    """Restore court and single-LLM usage separately from saved audit records.

    Include failed attempts with recorded usage so resumed runs retain their costs.
    """
    from potofgold.multi_agent.assessment.providers import provider_of
    def _zero():
        return {"input": 0, "output": 0, "calls": 0, "cache_read": 0, "cache_write": 0}
    court, court_by, base, base_by = _zero(), {}, _zero(), {}
    for ar in (audit_records or []):
        if ar.get("outcome") in ("single_llm_baseline", "single_llm_baseline_error"):
            bm = ar.get("metadata") or {}
            bp = base_by.setdefault(provider_of(ar.get("model") or ""), {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
            for sk, mk in (("input", "input_tokens"), ("output", "output_tokens"),
                           ("cache_read", "cache_read_input_tokens"), ("cache_write", "cache_creation_input_tokens")):
                n = int(bm.get(mk, 0) or 0); base[sk] += n; bp[sk] += n
            base["calls"] += 1
            continue                                         # Track comparator usage separately from court usage.
        for k in court:
            court[k] += int((ar.get("tokens") or {}).get(k, 0) or 0)
        for rec in (ar.get("agent_transcripts") or []):
            b = court_by.setdefault(provider_of(rec.get("model") or ""),
                                    {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
            b["input"] += int(rec.get("input_tokens", 0) or 0)
            b["output"] += int(rec.get("output_tokens", 0) or 0)
            b["cache_read"] += int(rec.get("cache_read_input_tokens", 0) or 0)
            b["cache_write"] += int(rec.get("cache_creation_input_tokens", 0) or 0)
    return court, court_by, base, base_by


def _allocation_explanation(action, multiplier, *, status=None, fallback=None, rationale=None, votes=None):
    """Summarize the percentage recommendation, vote status and recorded reasoning."""
    from potofgold.multi_agent.assessment.allocation_reasoning import allocation_label
    label = allocation_label(action)
    parts = [f"{label} ({100 * multiplier:g}%)" if multiplier is not None else label]
    if action == "CASH":
        parts.append("unallocated capital remains in cash")
    if votes:
        _v = ", ".join(f"{k}={n}" for k, n in votes.items() if n)
        if _v:
            parts.append(f"action votes: {_v}")
    if status:
        parts.append(f"status={status}")
    if fallback:
        parts.append(f"fallback={fallback}")
    if rationale and str(rationale).strip():
        from potofgold.multi_agent.assessment.allocation_reasoning import sentence_summary
        parts.append(sentence_summary(rationale))
    return "; ".join(parts)


def _court_allocation_explanation(v):
    """Explain the median percentage recommendation and any explicit fallback."""
    action, mult = v.allocation_action, v.allocation_multiplier
    if v.allocation_fallback_reason:
        return _allocation_explanation(action, mult, fallback=v.allocation_fallback_reason, votes=v.allocation_votes)
    recommendations = [j for j in v.panel if j.get("allocation_status") == "explicit"]
    sizes = ", ".join(f"{j['allocation_percent']:g}%" for j in recommendations)
    reasons = " | ".join(f"{j['allocation_percent']:g}%: {j.get('allocation_rationale', '')}" for j in recommendations)
    return _allocation_explanation(action, mult, status="explicit", rationale=
        f"Median of valid judge recommendations ({sizes}). {reasons}")


def _investigation_queue(v):
    from potofgold.multi_agent.assessment.allocation_reasoning import investigation_queue
    return investigation_queue(v)


def _preflight_probe(models, probe_fn=None):
    """Check connectivity and request parameters once per configured model.

    Expand fallback chains and stop at the first failed probe. These text-only
    requests check API access, not completion of the assessment protocol.
    """
    if probe_fn is None:
        from potofgold.multi_agent.assessment.agents import task_sync as probe_fn
    # Check fallback model IDs as well as primary models.
    for m in sorted({mid.strip() for spec in models if spec for mid in str(spec).split(">") if mid.strip()}):
        try:
            probe_fn(role="probe", user_message="Reply with the single word: OK.",
                     system_prompt="You are a connectivity probe. Reply with OK.",
                     tools=[], tool_handlers={}, model=m)
        except Exception as e:                              # noqa: BLE001
            raise SystemExit(f"[panel] PRE-FLIGHT FAILED for model {m!r}: {type(e).__name__}: {e}\n"
                             f"        fix the model id / API key / parameters before spending on the cohort")
        print(f"[panel] pre-flight OK: {m}", flush=True)


def _with_backup(m, *, enable=True):
    """Append the configured cross-provider fallback unless the caller supplied a chain."""
    if not m or not enable or ">" in str(m):
        return m
    from potofgold.multi_agent.assessment.providers import provider_of
    prim = str(m).strip()
    anchor = "gpt-5.6-terra" if provider_of(prim) == "anthropic" else "claude-opus-4-8"
    return prim if prim == anchor else f"{prim}>{anchor}"


def _resume_incompat(prev_fp, prev_dry, cur_fp, cur_dry):
    """Return reasons persisted verdicts cannot be reused under this run's identity."""
    reasons = []
    if bool(prev_dry) != bool(cur_dry):
        reasons.append(f"run mode changed (dry_run {prev_dry} -> {cur_dry}) = mock<->live")
    if prev_fp is None:
        reasons.append("prior output has no verified run fingerprint")
    elif prev_fp != cur_fp:
        reasons.append("run fingerprint changed (source / models / policy / cohort / evidence contents)")
    return reasons


def _configured_arguments(argv):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config", type=Path)
    options, remaining = parser.parse_known_args(argv)
    arguments = []
    if options.config is not None:
        config = yaml.safe_load(options.config.read_text())
        if not isinstance(config, dict) or set(config) != {"protocol", "arguments"}:
            parser.error("Court config requires protocol and arguments")
        if config["protocol"] != "training-informed-research-v12-no-analyst":
            parser.error("Use the v12 protocol without the analyst")
        if not isinstance(config["arguments"], dict):
            parser.error("arguments must be a mapping of original panel options")
        supplied = {value.split("=", 1)[0] for value in remaining if value.startswith("--")}
        for key, value in config["arguments"].items():
            flag = "--" + key.replace("_", "-")
            if flag in supplied or value is None or value is False:
                continue
            arguments.append(flag)
            if value is not True:
                arguments.append(",".join(map(str, value)) if isinstance(value, list) else str(value))
    return arguments + remaining


def main(argv=None):
    argv = _configured_arguments(argv)
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--champions", required=True)
    ap.add_argument("--exhibits", help="Combined training exhibits from potofgold-court-prepare")
    ap.add_argument("--g5", default=None)
    ap.add_argument("--g6", default=None)
    ap.add_argument("--g7", default=None)
    ap.add_argument("--intraday", default=None)
    ap.add_argument("--arm", default="C")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--peer-reference", default=None, help="Pinned cause-only peer reference, independent of execution batch")
    ap.add_argument("--peer-reference-sha256", default=None, help="Required content pin for --peer-reference")
    ap.add_argument("--champion-ids", default=None,
                    help="comma-separated champion_ids to run EXACTLY (applied after the --arm filter). Refuses if any "
                         "requested id is absent; cannot be combined with --limit.")
    ap.add_argument("--catalogue-runner-factory", default=None,
                    help="dotted 'module:attr' -> factory(champion_record) -> run_catalogue_test|None. "
                         "Enables diagnostic execution for all court roles and the single-LLM baseline.")
    ap.add_argument("--evidence-store", default=None,
                    help="development-reference EvidenceStore JSON. Lets the agents look "
                         "up whether a diagnostic separated held-out winners/losers on the DEV split. REQUIRES "
                         "--catalogue-runner-factory (a lookup keys off an executed test); absent => the "
                         "diagnostic_evidence_lookup tool stays inert.")
    ap.add_argument("--evidence-version", default="court-v2-boundary-5diag",
                    help="The frozen development rule-book version; must match the supplied store.")
    ap.add_argument("--evidence-scope", default="all",
                    help="Population scope of the frozen development rule book.")
    ap.add_argument("--evidence-mode", choices=("training_economics",), default="training_economics",
                    help="Use the reported training-economics evidence; held-out outcomes remain hidden.")
    ap.add_argument("--decision-policy", choices=("research_prediction",), default="research_prediction",
                    help="Research predictions use valid cited training evidence; development support informs weight.")
    ap.add_argument("--catalogue-allow", default=None,
                    help="comma-separated catalogue test_ids the court agents may request (trims the menu). "
                         "Absent => ALL. Use it to DISABLE the inert placeholders so no cycle is spent on them, e.g. "
                         "--catalogue-allow within_regime_repeatability (matched_allegation_control is separate and "
                         "always offered when its runner is wired).")
    ap.add_argument("--n-judges", type=int, default=3)
    ap.add_argument("--rounds", type=int, choices=(1,), default=1, help="The single H3 prosecution/defense round")
    ap.add_argument("--model", default=None, help="fallback model for any role not set below")
    ap.add_argument("--pros-model", default=None, help="prosecutor model (default: --model, else claude-opus-4-8)")
    ap.add_argument("--def-model", default=None, help="defender model (default: --model, else claude-opus-4-8)")
    ap.add_argument("--judge-model", default=None,
                    help="one primary model repeated across --n-judges seats; exclusive with --judge-models")
    ap.add_argument("--judge-models", default=None,
                    help="comma-separated one model PER judge, e.g. 'claude-opus-4-8,<openai-id>,<gemini-id>' "
                         "(default: --model, else claude-opus-4-8). Each judge (and --pros/def-model) may "
                         "be a '>'-separated FALLBACK CHAIN, e.g. 'gpt-5>gpt-4o-mini>claude-opus-4-8' — on a "
                         "quota/auth error the court walks to the next id (put Anthropic last as the anchor).")
    ap.add_argument("--preflight-only", action="store_true",
                    help="Prepare every selected training packet and write a receipt; no provider calls or verdicts.")
    ap.add_argument("--preflight-out", default=None, help="Preflight receipt path; defaults beside --out.")
    ap.add_argument("--preflight-receipt", default=None,
                    help="Verified receipt from --preflight-only; required for a full paid research run.")
    ap.add_argument("--no-backup", action="store_true",
                    help="Disable the per-role BACKUP model (ON by default): normally each agent gets a "
                         "'primary>anchor' chain so a failure (provider error OR a cite-rejected/empty verdict) "
                         "falls to a reliable anchor before any drop. Pass this to run primaries-only (no fallback).")
    ap.add_argument("--max-output-tokens", type=int, default=0,
                    help="Cumulative OUTPUT-token budget ceiling; abort (flushing earned verdicts) once the "
                         "cohort spend crosses it. One in-flight champion may exceed the ceiling. 0 = no cap.")
    ap.add_argument("--max-cost-usd", type=float, default=0.0,
                    help="ESTIMATED cumulative dollar ceiling (public list prices, see _PRICE_PER_MTOK); abort "
                         "(flushing earned verdicts) once the running estimate crosses it. 0 = no cap. Estimate only — "
                         "not a billing guarantee; one in-flight champion may exceed the ceiling. Probes are excluded.")
    ap.add_argument("--no-resume", action="store_true",
                    help="force a FRESH run — ignore + OVERWRITE any existing --out verdicts file. By default the "
                         "runner RESUMES: champions already scored in --out are skipped (a --limit smoke then a full "
                         "re-run is additive, not a re-pay). Combined with per-champion flush, the file grows in place.")
    ap.add_argument("--allocation-policy", choices=("percentage_v1",), default="percentage_v1",
                    help="The reported percentage recommendation contract.")
    ap.add_argument("--exploratory-ranking", action="store_true", default=True,
                    help="Require each judge's ordinal 0-100 risk score and report the panel median; "
                         "scores are separate from forecasts and allocation votes.")
    ap.add_argument("--single-llm-baseline", action="store_true",
                    help="ALSO run the single-LLM baseline (single_llm.assess_strategy) per champion on the "
                         "SAME packet, peer context and bounded diagnostic access as the court; it uses one agent. "
                         "The court-vs-baseline comparison; baseline results are recorded in the audit trail.")
    ap.add_argument("--one-llm-model", default=None,
                    help="single-LLM baseline model (default: single_llm.DEFAULT_MODEL).")
    ap.add_argument("--out", default=None,
                    help="verdicts output; default derives from the --champions stem so same-arm cohorts "
                         "(cs0/cs4/seed2 are all arm C) don't silently overwrite each other")
    a = ap.parse_args(argv)
    if a.exhibits and any((a.g5, a.g6, a.g7, a.intraday)):
        ap.error('Use --exhibits or the separate --g5/--g6/--g7/--intraday files')
    if a.judge_model and a.judge_models:
        ap.error("--judge-model and --judge-models are mutually exclusive")
    if a.champion_ids and a.limit:
        ap.error("--champion-ids and --limit are mutually exclusive: --limit truncates the pinned set (first-N by id), "
                 "silently running fewer than the pinned members. Drop --limit.")
    if not (math.isfinite(a.max_cost_usd) and a.max_cost_usd >= 0 and a.max_output_tokens >= 0
            and (a.max_cost_usd > 0 or a.max_output_tokens > 0)):
        ap.error("research prediction requires a positive finite spending or token ceiling")
    # Respect logging configuration established by the calling application.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not a.out:
        a.out = f"runs/{Path(a.champions).stem}/court.json"

    from potofgold.multi_agent.strategy_profiles.generalization_case_file import build_cohort_case_files
    from potofgold.multi_agent.assessment.court import validate_generalization
    from potofgold.multi_agent.assessment.court_citations import MalformedVerdict

    # Resume decisions and their audit records together.
    _resume_verdicts, _resume_audit, _resume_ids = [], [], set()
    _baseline_ids = set()
    _resume_existing = False
    _resume_fp, _resume_dry = None, None
    if not a.no_resume and a.out and Path(a.out.format(arm=a.arm)).exists():
        _resume_existing = True
        _rp = a.out.format(arm=a.arm)
        try:
            _prev = json.loads(Path(_rp).read_text())
        except (ValueError, OSError) as _e:
            raise SystemExit(f"[panel] REFUSING resume: cannot read {_rp}: {_e}. "
                             "Use a fresh --out or explicitly --no-resume after inspecting the artifact.") from _e
        _resume_fp = _prev.get("run_fingerprint")
        _resume_dry = _prev.get("dry_run")
        from potofgold.multi_agent.assessment.court import GeneralizationVerdict as _GV
        _vfields = set(_GV.__dataclass_fields__)
        for _v in _prev.get("verdicts", []):
            if not isinstance(_v, dict) or not _v.get("champion_id"):
                continue
            _resume_verdicts.append(_GV(**{k: _v[k] for k in _v if k in _vfields}))
            _resume_ids.add(_v["champion_id"])
        _rn = Path(_rp).name
        _ran = (_rn.replace("verdicts", "audit") if "verdicts" in _rn else Path(_rn).stem + "_audit" + Path(_rn).suffix)
        _rap = Path(_rp).with_name(_ran)
        if _rap.exists():
            try:
                _prior_audit = json.loads(_rap.read_text())
                if (_prior_audit.get("run_fingerprint") != _resume_fp
                        or _prior_audit.get("dry_run") != _resume_dry):
                    raise ValueError("audit and verdict fingerprints differ")
                # Failed attempts retain their usage after a successful retry.
                _resume_audit = [r for r in _prior_audit.get("champions", []) if isinstance(r, dict)]
            except (ValueError, OSError) as _e:
                raise SystemExit(f"[panel] REFUSING resume: incompatible audit {_rap}: {_e}") from _e
        elif _resume_ids or _prev.get("spend", {}).get("calls"):
            raise SystemExit(f"[panel] REFUSING resume: missing audit {_rap}; prior spend cannot be verified.")
        if len(_resume_ids) != len(_resume_verdicts):
            raise SystemExit("[panel] REFUSING resume: duplicate persisted champion verdicts.")
        if _resume_ids - {r.get("champion_id") for r in _resume_audit if r.get("outcome") == "verdict"}:
            raise SystemExit("[panel] REFUSING resume: one or more verdicts have no matching audit record.")
        _prior_reasoning = {r["champion_id"]: r for r in _resume_audit if r.get("outcome") == "verdict"}
        for _v in _resume_verdicts:
            # Restore full reasoning from the audit before rebuilding decision summaries.
            for _key in ("panel", "analysis", "prosecution", "defense"):
                if _key in _prior_reasoning[_v.champion_id]:
                    setattr(_v, _key, deepcopy(_prior_reasoning[_v.champion_id][_key]))
        _baseline_ids = {r.get("champion_id") for r in _resume_audit
                         if r.get("outcome") == "single_llm_baseline"}
        if _resume_ids:
            print(f"[panel] RESUME: {len(_resume_ids)} champion(s) already scored in {_rp} — skipping them "
                  f"(pass --no-resume for a fresh run)", flush=True)

    def _res(p):
        return json.loads(Path(p).read_text()).get("results", []) if p and Path(p).exists() else []

    input_members = json.loads(Path(a.champions).read_text())
    members = [m for m in input_members if m.get("arm") == a.arm]
    members.sort(key=lambda m: m["champion_id"])
    # Compute peers from the full input cohort, independently of execution limits.
    exhibits = json.loads(Path(a.exhibits).read_text()) if a.exhibits else {
        kind: _res(getattr(a, kind)) for kind in ('g5', 'g6', 'g7', 'intraday')}
    all_cfs = build_cohort_case_files(members, *(exhibits[kind] for kind in ('g5', 'g6', 'g7', 'intraday')))
    if a.champion_ids:
        _want = [c.strip() for c in a.champion_ids.split(",") if c.strip()]
        _have = {m.get("champion_id") for m in members}
        _absent = [c for c in _want if c not in _have]
        if _absent:
            raise SystemExit(f"[panel] --champion-ids: {len(_absent)} requested id(s) not in arm {a.arm} of "
                             f"{a.champions}: {_absent[:5]}")
        _want_set = set(_want)
        members = [m for m in members if m.get("champion_id") in _want_set]
    members.sort(key=lambda m: m["champion_id"])
    if not members:
        raise SystemExit(f"[panel] no champions for arm {a.arm} in {a.champions}")
    _selected_ids = {m["champion_id"] for m in members}
    cfs = [cf for cf in all_cfs if cf.champion_id in _selected_ids]
    if a.limit:
        cfs = cfs[:a.limit]
    members_by_id = {m.get("champion_id"): m for m in members}
    _rct_factory = None
    if a.catalogue_runner_factory:
        import importlib
        _mod, _attr = a.catalogue_runner_factory.split(":", 1)
        _rct_factory = getattr(importlib.import_module(_mod), _attr)
        print(f"[panel] catalogue runner ENABLED via {a.catalogue_runner_factory}", flush=True)
    _evidence = None
    if a.evidence_store:
        if not a.catalogue_runner_factory:
            print("[panel] WARNING: --evidence-store is set but --catalogue-runner-factory is NOT — "
                  "diagnostic_evidence_lookup stays INERT (nothing to look up without live tests).", flush=True)
        else:
            from potofgold.multi_agent.strategy_profiles.diagnostic_evidence import EvidenceStore
            from potofgold.multi_agent.input_tools.seed_firewall import firewall_manifest_id, load_manifest
            _ev_store = EvidenceStore.load(a.evidence_store)
            if len(_ev_store) == 0:
                raise SystemExit(f"[panel] --evidence-store {a.evidence_store} loaded 0 records (empty/corrupt) — "
                                 f"supply a development reference bound to the same cohort split.")
            _dev_manifest_id = firewall_manifest_id(load_manifest())
            _stored_ids = _ev_store.dev_manifest_ids()
            if _stored_ids != {_dev_manifest_id}:
                raise SystemExit(
                    f"[panel] evidence-store dev_manifest_id mismatch: store was frozen under {sorted(_stored_ids)} but "
                    f"this environment recomputes {_dev_manifest_id!r} (canonical cohort files unreadable here? "
                    f"re-freeze in this environment, or restore results/ so firewall_manifest_id hashes membership).")
            _evidence = {"store": _ev_store, "version": a.evidence_version, "dev_manifest_id": _dev_manifest_id,
                         "scope": a.evidence_scope}
            print(f"[panel] evidence-store ENABLED: {len(_ev_store)} records, version={a.evidence_version}, "
                  f"dev_manifest_id={_dev_manifest_id}", flush=True)
    if _evidence is None:
        raise SystemExit("[panel] H3 assessment requires --evidence-store and --catalogue-runner-factory.")
    # Check descriptions for every genome token before making model calls.
    from potofgold.multi_agent.strategy_profiles import terminal_registry as _treg
    _all_ident = set()
    for m in members:
        _g = (m.get("member") or {}).get("genome") or m.get("genome") or {}
        _all_ident |= _treg.identifiers(*[_g.get(k) for k in
                        ("entry_tree", "exit_tree", "size_tree", "delta_tree", "width_tree", "shape_tree")])
    _treg.assert_covers(_all_ident)
    run_agent = None
    print(f"[panel] {len(cfs)} champions | arm {a.arm} | judges {a.n_judges} | "
          f"{'PREFLIGHT (no provider calls)' if a.preflight_only else 'LIVE LLM'}", flush=True)
    ids = {cf.champion_id for cf in cfs}
    cov = {"margins": sum(cf.margins_available for cf in cfs),
           "ablation": sum(cf.ablation_available for cf in cfs),
           "regime": sum(cf.regime_available for cf in cfs),
           "intraday": sum(cf.intraday_available for cf in cfs),
           "rule_glossed": sum(bool(cf.rule_glossary) for cf in cfs)}
    print(f"[panel] exhibit coverage /{len(cfs)}: {cov}", flush=True)
    if all(v == 0 for v in cov.values()):
        print("[panel] WARNING: NO champion has ANY margin/perturbation/regime exhibit — wrong --g5/--g6/--g7 paths?", flush=True)

    # Run structural checks first, then verify each strategy's complete measurement packet.
    from potofgold.multi_agent.assessment.court_readiness import assert_court_ready
    from potofgold.multi_agent.strategy_profiles.generalization_case_file import resolve_basis_targets
    _by_id = {m["champion_id"]: m for m in members}
    _trip = []
    for cf in cfs:
        _m = _by_id.get(cf.champion_id, {})
        _g = (_m.get("member") or {}).get("genome") or _m.get("genome") or {}
        _trip.append((cf, [_g.get(k) for k in ("entry_tree", "exit_tree", "size_tree",
                                               "delta_tree", "width_tree", "shape_tree")],
                      resolve_basis_targets(_m.get("provenance") or {})))
    _gate = assert_court_ready(_trip, require_measurements=a.evidence_mode != "training_economics")
    _readiness_stage = "STRUCTURE READY; measurements pending"
    print(f"[panel] court-readiness GATE: {_readiness_stage} {_gate['n_ready']}/{_gate['n']} ready, "
          f"cite-enforcement ok={_gate['cite_enforcement']['ok']}", flush=True)

    def _check_training_packet(cf, runner):
        from potofgold.multi_agent.assessment.court import build_measurement_context
        from potofgold.multi_agent.assessment.court_readiness import check_champion
        packet = build_measurement_context(cf, runner, evidence=_evidence, evidence_mode=a.evidence_mode)
        model_detail = {}
        from potofgold.multi_agent.strategy_profiles.evidence_cards import validate_model_packet
        model_detail = validate_model_packet(packet)
        _, trees, targets = next(item for item in _trip if item[0].champion_id == cf.champion_id)
        readiness = check_champion(cf, trees, targets, measurement_context=packet)
        if not readiness["ready"]:
            raise ValueError("Training measurement readiness failed: " + "; ".join(readiness["issues"]))
        return {"ready": True, **model_detail, "records": [{"test_id": r.get("test_id"),
            "applicability": r.get("applicability"), "decision": r.get("decision"),
            "measurement_status": (r.get("measurement_interpretation") or {}).get("status")}
            for r in packet["records"]]}

    # Preserve failure counts and usage when a partial run resumes.
    court_drops = sum(1 for r in _resume_audit if r.get("outcome") == "court_drop")
    provider_errors = sum(1 for r in _resume_audit if r.get("outcome") == "provider_error")
    degraded_panels = sum(1 for r in _resume_audit
                          if r.get("outcome") == "verdict" and ((r.get("disagreement") or {}).get("degraded")))
    baseline_errors = sum(1 for r in _resume_audit if r.get("outcome") == "single_llm_baseline_error")
    verdicts = list(_resume_verdicts)
    audit_records = list(_resume_audit)
    _spent, _spent_by_provider, _baseline_spent, _baseline_by_provider = _spend_from_audit(_resume_audit)
    if _resume_audit:
        print(f"[panel] RESUME spend carried from prior run: court input_tok={_spent['input']} output_tok={_spent['output']} "
              f"cache_read={_spent['cache_read']} cache_write={_spent['cache_write']} (~${_est_cost_usd(_spent_by_provider):.2f}); "
              f"baseline output_tok={_baseline_spent['output']} (~${_est_cost_usd(_baseline_by_provider):.2f})", flush=True)
    _pros = a.pros_model or a.model or "claude-opus-4-8"
    _def = a.def_model or a.model or "claude-opus-4-8"
    # An explicit judge-model list takes precedence over n_judges.
    if a.judge_models:
        _judge_models = [m.strip() for m in a.judge_models.split(",") if m.strip()]
    else:
        _judge_models = [(a.judge_model or a.model or "claude-opus-4-8")] * max(1, a.n_judges)
    if len(_judge_models) < 2:
        print(f"[panel] WARNING: only {len(_judge_models)} judge seat(s) — the two-agreeing-judges rule needs >=2 to "
              f"decide, so EVERY champion will be UNSURE (defer to the filter). Use --n-judges 3 or --judge-models.",
              flush=True)
    # Use a fallback chain unless backup models are disabled.
    _bk = not a.no_backup
    _pros = _with_backup(_pros, enable=_bk)
    _def = _with_backup(_def, enable=_bk)
    _judge_models = [_with_backup(j, enable=_bk) for j in _judge_models]
    _cat_allow = [s.strip() for s in a.catalogue_allow.split(",") if s.strip()] if a.catalogue_allow else None
    if _cat_allow is not None:
        print(f"[panel] catalogue menu RESTRICTED to {_cat_allow} (inert tests disabled — no cycle spent on them)", flush=True)
    _compute = True
    # Peer statistics describe profiles; they are not an allocation rule.
    from potofgold.multi_agent.assessment.court_compute import build_cohort_stats
    _peer_reference_ids = [cf.champion_id for cf in all_cfs]
    if bool(a.peer_reference) != bool(a.peer_reference_sha256):
        raise SystemExit("--peer-reference and --peer-reference-sha256 must be supplied together")
    if a.peer_reference:
        from potofgold.multi_agent.strategy_profiles.peer_reference import load_reference
        _cohort_stats, _peer_reference_ids = load_reference(a.peer_reference,
            expected_sha256=a.peer_reference_sha256, arm=a.arm, cases=cfs)
    else:
        _cohort_stats = build_cohort_stats(all_cfs)
    from potofgold.multi_agent.assessment.reasoning_diagnostics import ACCESS_VERSION
    from potofgold.multi_agent.strategy_profiles.evidence_summaries import SUMMARY_VERSION
    from potofgold.multi_agent.assessment.allocation_reasoning import allocation_details, allocation_label
    _baseline_access_asymmetry = None
    if a.single_llm_baseline:
        _baseline_access_asymmetry = {
            "court_only_access": [], "version": ACCESS_VERSION,
            "baseline_shares": ["frozen_evidence_packet", "preloaded_measurement_context",
                                "development_reference_cards", "bounded_diagnostic_access"]
                               + (["inline_peer_percentiles"] if _compute else []),
            "new_measurement_budget_per_policy": 3,
            "note": "Same access, independently chosen requests. Court uses debate and voting; baseline uses one agent. "
                    "Token use and realized measurements can differ and are recorded."}
        print("[panel] Court and baseline share evidence, peer context and bounded diagnostic access.", flush=True)
    # Source, inputs and settings must match before saved decisions can be reused.
    from potofgold.multi_agent.assessment.run_readiness import assert_runner_ready, build_run_fingerprint
    if not callable(_rct_factory):
        raise SystemExit("[panel] H3 assessment requires an executable --catalogue-runner-factory.")
    if len(_judge_models) != 3:
        raise SystemExit("[panel] H3 assessment requires exactly three judge seats.")
    _run_mode = "research_prediction"
    _fp = build_run_fingerprint(
        config={"dry_run": False, "run_mode": _run_mode, "arm": a.arm,
                "pros": _pros, "def": _def, "judges": _judge_models,
                "diagnostic_access_version": ACCESS_VERSION,
                "exploratory_ranking": bool(a.exploratory_ranking),
                "evidence_summary_version": SUMMARY_VERSION,
                "rounds": a.rounds, "compute": _compute, "catalogue_runner_factory": a.catalogue_runner_factory,
                "peer_reference_ids": _peer_reference_ids, "peer_reference_sha256": a.peer_reference_sha256,
                "catalogue_allow": sorted(_cat_allow) if _cat_allow is not None else None,
                "evidence_version": a.evidence_version, "evidence_scope": a.evidence_scope,
                "evidence_mode": a.evidence_mode, "decision_policy": a.decision_policy,
                "allocation_policy": a.allocation_policy,
                "single_llm_baseline": bool(a.single_llm_baseline),
                "one_llm_model": a.one_llm_model,
                "dev_manifest_id": (_evidence or {}).get("dev_manifest_id")},
        input_paths={key: getattr(a, key) for key in
                     ("champions", "exhibits", "g5", "g6", "g7", "intraday", "evidence_store", "peer_reference")},
        repo_root=Path(__file__).resolve().parents[1], factory=_rct_factory)
    # Prepare every selected strategy before testing provider connectivity.
    from potofgold.multi_agent.assessment.panel_preflight import check_all, verify_receipt
    if a.preflight_only:
        try:
            receipt = check_all(cfs, members_by_id, _rct_factory, _check_training_packet, fingerprint=_fp)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        target = (Path(a.preflight_out) if a.preflight_out else
                  Path(a.out.format(arm=a.arm)).with_name(f"preflight_{a.arm}.json"))
        verdict_target = Path(a.out.format(arm=a.arm))
        if target.resolve() == verdict_target.resolve():
            raise SystemExit("The preflight receipt must use a separate path from verdicts")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(receipt, indent=2, default=str))
        print(f"[panel] PREFLIGHT {receipt['status']}: {receipt['n_ready']}/{receipt['n_attempted']} "
              f"training packets ready, zero provider calls; {target}", flush=True)
        return 0 if receipt["status"] == "ready" else 2
    if not a.preflight_receipt:
        raise SystemExit("Full paid research runs require --preflight-receipt. Run the same command first with "
                         "--preflight-only --preflight-out PATH; this prepares all selected training packets without providers.")
    if a.preflight_receipt:
        try:
            verify_receipt(json.loads(Path(a.preflight_receipt).read_text()), cfs, _fp)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"[panel] invalid preflight receipt: {exc}") from exc
    if _resume_existing:
        _incompat = _resume_incompat(_resume_fp, _resume_dry, _fp, False)
        _unknown_resumed = _resume_ids - set(members_by_id)
        if _unknown_resumed:
            _incompat.append("persisted verdicts are outside the current cohort")
        if _incompat:
            raise SystemExit(f"[panel] REFUSING resume from {a.out.format(arm=a.arm)}: {'; '.join(_incompat)}. "
                             f"Resuming would treat prior verdicts as if produced under this configuration. "
                             f"Use --no-resume, or a fresh --out.")
    _per_champ = max(1, a.rounds) * 2 + len(_judge_models)
    print(f"[panel] pros={_pros} def={_def} judges={_judge_models} rounds={a.rounds} "
          f"compute={'on' if _compute else 'off'} "
          f"(~{_per_champ} agent calls/champion, ~{_per_champ * len(cfs)} total)", flush=True)
    if a.rounds > 5:
        print(f"[panel] WARNING: rounds={a.rounds} is high — {_per_champ * len(cfs)} planned agent calls", flush=True)
    _flush_state = {"done": False}                           # Prevent signal and exit handlers from re-entering an active flush.

    def _atomic_write(path, text):
        """Write a temporary sibling and atomically replace the destination file."""
        _p = Path(path)
        _tmp = _p.with_name(_p.name + ".tmp")
        _tmp.write_text(text)
        _tmp.replace(_p)

    def _flush(aborted=False):
        """Save decisions, audit records and accumulated usage after each strategy or on exit."""
        if aborted:
            _flush_state["done"] = True
        _est = round(_est_cost_usd(_spent_by_provider), 4)
        _base_est = round(_est_cost_usd(_baseline_by_provider), 4)
        # Unresolved failures differ from failed attempts that later succeeded.
        _bl_err_ids = {r["champion_id"] for r in audit_records if r.get("outcome") == "single_llm_baseline_error"}
        _bl_ok_ids = {r["champion_id"] for r in audit_records if r.get("outcome") == "single_llm_baseline"}
        _n_unresolved_baseline = len(_bl_err_ids - _bl_ok_ids)
        _op = a.out.format(arm=a.arm)
        Path(_op).parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(_op, json.dumps({
            "arm": a.arm, "dry_run": False, "run_fingerprint": _fp,
            "baseline_access_asymmetry": _baseline_access_asymmetry,
            "run_mode": _run_mode,
            "n_judges": len(_judge_models), "aborted": aborted,
            "exploratory_ranking": bool(a.exploratory_ranking),
            "n_court_drops": court_drops, "n_provider_errors": provider_errors,
            "n_degraded_panels": degraded_panels,
            "n_baseline_errors": baseline_errors,
            "n_unresolved_baseline": _n_unresolved_baseline,
            "spend": dict(_spent),
            "spend_by_provider": {p: dict(t) for p, t in _spent_by_provider.items()},
            "est_cost_usd": _est,
            "baseline_spend": dict(_baseline_spent),
            "baseline_spend_by_provider": {p: dict(t) for p, t in _baseline_by_provider.items()},
            "baseline_est_cost_usd": _base_est,
            "allocation_policy": a.allocation_policy,
            "verdicts": [{"champion_id": v.champion_id, "decision": v.decision, "confidence": v.confidence,
                          "predicted_label": v.predicted_label,
                          "prediction_agreement_fraction": v.confidence,
                          "confidence_meaning": "Fraction of valid judges agreeing with the reported forecast; not a probability",
                          "stability_assessment": v.stability_assessment, "uncertainty_reason": v.uncertainty_reason,
                          "disagreement": v.disagreement,
                          "decisive_exhibit_ids": v.decisive_exhibit_ids,
                          "decisive_metric_values": v.decisive_metric_values, "rationale": v.rationale,
                          "allocation_action": v.allocation_action, "allocation_label": allocation_label(v.allocation_action, v.allocation_percent),
                          **({"allocation_percent": v.allocation_percent, "allocation_policy": a.allocation_policy}),
                          "allocation_multiplier": v.allocation_multiplier,
                          "allocation_fallback_reason": v.allocation_fallback_reason,
                          "allocation_votes": v.allocation_votes,
                          "allocation_status_counts": v.allocation_status_counts,
                          "allocation_explanation": _court_allocation_explanation(v),
                          "allocation_details": allocation_details(v.panel),
                          "investigation_queue": _investigation_queue(v)}
                         for v in verdicts]}, indent=2, default=str))
        print(f"[panel] wrote {_op}  court_drops={court_drops} provider_errors={provider_errors}"
              f"{' (ABORTED — partial)' if aborted else ''}", flush=True)
        # Store complete responses and tool results beside the decision summaries.
        _name = Path(_op).name
        _aname = (_name.replace("verdicts", "audit") if "verdicts" in _name
                  else Path(_name).stem + "_audit" + Path(_name).suffix)
        _ap = str(Path(_op).with_name(_aname))
        _atomic_write(_ap, json.dumps({"arm": a.arm, "dry_run": False, "run_fingerprint": _fp,
            "baseline_access_asymmetry": _baseline_access_asymmetry, "n_baseline_errors": baseline_errors,
            "n_unresolved_baseline": _n_unresolved_baseline,
            "run_mode": _run_mode, "aborted": aborted,
                                       "n_champions": len(audit_records), "champions": audit_records},
                                      indent=2, default=str))
        print(f"[panel] wrote audit trail {_ap} ({len(audit_records)} champions)", flush=True)
        # Save single-LLM assessments and usage separately.
        if a.single_llm_baseline:
            _base_rows = [r for r in audit_records if r.get("outcome") == "single_llm_baseline"]
            _bn = Path(_op).name
            _bname = (_bn.replace("verdicts", "baseline") if "verdicts" in _bn
                      else Path(_bn).stem + "_baseline" + Path(_bn).suffix)
            _bpath = str(Path(_op).with_name(_bname))
            # The single LLM supplies a risk score without a directional prediction.
            _atomic_write(_bpath, json.dumps({
                "arm": a.arm, "dry_run": False, "run_mode": _run_mode, "run_fingerprint": _fp,
                "policy": "single_llm", "model": a.one_llm_model, "allocation_policy": a.allocation_policy,
                "est_cost_usd": _base_est,
                "n_baseline_errors": baseline_errors, "n_unresolved_baseline": _n_unresolved_baseline,
                "baseline_spend": dict(_baseline_spent),
                "baseline_spend_by_provider": {p: dict(t) for p, t in _baseline_by_provider.items()},
                "baseline_est_cost_usd": _base_est,
                "verdicts": [{"champion_id": r["champion_id"], "decision": "UNSURE",
                              "risk_score": r.get("risk_score"),
                              "allocation_action": r.get("allocation_action"),
                              "allocation_label": allocation_label(r.get("allocation_action"), r.get("allocation_percent")),
                              **({"allocation_percent": r.get("allocation_percent"), "allocation_policy": a.allocation_policy}),
                              "allocation_multiplier": r.get("allocation_multiplier"),
                              "allocation_status": r.get("allocation_status"),
                              "allocation_fallback_reason": r.get("allocation_fallback_reason"),
                              "allocation_details": allocation_details([r.get("assessment") or {}]),
                              "diagnostic_access": r.get("diagnostic_access"),
                              "allocation_explanation": _allocation_explanation(
                                  r.get("allocation_action"), r.get("allocation_multiplier"),
                                  status=r.get("allocation_status"), fallback=r.get("allocation_fallback_reason"),
                                  rationale=(r.get("assessment") or {}).get("allocation_rationale"))}
                             for r in _base_rows]}, indent=2, default=str))
            print(f"[panel] wrote baseline handoff {_bpath} ({len(_base_rows)} single-LLM verdict(s), scorer-ready)", flush=True)
        return _op

    def _check_budget():
        """Check estimated court and baseline spending before starting another strategy.

        The current strategy can exceed the ceiling; connectivity probes are not included.
        """
        reason = None
        _tot_out = _spent["output"] + _baseline_spent["output"]   # The spending check includes both court and comparator usage.
        if a.max_output_tokens and _tot_out >= a.max_output_tokens:
            reason = (f"[panel] ABORT: output-token budget {a.max_output_tokens} reached "
                      f"(spent {_tot_out} = court {_spent['output']} + baseline {_baseline_spent['output']}) "
                      f"— raise --max-output-tokens to continue")
        else:
            estimate = _est_cost_usd(_spent_by_provider) + _est_cost_usd(_baseline_by_provider)
            if a.max_cost_usd and estimate >= a.max_cost_usd:
                reason = (f"[panel] ABORT: estimated total cost ${estimate:.2f} (court + baseline) reached the "
                          f"--max-cost-usd ${a.max_cost_usd:.2f} ceiling — raise it to continue "
                          "(estimate uses public list prices, not billing truth)")
        if reason:
            _flush(aborted=True)
            raise SystemExit(reason)

    if any(cf.champion_id not in _resume_ids for cf in cfs):
        _check_budget()  # Apply the budget to resumed usage before starting connectivity probes.

    _prepared_runner = {}
    _first = next((cf for cf in cfs if cf.champion_id not in _resume_ids), None)
    if _first is not None:
        try:
            _runner = _rct_factory(members_by_id[_first.champion_id])
            assert_runner_ready(_runner)
            _check_training_packet(_first, _runner)
        except Exception as exc:
            raise SystemExit(f"[panel] REFUSED before provider calls: catalogue preparation failed for "
                             f"{_first.champion_id}: {type(exc).__name__}: {exc}") from exc
        _prepared_runner[_first.champion_id] = _runner
    if any(cf.champion_id not in _resume_ids for cf in cfs):
        _preflight_probe({_pros, _def, *_judge_models})

    # Save completed work on interruption as well as normal exit.
    def _checkpoint(reason):
        """Save current progress from a signal or exit handler unless already completed."""
        if _flush_state["done"]:
            return False
        try:
            _flush(aborted=True)
            print(f"[panel] {reason}: checkpointed {len(verdicts)} verdict(s)", flush=True)
        except Exception as _e:                              # noqa: BLE001
            _flush_state["done"] = True
            print(f"[panel] {reason}: checkpoint FAILED ({type(_e).__name__}: {_e})", flush=True)
        return True

    def _signal_checkpoint_and_exit(signum, frame):
        _checkpoint(f"SIGNAL {signum}")
        raise SystemExit(130 if signum == getattr(signal, "SIGINT", None) else 143)

    # Signal handlers are available only in the main thread.
    try:
        signal.signal(signal.SIGTERM, _signal_checkpoint_and_exit)
        signal.signal(signal.SIGINT, _signal_checkpoint_and_exit)
    except (ValueError, OSError, AttributeError):
        pass
    atexit.register(_checkpoint, "atexit")

    _pf_consecutive = 0                                      # Reset the provider-fallback streak after a strategy completes without one.

    def _bucket_spend(v_or_err):
        """Group recorded agent usage by the provider of the model that ran."""
        from potofgold.multi_agent.assessment.providers import provider_of
        _tr = (getattr(v_or_err, "agent_transcripts", None)
               or getattr(v_or_err, "_court_transcripts", None) or [])
        for _rec in _tr:
            _prov = provider_of(_rec.get("model") or "")
            _b = _spent_by_provider.setdefault(_prov, {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
            _b["input"] += int(_rec.get("input_tokens", 0) or 0)
            _b["output"] += int(_rec.get("output_tokens", 0) or 0)
            _b["cache_read"] += int(_rec.get("cache_read_input_tokens", 0) or 0)
            _b["cache_write"] += int(_rec.get("cache_creation_input_tokens", 0) or 0)

    for i, cf in enumerate(cfs):
        _court_done = cf.champion_id in _resume_ids
        _need_baseline = (a.single_llm_baseline and _rct_factory is not None
                          and cf.champion_id not in _baseline_ids)
        if _court_done and not _need_baseline:
            print(f"[panel] {i+1}/{len(cfs)} {cf.champion_id} -> SKIP (already scored, resume)", flush=True)
            continue
        _check_budget()
        try:
            # A failed training preparation stops the assessment before model calls.
            _rct = None
            if _rct_factory is not None:
                try:
                    _rct = _prepared_runner.pop(cf.champion_id, None)
                    if _rct is None:
                        _rct = _rct_factory(members_by_id.get(cf.champion_id))
                    assert_runner_ready(_rct)
                    _check_training_packet(cf, _rct)
                except Exception as _e:
                    audit_records.append({"champion_id": cf.champion_id, "outcome": "preparation_error",
                                          "error": f"{type(_e).__name__}: {_e}"})
                    _flush(aborted=True)
                    raise SystemExit(f"[panel] ABORT before this champion's paid calls: catalogue preparation "
                                     f"failed for {cf.champion_id}: {type(_e).__name__}: {_e}") from _e
                    print(f"[panel] exploratory catalogue runner unavailable for {cf.champion_id} "
                          f"({type(_e).__name__}: {_e}) -> resolve-from-exhibits", flush=True)
            v = None
            if not _court_done:
                v = validate_generalization(cf, run_agent=run_agent, model=a.model,
                                            pros_model=_pros, def_model=_def,
                                            judge_models=_judge_models, n_judges=a.n_judges, rounds=a.rounds,
                                            cohort_stats=_cohort_stats, enable_compute=_compute, run_catalogue_test=_rct,
                                            evidence=_evidence, catalogue_allow=_cat_allow,
                                            evidence_mode=a.evidence_mode, decision_policy=a.decision_policy,
                                            exploratory_ranking=a.exploratory_ranking,
                                            allocation_policy=a.allocation_policy)
                verdicts.append(v)
                audit_records.append({
                    "champion_id": v.champion_id, "outcome": "verdict", "decision": v.decision,
                    "predicted_label": v.predicted_label, "prosecution": v.prosecution, "defense": v.defense,
                    "analysis": v.analysis, "panel": v.panel, "disagreement": v.disagreement,
                    "tokens": v.tokens, "agent_transcripts": v.agent_transcripts,
                    "tool_records": getattr(v, "tool_records", [])})
            if a.single_llm_baseline and _rct is not None and cf.champion_id not in _baseline_ids:
                # Both assessments receive the same prepared measurements and development references.
                # Retain failed baseline attempts so resumed spending includes them.
                from potofgold.multi_agent.assessment import single_llm as _comp
                from potofgold.multi_agent.assessment.providers import provider_of as _pof
                _rct.decision_policy = a.decision_policy
                _rct.evidence_mode = a.evidence_mode
                _base_model = a.one_llm_model or _comp.DEFAULT_MODEL
                _base_agent = __import__('potofgold.multi_agent.assessment.agents', fromlist=["task_sync"]).task_sync
                try:
                    _b = _comp.assess_strategy(cf, _rct, None, run_agent=_base_agent, model=_base_model, evidence=_evidence,
                                              cohort_stats=_cohort_stats, enable_compute=_compute, catalogue_allow=_cat_allow,
                                              allocation_policy=a.allocation_policy)
                    _bm = _b.get("metadata") or {}
                    audit_records.append({"champion_id": cf.champion_id, "outcome": "single_llm_baseline",
                        "model": _base_model, "risk_score": _b["risk_score"], "allocation_action": _b["allocation_action"],
                        "allocation_multiplier": _b["allocation_multiplier"],
                        **({"allocation_percent": _b["allocation_percent"]}),
                        "allocation_status": _b.get("allocation_status"),
                        "allocation_fallback_reason": _b.get("allocation_fallback_reason"),
                        "assessment": _b["assessment"], "metadata": _bm, "evidence_digest": _b["evidence_digest"],
                        "tool_records": _b.get("tool_records", []), "diagnostic_access": _b.get("diagnostic_access")})
                    _bb = _baseline_by_provider.setdefault(_pof(_base_model),
                                                           {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
                    for _sk, _mk in (("input", "input_tokens"), ("output", "output_tokens"),
                                     ("cache_read", "cache_read_input_tokens"), ("cache_write", "cache_creation_input_tokens")):
                        _n = int(_bm.get(_mk, 0) or 0); _baseline_spent[_sk] += _n; _bb[_sk] += _n
                    _baseline_spent["calls"] += 1
                except Exception as _be:
                    baseline_errors += 1
                    # A rejected response still contributes any usage returned by the provider.
                    _ra = getattr(_be, "_ranking_audit", None) or {}
                    _bm = _ra.get("metadata") or getattr(_be, "_agent_metadata", None) or {}
                    if _bm:
                        _bb = _baseline_by_provider.setdefault(_pof(_base_model),
                                                               {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
                        for _sk, _mk in (("input", "input_tokens"), ("output", "output_tokens"),
                                         ("cache_read", "cache_read_input_tokens"), ("cache_write", "cache_creation_input_tokens")):
                            _n = int(_bm.get(_mk, 0) or 0); _baseline_spent[_sk] += _n; _bb[_sk] += _n
                        _baseline_spent["calls"] += 1
                    print(f"[panel] single-LLM baseline FAILED for {cf.champion_id}: {type(_be).__name__}: {_be}", flush=True)
                    audit_records.append({"champion_id": cf.champion_id, "outcome": "single_llm_baseline_error",
                                          "model": _base_model, "error": f"{type(_be).__name__}: {_be}",
                                          "metadata": _bm, "validation_error": _ra.get("validation_error")})
            _est = _est_cost_usd(_spent_by_provider)
            if not _court_done:
                if (v.disagreement or {}).get("degraded"):
                    degraded_panels += 1
                for _k in ("input", "output", "calls", "cache_read", "cache_write"):
                    _spent[_k] += int((v.tokens or {}).get(_k, 0) or 0)
                _bucket_spend(v)
                _pf = int((v.disagreement or {}).get("provider_fallbacks", 0) or 0)
                _pf_consecutive = _pf_consecutive + 1 if _pf else 0
                _est = _est_cost_usd(_spent_by_provider)
            _flush()                                               # Checkpoint after each completed strategy.
            if not _court_done:
                print(f"[panel] {i+1}/{len(cfs)} {cf.champion_id} -> {v.decision} ({v.predicted_label}) "
                      f"conf={v.confidence:.2f} cite={list(v.decisive_metric_values)} "
                      f"[out_tok Σ={_spent['output']} ~${_est:.2f}"
                      f"{f' pf_streak={_pf_consecutive}' if _pf_consecutive else ''}]", flush=True)
                if _pf_consecutive >= _PROVIDER_FALLBACK_TRIPWIRE:
                    _dead = None
                    try:
                        from potofgold.multi_agent.assessment.providers import provider_of as _pof
                        _dead = _pof((_pros.split(">", 1)[0]))
                    except Exception:                                 # noqa: BLE001
                        pass
                    _flush(aborted=True)
                    raise SystemExit(f"[panel] ABORT: {_pf_consecutive} consecutive champions each fell PRIMARY->backup on a "
                                     f"provider error (primary provider '{_dead or _pros}' appears DEAD — its spend is cascading "
                                     f"onto the expensive backup anchor). Fix/replace the primary credit before continuing.")
            else:
                print(f"[panel] {i+1}/{len(cfs)} {cf.champion_id} -> baseline catch-up (court resumed) "
                      f"[out_tok Σ={_spent['output']} ~${_est:.2f}]", flush=True)
            _check_budget()
        except MalformedVerdict as e:
            court_drops += 1
            for _k in ("input", "output", "calls", "cache_read", "cache_write"):
                _spent[_k] += int((getattr(e, "_court_tokens", {}) or {}).get(_k, 0) or 0)
            _bucket_spend(e)
            _pf = int(getattr(e, "_court_provider_fallbacks", 0) or 0)
            _pf_consecutive = _pf_consecutive + 1 if _pf else 0
            audit_records.append({"champion_id": cf.champion_id, "outcome": "court_drop",
                                  "error": f"{type(e).__name__}: {e}", "tokens": getattr(e, "_court_tokens", {}),
                                  "agent_transcripts": getattr(e, "_court_transcripts", [])})
            _flush()
            print(f"[panel] {i+1}/{len(cfs)} {cf.champion_id} -> COURT-DROP {type(e).__name__}: {e}", flush=True)
            if _pf_consecutive >= _PROVIDER_FALLBACK_TRIPWIRE:
                _flush(aborted=True)
                raise SystemExit(f"[panel] ABORT: {_pf_consecutive} consecutive champions each fell PRIMARY->backup on a "
                                 f"provider error (primary provider '{_pros}' appears DEAD — spend is cascading onto the "
                                 f"expensive backup anchor). Fix/replace the primary credit before continuing.")
            if not verdicts and provider_errors == 0 and court_drops >= _ALL_DROP_TRIPWIRE:
                _flush(aborted=True)
                raise SystemExit(f"[panel] ABORT: first {court_drops} champions ALL court-dropped with 0 verdicts "
                                 f"(a systematically declining judge? bad model/tool protocol) — not spending on the rest")
        except Exception as e:
            provider_errors += 1
            _pf_consecutive = 0                             # A complete provider failure resets the separate fallback-streak counter.
            for _k in ("input", "output", "calls", "cache_read", "cache_write"):
                _spent[_k] += int((getattr(e, "_court_tokens", {}) or {}).get(_k, 0) or 0)
            _bucket_spend(e)
            audit_records.append({"champion_id": cf.champion_id, "outcome": "provider_error",
                                  "error": f"{type(e).__name__}: {e}", "tokens": getattr(e, "_court_tokens", {}),
                                  "agent_transcripts": getattr(e, "_court_transcripts", [])})
            _flush()
            import traceback
            print(f"[panel] {i+1}/{len(cfs)} {cf.champion_id} -> PROVIDER/HARNESS ERROR {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            if i == 0 or provider_errors >= 3:
                _flush(aborted=True)
                raise SystemExit(f"[panel] ABORT: {provider_errors} provider/harness error(s) — fix the provider "
                                 f"config; not spending on the remaining {len(cfs) - i - 1} champion(s)")

    out_path = _flush()
    _flush_state["done"] = True
    print(f"[panel] COURT SPEND: input_tok={_spent['input']} output_tok={_spent['output']} agent_calls={_spent['calls']} "
          f"cache_read_tok={_spent['cache_read']} cache_write_tok={_spent['cache_write']}", flush=True)
    if a.single_llm_baseline:
        print(f"[panel] BASELINE SPEND: input_tok={_baseline_spent['input']} output_tok={_baseline_spent['output']} "
              f"calls={_baseline_spent['calls']} (~${_est_cost_usd(_baseline_by_provider):.2f})", flush=True)
    print(f"[panel] ESTIMATED COURT COST ~${_est_cost_usd(_spent_by_provider):.2f} (public list prices per "
          f"provider, NOT billing truth): " +
          ", ".join(f"{p}={{i:{t.get('input',0)},o:{t.get('output',0)}}}" for p, t in _spent_by_provider.items()),
          flush=True)
    print("[panel] CACHE NOTE: cache_read mixes Anthropic reads (billed ~0.1x, ADDITIONAL to input_tok) with "
          "OpenAI/Gemini cached tokens (already INCLUDED in input_tok); cache_write (~1.25x) is Anthropic-only. "
          "Treat as a diagnostic of caching effect, not a single clean cost line — see per-agent audit for the split.", flush=True)
    if not verdicts:
        print("[panel] ERROR: no valid verdicts produced", flush=True)
        return 2
    if degraded_panels:
        _frac = degraded_panels / len(verdicts)
        print(f"[panel] ⚠ DEGRADED PANELS: {degraded_panels}/{len(verdicts)} ({_frac:.0%}) champions decided with "
              f"< the full {len(_judge_models)}-judge panel seated (provider errors or per-judge court-drops). "
              f"Check per-verdict disagreement.degraded / DEGRADED_PANEL before trusting the cohort result.", flush=True)
    # A later successful baseline resolves its earlier failure without deleting its usage.
    _unresolved_baseline = ({r["champion_id"] for r in audit_records if r.get("outcome") == "single_llm_baseline_error"}
                            - {r["champion_id"] for r in audit_records if r.get("outcome") == "single_llm_baseline"})
    if baseline_errors:
        print(f"[panel] BASELINE ATTEMPTS FAILED: {baseline_errors} single-LLM call(s) RAN but were rejected (paid; "
              f"spend in BASELINE SPEND above). Unresolved after retries: {len(_unresolved_baseline)}.", flush=True)
    if _unresolved_baseline:
        print(f"[panel] ⚠ UNRESOLVED BASELINE: {len(_unresolved_baseline)} champion(s) have no single-LLM baseline "
              f"(e.g. {sorted(_unresolved_baseline)[:3]}) — the court-vs-baseline comparison is incomplete for them. "
              f"Resume to retry.", flush=True)
    return 1 if (court_drops or provider_errors or degraded_panels or _unresolved_baseline) else 0


if __name__ == "__main__":
    raise SystemExit(main())
