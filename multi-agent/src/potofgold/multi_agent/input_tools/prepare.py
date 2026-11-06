"""Build training-diagnostic runners for each strategy input condition.

Use CAT_RUNNER_SCALAR_DATA for indicators, or CAT_RUNNER_DATA and
CAT_RUNNER_TARGETS for embeddings. CAT_RUNNER_NFOLDS defaults to five.
When a saved training P&L reference is supplied, verify the reconstruction
against it before running diagnostics.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from potofgold.multi_agent.diagnostics.catalogue_runner import genome_of, make_box_prep_factory

_ARM_GRAMMAR = {"A": None, "C": "emb", "AUC": "combined"}


def _resolve_template(champion_record: dict) -> Optional[str]:
    """Resolve the structure template from the genome, then from provenance."""
    g = genome_of(champion_record) or {}
    prov = champion_record.get("provenance") or {}
    return g.get("template_name") or g.get("template") or prov.get("template")


def _arm_factory(arm: str, *, recon: Optional[str], targets: Optional[str], n_folds: int, scalar: Optional[str] = None,
                 saved_report_for: Optional[Callable[[dict], Optional[dict]]] = None,
                 _make_factory: Callable[..., Any] = make_box_prep_factory) -> Callable[..., Any]:
    """Create a runner for one input condition.

    Cache one preparation factory at a time, keyed by template, basis and grammar.
    Pass saved training reports through to the runner for validated reuse.
    """
    arm = arm.upper()
    grammar = _ARM_GRAMMAR.get(arm)
    cache: Dict[str, Any] = {"key": None, "fac": None}

    def factory(champion_record: Optional[dict]) -> Any:
        if not champion_record:
            return None
        prov = champion_record.get("provenance") or {}
        template = _resolve_template(champion_record)
        if not template:
            raise ValueError(f"[prepare] no template resolvable for champion "
                             f"{champion_record.get('champion_id')!r} (arm {arm})")
        basis = prov.get("basis_method") or prov.get("basis") or "lean"
        key = f"{template}|{basis}|{grammar}"
        if cache["key"] != key:                            # Keep only one prepared market-data configuration in memory.
            if arm == "A":
                if not scalar:
                    raise ValueError('Indicator strategies require CAT_RUNNER_SCALAR_DATA or an explicit scalar path')
                scalar_args = {"data": scalar}
                cache["fac"] = _make_factory(arm="A", template=template, n_folds=n_folds,
                                             saved_report_for=saved_report_for, **scalar_args)
            else:
                if not (recon and targets):
                    raise KeyError("[prepare] embedding arm needs recon+targets (CLI --recon/--targets "
                                   "or CAT_RUNNER_DATA/CAT_RUNNER_TARGETS)")
                cache["fac"] = _make_factory(arm=arm, template=template, n_folds=n_folds, data=recon,
                                             targets=targets, basis=basis, grammar_mode=grammar,
                                             saved_report_for=saved_report_for)
            cache["key"] = key
        return cache["fac"](champion_record)

    return factory


def build_factory_for_arm(recon: Optional[str] = None, targets: Optional[str] = None, n_folds: int = 5,
                          scalar: Optional[str] = None,
                          saved_report_for: Optional[Callable[[dict], Optional[dict]]] = None,
                          _make_factory: Callable[..., Any] = make_box_prep_factory) -> Dict[str, Callable[..., Any]]:
    """Create indicator, embedding and combined-input factories.

    Explicit paths take precedence over environment variables.
    """
    scalar = scalar or os.environ.get("CAT_RUNNER_SCALAR_DATA")
    recon = recon or os.environ.get("CAT_RUNNER_DATA")
    targets = targets or os.environ.get("CAT_RUNNER_TARGETS")
    n_folds = int(n_folds or os.environ.get("CAT_RUNNER_NFOLDS", "5"))
    return {a: _arm_factory(a, recon=recon, targets=targets, n_folds=n_folds, scalar=scalar,
                            saved_report_for=saved_report_for, _make_factory=_make_factory)
            for a in ("A", "C", "AUC")}


def _env_saved_report_resolver() -> Optional[Callable[[dict], Optional[dict]]]:
    """Load saved training stress reports from CAT_RUNNER_SAVED_STRESS.

    The comma-separated files may contain JSON records or JSONL. The runner
    checks each report's strategy and split before reusing it.
    """
    raw = os.environ.get("CAT_RUNNER_SAVED_STRESS")
    if not raw:
        return None
    from potofgold.multi_agent.diagnostics.catalogue_runner import saved_stress_resolver
    records: list = []
    for p in (s.strip() for s in raw.split(",") if s.strip()):
        if p.endswith(".jsonl"):
            with open(p) as fh:
                records += [json.loads(line) for line in fh if line.strip()]
        else:
            records += (json.load(open(p)).get("records") or [])
    return saved_stress_resolver(records)


_factory_map: Optional[Dict[str, Callable[..., Any]]] = None
_factory_inputs: Optional[tuple] = None


def factory(champion_record: Optional[dict]) -> Any:
    """Bind a strategy to its market inputs and share one diagnostic snapshot between both assessments."""
    global _factory_map, _factory_inputs
    if not champion_record:
        return None
    inputs = tuple(os.environ.get(k) for k in (
        "CAT_RUNNER_SCALAR_DATA", "CAT_RUNNER_DATA", "CAT_RUNNER_TARGETS", "CAT_RUNNER_NFOLDS", "CAT_RUNNER_SAVED_STRESS"))
    if _factory_map is None or _factory_inputs != inputs:
        _factory_map = build_factory_for_arm(n_folds=int(os.environ.get("CAT_RUNNER_NFOLDS", "5")),
                                             saved_report_for=_env_saved_report_resolver())
        _factory_inputs = inputs
    arm = str(champion_record.get("arm") or "").upper()
    fac = _factory_map.get(arm)
    if fac is None:
        raise KeyError(f"[prepare] no factory for arm {arm!r} "
                       f"(champion {champion_record.get('champion_id')!r})")
    runner = fac(champion_record)
    # Establish the replay identity before caching measurements for both assessments.
    runner.get_training_economics()
    snapshot = None

    def measurements():
        nonlocal snapshot
        if snapshot is None:
            from potofgold.multi_agent.assessment.run_readiness import assert_runner_ready
            assert_runner_ready(runner)
            pending = []
            for test_id in ('regime_concentration', 'exit_behavior_profile', 'transaction_cost_sensitivity',
                            'entry_boundary_stability', 'entry_boundary_economics'):
                record = runner(test_id)
                pending.append(record.to_exhibit() if hasattr(record, 'to_exhibit') else record)
            snapshot = pending
        return deepcopy(snapshot)

    runner.get_preloaded_measurements = measurements
    return runner


def fingerprint_inputs():
    """Identify the market files and fold setting used for preflight and resume."""
    inputs = {key: os.environ[key] for key in
              ('CAT_RUNNER_SCALAR_DATA', 'CAT_RUNNER_DATA', 'CAT_RUNNER_TARGETS',
               'CAT_RUNNER_VIX_STATE') if os.environ.get(key)}
    inputs.update({f'saved_stress_{i}': path.strip() for i, path in
                   enumerate(os.environ.get('CAT_RUNNER_SAVED_STRESS', '').split(',')) if path.strip()})
    return {'files': inputs, 'metadata': {'n_folds': int(os.environ.get('CAT_RUNNER_NFOLDS', '5'))}}


factory.fingerprint_inputs = fingerprint_inputs


def main(argv=None):
    """Prepare one training-exhibit file for the public court command."""
    import argparse
    from potofgold.multi_agent.strategy_profiles.training_case_exhibits import KINDS
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--champions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    members = json.loads(args.champions.read_text())
    if not isinstance(members, list) or not members:
        parser.error('--champions must contain a nonempty list of strategy records')
    ids = [row['champion_id'] for row in members]
    if len(ids) != len(set(ids)):
        parser.error('Strategy IDs must be unique')
    result = {kind: [] for kind in KINDS}
    for member in members:
        member = {**member, 'shape': member.get('shape') or genome_of(member).get('shape') or _resolve_template(member)}
        runner = factory(member)
        exhibits = runner.get_training_case_exhibits(KINDS)
        for kind in KINDS:
            result[kind].append(exhibits[kind])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Wrote verified training exhibits for {len(members)} strategies to {args.output}')


if __name__ == '__main__':
    main()
