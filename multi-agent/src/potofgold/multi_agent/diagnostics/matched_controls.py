"""Compare boundary sensitivity and regime concentration with matched controls.

The boundary control compares one evolved comparison with nearby settings
at +/-0.25, 0.5, 0.75 and 1 training margin scale. Each setting is measured
against its own local perturbations and must have comparable entry activity.
The rank is descriptive: eight neighbors cannot support a 5% rank test.

The regime control compares the same combined concentration statistic with
random entry-day sets of equal size. Its corrected Monte Carlo p-value is
(1 + count(null >= observed)) / (K + 1). This null assumes exchangeability
within the eligible training-day universe and does not preserve clustering.

Results address the named allegation; they do not directly determine risk
scores, allocation or future profitability.
"""
from __future__ import annotations

import dataclasses
import math
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Boundary positions, in training margin scales; local nudges use a separate grid.
NEIGHBOURHOOD_OFFSETS: Tuple[float, ...] = (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)
LOCAL_NUDGE_EPS: Tuple[float, ...] = (0.1, 0.25, 0.5)
ACTIVITY_BAND: Tuple[float, float] = (0.5, 2.0)           # Keep entry rates between 0.5 and 2 times the evolved rate.
ACTIVITY_FLOOR = 0.02                                      # Exclude settings active on fewer than 2% of days.
ACTIVITY_CEIL = 0.98                                       # Exclude settings active on more than 98% of days.
MIN_VALID_NEIGHBOURS = 4                                   # Require this many comparable neighboring settings.
NULL_K_SMOKE = 999
NULL_K_FROZEN = 4999
REGIME_ALPHA = 0.05
MIN_FIRES = 10
ALLEGATIONS = ("boundary_brittleness", "regime_concentration")


class Verdict(str, Enum):
    EXCEPTIONAL = "EXCEPTIONAL"
    NOT_EXCEPTIONAL = "NOT_EXCEPTIONAL"
    INCONCLUSIVE = "INCONCLUSIVE"


def matched_allegation_applicability(allegation: str, genome: dict) -> Dict[str, Any]:
    """Check structural support without loading data, fitting a basis or replaying trades.

    This only rules out impossible measurements. A structurally supported tool
    must still pass the normal training replay, scale and activity checks.
    Boundary targets use the exact same tree parser and target identification
    as the executable perturbation, including direct-operand and float guards.
    """
    if allegation not in ALLEGATIONS:
        return {"allegation": allegation, "applicable": False,
                "verdict": Verdict.INCONCLUSIVE.value, "reason": "unknown_or_deferred_allegation"}
    if allegation != "boundary_brittleness":
        return {"allegation": allegation, "applicable": True}
    from potofgold.evolutionary_search.search.grammar import from_sexpr
    from potofgold.multi_agent.diagnostics.perturbation_prober import perturbation_targets
    expression = genome.get("entry_tree") if isinstance(genome, dict) else None
    try:
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("Missing entry expression")
        targets = perturbation_targets(from_sexpr(expression))["targets"]
    except (KeyError, TypeError, ValueError, IndexError):
        return {"allegation": allegation, "applicable": False,
                "verdict": Verdict.INCONCLUSIVE.value, "reason": "invalid_entry_tree",
                "method": "descriptive_rank"}
    if not targets:
        return {"allegation": allegation, "applicable": False,
                "verdict": Verdict.INCONCLUSIVE.value, "reason": "no_perturbable_boundary",
                "method": "descriptive_rank"}
    return {"allegation": allegation, "applicable": True}


def available_matched_allegations(genome: dict) -> Tuple[str, ...]:
    """Structural capabilities only; advertising a control never certifies its reliability."""
    return tuple(allegation for allegation in ALLEGATIONS
                 if matched_allegation_applicability(allegation, genome)["applicable"])


# Control statistics and decision rules.

def _boundary_verdict(center_local_phi: Optional[float], neighbours: Sequence[Dict[str, Any]], *,
                      min_valid: int = MIN_VALID_NEIGHBOURS) -> Dict[str, Any]:
    """Rank the evolved setting against valid neighboring boundary settings.

    EXCEPTIONAL means its local stability is strictly below every valid
    neighbor. Return INCONCLUSIVE for an invalid center or too few neighbors.
    The rank is descriptive, with one denoting the most sensitive setting.
    """
    if center_local_phi is None or not math.isfinite(center_local_phi):
        return {"verdict": Verdict.INCONCLUSIVE.value, "reason": "center_position_unusable",
                "n_valid_neighbours": 0, "method": "descriptive_rank"}
    valid = [n for n in neighbours if n.get("valid") and isinstance(n.get("local_phi"), (int, float))
             and math.isfinite(n["local_phi"])]
    if len(valid) < min_valid:
        return {"verdict": Verdict.INCONCLUSIVE.value, "reason": "too_few_comparable_neighbours",
                "n_valid_neighbours": len(valid), "min_valid": min_valid, "method": "descriptive_rank"}
    more_brittle = sum(1 for n in valid if n["local_phi"] <= center_local_phi)
    rank = more_brittle + 1
    exceptional = all(center_local_phi < n["local_phi"] for n in valid)
    return {"verdict": (Verdict.EXCEPTIONAL.value if exceptional else Verdict.NOT_EXCEPTIONAL.value),
            "center_local_phi": float(center_local_phi),
            "neighbour_local_phis": sorted(round(float(n["local_phi"]), 6) for n in valid),
            "rank": rank, "n_valid_neighbours": len(valid),
            "method": "descriptive_rank",
            "inference": "descriptive only — 8 neighbours cannot support a 5% test (exchangeable rank floor = 1/9)"}


def _corrected_mc_p(t_obs: float, t_nulls: Sequence[float]) -> float:
    """Return (1 + count(null >= observed)) / (K + 1), including ties.

    K is the number of finite null draws supplied.
    """
    k = len(t_nulls)
    ge = sum(1 for t in t_nulls if t >= t_obs)
    return (1 + ge) / (k + 1)


def _regime_verdict(t_obs: Optional[float], t_nulls: Sequence[float], *, n_fires: int, n_eligible: int,
                    k_requested: int, alpha: float = REGIME_ALPHA, min_fires: int = MIN_FIRES) -> Dict[str, Any]:
    """Compare observed concentration with the matched random-day draws.

    Expose the entry rate while keeping raw entry counts internal.
    """
    finite_nulls = [float(t) for t in t_nulls if isinstance(t, (int, float)) and math.isfinite(t)]
    rate = round(n_fires / n_eligible, 4) if n_eligible else None
    if t_obs is None or not math.isfinite(t_obs):
        return {"verdict": Verdict.INCONCLUSIVE.value, "reason": "observed_concentration_undefined",
                "fire_rate": rate, "method": "corrected_mc_p"}
    if n_fires < min_fires:
        return {"verdict": Verdict.INCONCLUSIVE.value, "reason": "too_few_fires_for_stable_null",
                "fire_rate": rate, "method": "corrected_mc_p"}
    if len(finite_nulls) < max(1, k_requested // 2):
        return {"verdict": Verdict.INCONCLUSIVE.value, "reason": "null_not_formable",
                "n_null_realized": len(finite_nulls), "k_requested": k_requested, "method": "corrected_mc_p"}
    p = _corrected_mc_p(float(t_obs), finite_nulls)
    return {"verdict": (Verdict.EXCEPTIONAL.value if p <= alpha else Verdict.NOT_EXCEPTIONAL.value),
            "p_value": float(p), "t_observed": float(t_obs), "alpha": alpha,
            "k_null": len(finite_nulls), "fire_rate": rate,
            "method": "corrected_mc_p (Phipson-Smyth: (1+#{null>=obs})/(K+1))",
            "null_model": "matched-fire-count, uniform over frozen eligible days",
            "null_assumption": "within-eligible exchangeability of fire days; discards clustering / changing "
                               "opportunity — a restricted (block/opportunity-aware) randomization is a later refinement"}


# Boundary control using the shared trade replay.

@dataclasses.dataclass
class _BoundaryEngine:
    """Hold shared training-replay functions for the boundary-neighborhood comparison."""
    targets: List[Dict[str, Any]]
    d_universe: int
    evolved_fires: set
    fires_at: Callable[[Tuple[int, ...], float], set]   # Map (constant path, value) to executed entry days.
    scale_of: Callable[[Tuple[int, ...]], Optional[float]]   # Map a comparison path to its training margin scale.
    base_val_of: Callable[[Tuple[int, ...]], float]
    cmp_name_of: Callable[[Tuple[int, ...]], str]
    phi: Callable[[set, set], float]                         # Chance-corrected agreement between entry-day sets.


def _build_boundary_engine(genome: dict, prep: dict) -> _BoundaryEngine:
    """Bind threshold perturbation functions to a reconstructed training strategy."""
    from potofgold.evolutionary_search.search.genetics import individual_from_genome_dict
    from potofgold.evolutionary_search.search.grammar import _node_at_path
    from potofgold.multi_agent.diagnostics.decision_tracer import build_evaluator, assert_disjoint_days
    from potofgold.multi_agent.diagnostics.perturbation_prober import perturbation_targets, perturbed_entry_tree, robust_scale, margin_series, _run_trades, _fired_days, phi_hat
    if prep.get("train_days") is not None and prep.get("heldout_days") is not None:
        assert_disjoint_days(prep["train_days"], prep["heldout_days"])

    base_ind = individual_from_genome_dict(genome)
    entry_tree = base_ind.entry_tree
    dates = np.asarray(prep["train_df"]["date"].values)
    d_universe = int(np.unique(dates).size)
    evolved_fires = _fired_days(_run_trades(base_ind, prep), dates)
    ev = build_evaluator(prep["train_df"], prep["td_tr"], bases_k=prep.get("bases_k"),
                         emb_cache_tr=prep.get("emb_cache_tr"))
    targets = perturbation_targets(entry_tree)["targets"]
    _scale_cache: Dict[Tuple[int, ...], Optional[float]] = {}

    def scale_of(cmp_path):
        if cmp_path not in _scale_cache:
            _scale_cache[cmp_path] = robust_scale(margin_series(ev, _node_at_path(entry_tree, cmp_path)))
        return _scale_cache[cmp_path]

    def base_val_of(const_path):
        return float(_node_at_path(entry_tree, const_path).value)

    def cmp_name_of(cmp_path):
        return _node_at_path(entry_tree, cmp_path).name

    def fires_at(const_path, value):
        tree = perturbed_entry_tree(entry_tree, const_path, float(value))
        pert = dataclasses.replace(base_ind, entry_tree=tree)
        return _fired_days(_run_trades(pert, prep), dates)

    return _BoundaryEngine(targets=targets, d_universe=d_universe, evolved_fires=evolved_fires,
                           fires_at=fires_at, scale_of=scale_of, base_val_of=base_val_of, cmp_name_of=cmp_name_of,
                           phi=lambda a, b: phi_hat([a, b], d_universe))


def _local_phi_at(engine: _BoundaryEngine, const_path, position: float, scale: float,
                  position_fires: Optional[set] = None) -> Optional[float]:
    """Compare a boundary setting's entries with its own local perturbations.

    Return the worst agreement over the nudge grid, or None without baseline entries.
    """
    base = position_fires if position_fires is not None else engine.fires_at(const_path, position)
    if not base:
        return None
    phis: List[float] = []
    for e in LOCAL_NUDGE_EPS:
        for sign in (+1.0, -1.0):
            fired = engine.fires_at(const_path, position + sign * float(e) * scale)
            ph = engine.phi(base, fired)
            if ph == ph:
                phis.append(float(ph))
    return min(phis) if phis else None


def boundary_brittleness_control(genome: dict, prep: dict, *, node_index: Optional[int] = None,
                                 engine: Optional[_BoundaryEngine] = None,
                                 min_valid: int = MIN_VALID_NEIGHBOURS) -> Dict[str, Any]:
    """Compare one evolved boundary with nearby settings at comparable entry rates.

    Use node_index when supplied; otherwise select the boundary with the lowest
    local stability at its evolved setting.
    """
    if engine is None:
        applicability = matched_allegation_applicability("boundary_brittleness", genome)
        if not applicability["applicable"]:
            return applicability
    eng = engine or _build_boundary_engine(genome, prep)
    if not eng.targets or not eng.evolved_fires:
        return {"allegation": "boundary_brittleness", "verdict": Verdict.INCONCLUSIVE.value,
                "reason": "no_perturbable_boundary_or_no_fires", "method": "descriptive_rank"}
    evolved_rate = len(eng.evolved_fires) / eng.d_universe if eng.d_universe else 0.0

    # Select one comparison at its evolved setting and retain it throughout.
    scored: List[Tuple[float, int, Dict[str, Any], float]] = []
    for i, tgt in enumerate(eng.targets):
        scale = eng.scale_of(tgt["cmp_path"])
        if scale is None or scale <= 0:
            continue
        center = eng.base_val_of(tgt["const_path"])
        cphi = _local_phi_at(eng, tgt["const_path"], center, scale, position_fires=eng.evolved_fires)
        if cphi is not None:
            scored.append((cphi, i, tgt, scale))
    if not scored:
        return {"allegation": "boundary_brittleness", "verdict": Verdict.INCONCLUSIVE.value,
                "reason": "no_node_with_valid_scale_and_center_stability", "method": "descriptive_rank"}
    if node_index is not None:
        pick = next((s for s in scored if s[1] == node_index), None)
        if pick is None:
            return {"allegation": "boundary_brittleness", "verdict": Verdict.INCONCLUSIVE.value,
                    "reason": "requested_node_not_perturbable", "node_index": node_index, "method": "descriptive_rank"}
    else:
        pick = min(scored, key=lambda s: s[0])
    center_phi, idx, tgt, scale = pick
    center_val = eng.base_val_of(tgt["const_path"])

    # Compare each neighboring setting with its own local perturbations.
    neighbours: List[Dict[str, Any]] = []
    for off in NEIGHBOURHOOD_OFFSETS:
        pos = center_val + off * scale
        pfires = eng.fires_at(tgt["const_path"], pos)
        rate = len(pfires) / eng.d_universe if eng.d_universe else 0.0
        rec: Dict[str, Any] = {"offset": off, "fire_rate": rate}
        if not pfires or rate < ACTIVITY_FLOOR or rate > ACTIVITY_CEIL:
            rec.update(valid=False, reason="near_inactive_or_always_active", local_phi=None)
        elif not (ACTIVITY_BAND[0] * evolved_rate <= rate <= ACTIVITY_BAND[1] * evolved_rate):
            rec.update(valid=False, reason="activity_not_comparable_to_evolved", local_phi=None)
        else:
            lp = _local_phi_at(eng, tgt["const_path"], pos, scale, position_fires=pfires)
            rec.update(valid=(lp is not None), reason=(None if lp is not None else "degenerate_local_nudges"),
                       local_phi=lp)
        neighbours.append(rec)

    out = _boundary_verdict(center_phi, neighbours, min_valid=min_valid)
    out.update(allegation="boundary_brittleness", node=eng.cmp_name_of(tgt["cmp_path"]), node_index=idx,
               evolved_fire_rate=round(evolved_rate, 4), neighbourhood_offsets=list(NEIGHBOURHOOD_OFFSETS))
    return out


# Regime concentration against random entry-day sets.

def _default_fires_and_axis(genome: dict, prep: dict):
    """Return training entry days, available days and optional volatility observations."""
    from potofgold.multi_agent.diagnostics.vulnerability_catalogue import _default_fires, _load_vix_rows_optional
    fires = _default_fires(genome, prep)
    return fires.get("fired_days") or [], fires.get("all_days") or [], _load_vix_rows_optional(None)


def regime_concentration_control(genome: dict, prep: dict, *, k: int = NULL_K_SMOKE, rng_seed: int = 0,
                                 alpha: float = REGIME_ALPHA, min_fires: int = MIN_FIRES,
                                 concentration_fn: Optional[Callable[[Sequence, Sequence], Optional[float]]] = None,
                                 fires_axis_fn: Optional[Callable[[dict, dict], Tuple]] = None) -> Dict[str, Any]:
    """Compare concentration with equal-sized random entry-day sets.

    Recompute the same combined statistic for each draw and report the
    corrected Monte Carlo p-value.
    """
    fired, all_days, vix_rows = (fires_axis_fn or _default_fires_and_axis)(genome, prep)
    eligible = sorted(set(str(d) for d in all_days))
    fired = [str(d) for d in fired if str(d) in set(eligible)]
    n_fires, n_eligible = len(fired), len(eligible)

    if concentration_fn is None:
        from potofgold.multi_agent.diagnostics.regime_features import regime_concentration_summary

        def concentration_fn(fd, ad):                       # Reuse the combined maximum over volatility and calendar axes.
            summ = regime_concentration_summary(list(fd), list(ad), vix_rows, min_fires=min_fires)
            return summ.get("fire_regime_concentration") if summ.get("available") else None

    t_obs = concentration_fn(fired, eligible)
    if t_obs is None or n_fires < min_fires or n_eligible < n_fires + 1:
        out = _regime_verdict(t_obs, [], n_fires=n_fires, n_eligible=n_eligible, k_requested=k, alpha=alpha,
                              min_fires=min_fires)
        out["allegation"] = "regime_concentration"
        return out

    rng = np.random.default_rng(rng_seed)                   # Use a fixed seed and the same statistic for every draw.
    elig_arr = np.asarray(eligible, dtype=object)
    t_nulls: List[float] = []
    for _ in range(k):
        draw = rng.choice(len(elig_arr), size=n_fires, replace=False)
        t = concentration_fn([elig_arr[j] for j in draw], eligible)
        if t is not None and math.isfinite(t):
            t_nulls.append(float(t))
    out = _regime_verdict(t_obs, t_nulls, n_fires=n_fires, n_eligible=n_eligible, k_requested=k, alpha=alpha,
                          min_fires=min_fires)
    out.update(allegation="regime_concentration", rng_seed=rng_seed)
    return out


MATCHED_CONTROL_TOOL = {
    "name": "matched_allegation_control",
    "description": (
        "Run a MATCHED control for a specific prosecution allegation on this champion; returns EXCEPTIONAL / "
        "NOT_EXCEPTIONAL / INCONCLUSIVE against a matched benchmark. allegation is one of: 'boundary_brittleness' "
        "(is the evolved boundary unusually sensitive versus its OWN neighbourhood of nearby boundaries — "
        "DESCRIPTIVE rank, not a significance test) or 'regime_concentration' (does it select regimes more strongly "
        "than a matched random-firing benchmark — corrected Monte-Carlo p). EXCEPTIONAL = the allegation holds "
        "against the control; NOT_EXCEPTIONAL = the control does not establish it; INCONCLUSIVE = it could not be "
        "measured. This resolves a FACTUAL point and does NOT by itself decide WILL/WONT (a genuine specialist may "
        "look concentrated; a whole neighbourhood may be brittle)."),
    "input_schema": {"type": "object", "properties": {
        "allegation": {"type": "string", "enum": list(ALLEGATIONS)},
        "resolves_uncertainty": {"type": "string"}},
        "required": ["allegation", "resolves_uncertainty"]}}


def matched_allegation_control(allegation: str, genome: dict, prep: dict, **params) -> Dict[str, Any]:
    """Run a boundary-sensitivity or regime-concentration control.

    Return EXCEPTIONAL, NOT_EXCEPTIONAL or INCONCLUSIVE for that allegation.
    """
    if allegation == "boundary_brittleness":
        return boundary_brittleness_control(genome, prep, **params)
    if allegation == "regime_concentration":
        return regime_concentration_control(genome, prep, **params)
    return {"allegation": allegation, "verdict": Verdict.INCONCLUSIVE.value,
            "reason": f"unknown_or_deferred_allegation (supported: {', '.join(ALLEGATIONS)})"}
