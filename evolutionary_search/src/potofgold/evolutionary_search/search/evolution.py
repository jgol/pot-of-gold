"""Evolve strategy populations, select saved fronts and evaluate fold champions."""

from __future__ import annotations
from typing import Optional
import hashlib
import json
import os
import random as _random
import signal
import time
from multiprocessing import TimeoutError as mp_TimeoutError
import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
from .grammar import (
    GType,
    TermDef,
    TermNode,
    to_str,
    node_count,
    is_structurally_degenerate,
    has_illegal_position_state,
)
from .genetics import (
    Individual,
    crossover,
    mutate,
    nsga3_select,
    seed_structure_genes,
    compose_mutate,
    seed_population_predicates,
    individual_from_genome_dict,
    canonical_key,
    _write_checkpoint_atomic,
    _load_checkpoint,
    _delete_checkpoint,
    _CHECKPOINT_FORMAT_VERSION,
)
from .fitness import (
    genome_capture_objectives,
    genome_capture_objectives_nested,
    INFEASIBLE,
    DEFAULT_MIN_TRADES,
    expectancy_trim_mismatch_warning,
    genome_input_sharpness,
    sam_feature_sigma,
    per_eligible_day_vector,
    per_day_sum_pnl,
    genome_trades,
    _entry_dates,
    capture_objectives_from_trades,
    cross_fold_agreement,
)
from .validation import (
    judge_champion,
    signal_gated_term,
    derive_cvar_ceiling,
    classify_abstention,
    genome_per_day_excess,
)
from .validation import _decision_mask as decision_mask

DECISION_OFFSET = 60
_INFeasible_FLOOR = 100000000.0
_FEAS_GUARD_CONSECUTIVE_GENS = 3
_FEAS_GUARD_MIN_POP = 64
_GATED_MIN_COUNT_DAYS = DEFAULT_MIN_TRADES


def _const(v: float) -> TermNode:
    return TermNode(defn=TermDef("EphReal", GType.REAL), value=float(v))


def _champion_sort_key(ind):
    return (
        round(float(ind.fitness[0]), 1),
        float(ind.fitness[1]),
        float(ind.fitness[2]),
        getattr(ind, "_nodes", 0.0),
    )


def _champion_tail_band():
    try:
        b = float(os.environ.get("GP_CHAMPION_TAIL_BAND", "") or "0")
    except ValueError:
        return None
    return b if b > 0 else None


def _tail_band_filter(feasible):
    band = _champion_tail_band()
    if band is None or not feasible:
        return feasible
    best_cvar = min((float(i.fitness[1]) for i in feasible))
    safe = [i for i in feasible if float(i.fitness[1]) <= best_cvar + band]
    return safe or feasible


def _pick_champion(feasible):
    if not feasible:
        return None
    feasible = _tail_band_filter(feasible)
    c0 = min(feasible, key=_champion_sort_key)
    ci = _cia_fitness_index()
    if ci is None:
        return c0
    agree0 = -float(c0.fitness[ci])
    if agree0 <= 0.0:
        return c0
    exp0 = -float(c0.fitness[0])
    band = _cia_champ_band()
    if band <= 0.0:
        er0 = round(exp0, 1)
        cand = [
            i
            for i in feasible
            if -float(i.fitness[ci]) > 0.0 and round(-float(i.fitness[0]), 1) == er0
        ]
    else:
        cand = [
            i
            for i in feasible
            if -float(i.fitness[ci]) > 0.0
            and exp0 - -float(i.fitness[0]) <= band + 1e-12
        ]
    return min(
        cand,
        key=lambda i: (
            float(i.fitness[ci]),
            float(i.fitness[0]),
            float(i.fitness[1]),
            getattr(i, "_nodes", 0.0),
        ),
    )


def _is_invalid(ind, grammar=None) -> bool:
    if is_structurally_degenerate(ind.entry_tree):
        return True
    if has_illegal_position_state(ind.entry_tree, "entry"):
        return True
    if is_structurally_degenerate(ind.exit_tree):
        return True
    if ind.delta_tree is not None and has_illegal_position_state(
        ind.delta_tree, "delta"
    ):
        return True
    if getattr(ind, "width_tree", None) is not None and has_illegal_position_state(
        ind.width_tree, "delta"
    ):
        return True
    if getattr(ind, "shape_tree", None) is not None and has_illegal_position_state(
        ind.shape_tree, "delta"
    ):
        return True
    if has_illegal_position_state(ind.size_tree, "delta"):
        return True
    if grammar is not None:
        for t in (
            ind.entry_tree,
            ind.exit_tree,
            ind.size_tree,
            ind.delta_tree,
            getattr(ind, "width_tree", None),
            getattr(ind, "shape_tree", None),
        ):
            if t is not None and node_count(t) > grammar.max_nodes:
                return True
    return False


def _random_individual(grammar, template, evolve_structure: bool, i: int):
    depth = 2 + i % 4
    ind = None
    for _attempt in range(8):
        entry = grammar._gen_capped(GType.BOOL, depth)
        for _ in range(5):
            if not (
                is_structurally_degenerate(entry)
                or has_illegal_position_state(entry, "entry")
            ):
                break
            entry = grammar._gen_capped(GType.BOOL, depth)
        exit_ = grammar._gen_capped(GType.BOOL, depth)
        for _ in range(5):
            if not is_structurally_degenerate(exit_):
                break
            exit_ = grammar._gen_capped(GType.BOOL, depth)
        delta = grammar._gen_capped(GType.REAL, depth) if evolve_structure else None
        if delta is not None:
            for _ in range(5):
                if not has_illegal_position_state(delta, "delta"):
                    break
                delta = grammar._gen_capped(GType.REAL, depth)
        if evolve_structure:
            size = grammar._gen_capped(GType.REAL, depth)
            for _ in range(5):
                if not has_illegal_position_state(size, "delta"):
                    break
                size = grammar._gen_capped(GType.REAL, depth)
        else:
            size = _const(0.5)
        ind = Individual(entry, exit_, size, template.name, delta_tree=delta)
        if evolve_structure:
            ind = seed_structure_genes(ind, grammar, max_d=depth)
        if not _is_invalid(ind, grammar):
            return ind
    return ind


def _set_infeasible(ind):
    inf = float(INFEASIBLE[0])
    ind.fitness = np.full(_fitness_dim(), inf, dtype=float)
    ind._nodes = inf


def _assign_fitness(
    ind,
    df,
    template,
    td,
    min_trades,
    cvar_frac,
    entry_decision_mask=None,
    grammar=None,
    warmup_bars=None,
    pca_bases=None,
    emb_proj_cache=None,
    decision_offset=DECISION_OFFSET,
    gated_min_count=_GATED_MIN_COUNT_DAYS,
    cvar_tail_max=None,
    cia_partition_seed=0,
):
    if _is_invalid(ind, grammar):
        _set_infeasible(ind)
        return
    if warmup_bars is None:
        warmup_bars = int(decision_offset)
    _entry_floor = None if entry_decision_mask is not None else int(decision_offset)
    _k_sub = _nested_cv_k()
    _champ_reuse = None
    if _k_sub > 1:
        o = genome_capture_objectives_nested(
            ind,
            df,
            template,
            td,
            k_sub=_k_sub,
            lam=_nested_cv_lambda(),
            partition_seed=_nested_cv_seed(),
            min_trades=min_trades,
            cvar_frac=cvar_frac,
            cvar_tail_max=cvar_tail_max,
            entry_decision_mask=entry_decision_mask,
            warmup_bars=warmup_bars,
            pca_bases=pca_bases,
            emb_proj_cache=emb_proj_cache,
            entry_floor_within_day=_entry_floor,
        )
    elif _obj1_excess_baseline():
        _champ_reuse = genome_trades(
            ind,
            df,
            template,
            td,
            entry_decision_mask=entry_decision_mask,
            warmup_bars=warmup_bars,
            pca_bases=pca_bases,
            emb_proj_cache=emb_proj_cache,
            entry_floor_within_day=_entry_floor,
        )
        o = capture_objectives_from_trades(
            _champ_reuse,
            _entry_dates(df),
            ind.total_nodes(),
            min_trades=min_trades,
            cvar_frac=cvar_frac,
            cvar_tail_max=cvar_tail_max,
        )
    else:
        o = genome_capture_objectives(
            ind,
            df,
            template,
            td,
            min_trades=min_trades,
            cvar_frac=cvar_frac,
            cvar_tail_max=cvar_tail_max,
            entry_decision_mask=entry_decision_mask,
            warmup_bars=warmup_bars,
            pca_bases=pca_bases,
            emb_proj_cache=emb_proj_cache,
            entry_floor_within_day=_entry_floor,
        )
    if float(o[0]) >= _INFeasible_FLOOR:
        _set_infeasible(ind)
        return
    _precomp = None
    _excess_neg_exp = None
    ind._raw_neg_exp = float(o[0])
    if _obj1_excess_baseline():
        (_exc_vec, _traded_mask, _precomp) = genome_per_day_excess(
            ind,
            df,
            template,
            td,
            decision_offset=decision_offset,
            entry_decision_mask=entry_decision_mask,
            warmup_bars=warmup_bars,
            pca_bases=pca_bases,
            emb_proj_cache=emb_proj_cache,
            entry_floor_within_day=_entry_floor,
            precomputed_champ=_champ_reuse,
        )
        _excess_neg_exp = _excess_obj1_neg_exp(_exc_vec, _traded_mask, ind._raw_neg_exp)
    reward = 0.0
    if float(o[0]) < 0.0:
        (z, _n_fired, _n_quiet, sg_thick) = signal_gated_term(
            ind,
            df,
            template,
            td,
            decision_offset=decision_offset,
            cvar_frac=cvar_frac,
            min_count=gated_min_count,
            entry_decision_mask=entry_decision_mask,
            warmup_bars=warmup_bars,
            pca_bases=pca_bases,
            emb_proj_cache=emb_proj_cache,
            entry_floor_within_day=_entry_floor,
            precomputed=_precomp,
        )
        if sg_thick and np.isfinite(z):
            reward = max(float(z), 0.0)
    _sam_n = _sam_n_pert()
    fit = [float(o[0]), float(o[1]), -reward]
    if _excess_neg_exp is not None:
        fit[0] = float(_excess_neg_exp)
    if _sam_n:
        sharp = genome_input_sharpness(
            ind,
            df,
            template,
            td,
            feature_sigma=sam_feature_sigma(td),
            base_neg_exp=float(o[0]),
            n_pert=_sam_n,
            radius=_sam_radius(),
            min_trades=min_trades,
            cvar_frac=cvar_frac,
            entry_decision_mask=entry_decision_mask,
            warmup_bars=warmup_bars,
            pca_bases=pca_bases,
            emb_proj_cache=emb_proj_cache,
            entry_floor_within_day=_entry_floor,
        )
        fit.append(float(sharp))
    if _cia_k():
        agree = 0.0
        if float(o[0]) < 0.0:
            # Reuse the candidate's executed trades from its return objective.
            # Nested-fold and non-excess modes still need a full-fold backtest.
            _cia_trades = _champ_reuse
            if _cia_trades is None:
                _cia_trades = genome_trades(
                    ind,
                    df,
                    template,
                    td,
                    entry_decision_mask=entry_decision_mask,
                    warmup_bars=warmup_bars,
                    pca_bases=pca_bases,
                    emb_proj_cache=emb_proj_cache,
                    entry_floor_within_day=_entry_floor,
                )
            agree = cross_fold_agreement(
                _cia_trades,
                _entry_dates(df),
                k_sub=_cia_k(),
                partition_seed=int(cia_partition_seed),
                cvar_frac=cvar_frac,
                min_days=_cia_min_days(),
            )
        fit.append(-float(agree))
    ind.fitness = np.asarray(fit, dtype=float)
    ind._nodes = float(o[2])
    if _sem_attach_on():
        ind._sem = per_eligible_day_vector(
            genome_trades(
                ind,
                df,
                template,
                td,
                entry_decision_mask=entry_decision_mask,
                warmup_bars=warmup_bars,
                pca_bases=pca_bases,
                emb_proj_cache=emb_proj_cache,
                entry_floor_within_day=_entry_floor,
            ),
            _entry_dates(df),
        )


def _ref_dirs(n_obj: int, pop_size: int) -> np.ndarray:
    n_part = 12
    rd = get_reference_directions("das-dennis", n_obj, n_partitions=n_part)
    while rd.shape[0] > pop_size and n_part > 1:
        n_part -= 1
        rd = get_reference_directions("das-dennis", n_obj, n_partitions=n_part)
    return rd


def _dominates(a, b) -> bool:
    (fa, fb) = (a.fitness, b.fitness)
    return bool(np.all(fa <= fb) and np.any(fa < fb))


def _feasible_front_2obj(pop):
    feas = [
        i
        for i in pop
        if i.fitness is not None and float(i.fitness[0]) < _INFeasible_FLOOR
    ]
    if not feas:
        return []
    obj = [(float(i.fitness[0]), float(i.fitness[1])) for i in feas]
    front = []
    for i, (ei, ti) in enumerate(obj):
        dominated = False
        for j, (ej, tj) in enumerate(obj):
            if j == i:
                continue
            if ej <= ei and tj <= ti and (ej < ei or tj < ti):
                dominated = True
                break
        if not dominated:
            front.append(feas[i])
    (seen, uniq) = (set(), [])
    for ind in front:
        try:
            sig = ind.signature()
        except Exception:
            sig = id(ind)
        if sig not in seen:
            seen.add(sig)
            uniq.append(ind)
    return sorted(uniq, key=_champion_sort_key)


def _attach_front(champion, pop):
    if champion is not None:
        champion._front = _feasible_front_2obj(pop)
    return champion


def fold_degenerate_shape_routers(sexpr: str) -> str:
    if sexpr is None:
        return None
    from .grammar import from_sexpr, to_str, fold_degenerate_shape_router

    return to_str(fold_degenerate_shape_router(from_sexpr(sexpr)))


_PARSIMONY_TOURNAMENT = os.environ.get(
    "GP_PARSIMONY_TOURNAMENT", "0"
).strip().lower() in ("1", "true", "yes")


def _nested_cv_k() -> int:
    try:
        return max(0, int(os.environ.get("GP_NESTED_CV_K", "0") or "0"))
    except ValueError:
        return 0


def _nested_cv_lambda() -> float:
    try:
        return max(0.0, float(os.environ.get("GP_NESTED_CV_LAMBDA", "1.0") or "1.0"))
    except ValueError:
        return 1.0


def _nested_cv_seed() -> int:
    try:
        return int(os.environ.get("GP_NESTED_CV_SEED", "0") or "0")
    except ValueError:
        return 0


def _sam_n_pert() -> int:
    try:
        return max(0, int(os.environ.get("GP_SAM_N_PERT", "0") or "0"))
    except ValueError:
        return 0


def _sam_radius() -> float:
    try:
        return max(0.0, float(os.environ.get("GP_SAM_RADIUS", "0.10") or "0.10"))
    except ValueError:
        return 0.1


def _cia_k() -> int:
    try:
        return max(0, int(os.environ.get("GP_CIA_K", "0") or "0"))
    except ValueError:
        return 0


def _excess_obj1_neg_exp(exc_vec, traded_mask, raw_neg_exp):
    if not (exc_vec.size and bool(np.all(np.isfinite(exc_vec)))):
        return None
    if raw_neg_exp is None or float(raw_neg_exp) >= 0.0:
        return None
    t = exc_vec[traded_mask]
    if t.size < 2:
        return None
    m = float(np.mean(t))
    s = float(np.std(t, ddof=1))
    if not (m > 0.0 and s > 0.0):
        return None
    if m / (s / np.sqrt(t.size)) < 1.0:
        return None
    return -float(np.mean(exc_vec))


def _obj1_excess_baseline() -> bool:
    return os.environ.get("GP_OBJ1_EXCESS_BASELINE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _cia_min_days():
    raw = os.environ.get("GP_CIA_MIN_DAYS")
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError:
        return None
    return v if v > 0 else None


def _cia_champ_band() -> float:
    try:
        return max(0.0, float(os.environ.get("GP_CIA_CHAMP_BAND", "0.0") or "0.0"))
    except ValueError:
        return 0.0


def _cia_fitness_index():
    if not _cia_k():
        return None
    return 3 + (1 if _sam_n_pert() else 0)


def _fitness_dim() -> int:
    return 3 + (1 if _sam_n_pert() else 0) + (1 if _cia_k() else 0)


def _sem_attach_on() -> bool:
    return os.environ.get("GP_SEM_NICHING", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    ) or os.environ.get("GP_SEM_DEDUP", "0").strip().lower() in ("1", "true", "yes")


def _sem_niching_enabled(pop_size: int) -> bool:
    if pop_size < 64:
        return False
    return os.environ.get("GP_SEM_NICHING", "0").strip().lower() in ("1", "true", "yes")


def _sem_dedup_enabled(pop_size: int) -> bool:
    if pop_size < 64:
        return False
    return os.environ.get("GP_SEM_DEDUP", "0").strip().lower() in ("1", "true", "yes")


def _sem_max_per(pop_size: int) -> int:
    env = os.environ.get("GP_SEM_MAX_PER", "")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(2, pop_size // 16)


def _sem_bucket_frac() -> float:
    try:
        return max(1e-09, float(os.environ.get("GP_SEM_BUCKET_FRAC", "0.25") or "0.25"))
    except ValueError:
        return 0.25


def _tournament(pop, rng, k: int = 4):
    best = pop[int(rng.integers(len(pop)))]
    for _ in range(k - 1):
        c = pop[int(rng.integers(len(pop)))]
        if _dominates(c, best):
            best = c
        elif _PARSIMONY_TOURNAMENT and (not _dominates(best, c)):
            if getattr(c, "_nodes", 0.0) < getattr(best, "_nodes", 0.0):
                best = c
    return best


def _routed_shape_desc(ind):
    st = getattr(ind, "shape_tree", None)
    if st is not None:
        from .grammar import to_str

        return to_str(st)
    return getattr(ind, "shape", None)


def _write_heartbeat(path, tag, gen, n_gen, pop, t0):
    feas = [
        i
        for i in pop
        if i.fitness is not None and float(i.fitness[0]) < _INFeasible_FLOOR
    ]
    best = min(feas, key=_champion_sort_key) if feas else None
    snap = {
        "tag": tag,
        "gen": gen,
        "n_gen": n_gen,
        "pop": len(pop),
        "n_feasible": len(feas),
        "best_expectancy": (
            -float(getattr(best, "_raw_neg_exp", best.fitness[0]))
            if best is not None
            else None
        ),
        "best_tail": float(best.fitness[1]) if best is not None else None,
        "best_safety_reward": -float(best.fitness[2]) if best is not None else None,
        "best_shape": _routed_shape_desc(best) if best is not None else None,
        "obj1_is_excess": bool(_obj1_excess_baseline()),
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(snap, f)
    os.replace(tmp, path)
    with open(f"{path}.jsonl", "a") as hf:
        hf.write(json.dumps(snap) + "\n")


def _print_gen_progress(tag, gen, n_gen, pop, pop_size, t0):
    feas = [
        i
        for i in pop
        if i.fitness is not None and float(i.fitness[0]) < _INFeasible_FLOOR
    ]
    best = min(feas, key=_champion_sort_key) if feas else None
    best_exp = (
        -float(getattr(best, "_raw_neg_exp", best.fitness[0]))
        if best is not None
        else None
    )
    best_tail = float(best.fitness[1]) if best is not None else None
    best_reward = -float(best.fitness[2]) if best is not None else None
    best_shape = _routed_shape_desc(best) if best is not None else None
    print(
        f"[{tag}] gen {gen}/{n_gen} best_exp={best_exp} best_tail={best_tail} best_reward={best_reward} best_shape={best_shape} n_feasible={len(feas)}/{pop_size} elapsed={round(time.time() - t0, 1)}s",
        flush=True,
    )


def _feasibility_guard(
    pop, pop_size, consecutive, gen, template_name, min_trades
) -> int:
    n_feas = sum(
        (
            1
            for i in pop
            if i.fitness is not None and float(i.fitness[0]) < _INFeasible_FLOOR
        )
    )
    consecutive = consecutive + 1 if n_feas == 0 else 0
    if (
        pop_size >= _FEAS_GUARD_MIN_POP
        and consecutive >= _FEAS_GUARD_CONSECUTIVE_GENS
        and (not os.environ.get("GP_DISABLE_FEASIBILITY_GUARD"))
    ):
        raise RuntimeError(
            f"[feasibility guard] {template_name}: 0/{len(pop)} individuals feasible for {consecutive} consecutive generations (through gen {gen}) -- every individual is at the INFEASIBLE sentinel, so NSGA-III cannot select and the run would silently return an arbitrary front. In the capture loop, feasible == makes >= min_trades({min_trades}) trades, so 0-feasible is a STRUCTURAL fault (e.g. decision_offset past the day's bars, a fixture/slice with too few bars, or min_trades too high for the data) -- NOT a legitimate N1/N2 rejection (those trade, just unprofitably). Aborting fast. (Set GP_DISABLE_FEASIBILITY_GUARD=1 to override.)"
        )
    return consecutive


def _cvar_ceiling_enabled():
    return os.environ.get("GP_CVAR_TAIL_MAX_AUTO", "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )


_WORKER_CTX: dict = {}


def _eval_worker(genome_dict):
    c = _WORKER_CTX
    _cia_seed = genome_dict.pop("_cia_seed", 0)
    ind = individual_from_genome_dict(genome_dict)
    _assign_fitness(
        ind,
        c["df"],
        c["template"],
        c["td"],
        c["min_trades"],
        c["cvar_frac"],
        entry_decision_mask=c["edm"],
        grammar=c["grammar"],
        warmup_bars=c["warmup"],
        pca_bases=c["pca_bases"],
        emb_proj_cache=c["emb_proj_cache"],
        decision_offset=c["decision_offset"],
        gated_min_count=c["gated_min_count"],
        cvar_tail_max=c["cvar_tail_max"],
        cia_partition_seed=_cia_seed,
    )
    _sem = getattr(ind, "_sem", None)
    _raw_ne = getattr(ind, "_raw_neg_exp", None)
    return ([float(x) for x in ind.fitness], float(ind._nodes), _sem, _raw_ne)


class _EvalTimeout(BaseException):
    pass


def _eval_timeout_s():
    raw = os.environ.get("GP_EVAL_TIMEOUT_S")
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v > 0.0 else None


def _gen_timeout_s():
    raw = os.environ.get("GP_GEN_TIMEOUT_S")
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v > 0.0 else None


def _snap_population_thresholds(pop, grid_map=None):
    from .grammar import snap_tree_thresholds
    from .thresholds import grid_enabled

    if not grid_enabled():
        return
    for ind in pop:
        for attr in (
            "entry_tree",
            "exit_tree",
            "shape_tree",
            "size_tree",
            "delta_tree",
            "width_tree",
        ):
            t = getattr(ind, attr, None)
            if t is not None:
                snap_tree_thresholds(t, grid_map=grid_map)


def _e1_psnap() -> float:
    try:
        v = float(os.environ.get("GP_THRESHOLD_E1_PSNAP", "0.5"))
    except ValueError:
        return 0.5
    return v if v == v and v not in (float("inf"), float("-inf")) else 0.5


def _cmp_carriers():
    v = os.environ.get("GP_THRESHOLD_CMP_CARRIERS", "").strip()
    return frozenset((x.strip() for x in v.split(",") if x.strip())) if v else None


def _e1_bias_new_genomes(pop):
    from .grammar import (
        bias_comparison_data_terminals,
        forbid_noncarrier_comparison_terminals,
    )
    from .thresholds import grid_enabled

    carriers = _cmp_carriers()
    e1_on = grid_enabled()
    if not e1_on and (not carriers):
        return
    p = _e1_psnap()
    for ind in pop:
        if getattr(ind, "_e1_applied", False):
            continue
        for attr in ("entry_tree", "exit_tree", "shape_tree"):
            t = getattr(ind, attr, None)
            if t is None:
                continue
            if carriers:
                forbid_noncarrier_comparison_terminals(t, carriers)
            if e1_on:
                bias_comparison_data_terminals(t, p)
        ind._e1_applied = True


def _eval_population(
    pop,
    pool,
    df,
    template,
    td,
    min_trades,
    cvar_frac,
    edm,
    grammar,
    warmup,
    pca_bases,
    emb_proj_cache,
    decision_offset,
    gated_min_count,
    cvar_tail_max,
    grid_map=None,
    cia_gen=0,
    cia_base_seed=0,
):
    _snap_population_thresholds(pop, grid_map)
    _e1_bias_new_genomes(pop)
    _cia_seed = int(cia_base_seed) + int(cia_gen)
    if pool is None:
        timeout_s = _eval_timeout_s()
        if timeout_s is None:
            for ind in pop:
                _assign_fitness(
                    ind,
                    df,
                    template,
                    td,
                    min_trades,
                    cvar_frac,
                    edm,
                    grammar,
                    warmup_bars=warmup,
                    pca_bases=pca_bases,
                    emb_proj_cache=emb_proj_cache,
                    decision_offset=decision_offset,
                    gated_min_count=gated_min_count,
                    cvar_tail_max=cvar_tail_max,
                    cia_partition_seed=_cia_seed,
                )
            return

        def _on_alarm(signum, frame):
            raise _EvalTimeout()

        prev = signal.signal(signal.SIGALRM, _on_alarm)
        try:
            for ind in pop:
                signal.setitimer(signal.ITIMER_REAL, timeout_s)
                try:
                    _assign_fitness(
                        ind,
                        df,
                        template,
                        td,
                        min_trades,
                        cvar_frac,
                        edm,
                        grammar,
                        warmup_bars=warmup,
                        pca_bases=pca_bases,
                        emb_proj_cache=emb_proj_cache,
                        decision_offset=decision_offset,
                        gated_min_count=gated_min_count,
                        cvar_tail_max=cvar_tail_max,
                        cia_partition_seed=_cia_seed,
                    )
                except _EvalTimeout:
                    _set_infeasible(ind)
                    print(
                        f"[eval-timeout] culled 1 individual after {timeout_s:.3f}s (entry={to_str(ind.entry_tree)[:60]})",
                        flush=True,
                    )
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
        finally:
            signal.signal(signal.SIGALRM, prev)
        return
    genomes = [ind.to_genome_dict() for ind in pop]
    if _cia_k():
        for gd in genomes:
            gd["_cia_seed"] = _cia_seed
    chunk = max(1, len(genomes) // (pool._processes * 4))
    gen_timeout = _gen_timeout_s()
    async_res = pool.map_async(_eval_worker, genomes, chunksize=chunk)
    try:
        results = async_res.get(timeout=gen_timeout)
    except mp_TimeoutError:
        pool.terminate()
        raise RuntimeError(
            f"[gen-timeout] population eval exceeded GP_GEN_TIMEOUT_S={gen_timeout}s across {pool._processes} workers — a worker is hung; terminated the pool and aborting the fold (the checkpoint lets it resume). Raise/unset GP_GEN_TIMEOUT_S if this is a false positive."
        )
    for ind, (fit, nodes, sem, raw_ne) in zip(pop, results):
        ind.fitness = np.asarray(fit, dtype=float)
        ind._nodes = nodes
        if sem is not None:
            ind._sem = np.asarray(sem, dtype=float)
        if raw_ne is not None:
            ind._raw_neg_exp = float(raw_ne)


def _make_eval_pool(eval_workers, ctx):
    if not eval_workers or eval_workers <= 1:
        return None
    import multiprocessing as _mp
    from .grammar import _ensure_lookups

    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    global _WORKER_CTX
    _WORKER_CTX = ctx
    _ensure_lookups()
    return _mp.get_context("fork").Pool(eval_workers)


def _close_eval_pool(pool):
    global _WORKER_CTX
    if pool is not None:
        pool.close()
        pool.join()
    _WORKER_CTX = {}


_CHECKPOINT_EVERY_DEFAULT = 5


def _capture_run_fingerprint(
    grammar,
    template,
    pop_size,
    n_gen,
    seed,
    gate,
    decision_offset,
    min_trades,
    evolve_structure,
):
    _dedup_on = _entry_dedup_enabled(pop_size)
    payload = {
        "fmt": _CHECKPOINT_FORMAT_VERSION,
        "template": getattr(template, "name", str(template)),
        "grammar_funcs": tuple(sorted((f.name for f in grammar.functions))),
        "grammar_terms": tuple(sorted((t.name for t in grammar.terminals))),
        "grammar_max_depth": grammar.max_depth,
        "grammar_max_nodes": grammar.max_nodes,
        "pop_size": int(pop_size),
        "n_gen": int(n_gen),
        "seed": int(seed),
        "gate": bool(gate),
        "decision_offset": int(decision_offset),
        "min_trades": int(min_trades),
        "evolve_structure": bool(evolve_structure),
        "entry_dedup": bool(_dedup_on),
        "entry_max_per": int(_entry_max_per_for(pop_size)) if _dedup_on else 0,
    }
    if _genome_dedup_enabled(pop_size):
        payload["genome_dedup"] = True
    _shape_floor_active = _shape_floor_max_per(pop_size)
    if _shape_floor_active:
        payload["shape_floor_max_per"] = int(_shape_floor_active)
    if os.environ.get("GP_LEAN_PROTO_NONLINEAR", "0") not in (
        "0",
        "false",
        "False",
    ) and getattr(grammar, "_emb_proto_leaf_pairs", None):
        payload["lean_proto_nonlinear"] = True
    if _obj1_excess_baseline():
        payload["obj1_excess_baseline"] = True
    if os.environ.get("GP_ENTRY_ROBUST", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        payload["entry_robust"] = True
        payload["entry_max_conj"] = (
            os.environ.get("GP_ENTRY_MAX_CONJ", "").strip() or "2"
        )
    _k_sub = _nested_cv_k()
    if _k_sub > 1:
        payload["nested_cv_k"] = int(_k_sub)
        payload["nested_cv_lambda"] = float(_nested_cv_lambda())
        payload["nested_cv_seed"] = int(_nested_cv_seed())
    from .thresholds import grid_enabled

    if grid_enabled():
        payload["threshold_grid"] = True
        payload["threshold_e1_psnap"] = float(_e1_psnap())
    _f4 = _cmp_carriers()
    if _f4:
        payload["threshold_cmp_carriers"] = sorted(_f4)
    if _sam_n_pert():
        payload["sam_n_pert"] = int(_sam_n_pert())
        payload["sam_radius"] = float(_sam_radius())
    if _cia_k():
        payload["cia_k"] = int(_cia_k())
        payload["cia_min_days"] = (
            "derived" if _cia_min_days() is None else int(_cia_min_days())
        )
        payload["cia_champ_band"] = float(_cia_champ_band())
    if _sem_niching_enabled(pop_size):
        payload["sem_niching"] = True
    if _sem_dedup_enabled(pop_size):
        payload["sem_dedup"] = True
        payload["sem_max_per"] = int(_sem_max_per(pop_size))
        payload["sem_bucket_frac"] = float(_sem_bucket_frac())
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _serialize_capture_state(
    pop, completed_gen, rng, consecutive_zero_feasible, fingerprint
):
    return {
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "gen": int(completed_gen),
        "population": [
            (
                ind.to_genome_dict(),
                [float(x) for x in ind.fitness] if ind.fitness is not None else None,
                float(getattr(ind, "_nodes", 0.0)),
                (
                    float(getattr(ind, "_raw_neg_exp", ind.fitness[0]))
                    if ind.fitness is not None
                    else None
                ),
            )
            for ind in pop
        ],
        "np_rng_state": rng.bit_generator.state,
        "py_random_state": _random.getstate(),
        "consecutive_zero_feasible": int(consecutive_zero_feasible),
    }


def _restore_capture_pop(state):
    pop = []
    for rec in state["population"]:
        (genome_dict, fit, nodes) = (rec[0], rec[1], rec[2])
        raw_ne = rec[3] if len(rec) > 3 else None
        ind = individual_from_genome_dict(genome_dict)
        ind.fitness = np.asarray(fit, dtype=float) if fit is not None else None
        ind._nodes = float(nodes)
        if raw_ne is not None:
            ind._raw_neg_exp = float(raw_ne)
        pop.append(ind)
    return pop


def _entry_dedup_enabled(pop_size: int) -> bool:
    if pop_size < 64:
        return False
    return os.environ.get("GP_ENTRY_DEDUP", "1") not in ("0", "false", "False")


def _entry_max_per_for(pop_size: int) -> int:
    cap = max(2, pop_size // 16)
    env = os.environ.get("GP_ENTRY_MAX_PER")
    if env:
        try:
            cap = max(1, int(env))
        except ValueError:
            pass
    return cap


def _cap_entry_monoculture(pop, grammar, template, evolve_structure, max_per_entry):
    groups = {}
    for idx, ind in enumerate(pop):
        groups.setdefault(canonical_key(ind.entry_tree, _eph_round=1), []).append(idx)
    reseeded = 0
    for idxs in groups.values():
        if len(idxs) <= max_per_entry:
            continue
        for ri in sorted(idxs, key=lambda i: _champion_sort_key(pop[i]))[
            max_per_entry:
        ]:
            pop[ri] = _random_individual(grammar, template, evolve_structure, ri)
            pop[ri].fitness = None
            reseeded += 1
    return reseeded


def _genome_dedup_enabled(pop_size: int) -> bool:
    return os.environ.get("GP_GENOME_DEDUP", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _genome_signature(ind) -> str:
    parts = []
    for attr in (
        "entry_tree",
        "exit_tree",
        "size_tree",
        "delta_tree",
        "width_tree",
        "shape_tree",
    ):
        t = getattr(ind, attr, None)
        parts.append(canonical_key(t, _eph_round=1) if t is not None else "")
    parts.append(str(getattr(ind, "shape", "")))
    parts.append(repr(getattr(ind, "stop_mult", None)))
    return "|".join(parts)


def _cap_genome_clones(pop, grammar, template, evolve_structure):
    groups = {}
    for idx, ind in enumerate(pop):
        groups.setdefault(_genome_signature(ind), []).append(idx)
    reseeded = 0
    for idxs in groups.values():
        if len(idxs) <= 1:
            continue
        for ri in sorted(idxs, key=lambda i: _champion_sort_key(pop[i]))[1:]:
            pop[ri] = _random_individual(grammar, template, evolve_structure, ri)
            pop[ri].fitness = None
            reseeded += 1
    return reseeded


def _shape_floor_max_per(pop_size: int) -> int:
    try:
        n = int(os.environ.get("GP_SHAPE_MAX_PER", "0"))
    except ValueError:
        n = 0
    if n <= 0:
        return 0
    from .structure import allowed_shapes

    if len(allowed_shapes()) <= 1:
        return 0
    return n


def _shape_family_key(ind) -> Optional[str]:
    from .structure import routed_shape

    st = getattr(ind, "shape_tree", None)
    rs = routed_shape(
        to_str(st) if st is not None else None, fallback=getattr(ind, "shape", None)
    )
    if rs is None or rs == "router(multi)":
        return None
    return str(rs)


def _cap_shape_monoculture(pop, grammar, template, evolve_structure, max_per_shape):
    groups = {}
    for idx, ind in enumerate(pop):
        k = _shape_family_key(ind)
        if k is not None:
            groups.setdefault(k, []).append(idx)
    reseeded = 0
    for idxs in groups.values():
        if len(idxs) <= max_per_shape:
            continue
        for ri in sorted(idxs, key=lambda i: _champion_sort_key(pop[i]))[
            max_per_shape:
        ]:
            pop[ri] = _random_individual(grammar, template, evolve_structure, ri)
            pop[ri].fitness = None
            reseeded += 1
    return reseeded


def _semantic_signature(ind, bucket) -> str:
    s = getattr(ind, "_sem", None)
    if s is None:
        return "none"
    return ",".join((str(int(round(float(v) / bucket))) for v in np.asarray(s)))


def _cap_semantic_clones(pop, grammar, template, evolve_structure, bucket, max_per):
    groups = {}
    for idx, ind in enumerate(pop):
        groups.setdefault(_semantic_signature(ind, bucket), []).append(idx)
    reseeded = 0
    for sig, idxs in groups.items():
        if sig == "none" or len(idxs) <= max_per:
            continue
        for ri in sorted(idxs, key=lambda i: _champion_sort_key(pop[i]))[max_per:]:
            pop[ri] = _random_individual(grammar, template, evolve_structure, ri)
            pop[ri].fitness = None
            reseeded += 1
    return reseeded


_EMB_TREE_MARKERS = (
    "EmbProj_",
    "EmbDir_",
    "EmbCos_",
    "EmbNorm_",
    "EmbSub_",
    "EmbLag_",
    "PROTO_",
)


def _individual_uses_emb(ind) -> bool:
    for attr in (
        "entry_tree",
        "exit_tree",
        "delta_tree",
        "size_tree",
        "width_tree",
        "shape_tree",
    ):
        t = getattr(ind, attr, None)
        if t is not None:
            s = to_str(t)
            if any((m in s for m in _EMB_TREE_MARKERS)):
                return True
    return False


def _enforce_gen0_emb_floor(pop, grammar, template, evolve_structure, n_protected):

    def _emb_idx():
        return [i for (i, ind) in enumerate(pop) if _individual_uses_emb(ind)]

    before = _emb_idx()
    if not getattr(grammar, "_emb_proj_leaf_pairs", None):
        return (len(before), len(before))
    frac = float(os.environ.get("GP_EMB_GEN0_FLOOR", "0.5"))
    if frac <= 0.0:
        return (len(before), len(before))
    target = int(len(pop) * frac)
    if len(before) >= target:
        return (len(before), len(before))
    emb_set = set(before)
    replaceable = [i for i in range(int(n_protected), len(pop)) if i not in emb_set]
    need = target - len(before)
    for i in replaceable:
        if need <= 0:
            break
        for _try in range(25):
            cand = _random_individual(grammar, template, evolve_structure, i)
            if _individual_uses_emb(cand):
                pop[i] = cand
                need -= 1
                break
    return (len(before), len(_emb_idx()))


def evolve_capture(
    grammar,
    df,
    template,
    td,
    *,
    pop_size=40,
    n_gen=20,
    seed=0,
    evolve_structure=False,
    min_trades=10,
    cvar_frac=0.05,
    gate=True,
    decision_offset=DECISION_OFFSET,
    seed_predicates=0.0,
    compose_prob=0.0,
    heartbeat_path=None,
    heartbeat_tag="",
    pca_bases=None,
    emb_proj_cache=None,
    seed_individuals=None,
    eval_workers=1,
    _stop_after_gen=None,
):
    _trim_warn = expectancy_trim_mismatch_warning()
    if _trim_warn:
        print(_trim_warn, flush=True)
    _random.seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    _consecutive_zero_feasible = 0
    _ckpt_dir = os.environ.get("GP_CHECKPOINT_DIR") or None
    _ckpt_name = heartbeat_tag or "evolve_capture"
    _ckpt_every = _CHECKPOINT_EVERY_DEFAULT
    _env_every = os.environ.get("GP_CHECKPOINT_EVERY")
    if _env_every:
        try:
            _ckpt_every = max(1, int(_env_every))
        except ValueError:
            pass
    _ckpt_fp = (
        _capture_run_fingerprint(
            grammar,
            template,
            pop_size,
            n_gen,
            seed,
            gate,
            decision_offset,
            min_trades,
            evolve_structure,
        )
        if _ckpt_dir
        else None
    )
    edm = decision_mask(df, decision_offset) if gate else None
    _warmup = decision_offset
    from .thresholds import build_landmark_grid, grid_enabled, landmark_enabled

    grid_map = None
    if grid_enabled() and landmark_enabled():
        grid_map = {}
        for _name, _arr in (td or {}).items():
            _a = np.asarray(_arr)
            if _a.ndim == 1 and _a.size > 0 and np.issubdtype(_a.dtype, np.number):
                grid_map[_name] = build_landmark_grid(_a)
    _cvar_tail_max = None
    if _cvar_ceiling_enabled():
        _cvar_tail_max = derive_cvar_ceiling(
            df, template, td, decision_offset=decision_offset, cvar_frac=cvar_frac
        )
        _c = (
            "DISABLED (benign / no-trade baseline)"
            if _cvar_tail_max is None
            else f"{_cvar_tail_max:.3f}"
        )
        print(
            f"[{heartbeat_tag or 'evolve'}] Training-fold CVaR ceiling (always-enter per-day) = {_c}"
        )
    ref_dirs = _ref_dirs(_fitness_dim(), pop_size)
    _entry_dedup = _entry_dedup_enabled(pop_size)
    _entry_max_per = _entry_max_per_for(pop_size)
    _genome_dedup = _genome_dedup_enabled(pop_size)
    _shape_floor = _shape_floor_max_per(pop_size)
    _sem_niching = _sem_niching_enabled(pop_size)
    _sem_dedup = _sem_dedup_enabled(pop_size)
    _sem_maxper = _sem_max_per(pop_size)
    _sem_bucket = None
    if _sem_dedup:
        from .validation import _always_enter_trades

        _ae = _always_enter_trades(df, template, td, decision_offset)
        _ae_day = (
            np.array(list(per_day_sum_pnl(_ae, _entry_dates(df)).values()), dtype=float)
            if _ae
            else np.array([], dtype=float)
        )
        _ae_std = float(np.std(_ae_day)) if _ae_day.size else 0.0
        _sem_bucket = max(_sem_bucket_frac() * (_ae_std if _ae_std > 0 else 1.0), 1e-09)
        print(
            f"[{heartbeat_tag or 'evolve'}] Behavior deduplication: bucket={_sem_bucket:.4f} (always-enter per-day std {_ae_std:.4f} x frac {_sem_bucket_frac()}), max_per={_sem_maxper}"
        )
    _start_gen = 0
    _resumed = False
    pop = None
    if _ckpt_dir:
        _state = _load_checkpoint(_ckpt_dir, _ckpt_name, _ckpt_fp)
        if _state is not None:
            pop = _restore_capture_pop(_state)
            rng.bit_generator.state = _state["np_rng_state"]
            _random.setstate(_state["py_random_state"])
            _consecutive_zero_feasible = int(_state["consecutive_zero_feasible"])
            _start_gen = int(_state["gen"])
            _resumed = True
            print(
                f"[{heartbeat_tag or 'evolve'}] RESUMED from gen {_start_gen}",
                flush=True,
            )
            if _sem_attach_on():
                _ef = None if edm is not None else int(decision_offset)
                for _ind in pop:
                    if (
                        getattr(_ind, "_sem", None) is None
                        and _ind.fitness is not None
                        and (float(_ind.fitness[0]) < _INFeasible_FLOOR)
                    ):
                        _ind._sem = per_eligible_day_vector(
                            genome_trades(
                                _ind,
                                df,
                                template,
                                td,
                                entry_decision_mask=edm,
                                warmup_bars=_warmup,
                                pca_bases=pca_bases,
                                emb_proj_cache=emb_proj_cache,
                                entry_floor_within_day=_ef,
                            ),
                            _entry_dates(df),
                        )
    if not _resumed:
        pop = [
            _random_individual(grammar, template, evolve_structure, i)
            for i in range(pop_size)
        ]
        seed_population_predicates(pop, grammar, seed_predicates)
        if seed_individuals:
            n_inj = min(len(seed_individuals), pop_size)
            pop[:n_inj] = list(seed_individuals[:n_inj])
        _n_protected = min(len(seed_individuals), pop_size) if seed_individuals else 0
        (_emb0_before, _emb0_after) = _enforce_gen0_emb_floor(
            pop, grammar, template, evolve_structure, _n_protected
        )
        if getattr(grammar, "_emb_proj_leaf_pairs", None):
            print(
                f"[evolve] gen-0 embedding presence: {_emb0_after}/{pop_size} ({100.0 * _emb0_after / max(pop_size, 1):.0f}%) carry an emb op (floor={os.environ.get('GP_EMB_GEN0_FLOOR', '0.5')}; {_emb0_before} before enforcement)",
                flush=True,
            )
    _pool = _make_eval_pool(
        eval_workers,
        {
            "df": df,
            "template": template,
            "td": td,
            "min_trades": min_trades,
            "cvar_frac": cvar_frac,
            "edm": edm,
            "grammar": grammar,
            "warmup": _warmup,
            "pca_bases": pca_bases,
            "emb_proj_cache": emb_proj_cache,
            "decision_offset": decision_offset,
            "gated_min_count": _GATED_MIN_COUNT_DAYS,
            "cvar_tail_max": _cvar_tail_max,
        },
    )
    try:
        if not _resumed:
            _eval_population(
                pop,
                _pool,
                df,
                template,
                td,
                min_trades,
                cvar_frac,
                edm,
                grammar,
                _warmup,
                pca_bases,
                emb_proj_cache,
                decision_offset,
                _GATED_MIN_COUNT_DAYS,
                _cvar_tail_max,
                grid_map=grid_map,
                cia_gen=0,
                cia_base_seed=seed,
            )
            _consecutive_zero_feasible = _feasibility_guard(
                pop, pop_size, _consecutive_zero_feasible, 0, template.name, min_trades
            )
            if heartbeat_path:
                _write_heartbeat(heartbeat_path, heartbeat_tag, 0, n_gen, pop, t0)
            _print_gen_progress(heartbeat_tag or "evolve", 0, n_gen, pop, pop_size, t0)
            if _ckpt_dir:
                _write_checkpoint_atomic(
                    _ckpt_dir,
                    _ckpt_name,
                    _serialize_capture_state(
                        pop, 0, rng, _consecutive_zero_feasible, _ckpt_fp
                    ),
                )
                if _stop_after_gen == 0:
                    _close_eval_pool(_pool)
                    _pool = None
                    feasible0 = [
                        i
                        for i in pop
                        if i.fitness is not None
                        and float(i.fitness[0]) < _INFeasible_FLOOR
                    ]
                    return _attach_front(_pick_champion(feasible0), pop)
        for _gen in range(_start_gen, n_gen):
            offspring = []
            while len(offspring) < pop_size:
                (pa, pb) = (_tournament(pop, rng), _tournament(pop, rng))
                (c1, c2) = crossover(pa, pb, grammar, evolve_size=evolve_structure)
                offspring.append(
                    compose_mutate(
                        mutate(c1, grammar, evolve_size=evolve_structure),
                        grammar,
                        compose_prob,
                    )
                )
                offspring.append(
                    compose_mutate(
                        mutate(c2, grammar, evolve_size=evolve_structure),
                        grammar,
                        compose_prob,
                    )
                )
            offspring = offspring[:pop_size]
            _eval_population(
                offspring,
                _pool,
                df,
                template,
                td,
                min_trades,
                cvar_frac,
                edm,
                grammar,
                _warmup,
                pca_bases,
                emb_proj_cache,
                decision_offset,
                _GATED_MIN_COUNT_DAYS,
                _cvar_tail_max,
                grid_map=grid_map,
                cia_gen=_gen + 1,
                cia_base_seed=seed,
            )
            pop = nsga3_select(
                pop + offspring, ref_dirs, pop_size, rng, semantic_niching=_sem_niching
            )
            if _entry_dedup and _gen + 1 < n_gen:
                if _cap_entry_monoculture(
                    pop, grammar, template, evolve_structure, _entry_max_per
                ):
                    _eval_population(
                        [i for i in pop if i.fitness is None],
                        _pool,
                        df,
                        template,
                        td,
                        min_trades,
                        cvar_frac,
                        edm,
                        grammar,
                        _warmup,
                        pca_bases,
                        emb_proj_cache,
                        decision_offset,
                        _GATED_MIN_COUNT_DAYS,
                        _cvar_tail_max,
                        grid_map=grid_map,
                        cia_gen=_gen + 1,
                        cia_base_seed=seed,
                    )
            if _genome_dedup and _gen + 1 < n_gen:
                if _cap_genome_clones(pop, grammar, template, evolve_structure):
                    _eval_population(
                        [i for i in pop if i.fitness is None],
                        _pool,
                        df,
                        template,
                        td,
                        min_trades,
                        cvar_frac,
                        edm,
                        grammar,
                        _warmup,
                        pca_bases,
                        emb_proj_cache,
                        decision_offset,
                        _GATED_MIN_COUNT_DAYS,
                        _cvar_tail_max,
                        grid_map=grid_map,
                        cia_gen=_gen + 1,
                        cia_base_seed=seed,
                    )
            if _shape_floor and _gen + 1 < n_gen:
                if _cap_shape_monoculture(
                    pop, grammar, template, evolve_structure, _shape_floor
                ):
                    _eval_population(
                        [i for i in pop if i.fitness is None],
                        _pool,
                        df,
                        template,
                        td,
                        min_trades,
                        cvar_frac,
                        edm,
                        grammar,
                        _warmup,
                        pca_bases,
                        emb_proj_cache,
                        decision_offset,
                        _GATED_MIN_COUNT_DAYS,
                        _cvar_tail_max,
                        grid_map=grid_map,
                        cia_gen=_gen + 1,
                        cia_base_seed=seed,
                    )
            if _sem_dedup and _sem_bucket is not None and (_gen + 1 < n_gen):
                if _cap_semantic_clones(
                    pop, grammar, template, evolve_structure, _sem_bucket, _sem_maxper
                ):
                    _eval_population(
                        [i for i in pop if i.fitness is None],
                        _pool,
                        df,
                        template,
                        td,
                        min_trades,
                        cvar_frac,
                        edm,
                        grammar,
                        _warmup,
                        pca_bases,
                        emb_proj_cache,
                        decision_offset,
                        _GATED_MIN_COUNT_DAYS,
                        _cvar_tail_max,
                        grid_map=grid_map,
                        cia_gen=_gen + 1,
                        cia_base_seed=seed,
                    )
            _consecutive_zero_feasible = _feasibility_guard(
                pop,
                pop_size,
                _consecutive_zero_feasible,
                _gen + 1,
                template.name,
                min_trades,
            )
            if heartbeat_path:
                _write_heartbeat(
                    heartbeat_path, heartbeat_tag, _gen + 1, n_gen, pop, t0
                )
            _print_gen_progress(
                heartbeat_tag or "evolve", _gen + 1, n_gen, pop, pop_size, t0
            )
            if _ckpt_dir and (
                (_gen + 1) % _ckpt_every == 0 or _gen + 1 == _stop_after_gen
            ):
                _write_checkpoint_atomic(
                    _ckpt_dir,
                    _ckpt_name,
                    _serialize_capture_state(
                        pop, _gen + 1, rng, _consecutive_zero_feasible, _ckpt_fp
                    ),
                )
            if _stop_after_gen is not None and _gen + 1 == _stop_after_gen:
                _close_eval_pool(_pool)
                _pool = None
                feasible_s = [
                    i
                    for i in pop
                    if i.fitness is not None and float(i.fitness[0]) < _INFeasible_FLOOR
                ]
                return _attach_front(_pick_champion(feasible_s), pop)
        if _ckpt_dir:
            _delete_checkpoint(_ckpt_dir, _ckpt_name)
    finally:
        _close_eval_pool(_pool)
    feasible = [
        ind
        for ind in pop
        if ind.fitness is not None and float(ind.fitness[0]) < _INFeasible_FLOOR
    ]
    if not feasible:
        return None
    return _attach_front(_pick_champion(feasible), pop)


def router_folded_genome(genome: dict) -> dict:
    g = dict(genome)
    st = g.get("shape_tree")
    if st is not None:
        g["shape_tree"] = fold_degenerate_shape_routers(st)
        from .structure import resolve_constant_shape

        _rs = resolve_constant_shape(g["shape_tree"])
        if _rs:
            g["shape"] = _rs
    return g


def build_survivor_record(
    ind,
    test_df,
    template,
    test_td,
    *,
    gate,
    decision_offset,
    min_trades,
    perm_seed,
    pca_bases=None,
    emb_proj_cache=None,
    rank=0,
    train_df=None,
    train_td=None,
    train_emb_proj_cache=None,
):
    v = judge_champion(
        ind,
        test_df,
        template,
        test_td,
        perm_seed=perm_seed,
        gate_decisions=gate,
        decision_offset=decision_offset,
        min_trades=min_trades,
        pca_bases=pca_bases,
        emb_proj_cache=emb_proj_cache,
    )
    rec = {
        "_rank_in_fold": rank,
        "train_objectives": (
            [float(x) for x in ind.fitness] if ind.fitness is not None else None
        ),
        "train_neg_exp_raw": (
            float(getattr(ind, "_raw_neg_exp", ind.fitness[0]))
            if ind.fitness is not None
            else None
        ),
        "obj1_is_excess": bool(_obj1_excess_baseline()),
        "accepted": v["accepted"],
        "champion_oos_mean_pnl": v["champion_oos_mean_pnl"],
        "mean_per_day_excess": v["mean_per_day_excess"],
        "permutation_p": v["permutation_p"],
        "n_champion_trades": v["n_champion_trades"],
        "tail_ok": v["tail_ok"],
        "champion_day_cvar": v["champion_day_cvar"],
        "baseline_day_cvar": v["baseline_day_cvar"],
        "traded_day_edge": v["traded_day_edge"],
        "selection_credit": v["selection_credit"],
        "traded_coverage": v["traded_coverage"],
        "held_out_day_excess": v["held_out_day_excess"],
        "held_out_days": v["held_out_days"],
        "held_out_traded_day_pnl": v["held_out_traded_day_pnl"],
        "held_out_traded_days": v["held_out_traded_days"],
        "traded_day_p": v["traded_day_p"],
        "traded_day_obs": v["traded_day_obs"],
        "honesty": classify_abstention(v),
        **{
            kk: v[kk]
            for kk in (
                "held_out_day_pnl_full",
                "held_out_fire_mask",
                "coverage_matched_obs",
                "coverage_matched_p",
                "cms_resolution_ok",
                "cms_stratified",
            )
            if kk in v
        },
        "shape": ind.shape,
        "entry_rule": to_str(ind.entry_tree),
        "genome": router_folded_genome(ind.to_genome_dict()),
        "intraday": v["intraday"],
    }
    if train_df is not None and train_td is not None:
        from .validation import _champion_trades as _ct
        from .fitness import per_day_sum_pnl as _pdsp

        _ef = None if gate else int(decision_offset)
        _tr = _ct(
            ind,
            train_df,
            template,
            train_td,
            warmup_bars=int(decision_offset),
            entry_floor_within_day=_ef,
            pca_bases=pca_bases,
            emb_proj_cache=train_emb_proj_cache,
        )
        rec["train_traded_day_pnl"] = [
            float(x) for x in _pdsp(_tr, train_df["date"].values).values()
        ]
    return rec


def build_surviving_set(
    champion,
    test_df,
    template,
    test_td,
    *,
    gate,
    decision_offset,
    min_trades,
    perm_seed,
    pca_bases=None,
    emb_proj_cache=None,
    train_df=None,
    train_td=None,
    train_emb_proj_cache=None,
):
    front = getattr(champion, "_front", None) or []
    out = []
    for rank, ind in enumerate(front):
        out.append(
            build_survivor_record(
                ind,
                test_df,
                template,
                test_td,
                gate=gate,
                decision_offset=decision_offset,
                min_trades=min_trades,
                perm_seed=perm_seed,
                pca_bases=pca_bases,
                emb_proj_cache=emb_proj_cache,
                rank=rank,
                train_df=train_df,
                train_td=train_td,
                train_emb_proj_cache=train_emb_proj_cache,
            )
        )
    return out
