"""Typed crossover, mutation and NSGA-III selection for strategy populations."""

from __future__ import annotations
import os
import pickle
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga3 import (
    HyperplaneNormalization,
    associate_to_niches,
    calc_niche_count,
    niching,
)
from .grammar import (
    GType,
    Grammar,
    Node,
    canonical_key,
    deep_copy,
    is_structurally_degenerate,
    has_illegal_position_state,
    fold_degenerate_shape_router,
    node_count,
    to_str,
    simplify_tree,
    GT,
    LT,
    AND,
    OR,
    FuncNode,
    TermNode,
    TermDef,
)

STOP_MULT_MAX: float = 4.0
STOP_MULT_DEFAULT: float = 2.5


@dataclass
class Individual:
    entry_tree: Node
    exit_tree: Node
    size_tree: Node
    template_name: str
    delta_tree: Optional[Node] = None
    width_tree: Optional[Node] = None
    shape: Optional[str] = None
    shape_tree: Optional[Node] = None
    stop_mult: float = STOP_MULT_DEFAULT
    fitness: Optional[np.ndarray] = None

    def total_nodes(self) -> int:
        total = (
            node_count(self.entry_tree)
            + node_count(self.exit_tree)
            + node_count(self.size_tree)
        )
        if self.delta_tree is not None:
            total += node_count(self.delta_tree)
        if self.width_tree is not None:
            total += node_count(self.width_tree)
        if self.shape_tree is not None:
            total += node_count(self.shape_tree)
        return total

    def signature(self) -> str:
        base = f"E:{canonical_key(self.entry_tree)}|X:{canonical_key(self.exit_tree)}|S:{canonical_key(self.size_tree)}"
        if self.delta_tree is not None:
            base += f"|D:{canonical_key(self.delta_tree)}"
        if self.width_tree is not None:
            base += f"|W:{canonical_key(self.width_tree)}"
        if self.shape_tree is not None:
            base += f"|ST:{canonical_key(self.shape_tree)}"
        elif self.shape is not None:
            base += f"|SH:{self.shape}"
        base += f"|SM:{self.stop_mult:.3f}"
        return base

    def to_genome_dict(self) -> dict:

        return {
            "entry_tree": to_str(self.entry_tree),
            "exit_tree": to_str(self.exit_tree),
            "size_tree": to_str(self.size_tree),
            "delta_tree": (
                to_str(self.delta_tree) if self.delta_tree is not None else None
            ),
            "width_tree": (
                to_str(self.width_tree) if self.width_tree is not None else None
            ),
            "shape": self.shape,
            "shape_tree": (
                to_str(self.shape_tree) if self.shape_tree is not None else None
            ),
            "stop_mult": float(self.stop_mult),
            "template_name": self.template_name,
            "fitness": (
                [float(x) for x in self.fitness] if self.fitness is not None else None
            ),
        }


def individual_from_genome_dict(d: dict) -> "Individual":
    from .grammar import from_sexpr

    return Individual(
        entry_tree=from_sexpr(d["entry_tree"]),
        exit_tree=from_sexpr(d["exit_tree"]),
        size_tree=from_sexpr(d["size_tree"]),
        template_name=d["template_name"],
        delta_tree=from_sexpr(d["delta_tree"]) if d.get("delta_tree") else None,
        width_tree=from_sexpr(d["width_tree"]) if d.get("width_tree") else None,
        shape=d.get("shape"),
        shape_tree=from_sexpr(d["shape_tree"]) if d.get("shape_tree") else None,
        stop_mult=float(d.get("stop_mult", STOP_MULT_DEFAULT)),
    )


def _reflect_into_range(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return lo
    span = hi - lo
    y = x - lo
    y = y % (2.0 * span)
    if y < 0.0:
        y += 2.0 * span
    if y > span:
        y = 2.0 * span - y
    return lo + y


def crossover(
    parent_a: Individual,
    parent_b: Individual,
    grammar: Grammar,
    evolve_size: bool = False,
) -> Tuple[Individual, Individual]:
    (e1, e2) = grammar.crossover(parent_a.entry_tree, parent_b.entry_tree)
    (x1, x2) = grammar.crossover(parent_a.exit_tree, parent_b.exit_tree)
    if evolve_size:
        (s1, s2) = grammar.crossover(parent_a.size_tree, parent_b.size_tree)
    else:
        (s1, s2) = (deep_copy(parent_a.size_tree), deep_copy(parent_b.size_tree))
    (d1, d2) = (None, None)
    if parent_a.delta_tree is not None and parent_b.delta_tree is not None:
        (d1, d2) = grammar.crossover(parent_a.delta_tree, parent_b.delta_tree)
    elif parent_a.delta_tree is not None:
        (d1, d2) = (deep_copy(parent_a.delta_tree), deep_copy(parent_a.delta_tree))
    elif parent_b.delta_tree is not None:
        (d1, d2) = (deep_copy(parent_b.delta_tree), deep_copy(parent_b.delta_tree))
    (w1, w2) = (None, None)
    if parent_a.width_tree is not None and parent_b.width_tree is not None:
        (w1, w2) = grammar.crossover(parent_a.width_tree, parent_b.width_tree)
    elif parent_a.width_tree is not None:
        (w1, w2) = (deep_copy(parent_a.width_tree), deep_copy(parent_a.width_tree))
    elif parent_b.width_tree is not None:
        (w1, w2) = (deep_copy(parent_b.width_tree), deep_copy(parent_b.width_tree))
    (st1, st2) = (None, None)
    if parent_a.shape_tree is not None and parent_b.shape_tree is not None:
        (st1, st2) = grammar.crossover(parent_a.shape_tree, parent_b.shape_tree)
    elif parent_a.shape_tree is not None:
        (st1, st2) = (deep_copy(parent_a.shape_tree), deep_copy(parent_a.shape_tree))
    elif parent_b.shape_tree is not None:
        (st1, st2) = (deep_copy(parent_b.shape_tree), deep_copy(parent_b.shape_tree))
    from .grammar import simplify_tree

    (e1, e2) = (simplify_tree(e1), simplify_tree(e2))
    (x1, x2) = (simplify_tree(x1), simplify_tree(x2))
    if d1 is not None:
        d1 = simplify_tree(d1)
    if d2 is not None:
        d2 = simplify_tree(d2)
    if w1 is not None:
        w1 = simplify_tree(w1)
    if w2 is not None:
        w2 = simplify_tree(w2)
    if evolve_size:
        (s1, s2) = (simplify_tree(s1), simplify_tree(s2))
    max_n = grammar.max_nodes
    if node_count(e1) > max_n:
        e1 = deep_copy(parent_a.entry_tree)
    if node_count(e2) > max_n:
        e2 = deep_copy(parent_b.entry_tree)
    if node_count(x1) > max_n:
        x1 = deep_copy(parent_a.exit_tree)
    if node_count(x2) > max_n:
        x2 = deep_copy(parent_b.exit_tree)
    if node_count(s1) > max_n:
        s1 = deep_copy(parent_a.size_tree)
    if node_count(s2) > max_n:
        s2 = deep_copy(parent_b.size_tree)
    if d1 is not None and node_count(d1) > max_n:
        d1 = deep_copy(parent_a.delta_tree)
    if d2 is not None and node_count(d2) > max_n:
        d2 = deep_copy(parent_b.delta_tree)
    if w1 is not None and node_count(w1) > max_n:
        w1 = deep_copy(parent_a.width_tree)
    if w2 is not None and node_count(w2) > max_n:
        w2 = deep_copy(parent_b.width_tree)
    if st1 is not None and node_count(st1) > max_n:
        st1 = deep_copy(parent_a.shape_tree)
    if st2 is not None and node_count(st2) > max_n:
        st2 = deep_copy(parent_b.shape_tree)
    st1 = fold_degenerate_shape_router(st1) if st1 is not None else None
    st2 = fold_degenerate_shape_router(st2) if st2 is not None else None
    _alpha = random.random()
    sm1 = _alpha * parent_a.stop_mult + (1.0 - _alpha) * parent_b.stop_mult
    sm2 = _alpha * parent_b.stop_mult + (1.0 - _alpha) * parent_a.stop_mult
    sm1 = min(STOP_MULT_MAX, max(0.0, sm1))
    sm2 = min(STOP_MULT_MAX, max(0.0, sm2))
    return (
        Individual(
            e1,
            x1,
            s1,
            parent_a.template_name,
            delta_tree=d1,
            width_tree=w1,
            shape=parent_a.shape,
            shape_tree=st1,
            stop_mult=sm1,
        ),
        Individual(
            e2,
            x2,
            s2,
            parent_b.template_name,
            delta_tree=d2,
            width_tree=w2,
            shape=parent_b.shape,
            shape_tree=st2,
            stop_mult=sm2,
        ),
    )


def mutate(
    ind: Individual,
    grammar: Grammar,
    mutation_rate: float = 0.15,
    subtree_mutation_rate: float = 0.1,
    evolve_size: bool = False,
) -> Individual:
    new_entry = ind.entry_tree
    new_exit = ind.exit_tree
    new_size = ind.size_tree
    _mut_fails = 0
    if random.random() < mutation_rate:
        try:
            new_entry = grammar.point_mutate(ind.entry_tree, p=0.2)
        except (ValueError, RecursionError):
            _mut_fails += 1
    if random.random() < mutation_rate:
        try:
            new_exit = grammar.point_mutate(ind.exit_tree, p=0.2)
        except (ValueError, RecursionError):
            _mut_fails += 1
    if random.random() < subtree_mutation_rate:
        try:
            new_entry = grammar.subtree_mutate(new_entry, max_d=3)
        except (ValueError, RecursionError):
            _mut_fails += 1
    if random.random() < subtree_mutation_rate:
        try:
            new_exit = grammar.subtree_mutate(new_exit, max_d=3)
        except (ValueError, RecursionError):
            _mut_fails += 1
    if evolve_size:
        if random.random() < mutation_rate:
            try:
                new_size = grammar.point_mutate(new_size, p=0.2)
            except (ValueError, RecursionError):
                _mut_fails += 1
        if random.random() < subtree_mutation_rate:
            try:
                new_size = grammar.subtree_mutate(new_size, max_d=3)
            except (ValueError, RecursionError):
                _mut_fails += 1
    new_delta = ind.delta_tree
    if new_delta is not None:
        if random.random() < mutation_rate:
            try:
                new_delta = grammar.point_mutate(new_delta, p=0.2)
            except (ValueError, RecursionError):
                _mut_fails += 1
        if random.random() < subtree_mutation_rate:
            try:
                new_delta = grammar.subtree_mutate(new_delta, max_d=3)
            except (ValueError, RecursionError):
                _mut_fails += 1
    new_width = ind.width_tree
    if new_width is not None:
        if random.random() < mutation_rate:
            try:
                new_width = grammar.point_mutate(new_width, p=0.2)
            except (ValueError, RecursionError):
                _mut_fails += 1
        if random.random() < subtree_mutation_rate:
            try:
                new_width = grammar.subtree_mutate(new_width, max_d=3)
            except (ValueError, RecursionError):
                _mut_fails += 1
    new_shape_tree = ind.shape_tree
    if new_shape_tree is not None:
        if random.random() < mutation_rate:
            try:
                new_shape_tree = grammar.point_mutate(new_shape_tree, p=0.2)
            except (ValueError, RecursionError):
                _mut_fails += 1
        if random.random() < subtree_mutation_rate:
            try:
                new_shape_tree = grammar.subtree_mutate(new_shape_tree, max_d=3)
            except (ValueError, RecursionError):
                _mut_fails += 1
    new_shape = ind.shape
    if new_shape is not None:
        from .structure import allowed_shapes

        _allowed = allowed_shapes()
        if random.random() < mutation_rate or new_shape not in _allowed:
            _alts = [s for s in _allowed if s != new_shape]
            if _alts:
                new_shape = random.choice(_alts)
    new_stop = ind.stop_mult
    if random.random() < mutation_rate:
        new_stop = _reflect_into_range(
            ind.stop_mult + random.gauss(0.0, 0.1 * STOP_MULT_MAX), 0.0, STOP_MULT_MAX
        )
    if _mut_fails > 0:
        if not hasattr(mutate, "_total_fails"):
            mutate._total_fails = 0
            mutate._total_calls = 0
        mutate._total_fails += _mut_fails
        mutate._total_calls += 1
    from .grammar import simplify_tree

    new_entry = simplify_tree(new_entry)
    new_exit = simplify_tree(new_exit)
    if new_delta is not None:
        new_delta = simplify_tree(new_delta)
    if new_width is not None:
        new_width = simplify_tree(new_width)
    if evolve_size:
        new_size = simplify_tree(new_size)
    max_n = grammar.max_nodes
    if node_count(new_entry) > max_n:
        new_entry = ind.entry_tree
    if node_count(new_exit) > max_n:
        new_exit = ind.exit_tree
    if node_count(new_size) > max_n:
        new_size = ind.size_tree
    if new_delta is not None and node_count(new_delta) > max_n:
        new_delta = ind.delta_tree
    if new_width is not None and node_count(new_width) > max_n:
        new_width = ind.width_tree
    if new_shape_tree is not None and node_count(new_shape_tree) > max_n:
        new_shape_tree = ind.shape_tree
    new_shape_tree = (
        fold_degenerate_shape_router(new_shape_tree)
        if new_shape_tree is not None
        else None
    )
    return Individual(
        new_entry,
        new_exit,
        new_size,
        ind.template_name,
        delta_tree=new_delta,
        width_tree=new_width,
        shape=new_shape,
        shape_tree=new_shape_tree,
        stop_mult=new_stop,
    )


def _seed_shape(r):
    from .structure import allowed_shapes

    shapes = allowed_shapes()
    if not os.getenv("GP_VERTICALS_PRIOR"):
        return r.choice(shapes)
    try:
        w = float(os.getenv("GP_VERTICALS_PRIOR_WEIGHT", "2.0"))
    except ValueError:
        w = 2.0
    w = max(w, 0.0)
    weights = [w if s in ("put_vertical", "call_vertical") else 1.0 for s in shapes]
    return r.choices(shapes, weights=weights, k=1)[0]


def seed_structure_genes(
    ind: Individual,
    grammar: Grammar,
    max_d: int = 3,
    rng: Optional[random.Random] = None,
) -> Individual:
    from dataclasses import replace
    from .grammar import has_illegal_position_state

    r = rng or random

    def _capped_clean(gtype):
        t = grammar._gen_capped(gtype, max_d)
        for _ in range(5):
            if not has_illegal_position_state(t, "delta"):
                return t
            t = grammar._gen_capped(gtype, max_d)
        for _ in range(8):
            leaf = grammar._random_real_leaf(gtype)
            if not has_illegal_position_state(leaf, "delta"):
                return leaf
        return t

    width = _capped_clean(GType.REAL)
    shape_tree = _capped_clean(GType.SHAPE_ID)
    shape_tree = (
        fold_degenerate_shape_router(shape_tree) if shape_tree is not None else None
    )
    return replace(ind, width_tree=width, shape=_seed_shape(r), shape_tree=shape_tree)


_EPH_THRESHOLD_RANGE = (-3.0, 3.0)


def _data_real_terminals(grammar: Grammar) -> List[TermDef]:
    out: List[TermDef] = []
    zero = TermNode(defn=TermDef("EphReal", GType.REAL), value=0.0)
    for t in grammar.terminals:
        if not (t.ret_type == GType.REAL and t.value is None and (t.sampler is None)):
            continue
        probe = FuncNode(defn=GT, children=[TermNode(defn=t), zero])
        if has_illegal_position_state(probe, "entry") or is_structurally_degenerate(
            probe
        ):
            continue
        out.append(t)
    return out


def _eph_threshold(r) -> TermNode:
    return TermNode(
        defn=TermDef("EphReal", GType.REAL),
        value=round(r.uniform(*_EPH_THRESHOLD_RANGE), 4),
    )


def _tree_terminal_names(node) -> set:
    if isinstance(node, TermNode):
        return {node.name}
    names: set = set()
    for c in getattr(node, "children", ()):
        names |= _tree_terminal_names(c)
    return names


def _atomic_predicate(term: TermDef, r) -> FuncNode:
    op = GT if r.random() < 0.5 else LT
    return FuncNode(defn=op, children=[TermNode(defn=term), _eph_threshold(r)])


def seed_predicate_library(
    grammar: Grammar, rng: Optional[random.Random] = None
) -> List[FuncNode]:
    r = rng or random
    lib: List[FuncNode] = []
    for t in _data_real_terminals(grammar):
        lib.append(FuncNode(defn=GT, children=[TermNode(defn=t), _eph_threshold(r)]))
        lib.append(FuncNode(defn=LT, children=[TermNode(defn=t), _eph_threshold(r)]))
    return lib


def compose_mutate(
    ind: Individual,
    grammar: Grammar,
    compose_prob: float = 0.0,
    rng: Optional[random.Random] = None,
) -> Individual:
    r = rng or random
    if compose_prob <= 0.0 or r.random() >= compose_prob:
        return ind
    terms = _data_real_terminals(grammar)
    if not terms:
        return ind
    present = _tree_terminal_names(ind.entry_tree)
    candidates = [t for t in terms if t.name not in present] or terms
    op = AND if r.random() < 0.5 else OR
    new_entry = simplify_tree(
        FuncNode(
            defn=op,
            children=[
                deep_copy(ind.entry_tree),
                _atomic_predicate(r.choice(candidates), r),
            ],
        )
    )
    if node_count(new_entry) > grammar.max_nodes:
        return ind
    from dataclasses import replace

    return replace(ind, entry_tree=new_entry)


def seed_population_predicates(
    pop: List[Individual],
    grammar: Grammar,
    frac: float,
    rng: Optional[random.Random] = None,
) -> None:
    if frac <= 0.0:
        return
    r = rng or random
    terms = _data_real_terminals(grammar)
    lib = seed_predicate_library(grammar, r)
    if not lib:
        return
    from dataclasses import replace

    n_seed = min(int(round(frac * len(pop))), len(pop))
    for k in range(n_seed):
        if k % 3 == 2 and terms:
            op = AND if r.random() < 0.5 else OR
            (t1, t2) = r.sample(terms, 2) if len(terms) >= 2 else (terms[0], terms[0])
            entry = FuncNode(
                defn=op, children=[_atomic_predicate(t1, r), _atomic_predicate(t2, r)]
            )
        else:
            entry = deep_copy(lib[k % len(lib)])
        pop[k] = replace(pop[k], entry_tree=entry)


def _semantic_maxmin_local(
    splitting_front, until_last_front, population, n_remaining, rng
):
    ref = [
        np.asarray(population[int(i)]._sem, dtype=np.float64) for i in until_last_front
    ]
    front_sem = [
        np.asarray(population[int(g)]._sem, dtype=np.float64) for g in splitting_front
    ]
    n_front = len(front_sem)
    n_take = int(min(n_remaining, n_front))
    (chosen, chosen_set) = ([], set())
    for _ in range(n_take):
        (best_d, tied) = (-np.inf, [])
        for pos in range(n_front):
            if pos in chosen_set:
                continue
            s = front_sem[pos]
            d = (
                min((float(np.linalg.norm(s - r)) for r in ref))
                if ref
                else float(np.linalg.norm(s))
            )
            if d > best_d * (1.0 + 1e-09) + 1e-12:
                (best_d, tied) = (d, [pos])
            elif abs(d - best_d) <= abs(best_d) * 1e-09 + 1e-12:
                tied.append(pos)
        pick = (
            int(tied[0]) if len(tied) == 1 else int(tied[int(rng.integers(len(tied)))])
        )
        chosen.append(pick)
        chosen_set.add(pick)
        ref.append(front_sem[pick])
    return chosen


def nsga3_select(
    population: List[Individual],
    ref_dirs: np.ndarray,
    n_survivors: int,
    rng: Optional[np.random.Generator] = None,
    *,
    semantic_niching: bool = False,
) -> List[Individual]:
    if not population:
        return []
    if rng is None:
        raise ValueError(
            "nsga3_select requires an explicit numpy.random.Generator for reproducible selection. "
            "Pass the caller's generator initialized with the recorded search seed."
        )
    F = np.array([ind.fitness for ind in population])
    nds = NonDominatedSorting()
    fronts = nds.do(F, return_rank=False, n_stop_if_ranked=n_survivors)
    survivors_indices: List[int] = []
    last_front_idx = 0
    for i, front in enumerate(fronts):
        if len(survivors_indices) + len(front) <= n_survivors:
            survivors_indices.extend((int(x) for x in front))
            last_front_idx = i
            if len(survivors_indices) == n_survivors:
                return [population[i] for i in survivors_indices]
        else:
            last_front_idx = i
            break
    splitting_front = fronts[last_front_idx]
    until_last_front = (
        np.concatenate(fronts[:last_front_idx])
        if last_front_idx > 0
        else np.array([], dtype=int)
    )
    norm = HyperplaneNormalization(ref_dirs.shape[1])
    norm.update(F, nds=fronts[0])
    (ideal, nadir) = (norm.ideal_point, norm.nadir_point)
    all_indices = np.concatenate((until_last_front, splitting_front)).astype(int)
    F_all = F[all_indices]
    (niche_of_individuals, dist_to_niche, _) = associate_to_niches(
        F_all, ref_dirs, ideal, nadir
    )
    n_until = len(until_last_front)
    niche_count = calc_niche_count(len(ref_dirs), niche_of_individuals[:n_until])
    n_remaining = n_survivors - n_until
    _ref_idx = list(until_last_front) + list(splitting_front)
    _sems = [getattr(population[int(i)], "_sem", None) for i in _ref_idx]
    _sem_ok = (
        semantic_niching
        and len(splitting_front) > 0
        and all((s is not None for s in _sems))
        and (len({np.asarray(s).shape for s in _sems}) == 1)
    )
    if _sem_ok:
        chosen_local = _semantic_maxmin_local(
            splitting_front, until_last_front, population, n_remaining, rng
        )
    else:
        last_niche_of = niche_of_individuals[n_until:]
        last_dist = dist_to_niche[n_until:]
        chosen_local = niching(
            pop=np.arange(len(splitting_front)),
            n_remaining=n_remaining,
            niche_count=niche_count,
            niche_of_individuals=last_niche_of,
            dist_to_niche=last_dist,
            random_state=rng,
        )
    survivors_indices.extend((int(splitting_front[i]) for i in chosen_local))
    return [population[i] for i in survivors_indices]


_CHECKPOINT_FORMAT_VERSION = 5


def _checkpoint_path(checkpoint_dir: "Path", template_name: str) -> "Path":
    return Path(checkpoint_dir) / f"checkpoint_{template_name}.pkl"


def _write_checkpoint_atomic(
    checkpoint_dir: "Path", template_name: str, state: Dict
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = _checkpoint_path(checkpoint_dir, template_name)
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


def _delete_checkpoint(checkpoint_dir: "Path", template_name: str) -> None:
    try:
        _checkpoint_path(checkpoint_dir, template_name).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _load_checkpoint(
    checkpoint_dir: "Path", template_name: str, expected_fingerprint: str
) -> Optional[Dict]:
    path = _checkpoint_path(checkpoint_dir, template_name)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            state = pickle.load(fh)
    except Exception as exc:
        warnings.warn(
            f"[checkpoint] {path} is unreadable ({type(exc).__name__}: {exc}); ignoring and starting fresh.",
            stacklevel=2,
        )
        return None
    if state.get("format_version") != _CHECKPOINT_FORMAT_VERSION:
        warnings.warn(
            f"[checkpoint] {path} has format_version {state.get('format_version')} != current {_CHECKPOINT_FORMAT_VERSION}; ignoring stale checkpoint.",
            stacklevel=2,
        )
        return None
    if state.get("fingerprint") != expected_fingerprint:
        warnings.warn(
            f"[checkpoint] {path} fingerprint mismatch (config/grammar changed since it was written); ignoring stale checkpoint and starting fresh.",
            stacklevel=2,
        )
        return None
    return state
