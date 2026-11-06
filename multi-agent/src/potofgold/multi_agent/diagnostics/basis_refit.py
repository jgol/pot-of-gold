"""Measure entry sensitivity to training-day basis jackknife refits.

Refit the recorded basis on reduced training masks and replay the unchanged
strategy with each fitted basis. Clear projection caches between replays.
"""
from __future__ import annotations

from typing import Any, Callable
import threading
import warnings

import numpy as np

MAX_RESAMPLES = 16


def _day_keys(values):
    from potofgold.multi_agent.diagnostics.decision_tracer import _daykey
    return np.asarray([_daykey(v) for v in values], dtype=str)


def day_block_masks(dates, train_mask, n_resamples=8):
    """Delete consecutive groups of whole training days; never add held-out rows."""
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)):
        raise ValueError("n_resamples must be an integer")
    if not 2 <= int(n_resamples) <= MAX_RESAMPLES:
        raise ValueError(f"n_resamples must be between 2 and {MAX_RESAMPLES}")
    days = _day_keys(dates)
    mask = np.asarray(train_mask, dtype=bool)
    if mask.shape != days.shape:
        raise ValueError("training mask must match the row calendar")
    selected = np.unique(days[mask])
    if len(selected) < 2:
        raise ValueError("at least two training days are required")
    # Require whole days so each reduced fit remains a day jackknife.
    if np.any(np.isin(days, selected) != mask):
        raise ValueError("training mask must select complete days")
    blocks = np.array_split(selected, min(int(n_resamples), len(selected)))
    return [mask & ~np.isin(days, block) for block in blocks]


def _entry_uses_basis(genome):
    def walk(node):
        if isinstance(node, str):
            return any(prefix in node for prefix in ("EmbProj_", "EmbDir_", "PROTO_"))
        if not isinstance(node, dict):
            return False
        name = str(node.get("name", ""))
        return (name.startswith(("EmbProj_", "EmbDir_", "PROTO_"))
                or any(walk(c) for c in node.get("children", [])))
    return walk(genome.get("entry_tree", {}))


def _replay_fired(genome, prep):
    from potofgold.evolutionary_search.search.genetics import individual_from_genome_dict
    from potofgold.multi_agent.diagnostics.perturbation_prober import _run_trades, _fired_days
    trades = _run_trades(individual_from_genome_dict(genome), prep)
    return set(_day_keys(_fired_days(trades, prep["train_df"]["date"].values)))


def _projection_parity(genome, original, refitted):
    """Check used basis coordinates in all strategy trees through the actual evaluator."""
    from potofgold.evolutionary_search.search.genetics import individual_from_genome_dict
    from potofgold.evolutionary_search.search.grammar import all_nodes_indexed
    from potofgold.multi_agent.diagnostics.decision_tracer import build_evaluator
    individual = individual_from_genome_dict(genome)
    def evaluator(prep):
        return build_evaluator(prep["train_df"], prep["td_tr"],
                               bases_k=prep["bases_k"], emb_cache_tr=None)
    a, b = evaluator(original), evaluator(refitted)
    for tree_name in ("entry_tree", "exit_tree", "size_tree", "delta_tree", "width_tree", "shape_tree"):
        tree = getattr(individual, tree_name, None)
        if tree is None:
            continue
        for node, _, _ in all_nodes_indexed(tree):
            if node.name.startswith(("EmbProj_", "EmbDir_", "PROTO_")):
                x, y = np.asarray(a.evaluate(node)), np.asarray(b.evaluate(node))
                if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.array_equal(x, y)):
                    return False
    return True


def _same_bases(a, b):
    """Full-mask refitting must reproduce every basis field, including direction banks."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_bases(a[k], b[k]) for k in a)
    try:
        x, y = np.asarray(a), np.asarray(b)
        return x.shape == y.shape and bool(np.array_equal(x, y, equal_nan=False))
    except (TypeError, ValueError):
        return a == b


def _finite_basis(value):
    """Reject nonfinite fitted fields, including banks that the entry may not read."""
    if isinstance(value, dict):
        return all(_finite_basis(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_basis(v) for v in value)
    array = np.asarray(value)
    return bool(np.all(np.isfinite(array))) if np.issubdtype(array.dtype, np.number) else True


def make_basis_refit_fn(*, df, train_mask, basis_method, targets, targets_valid,
                        primary_target_idx, fit_fn: Callable, config_check: Callable,
                        replay_fn=None, projection_check=None):
    """Bind basis refits to the original aligned fold context.

    Use embedding_data._fit_fold_basis to preserve target order, residualization
    and basis conventions. Refit residualization on each reduced mask. Cache
    bases within this fold and discard per-replay projection caches.
    """
    mask = np.array(train_mask, dtype=bool, copy=True)
    dates = _day_keys(df["date"].values)
    if mask.shape != dates.shape or not mask.any():
        raise ValueError("empty or misaligned training mask")
    if basis_method != "lean":
        raise ValueError("only the recorded lean basis is supported")
    if targets is None or targets_valid is None:
        raise ValueError("supervised basis requires its original aligned targets")
    if len(targets) != len(df) or len(targets_valid) != len(df):
        raise ValueError("basis targets must be row-aligned to the original corpus")
    universe = set(dates[mask])
    heldout = set(dates[~mask])
    if universe & heldout:
        raise ValueError("training and held-out calendars overlap")
    replay = replay_fn or _replay_fired
    parity = projection_check or _projection_parity
    fitted: dict[Any, Any] = {}
    lock = threading.Lock()

    def fit(key, subset):
        if key not in fitted:
            config_check()
            with warnings.catch_warnings():
                # Residualization must fit successfully for this reconstruction to be usable.
                warnings.filterwarnings("error", message=".*residualization NO-OPed.*", category=RuntimeWarning)
                bases, info = fit_fn(df, subset.copy(), basis_method, targets, targets_valid,
                                     primary_target_idx=primary_target_idx)
            if not bases or info.get("basis_method") != basis_method:
                raise ValueError("basis fitter returned incompatible configuration")
            if not _finite_basis(bases):
                raise ValueError("basis fitter returned nonfinite fields")
            fitted[key] = bases
        return fitted[key]

    def refit(genome, prep, *, n_resamples=8, **params):
        def fail(reason, **detail):
            return {"applicable": False, "reason": reason, "detail": detail}
        if params:
            return fail("unsupported_basis_refit_parameters")
        if not _entry_uses_basis(genome):
            return fail("no_basis_projected_operand")
        if not prep.get("bases_k"):
            return fail("basis_refit_context_unavailable")
        eval_days = set(_day_keys(prep["train_df"]["date"].values))
        if eval_days != universe or eval_days & heldout:
            return fail("basis_refit_calendar_mismatch")
        if prep.get("heldout_days") is not None and universe & set(_day_keys(prep["heldout_days"])):
            return fail("basis_refit_calendar_overlap")
        if prep.get("replay_fidelity") == 0:
            return fail("train_replay_fidelity_failed")
        try:
            masks = day_block_masks(dates, mask, n_resamples)
        except ValueError as exc:
            return fail("invalid_basis_refit_budget", error=str(exc))
        # Copy per-strategy state because the fold closure is shared.
        with lock:
            try:
                config_check()
                reference = dict(prep, emb_cache_tr=None)
                baseline = set(_day_keys(replay(genome, reference)))
                if not baseline or not baseline <= universe:
                    return fail("empty_or_invalid_baseline_fires")
                recorded = prep.get("recorded_train_fired_days")
                if recorded is not None and baseline != set(_day_keys(recorded)):
                    return fail("recorded_training_fires_mismatch")
                full_basis = fit("baseline", mask)
                if not _same_bases(reference["bases_k"], full_basis):
                    return fail("baseline_refit_basis_mismatch")
                full_prep = dict(reference, bases_k=full_basis, emb_cache_tr=None)
                if not parity(genome, reference, full_prep):
                    return fail("baseline_projection_mismatch")
                if set(_day_keys(replay(genome, full_prep))) != baseline:
                    return fail("baseline_refit_fires_mismatch")
                reconstructions = []
                for index, subset in enumerate(masks):
                    bases = fit((len(masks), index), subset)
                    child = dict(reference, bases_k=bases, emb_cache_tr=None)
                    # Finite coefficients can overflow when projected; treat that as a failed refit.
                    if not parity(genome, child, child):
                        return fail("refit_projection_nonfinite")
                    fires = set(_day_keys(replay(genome, child)))
                    if not fires <= universe:
                        return fail("refit_fires_outside_training_calendar")
                    reconstructions.append(sorted(fires))
            except (Exception, SystemExit) as exc:
                return fail("basis_refit_failed", error_type=type(exc).__name__)
        return {"applicable": True, "baseline_fired": sorted(baseline),
                "reconstructions": reconstructions, "all_days": sorted(universe),
                "detail": {"baseline_refit_fires_match": True, "projection_parity_ok": True,
                           "recorded_fire_identity_verified": recorded is not None,
                           "reference": "recorded_training_fires" if recorded is not None else "reconstructed_baseline",
                           "intervention": "whole_strategy_basis_refit",
                           "refit_day_blocks": len(masks), "n_refit_failures": 0,
                           "projection_mode": "canonical_per_eval_no_cache"}}
    return refit
