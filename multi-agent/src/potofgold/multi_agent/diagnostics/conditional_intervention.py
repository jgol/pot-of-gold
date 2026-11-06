"""Conditional interventions on computed entry operands using training data.

Fit additive linear conditional means and exchange whole-day residual
blocks. Withhold draws when support, residual dependence, temporal drift
or near-deterministic operands fail validation. These empirical checks do
not establish conditional independence; no hypothesis-test p-value is
reported. Held-out day identities are used only to check split separation.

Methodological references:
https://pmc.ncbi.nlm.nih.gov/articles/PMC2491635/
https://pmc.ncbi.nlm.nih.gov/articles/PMC10323874/
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


METHOD = "conditional_residual_day_blocks_v1"
SEMANTIC_DOMAIN_POLICY = "direct_computed_operand_domains_v1"
# Allow fixed float64 round-off tolerance without clipping generated draws.
SEMANTIC_DOMAIN_TOLERANCE = float(64 * np.finfo(np.float64).eps)


def operand_semantic_bounds(node):
    """Return bounds for direct operators whose evaluator defines a hard domain.

    EmbDir and EmbProj are continuous projections without a [-1, 1] bound.
    An arithmetic expression does not inherit the bounds of a child operator.
    """
    from potofgold.evolutionary_search.search.grammar import FuncNode
    if not isinstance(node, FuncNode):
        return None
    if node.defn.name.startswith("EmbCos_"):
        return (-1.0, 1.0)
    if node.defn.name.startswith("EmbNorm_"):
        return (0.0, None)
    return None


def _valid_semantic_bounds(bounds):
    return (bounds is None or (isinstance(bounds, tuple) and len(bounds) == 2
        and all(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)
                                 and np.isfinite(value)) for value in bounds)
        and (bounds[0] is None or bounds[1] is None or bounds[0] <= bounds[1])))


def _semantic_violation(values, bounds):
    if bounds is None:
        return False
    lower, upper = bounds
    return bool((lower is not None and np.any(values < lower - SEMANTIC_DOMAIN_TOLERANCE))
                or (upper is not None and np.any(values > upper + SEMANTIC_DOMAIN_TOLERANCE)))


def entry_signal_eligibility(prep):
    """Find signal bars meeting replay floor, data and cutoff requirements.

    Position state, pending orders, trade caps, size and capital constraints
    are checked by the separate trade replay.
    """
    from potofgold.evolutionary_search.search.simulator import prepare_static_arrays, ENTRY_CUTOFF_MTC
    data = prep["train_df"]
    if "date" not in data.columns or len(data) == 0:
        raise ValueError("Entry-disagreement scope requires dated training bars")
    offset = prep["off"]
    if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)) or offset < 0:
        raise ValueError("Entry-disagreement scope requires a nonnegative integer replay floor")
    arrays = prepare_static_arrays(data)
    spot, mtc = arrays["spot_arr"], arrays["mtc_arr"]
    return ((arrays["within_day_pos"] >= int(offset)) & (np.arange(len(data)) >= int(offset))
            & np.isfinite(spot) & (spot > 0) & np.isfinite(mtc) & (mtc > ENTRY_CUTOFF_MTC))


@dataclass(frozen=True)
class ConditionalConfig:
    n_draws: int = 4
    seed: int = 1729
    folds: int = 5
    min_days: int = 40
    min_donor_days: int = 12
    min_rows: int = 160
    max_conditioners: int = 12
    min_residual_fraction: float = 0.03
    max_residual_dependence: float = 0.20
    max_fold_mean_shift: float = 0.60
    max_fold_scale_ratio: float = 2.0
    max_day_residual_autocorrelation: float = 0.30
    max_correlation_shift: float = 0.12
    max_outside_fraction: float = 0.01


@dataclass
class ConditionalDraws:
    reliable: bool
    reason: str
    values: list[np.ndarray] = field(default_factory=list)
    controls: dict[str, Any] = field(default_factory=dict)


def _max_abs_corr(y: np.ndarray, x: np.ndarray) -> float:
    if x.shape[1] == 0:
        return 0.0
    yc = y - y.mean()
    xc = x - x.mean(axis=0)
    den = np.sqrt(np.sum(yc * yc) * np.sum(xc * xc, axis=0))
    good = den > 1e-12
    if not good.any():
        return 0.0
    return float(np.max(np.abs((yc @ xc[:, good]) / den[good])))


def _correlations(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 0:
        return np.empty(0)
    yc, xc = y - y.mean(), x - x.mean(axis=0)
    den = np.sqrt(np.sum(yc * yc) * np.sum(xc * xc, axis=0))
    return np.divide(yc @ xc, den, out=np.zeros(x.shape[1]), where=den > 1e-12)


def _validation_features(x: np.ndarray) -> np.ndarray:
    """Check linear and simple nonlinear residual dependence, including interactions."""
    cols = [x, x * x]
    if x.shape[1] > 1:
        cols.append(np.column_stack([x[:, i] * x[:, j]
                                     for i in range(x.shape[1])
                                     for j in range(i + 1, x.shape[1])]))
    return np.column_stack(cols)


def conditional_day_block_draws(
    target: np.ndarray,
    conditioners: np.ndarray,
    days: np.ndarray,
    *,
    warmup_bars: int = 0,
    config: ConditionalConfig = ConditionalConfig(),
    semantic_bounds: tuple[float | None, float | None] | None = None,
) -> ConditionalDraws:
    """Replace target innovations conditional on other entry operands.

    Cross-fit predictions and donor residuals by whole days. Copy a different
    training day's residual trajectory at matching within-day positions; a
    short session may use a longer donor. Leave pre-entry-floor bars unchanged.
    Withhold all draws if validation fails.
    """
    controls: dict[str, Any] = {"method": METHOD, "n_draws": config.n_draws,
                                "seed": config.seed, "conditional_model": "linear_additive",
                                "semantic_domain_policy": SEMANTIC_DOMAIN_POLICY,
                                "semantic_lower_bound": semantic_bounds[0] if _valid_semantic_bounds(semantic_bounds) and semantic_bounds else None,
                                "semantic_upper_bound": semantic_bounds[1] if _valid_semantic_bounds(semantic_bounds) and semantic_bounds else None,
                                "semantic_domain_tolerance": SEMANTIC_DOMAIN_TOLERANCE}

    def fail(reason: str) -> ConditionalDraws:
        return ConditionalDraws(False, reason, controls=controls)

    if (config.n_draws < 2 or config.folds < 2 or config.min_days < 2 * config.folds
            or config.min_donor_days < 2 or warmup_bars < 0 or not _valid_semantic_bounds(semantic_bounds)):
        return fail("invalid_conditional_configuration")
    y = np.asarray(target, dtype=float)
    z = np.asarray(conditioners, dtype=float)
    dates = np.asarray(days)
    if (y.ndim != 1 or z.ndim != 2 or dates.ndim != 1
            or len(y) != len(z) or len(y) != len(dates) or not len(y)):
        return fail("invalid_operand_shapes")
    if z.shape[1] > config.max_conditioners:
        return fail("too_many_conditioning_operands")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(z)):
        return fail("nonfinite_operand_values")
    if _semantic_violation(y, semantic_bounds):
        return fail("operand_outside_semantic_domain")
    # Each day must occupy one contiguous block for donor positions to align.
    starts = np.r_[0, 1 + np.flatnonzero(dates[1:] != dates[:-1])]
    ends = np.r_[starts[1:], len(y)]
    day_keys = [str(dates[i]) for i in starts]
    if len(set(day_keys)) != len(day_keys):
        return fail("noncontiguous_training_days")
    if len(starts) < config.min_days:
        return fail("insufficient_training_days")
    lengths = ends - starts
    if np.any(lengths <= warmup_bars):
        return fail("session_has_no_supported_entry_bars")
    groups = np.repeat(np.arange(len(starts)), lengths)
    positions = np.arange(len(y)) - np.repeat(starts, lengths)
    active = positions >= warmup_bars
    if int(active.sum()) < config.min_rows:
        return fail("insufficient_training_rows")
    ya, za, ga, pa = y[active], z[active], groups[active], positions[active]
    target_scale = float(np.std(ya))
    if target_scale <= 1e-10:
        return fail("constant_target_operand")
    # Include time of day and gradual chronological drift as nuisance variables.
    nuisance = np.column_stack([za, pa / float(max(lengths)),
                                (pa / float(max(lengths))) ** 2,
                                ga / float(len(starts) - 1)])
    std = nuisance.std(axis=0)
    nuisance = nuisance[:, std > 1e-10]
    std = std[std > 1e-10]
    xn = (nuisance - nuisance.mean(axis=0)) / std
    # Cross-fit on training-day blocks to check temporal stability.
    fold = np.minimum(ga * config.folds // len(starts), config.folds - 1)
    pred = np.empty(len(ya), dtype=float)
    for f in range(config.folds):
        held = fold == f
        fit = ~held
        if int(fit.sum()) <= 3 * (xn.shape[1] + 1):
            return fail("insufficient_conditional_fit_support")
        # Fit standardization on fitting days only.
        center = nuisance[fit].mean(axis=0)
        scale = nuisance[fit].std(axis=0)
        if np.any(scale <= 1e-10):
            return fail("unsupported_conditioning_dimension")
        xf = np.column_stack([np.ones(int(fit.sum())), (nuisance[fit] - center) / scale])
        xh = np.column_stack([np.ones(int(held.sum())), (nuisance[held] - center) / scale])
        coef = np.linalg.lstsq(xf, ya[fit], rcond=1e-10)[0]
        pred[held] = xh @ coef
    residual = ya - pred
    residual_scale = float(np.std(residual))
    fraction = residual_scale / target_scale
    controls["residual_scale_fraction"] = fraction
    if fraction < config.min_residual_fraction:
        return fail("target_determined_by_other_operands")
    checks = _validation_features(xn)
    mean_dependence = _max_abs_corr(residual, checks)
    variance_dependence = _max_abs_corr(residual * residual, checks)
    controls.update(residual_mean_dependence=mean_dependence,
                    residual_variance_dependence=variance_dependence)
    if max(mean_dependence, variance_dependence) > config.max_residual_dependence:
        return fail("conditional_residual_dependence")
    means = [float(np.mean(residual[fold == f])) for f in range(config.folds)]
    scales = [float(np.std(residual[fold == f])) for f in range(config.folds)]
    mean_shift = max(abs(v) for v in means) / residual_scale
    scale_ratio = max(scales) / max(min(scales), 1e-12)
    controls.update(fold_mean_shift=mean_shift, fold_scale_ratio=scale_ratio)
    if mean_shift > config.max_fold_mean_shift or scale_ratio > config.max_fold_scale_ratio:
        return fail("conditional_residual_temporal_drift")
    day_means = np.array([residual[ga == d].mean() for d in range(len(starts))])
    day_serial = max(_max_abs_corr(day_means[lag:], day_means[:-lag, None])
                     for lag in range(1, min(6, len(starts) // 4)))
    controls["day_residual_autocorrelation"] = day_serial
    if day_serial > config.max_day_residual_autocorrelation:
        return fail("conditional_residual_cross_day_dependence")
    full_residual = np.full(len(y), np.nan)
    full_residual[active] = residual
    full_prediction = np.array(y, copy=True)
    full_prediction[active] = pred
    donor_sets = [np.flatnonzero((lengths >= length) & (np.arange(len(starts)) != d))
                  for d, length in enumerate(lengths)]
    if min(len(donors) for donors in donor_sets) < config.min_donor_days:
        return fail("insufficient_matching_donor_days")
    rng = np.random.default_rng(config.seed)
    baseline_corr = _correlations(ya, za)
    low, high = float(ya.min()), float(ya.max())
    draws: list[np.ndarray] = []
    max_shift = max_outside = 0.0
    for _ in range(config.n_draws):
        draw = np.array(y, copy=True)
        for d, donors in enumerate(donor_sets):
            donor = int(rng.choice(donors))
            dst = np.arange(starts[d] + warmup_bars, ends[d])
            src = starts[donor] + np.arange(warmup_bars, lengths[d])
            draw[dst] = full_prediction[dst] + full_residual[src]
        da = draw[active]
        if not np.all(np.isfinite(draw)):
            return fail("nonfinite_conditional_draw")
        if _semantic_violation(draw, semantic_bounds):
            controls["validation"] = "failed"
            return fail("conditional_draw_outside_semantic_domain")
        outside = float(np.mean((da < low) | (da > high)))
        shift = float(np.max(np.abs(_correlations(da, za) - baseline_corr), initial=0.0))
        max_shift, max_outside = max(max_shift, shift), max(max_outside, outside)
        draws.append(draw)
    controls.update(max_correlation_shift=max_shift, max_outside_fraction=max_outside,
                    n_conditioners=int(z.shape[1]), validation="passed")
    if max_outside > config.max_outside_fraction:
        controls["validation"] = "failed"
        return fail("conditional_draw_outside_observed_support")
    if max_shift > config.max_correlation_shift:
        controls["validation"] = "failed"
        return fail("conditional_draw_dependence_shift")
    return ConditionalDraws(True, "validated_conditional_model", draws, controls)


def conditional_signal_profile(individual, prep, evaluator, baseline: set, *, config=None) -> dict:
    """Replay conditional operand interventions through the trade engine.

    Check original signal and entry-day parity before intervening. Replace
    each identical operand occurrence together, leaving other trees unchanged.
    Report the worst mean agreement across operands, averaging all fixed
    draws for each operand.
    """
    import dataclasses
    from potofgold.evolutionary_search.search.grammar import GType, TermDef, TermNode, _node_at_path, _replace_at_path
    from potofgold.multi_agent.diagnostics.decision_tracer import build_evaluator, _eval_1d, is_exogenous, assert_disjoint_days
    from potofgold.multi_agent.diagnostics.perturbation_prober import comparison_operands, _run_trades, _fired_days, phi_hat

    config = config or ConditionalConfig()
    result = {"phi_min": float("nan"), "phi_mean": float("nan"), "n_ablated": 0,
              "n_attempted": 0, "n_validated": 0, "method": METHOD,
              "reliability_ok": False, "confounded_by_correlation": False,
              "reasons": [], "per_operand": [],
              "semantic_domain_policy": SEMANTIC_DOMAIN_POLICY,
              "entry_signal_scope": "training_bars_after_replay_floor_with_valid_spot_and_before_entry_cutoff; not fill eligibility",
              "entry_disagreement_mean": None, "entry_disagreement_max": None}
    tree = individual.entry_tree
    if prep.get("train_days") is not None and prep.get("heldout_days") is not None:
        assert_disjoint_days(prep["train_days"], prep["heldout_days"])
    if tree is not None and not is_exogenous(tree):
        result["reasons"].append("outcome_dependent_entry_operand")
        return result
    ops = comparison_operands(tree)
    grouped: dict[str, list[tuple[int, ...]]] = {}
    for op in ops:
        grouped.setdefault(op["label"], []).append(op["path"])
    result["n_attempted"] = len(grouped)
    if not grouped or not baseline:
        result["reasons"].append("no_operand_or_no_baseline_fires")
        return result
    paths = [op["path"] for op in ops]
    if any(p != q and len(p) < len(q) and q[:len(p)] == p for p in paths for q in paths):
        result["reasons"].append("overlapping_comparison_operands")
        return result
    labels = list(grouped)
    try:
        values = np.column_stack([_eval_1d(evaluator, _node_at_path(tree, grouped[label][0]))
                                  for label in labels])
    except (TypeError, ValueError):
        result["reasons"].append("non_numeric_comparison_operand")
        return result
    dates = np.asarray(prep["train_df"]["date"].values)
    if len(values) != len(dates):
        result["reasons"].append("operand_calendar_mismatch")
        return result
    universe = set(np.unique(dates))
    if not baseline.issubset(universe):
        result["reasons"].append("baseline_fires_outside_training_calendar")
        return result
    original_signals = _eval_1d(evaluator, tree)
    try:
        eligible = entry_signal_eligibility(prep)
    except (TypeError, ValueError, KeyError):
        result["reasons"].append("invalid_entry_disagreement_scope")
        return result
    if not eligible.any():
        result["reasons"].append("no_eligible_entry_signal_bars")
        return result
    if len(original_signals) != len(dates) or not np.isfinite(original_signals[eligible]).all():
        result["reasons"].append("invalid_baseline_entry_signals")
        return result
    bounds = [operand_semantic_bounds(_node_at_path(tree, grouped[label][0])) for label in labels]
    if any(_semantic_violation(values[:, i], domain) for i, domain in enumerate(bounds)):
        result["reasons"].append("operand_outside_semantic_domain")
        return result
    mean_phis: list[float] = []
    mean_disagreements: list[float] = []
    max_disagreements: list[float] = []
    for i, label in enumerate(labels):
        draws = conditional_day_block_draws(
            values[:, i], np.delete(values, i, axis=1), dates,
            warmup_bars=int(prep["off"]), config=dataclasses.replace(config, seed=config.seed + i),
            semantic_bounds=bounds[i])
        detail = {"operand": label, "reliability_ok": draws.reliable,
                  "reason": draws.reason, **draws.controls}
        result["per_operand"].append(detail)
        if not draws.reliable:
            result["reasons"].append(draws.reason)
            continue
        # Validate draws again at the trade-replay boundary.
        try:
            draw_arrays = [np.asarray(draw, dtype=float) for draw in draws.values]
            valid_draws = len(draw_arrays) == config.n_draws and all(
                draw.shape == values[:, i].shape and np.isfinite(draw).all() for draw in draw_arrays)
        except (TypeError, ValueError):
            valid_draws = False
        if not valid_draws:
            detail.update(reliability_ok=False, reason="invalid_conditional_draw_values")
            result["reasons"].append("invalid_conditional_draw_values")
            continue
        if any(_semantic_violation(draw, bounds[i]) for draw in draw_arrays):
            detail.update(reliability_ok=False, reason="conditional_draw_outside_semantic_domain")
            result["reasons"].append("conditional_draw_outside_semantic_domain")
            continue
        terminal_name = f"__court_conditional_operand_{i}"
        if terminal_name in prep["td_tr"]:
            detail.update(reliability_ok=False, reason="conditional_terminal_name_collision")
            result["reasons"].append("conditional_terminal_name_collision")
            continue
        replacement = TermNode(defn=TermDef(terminal_name, GType.REAL))
        rewritten = tree
        for path in grouped[label]:
            rewritten = _replace_at_path(rewritten, path, replacement)
        perturbed = dataclasses.replace(individual, entry_tree=rewritten)
        local_prep = dict(prep, td_tr=dict(prep["td_tr"]))
        local_prep["td_tr"][terminal_name] = values[:, i]
        replay_evaluator = build_evaluator(
            prep["train_df"], local_prep["td_tr"], bases_k=prep.get("bases_k"),
            emb_cache_tr=prep.get("emb_cache_tr"))
        replay_signals = _eval_1d(replay_evaluator, rewritten)
        if not np.array_equal(original_signals, replay_signals, equal_nan=True):
            detail.update(reliability_ok=False, reason="conditional_entry_signal_replay_mismatch")
            result["reasons"].append("conditional_entry_signal_replay_mismatch")
            continue
        if _fired_days(_run_trades(perturbed, local_prep), dates) != baseline:
            detail.update(reliability_ok=False, reason="conditional_fired_day_replay_mismatch")
            result["reasons"].append("conditional_fired_day_replay_mismatch")
            continue
        phis, disagreements = [], []
        for draw in draw_arrays:
            local_prep["td_tr"][terminal_name] = draw
            # Use a fresh evaluator to avoid cached baseline node values.
            draw_evaluator = build_evaluator(
                prep["train_df"], local_prep["td_tr"], bases_k=prep.get("bases_k"),
                emb_cache_tr=prep.get("emb_cache_tr"))
            draw_signals = _eval_1d(draw_evaluator, rewritten)
            if len(draw_signals) != len(dates) or not np.isfinite(draw_signals[eligible]).all():
                detail.update(reliability_ok=False, reason="invalid_conditional_entry_signals")
                result["reasons"].append("invalid_conditional_entry_signals")
                break
            disagreement = float(np.mean((draw_signals[eligible] > 0.5) != (original_signals[eligible] > 0.5)))
            fired = _fired_days(_run_trades(perturbed, local_prep), dates)
            if not fired.issubset(universe):
                detail.update(reliability_ok=False, reason="conditional_fires_outside_training_calendar")
                result["reasons"].append("conditional_fires_outside_training_calendar")
                break
            phis.append(phi_hat([baseline, fired], len(universe)))
            disagreements.append(disagreement)
        if len(phis) != config.n_draws or not np.all(np.isfinite(phis)):
            detail.update(reliability_ok=False, reason="undefined_conditional_fireset_agreement")
            result["reasons"].append("undefined_conditional_fireset_agreement")
            continue
        mean_phis.append(float(np.mean(phis)))
        detail["phi_mean"] = mean_phis[-1]
        detail["entry_disagreement_by_draw"] = disagreements
        detail["entry_disagreement_mean"] = float(np.mean(disagreements))
        detail["entry_disagreement_max"] = float(max(disagreements))
        mean_disagreements.append(detail["entry_disagreement_mean"])
        max_disagreements.append(detail["entry_disagreement_max"])
        result["n_validated"] += 1
    # Require complete operand coverage before reporting the worst score.
    if result["reasons"] or result["n_validated"] != len(labels):
        result["reasons"] = sorted(set(result["reasons"]))
        return result
    result.update(phi_min=float(min(mean_phis)), phi_mean=float(np.mean(mean_phis)),
                  n_ablated=len(mean_phis), reliability_ok=True,
                  entry_disagreement_mean=float(np.mean(mean_disagreements)),
                  entry_disagreement_max=float(max(max_disagreements)))
    return result
