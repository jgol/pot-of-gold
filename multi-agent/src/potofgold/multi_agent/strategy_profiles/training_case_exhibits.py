"""Build training diagnostics from an existing strategy preparation."""
from functools import lru_cache
import math
import os

KINDS = ("g5", "g6", "g7", "intraday")

# Approved fields for the exported training diagnostics.
_ALLOWED_FIELDS = {
    'g5': frozenset({
        'arm', 'binding', 'champion_id', 'error',
        'fidelity_ok', 'fire_temporal', 'fold', 'n_cmp_nodes',
        'n_degenerate_scale', 'n_excluded_position_state', 'n_traced', 'news',
        'parity_ok', 'per_node', 'shape', 'traceable',
    }),
    'g6': frozenset({
        'ablation', 'arm', 'champion_id', 'correlation',
        'curve', 'error', 'fidelity_ok', 'fold',
        'n_degenerate_scale', 'n_int_skipped', 'n_nested_skipped', 'n_no_comparison',
        'n_targets', 'parity_ok', 'per_node', 'shape',
        'traceable',
    }),
    'g7': frozenset({
        'arm', 'atm_iv_lift', 'available', 'champion_id',
        'dow_tv_distance', 'error', 'fidelity_ok', 'fires_present',
        'fold', 'fomc_near_base_frac', 'fomc_near_fire_frac', 'fomc_near_lift',
        'opex_base_frac', 'opex_fire_frac', 'opex_lift', 'shape',
        'term_slope_lift', 'tom_base_frac', 'tom_fire_frac', 'tom_lift',
        'vix_level_lift', 'vix_pctile_iqr', 'vix_pctile_mean', 'vix_pctile_spread',
        'vol_available', 'vol_fires_present',
    }),
    'intraday': frozenset({
        'arm', 'available', 'bucket_margin_range', 'bucket_sign_consistency',
        'champion_id', 'error', 'fidelity_ok', 'fold',
        'n_buckets_populated', 'shape',
    }),
}


def json_safe_exhibit(value):
    """Encode undefined diagnostic summaries as JSON null."""
    if isinstance(value, dict):
        return {key: json_safe_exhibit(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_exhibit(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@lru_cache(maxsize=1)
def _vix_rows(path):
    from potofgold.multi_agent.diagnostics.regime_features import load_vix_rows
    return load_vix_rows(path)


def validate_exhibit(kind, record):
    """Admit only the recorded training fields and a successful replay check."""
    if kind not in _ALLOWED_FIELDS:
        raise ValueError(f"Unknown case exhibit: {kind}")
    extra = set(record) - _ALLOWED_FIELDS[kind]
    if extra:
        raise ValueError(f"result for {record.get('champion_id')} has non-whitelisted field(s) "
                         f"(possible leak): {sorted(extra)}")
    if kind == "g5":
        nested = {
            "fire_temporal": {"fire_day_gap_gini", "behavior_half_consistency"},
            "news": {"news_tone_lift", "news_neg_frac_lift"},
        }
        for field, allowed in nested.items():
            extra = set(record.get(field) or {}) - allowed
            if extra:
                raise ValueError(f"result for {record.get('champion_id')} has non-whitelisted "
                                 f"{field} sub-field(s) (possible leak): {sorted(extra)}")
    if record.get("fidelity_ok") is not True or record.get("parity_ok") is False or record.get("error"):
        raise ValueError(f"Unverified training case exhibit: {kind}/{record.get('champion_id')}")


def measure_case_exhibits(champion, prep, *, stored_day_pnl, names):
    from potofgold.multi_agent.diagnostics import decision_tracer, perturbation_prober, regime_features, intraday_stability
    from potofgold.multi_agent.diagnostics.catalogue_runner import genome_of
    if not set(names) <= set(KINDS) or stored_day_pnl is None:
        raise ValueError("Known case exhibits and a stored training reference are required")
    genome = genome_of(champion)
    calls = {
        "g5": lambda: decision_tracer.trace_champion(genome, prep, stored_day_pnl=stored_day_pnl),
        "g6": lambda: perturbation_prober.probe_champion(genome, prep, ablate=True, stored_day_pnl=stored_day_pnl),
        "g7": lambda: regime_features.regime_for_champion(genome, prep, vix_rows=_vix_rows(os.environ.get("CAT_RUNNER_VIX_STATE")),
                                                          stored_day_pnl=stored_day_pnl),
        "intraday": lambda: intraday_stability.intraday_for_champion(genome, prep, n_buckets=5,
                                                                    stored_day_pnl=stored_day_pnl),
    }
    results = {}
    for kind in names:
        output = calls[kind]()
        for internal in ("n_trades", "_fidelity_maxdelta", "_vol_covered_fired", "_vol_total_fired"):
            output.pop(internal, None)
        record = {k: champion[k] for k in ("champion_id", "arm", "fold", "shape")}
        record.update(output)
        validate_exhibit(kind, record)
        results[kind] = json_safe_exhibit(record)
    return results
