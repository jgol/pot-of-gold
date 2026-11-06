"""Assemble strategy rules and training diagnostics for court and single-LLM assessment.

Profiles describe entry margins, perturbation sensitivity, market regimes and
intraday stability. They exclude held-out outcomes, labels and raw trading
ledgers. Approved training-economics summaries are attached separately by the
assessment packet builder. Numeric citations are checked against these inputs.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass, asdict, fields as _dc_fields
from typing import Optional, Dict, Any, List

from potofgold.multi_agent.strategy_profiles import terminal_registry as _treg

_EMB_MARKERS = ("EmbProj", "EmbDir", "EMB_", "PROTO")


def _rules_from_genome(g: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Extract the rule text from the executable genome."""
    if not isinstance(g, dict):
        return {"entry_rule": None, "exit_rule": None, "size_rule": None, "structure_rule": None}
    _s = lambda k: (str(g.get(k)) if g.get(k) is not None else None)
    struct = None
    if g.get("shape") or g.get("delta_tree") or g.get("width_tree"):
        struct = f"shape={g.get('shape')} delta={_s('delta_tree')} width={_s('width_tree')}"
    return {"entry_rule": _s("entry_tree"), "exit_rule": _s("exit_tree"),
            "size_rule": _s("size_tree"), "structure_rule": struct}

_LEAKY_PATTERNS = ("pnl", "held_out", "heldout", "label", "permut", "qc", "oos", "n_trade", "fire_count",
                   "fired", "coverage", "cvar", "sharpe", "expectancy", "reward")

_LLM_WITHHELD = ("arm", "uses_embedding", "fold", "code_sha")


def _opaque_alias(champion_id: str) -> str:
    """Hash the strategy ID for use in model context.

    The original ID encodes the input condition, seed and fold.
    """
    return "champion_" + hashlib.sha256(str(champion_id).encode()).hexdigest()[:10]


def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def resolve_basis_targets(provenance: Dict[str, Any]) -> tuple:
    """Read projection target names from basis provenance.

    Accept the historical per_group_k mapping or its serialized form.
    """
    bt = provenance.get("basis_targets")
    if bt:
        return tuple(bt)
    pgk = provenance.get("per_group_k")
    if isinstance(pgk, str):
        try:
            pgk = ast.literal_eval(pgk)
        except (ValueError, SyntaxError):
            return ()
    return tuple(pgk.keys()) if isinstance(pgk, dict) else ()


@dataclass
class GeneralizationCaseFile:
    # Strategy identity.
    champion_id: str
    arm: Optional[str] = None
    fold: Optional[int] = None
    shape: Optional[str] = None
    uses_embedding: bool = False
    # Executable rules.
    entry_rule: Optional[str] = None
    exit_rule: Optional[str] = None
    size_rule: Optional[str] = None
    structure_rule: Optional[str] = None
    rule_glossary: Optional[Dict[str, str]] = None
    # Entry margins.
    margins_available: bool = False
    fire_margin_median: Optional[float] = None
    fire_margin_mean: Optional[float] = None
    fire_margin_std: Optional[float] = None
    fire_margin_abs_median: Optional[float] = None
    fire_margin_p10: Optional[float] = None
    fire_margin_p90: Optional[float] = None
    fire_knife_edge_frac: Optional[float] = None      # Fraction of firing rows within 0.1 robust scale of the boundary.
    fire_positive_frac: Optional[float] = None
    n_comparison_nodes: Optional[int] = None
    n_margins_traced: Optional[int] = None
    # Margins for individual comparisons.
    per_node_margins: Optional[List[Dict[str, Any]]] = None
    # Comparisons whose margins are constant.
    n_inert_comparisons: Optional[int] = None
    dead_comparison_frac: Optional[float] = None
    # Response to signal interventions.
    ablation_available: bool = False
    ablation_phi_min: Optional[float] = None          # Worst agreement between original and intervened firing sets.
    ablation_phi_mean: Optional[float] = None
    ablation_n_signals: Optional[int] = None
    ablation_confounded: Optional[bool] = None
    # Response to threshold perturbations.
    nudge_stability_available: bool = False
    nudge_phi_min: Optional[float] = None             # Wider perturbation steps.
    nudge_phi_mean: Optional[float] = None
    # Operand correlation.
    signal_max_abs_corr: Optional[float] = None
    n_signals: Optional[int] = None
    signals: Optional[List[str]] = None
    # Calendar exposure.
    regime_available: bool = False
    fomc_near_lift: Optional[float] = None
    tom_lift: Optional[float] = None
    opex_lift: Optional[float] = None
    dow_tv_distance: Optional[float] = None
    # Volatility exposure.
    vol_regime_available: bool = False
    vix_level_lift: Optional[float] = None
    atm_iv_lift: Optional[float] = None
    term_slope_lift: Optional[float] = None
    vix_pctile_mean: Optional[float] = None           # Expanding percentile uses only observations available at each row.
    vix_pctile_spread: Optional[float] = None         # Population standard deviation (ddof=0).
    vix_pctile_iqr: Optional[float] = None
    # Intraday margin stability.
    intraday_available: bool = False
    intraday_sign_consistency: Optional[float] = None
    intraday_margin_range: Optional[float] = None
    # Temporal pattern of entry days.
    fire_temporal_available: bool = False
    fire_day_gap_gini: Optional[float] = None          # Gini coefficient of gaps between unique entry days.
    behavior_half_consistency: Optional[float] = None  # Margin magnitude in the first and second halves of training.
    # Historical news fields retained for saved profile input.
    news_available: bool = False
    news_tone_lift: Optional[float] = None
    news_neg_frac_lift: Optional[float] = None
    # Structure and replay checks.
    n_evolved_thresholds: Optional[int] = None        # Number of constants that can be perturbed.
    fidelity_ok: Optional[bool] = None
    code_sha: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def to_llm_dict(self) -> Dict[str, Any]:
        """Build model context without cohort identity or availability flags.

        Replace the strategy ID with an alias and expose numeric margin summaries
        as fields that agents can cite.
        """
        # Availability flags can disclose activity counts.
        d = {k: v for k, v in asdict(self).items()
             if k not in _LLM_WITHHELD and not k.endswith("_available")}
        d["champion_id"] = _opaque_alias(self.champion_id)
        # Expose margin statistics as numeric citation fields.
        for i, node in enumerate(self.per_node_margins or []):
            for k, v in (node or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    d[f"per_node_{i}_{k}"] = v
        # Remove nested availability flags as well.
        if self.per_node_margins:
            d["per_node_margins"] = [{k: v for k, v in (node or {}).items() if k != "n_available"}
                                     for node in self.per_node_margins]
        leaked = [k for k in d if k.endswith("_available")]
        if leaked:
            raise ValueError(f"LLM view leaked availability/gate flag(s): {leaked}")
        d["exhibit_glossary"] = _exhibit_glossary(d)
        return d

    def to_llm_json(self) -> str:
        return json.dumps(self.to_llm_dict(), indent=2, default=str)


# Add descriptions and units for the rules' tokens.
EXHIBIT_LEGEND: Dict[str, str] = {
    "shape": "the defined-risk options structure of the strategy (e.g. iron_butterfly / long_strangle) [enum; see rule_glossary]",
    "signals": "the strategy's signal operands, as s-expressions [labels; decode each via rule_glossary]",
    "fire_margin_median": "median SIGNED distance of the fired-day decisions from their flip boundary [standardized (z-score); |value| larger = more room past the threshold]",
    "fire_margin_mean": "mean of the fired-day signed decision margins [standardized (z-score)]",
    "fire_margin_std": "spread (std) of the fired-day decision margins [standardized (z-score)]",
    "fire_margin_abs_median": "median of |decision margin| — typical distance from the flip regardless of side [standardized (z-score); SMALL = knife-edge]",
    "fire_margin_p10": "10th percentile of the fired-day decision margins [standardized (z-score)]",
    "fire_margin_p90": "90th percentile of the fired-day decision margins [standardized (z-score)]",
    "fire_knife_edge_frac": "share of fires within a hair (~0.1 standardized units) of the flip [0-1; HIGH = knife-edge]",
    "fire_positive_frac": "share of fires with a positive signed margin [0-1; NEUTRAL direction balance — not a tell]",
    "n_comparison_nodes": "number of comparison nodes in the strategy's rules [count]",
    "n_margins_traced": "number of comparison nodes with a traced margin distribution [count]",
    "per_node_margins": "the FULL signed-margin distribution for EACH comparison node (nested; READ it, but CITE the flat per_node_<i>_<stat> aliases)",
    "n_inert_comparisons": "comparison nodes with constant margin over train (never gate a decision = dead) [count]",
    "dead_comparison_frac": "inert / total comparison nodes [0-1; HIGH = dead-tree bloat]",
    "ablation_phi_min": "WORST-case chance-corrected fired-set stability (Nogueira Φ̂) when each single signal is neutralized in turn [1 = unchanged, ~0 = chance, NEGATIVE = neutralizing the signal INVERTS the fired set; LOW or NEGATIVE = single-signal-fragile — a valid negative reading is NOT out-of-range/invalid]",
    "ablation_phi_mean": "AVERAGE Nogueira Φ̂ fired-set stability across single-signal neutralizations [1 = unchanged, ~0 = chance, NEGATIVE = worse-than-chance; LOW = fragile]",
    "ablation_n_signals": "number of signals ablated [count]",
    "ablation_confounded": "TRUE => the signals are correlated (|corr|>=0.30) so the ablation_phi numbers are UNRELIABLE (you cannot cleanly isolate one signal when signals move together) [bool]",
    "nudge_phi_min": "worst-case Nogueira Φ̂ fired-set stability when each evolved threshold is nudged +/-e [1 = unchanged, ~0 = chance, NEGATIVE possible; LOW or NEGATIVE = boundary fit to noise]; null = no evolved threshold",
    "nudge_phi_mean": "average Nogueira Φ̂ fired-set stability across threshold nudges [1 = unchanged, ~0 = chance, NEGATIVE possible]; null = no evolved threshold",
    "signal_max_abs_corr": "max |correlation| among the strategy's signal operands [0-1; HIGH = redundant/near-duplicate signals = fewer effective degrees of freedom]",
    "n_signals": "number of distinct signal operands [count]",
    "fomc_near_lift": "fire-rate within +/-3 trading days of FOMC minus the all-train base [fraction; + = fires MORE near FOMC, - = AVOIDS]",
    "tom_lift": "turn-of-month fire-rate lift vs base [fraction]",
    "opex_lift": "OpEx-week fire-rate lift vs base [fraction]",
    "dow_tv_distance": "total-variation distance of the fires' day-of-week mix from the train base [0-1; 0 = agnostic, 1 = concentrated]",
    "vix_level_lift": "mean VIX on fired days minus the covered-train base [VIX POINTS; can be large +/-]",
    "atm_iv_lift": "mean ATM-IV on fired days minus base [IV fraction; ONE axis with vix_level_lift]",
    "term_slope_lift": "mean VIX term-structure slope on fired days minus base [slope units]",
    "vix_pctile_mean": "mean VIX percentile of the fired days, each ranked within the history up to that day [0-1; 0 = calmest, 1 = most volatile]",
    "vix_pctile_spread": "std of that VIX-percentile across fires [0-1; LOW = fires cluster in a THIN slice of the VIX distribution]",
    "vix_pctile_iqr": "inter-quartile range of the fires' VIX-percentile [0-1; robust VIX-dispersion companion to vix_pctile_spread]",
    "intraday_sign_consistency": "fraction of intraday buckets whose margin sign matches the pooled sign [0-1; LOW = flips by time-of-day]. =1.0 is TRIVIAL when the strategy fires in only one intraday bucket",
    "intraday_margin_range": "max-min of the per-bucket mean margin across the trading day [normalized; larger = more time-of-day variation]",
    "fire_day_gap_gini": "Gini of the gaps between UNIQUE fired days [0-1; HIGH = fires cluster in a few calendar bursts]",
    "behavior_half_consistency": "|margin| stability between the first and second half of the fired window [0-1; 1 = stable, LOW = drift]",
    "news_tone_lift": "mean GDELT news tone on fired days minus the non-fired base [tone units; + = fires on better-toned days]",
    "news_neg_frac_lift": "negative-tone fraction on fired days minus base [fraction; + = fires on more-hostile-news days]",
    "n_evolved_thresholds": "number of evolved numeric cut-points (EphReal) in the rules [count; 0 = signal-vs-signal rule]",
    "fidelity_ok": "TRUE if the trace reconstructed the sealed run faithfully; if FALSE, treat ALL exhibits as UNRELIABLE [bool]",
}
_LEGEND_SKIP = {"champion_id", "entry_rule", "exit_rule", "size_rule", "structure_rule", "rule_glossary",
                "exhibit_glossary"}


def _exhibit_glossary(view: Dict[str, Any]) -> Dict[str, str]:
    """Describe diagnostic fields included in the model context."""
    gloss: Dict[str, str] = {}
    missing = []
    for k in view:
        if k in _LEGEND_SKIP:
            continue
        if k in EXHIBIT_LEGEND:
            gloss[k] = EXHIBIT_LEGEND[k]
        elif k.startswith("per_node_") and k[len("per_node_"):].split("_", 1)[0].isdigit():
            idx, _, stat = k[len("per_node_"):].partition("_")
            gloss[k] = (f"comparison node {idx}: {stat} of its signed-margin distribution "
                        f"[flat, citable alias of per_node_margins[{idx}]]")
        else:
            missing.append(k)
    if missing:
        raise ValueError(f"exhibit_glossary: {len(missing)} exhibit(s) have NO legend entry — add each to "
                         f"EXHIBIT_LEGEND (no field in the LLM view may be unlabeled): {sorted(missing)}")
    return gloss


def assert_cause_only(cf: "GeneralizationCaseFile") -> bool:
    """Reject fields whose names expose evaluation outcomes."""
    bad = [f.name for f in _dc_fields(cf)
           if any(p in f.name.lower() for p in _LEAKY_PATTERNS)]
    if bad:
        raise ValueError(f"GeneralizationCaseFile has leaky field name(s) (outcome/label leak): {bad}")
    return True


_ALLOWED_MEMBER_INNER = frozenset({"genome", "train_traded_day_pnl"})


def assert_member_leak_safe(member: Dict[str, Any]) -> bool:
    """Restrict the profile to approved training measurements."""
    inner = member.get("member")
    if isinstance(inner, dict):
        extra = set(inner) - _ALLOWED_MEMBER_INNER
        if extra:
            raise ValueError(f"member inner carries non-allowlisted key(s) (leak risk): {sorted(extra)} — "
                             f"supply only the documented training fields")
    return True


def build_generalization_case_file(member: Dict[str, Any], g5: Optional[Dict[str, Any]],
                                   g6: Optional[Dict[str, Any]],
                                   g7: Optional[Dict[str, Any]] = None,
                                   intraday: Optional[Dict[str, Any]] = None) -> GeneralizationCaseFile:
    """Assemble a strategy profile from its training diagnostics.

    Leave missing measurements as None. Training profit measurements are
    validated separately by training_economics.py.
    """
    assert_member_leak_safe(member)
    g5 = g5 or {}
    g6 = g6 or {}
    g7 = g7 or {}
    intraday = intraday or {}
    genome = (member.get("member") or {}).get("genome") or member.get("genome") or {}
    gtxt = " ".join(str(genome.get(k)) for k in ("entry_tree", "exit_tree", "size_tree",
                                                 "delta_tree", "width_tree", "shape_tree")
                    if genome.get(k) is not None)
    binding = g5.get("binding") or {}
    ftemp = g5.get("fire_temporal") or {}
    news = g5.get("news") or {}
    abl = g6.get("ablation") or {}
    corr = g6.get("correlation") or {}
    curve = g6.get("curve") or {}
    _ekeys = [k for k in curve if _f(k) is not None]
    wide = curve.get(max(_ekeys, key=lambda k: float(k))) if _ekeys else {}  # Summarize the widest threshold perturbation.
    signals = [o.get("label") for o in (corr.get("per_operand") or []) if o.get("label")]

    _tree_keys = ("entry_tree", "exit_tree", "size_tree", "delta_tree", "width_tree", "shape_tree")
    _basis = resolve_basis_targets(member.get("provenance") or {})
    _idents = _treg.identifiers(*[genome.get(k) for k in _tree_keys])
    rule_glossary = {t: g for t in sorted(_idents) if (g := _treg.describe(t))} or None
    if rule_glossary and _basis:                          # Include projection targets once for the full basis.
        rule_glossary["(encoder note)"] = (
            "the L1 encoder behind every EMB_* embedding was trained to reconstruct: " + ", ".join(str(x) for x in _basis))

    cf = GeneralizationCaseFile(
        champion_id=member["champion_id"],
        arm=member.get("arm"),
        fold=member.get("fold"),
        shape=member.get("shape") or g6.get("shape") or g5.get("shape"),
        uses_embedding=any(mk in gtxt for mk in _EMB_MARKERS),
        **_rules_from_genome(genome),
        rule_glossary=rule_glossary,
        margins_available=bool(binding.get("n_available")),
        fire_margin_median=_f(binding.get("median")),
        fire_margin_mean=_f(binding.get("mean")),
        fire_margin_std=_f(binding.get("std")),
        fire_margin_abs_median=_f(binding.get("abs_median")),
        fire_margin_p10=_f(binding.get("p10")),
        fire_margin_p90=_f(binding.get("p90")),
        fire_knife_edge_frac=_f(binding.get("knife_edge_frac")),
        fire_positive_frac=_f(binding.get("positive_frac")),
        n_comparison_nodes=g5.get("n_cmp_nodes"),
        n_margins_traced=g5.get("n_traced"),
        per_node_margins=(g5.get("per_node") or None),
        n_inert_comparisons=g5.get("n_degenerate_scale"),
        dead_comparison_frac=(_f(g5["n_degenerate_scale"] / g5["n_cmp_nodes"])
                              if (g5.get("n_cmp_nodes") and g5.get("n_degenerate_scale") is not None)
                              else None),
        ablation_available=bool(abl),
        ablation_phi_min=_f(abl.get("phi_min")),
        ablation_phi_mean=_f(abl.get("phi_mean")),
        ablation_n_signals=abl.get("n_ablated"),
        ablation_confounded=abl.get("confounded_by_correlation"),
        nudge_stability_available=bool(curve),
        nudge_phi_min=_f(wide.get("phi_min")),
        nudge_phi_mean=_f(wide.get("phi_mean")),
        signal_max_abs_corr=_f(corr.get("max_abs_corr")),
        n_signals=corr.get("n_operands"),
        signals=signals or None,
        regime_available=bool(g7.get("fires_present")),
        fomc_near_lift=_f(g7.get("fomc_near_lift")),
        tom_lift=_f(g7.get("tom_lift")),
        opex_lift=_f(g7.get("opex_lift")),
        dow_tv_distance=_f(g7.get("dow_tv_distance")),
        vol_regime_available=bool(g7.get("vol_fires_present")),
        vix_level_lift=_f(g7.get("vix_level_lift")),
        atm_iv_lift=_f(g7.get("atm_iv_lift")),
        term_slope_lift=_f(g7.get("term_slope_lift")),
        vix_pctile_mean=_f(g7.get("vix_pctile_mean")),
        vix_pctile_spread=_f(g7.get("vix_pctile_spread")),
        vix_pctile_iqr=_f(g7.get("vix_pctile_iqr")),
        intraday_available=bool(intraday.get("available")),
        intraday_sign_consistency=_f(intraday.get("bucket_sign_consistency")),
        intraday_margin_range=_f(intraday.get("bucket_margin_range")),
        fire_temporal_available=(ftemp.get("fire_day_gap_gini") is not None),
        fire_day_gap_gini=_f(ftemp.get("fire_day_gap_gini")),
        behavior_half_consistency=_f(ftemp.get("behavior_half_consistency")),
        news_available=(news.get("news_tone_lift") is not None),
        news_tone_lift=_f(news.get("news_tone_lift")),
        news_neg_frac_lift=_f(news.get("news_neg_frac_lift")),
        n_evolved_thresholds=g6.get("n_targets"),
        fidelity_ok=(g6.get("fidelity_ok") if g6.get("fidelity_ok") is not None else g5.get("fidelity_ok")),
        code_sha=(member.get("provenance") or {}).get("code_sha"),
    )
    assert_cause_only(cf)
    return cf


def build_cohort_case_files(members: List[Dict[str, Any]], g5_results: List[Dict[str, Any]],
                            g6_results: List[Dict[str, Any]],
                            g7_results: Optional[List[Dict[str, Any]]] = None,
                            intraday_results: Optional[List[Dict[str, Any]]] = None
                            ) -> List[GeneralizationCaseFile]:
    """Join strategy records and diagnostics by champion_id.

    Leave missing diagnostics as None.
    """
    g5_by = {r["champion_id"]: r for r in g5_results if r.get("champion_id")}
    g6_by = {r["champion_id"]: r for r in g6_results if r.get("champion_id")}
    g7_by = {r["champion_id"]: r for r in (g7_results or []) if r.get("champion_id")}
    id_by = {r["champion_id"]: r for r in (intraday_results or []) if r.get("champion_id")}
    return [build_generalization_case_file(m, g5_by.get(m["champion_id"]), g6_by.get(m["champion_id"]),
                                           g7_by.get(m["champion_id"]), id_by.get(m["champion_id"]))
            for m in members if m.get("champion_id")]
