"""Compare entry-margin signs across intraday time buckets.

Reconstruct training entries with the shared evaluator and summarize the
mean signed binding margin in each populated bucket. The results describe
time-of-day variation, without predicting profitability.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def bucket_stability(bucket_ids: Sequence[int], margins: Sequence[float], n_buckets: int = 5) -> Dict:
    """Summarize signed margins by intraday bucket.

    Return the fraction of populated buckets matching the pooled sign, the
    range of bucket means and the number of populated buckets. Ignore NaNs.
    """
    b = np.asarray(bucket_ids)
    m = np.asarray(margins, dtype=float)
    ok = np.isfinite(m)
    b, m = b[ok], m[ok]
    if m.size == 0:
        return {"available": False}
    pooled = float(m.mean())
    pooled_sign = 1 if pooled > 0 else (-1 if pooled < 0 else 0)
    means = [float(m[b == k].mean()) for k in range(n_buckets) if np.any(b == k)]
    if not means:
        return {"available": False}
    signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in means]
    ref = pooled_sign if pooled_sign != 0 else signs[0]
    consistency = float(np.mean([s == ref for s in signs]))
    rng = float(max(means) - min(means)) if len(means) > 1 else 0.0
    return {"available": True, "bucket_sign_consistency": consistency,
            "bucket_margin_range": rng, "n_buckets_populated": len(means)}


def intraday_for_champion(genome: dict, prep: dict, *, stored_day_pnl=None, fidelity_tol: float = 1e-4,
                          n_buckets: int = 5) -> Dict:
    """Reconstruct training entries and summarize their margins by within-day bucket.

    Check split separation and reconstruction fidelity; keep trade counts and
    validation deltas internal.
    """
    from potofgold.evolutionary_search.search.genetics import individual_from_genome_dict
    from potofgold.evolutionary_search.search.validation import _champion_trades
    from potofgold.evolutionary_search.search.simulator import prepare_static_arrays
    from potofgold.multi_agent.diagnostics.decision_tracer import build_evaluator, classify_comparison_nodes, normalized_margins_at_bars, binding_margin, decision_bars_from_trades, reconstructed_day_pnl, fidelity_vs_stored, assert_disjoint_days
    if prep.get("train_days") is not None and prep.get("heldout_days") is not None:
        assert_disjoint_days(prep["train_days"], prep["heldout_days"])
    ind = individual_from_genome_dict(genome)
    off = prep["off"]
    trades = _champion_trades(ind, prep["train_df"], prep["tmpl"], prep["td_tr"], entry_decision_mask=None,
                              warmup_bars=off, pca_bases=prep.get("bases_k"),
                              emb_proj_cache=prep.get("emb_cache_tr"), entry_floor_within_day=int(off))
    dates = np.asarray(prep["train_df"]["date"].values)
    fid = (fidelity_vs_stored(reconstructed_day_pnl(trades, dates), stored_day_pnl, fidelity_tol)
           if stored_day_pnl is not None else None)
    db = decision_bars_from_trades(trades)
    ev = build_evaluator(prep["train_df"], prep["td_tr"], bases_k=prep.get("bases_k"),
                         emb_cache_tr=prep.get("emb_cache_tr"))
    kept, _ = classify_comparison_nodes(ind.entry_tree)
    margins, _ = normalized_margins_at_bars(ev, kept, db)      # One row per entry and one column per nonconstant comparison.
    bind = binding_margin(margins)                              # Select the signed margin closest to its boundary for each entry.
    wd = np.asarray(prepare_static_arrays(prep["train_df"], 0.20)["within_day_pos"], dtype=float)
    day_len = float(np.max(wd)) + 1.0 if wd.size else 1.0
    buckets = np.clip((wd[db] / day_len * n_buckets).astype(int), 0, n_buckets - 1) if len(db) else np.empty(0, int)
    out = bucket_stability(buckets, bind, n_buckets)
    if fid is not None:
        out["fidelity_ok"] = fid["ok"]
        out["_fidelity_maxdelta"] = fid["maxdelta"]
    out["n_trades"] = int(len(db))
    return out
