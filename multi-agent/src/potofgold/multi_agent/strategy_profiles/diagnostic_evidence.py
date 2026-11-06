"""Build and retrieve diagnostic reference findings from development strategies.

Fit each rejection threshold on the other development folds and apply it to
the excluded fold. Summarize profitability, discrimination and portfolio
changes separately, including results by input condition. The assessed
strategies and their held-out outcomes do not enter this reference.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence

from potofgold.multi_agent.analysis.court_statistics import cvar_left, _rejects
# Use the same fixed-calendar portfolio convention as the court statistics.

STATUSES = ("supported_on_dev_validation", "no_relationship_detected", "contrary_evidence",
            "insufficient_evidence", "out_of_scope")


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x


def _uid(r: dict) -> tuple:
    """Identify a strategy within its fold; the same genome can occur in several folds."""
    return (str(r["cid"]), r["fold"])


def _fold_key(x: Any) -> tuple:
    """Numeric-aware fold ordering: ints sort numerically (0,1,2,10), non-ints by str, ints before non-ints."""
    return (0, x) if isinstance(x, int) and not isinstance(x, bool) else (1, str(x))


def _winner(day_pnl: Dict[Any, float]) -> Optional[bool]:
    """True for positive net, False for a realized loss, None for zero or no valid days."""
    vals = [float(v) for v in day_pnl.values() if _is_num(v)]
    if not vals:
        return None
    net = sum(vals)
    return True if net > 0 else False if net < 0 else None


FOLD_POLICY_VERSION = "original-fold-purge-v1"


def _genome_key(cid: Any) -> str:
    """Legacy full-genome hash suffix; fitness makes this incomplete executable identity."""
    return str(cid).rsplit("_", 1)[-1]


def executable_genome_group(genome: Mapping[str, Any]) -> str:
    """Hash executable genome fields using the scorer's identity convention."""
    if not isinstance(genome, Mapping) or not genome:
        raise ValueError("Executable identity requires a nonempty genome")
    return json.dumps({key: value for key, value in genome.items() if key != "fitness"}, sort_keys=True)


def _relationship_keys(record: Mapping[str, Any]) -> tuple[set[tuple], bool]:
    """Identify executable duplicates and caller-declared strategy groups.

    Groups must be defined from structure or training evidence. Validation
    returns are not used to establish relatedness.
    """
    keys: set[tuple] = set()
    genome = record.get("genome") or (record.get("member") or {}).get("genome")
    if isinstance(genome, Mapping) and genome:
        # Compute identity locally to avoid importing the evaluator before its environment is set.
        key = executable_genome_group(genome)
        keys.add(("executable", key))
    for field in ("genome_group", "duplicate_group"):
        value = record.get(field)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a nonempty outcome-independent group")
            keys.add(("executable", value))
    complete = bool(keys)
    if not complete:
        keys.add(("legacy_champion_suffix", _genome_key(record["cid"])))
    related = record.get("related_groups", ())
    if isinstance(related, str):
        related = [related]
    if not isinstance(related, (list, tuple, set)):
        raise ValueError("related_groups must contain outcome-independent nonempty group strings")
    if record.get("related_group") is not None:
        related = [*related, record["related_group"]]
    for group in related:
        if not isinstance(group, str) or not group:
            raise ValueError("related_groups must contain outcome-independent nonempty group strings")
        keys.add(("related", group))
    return keys, complete


def _cluster_near_dups(recs: Sequence[dict], dpc=None,
                       corr_threshold: float = 0.99, min_overlap: int = 10) -> Dict[tuple, tuple]:
    """Group strategies by executable identity and declared relationships.

    The legacy dpc, corr_threshold and min_overlap arguments are unused.
    """
    uids = [_uid(r) for r in recs]
    if len(set(uids)) != len(uids):
        raise ValueError("Duplicate (champion, original fold) evidence records")
    parent = {u: u for u in uids}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    by_key: dict[tuple, tuple] = {}
    for record in recs:
        uid = _uid(record)
        for key in _relationship_keys(record)[0]:
            if key in by_key:
                parent[find(uid)] = find(by_key[key])
            else:
                by_key[key] = uid
    return {uid: find(uid) for uid in uids}


def original_fold_purge_plan(recs: Sequence[dict], *, enabled: bool = True) -> list[dict]:
    """Preserve validation folds and exclude each fold's relatives from threshold fitting.

    Return composite strategy-and-fold IDs for the fitting and evaluation sets.
    """
    clusters = _cluster_near_dups(recs)
    plan = []
    for fold in sorted({record["fold"] for record in recs}, key=_fold_key):
        validation_ids = [_uid(record) for record in recs if record["fold"] == fold]
        related = {clusters[uid] for uid in validation_ids}
        candidates = [_uid(record) for record in recs if record["fold"] != fold]
        purged = [uid for uid in candidates if enabled and clusters[uid] in related]
        excluded = set(purged)
        plan.append({"fold": fold, "validation_ids": validation_ids,
                     "fit_ids": [uid for uid in candidates if uid not in excluded], "purged_ids": purged})
    return plan


def original_fold_policy(recs: Sequence[dict], *, enabled: bool = True) -> dict:
    return {"version": FOLD_POLICY_VERSION, "regroup": False, "purge": bool(enabled),
            "preserve_original_folds": True, "heldout_outcomes_used_for_grouping": False,
            "relationship_source": "executable_genome_or_declared_cause_only_groups",
            "identity_complete": all(_relationship_keys(record)[1] for record in recs),
            "limitation": "Unspecified near-twins with different executable genomes are not detected."}


def current_fold_policy(policy: Any) -> bool:
    """Check that the reference uses complete, outcome-independent fold protection."""
    return (isinstance(policy, Mapping) and policy.get("version") == FOLD_POLICY_VERSION
            and policy.get("regroup") is False and policy.get("purge") is True
            and policy.get("preserve_original_folds") is True
            and policy.get("heldout_outcomes_used_for_grouping") is False
            and policy.get("identity_complete") is True)


def _reference_portfolio(day_pnl_by_champ, kept_ids, tail_q, days):
    """The fixed-calendar, responsible-strategy mean used to fit the saved rule book."""
    from potofgold.multi_agent.analysis.court_statistics import portfolio_series
    xs = portfolio_series(day_pnl_by_champ, kept_ids, days, mode="mean")
    return ((float(cvar_left(xs, tail_q)), sum(xs) / len(xs)) if xs
            else (float("nan"), float("nan")))


def threshold_fit_policy(*, min_participation: float = 0.1, min_calib_n: int = 10,
                         portfolio_mode: str = "mean") -> Dict[str, Any]:
    """Describe the candidate threshold search and participation requirements."""
    return {"id": "cvar_net_preserving_both_sides_v1", "portfolio_mode": portfolio_mode,
            "objective": "maximize_cvar_gain_subject_to_mean_not_decreasing",
            "candidate_thresholds": "observed_statistic_values_ascending",
            "tie_break": "first_maximum", "min_calib_n": min_calib_n,
            "min_participation": min_participation, "participation_rounding": "ceil",
            "minimum_per_side": 1, "participation_sides": ["kept", "rejected"]}


def _participation_need(n: int, min_participation: float) -> int:
    return max(1, math.ceil(min_participation * n))


def _best_cvar_threshold(calib: Sequence[tuple[float, str]], day_pnl_by_champ, reject_when: str, tail_q: float,
                         need: int) -> tuple:
    """Choose the cutoff with the largest improvement in portfolio CVaR.

    Require nondecreasing mean daily P&L and at least need strategies on each
    side. Return (threshold, cvar_gain), or (None, -inf) if no cutoff qualifies.
    """
    all_ids = [i for _, i in calib]
    days = sorted({d for i in all_ids for d in (day_pnl_by_champ.get(i) or {})},   # Use one calendar for the baseline and every candidate portfolio.
                  key=lambda x: (type(x).__name__, str(x)))
    base_cvar, base_net = _reference_portfolio(day_pnl_by_champ, all_ids, tail_q, days)
    best_thr, best_gain = None, float("-inf")
    for cand in sorted({s for s, _ in calib}):
        kept = [i for s, i in calib if not _rejects(s, cand, reject_when)]
        rej = [i for s, i in calib if _rejects(s, cand, reject_when)]
        if len(kept) < need or len(rej) < need:
            continue
        k_cvar, k_net = _reference_portfolio(day_pnl_by_champ, kept, tail_q, days)
        if not (_is_num(k_cvar) and _is_num(k_net)) or k_net < base_net:   # Reject cutoffs that reduce mean daily net P&L.
            continue
        gain = k_cvar - base_cvar
        if gain > best_gain:
            best_gain, best_thr = gain, float(cand)
    return best_thr, best_gain


def _best_cvar_threshold_meanbook(calib: Sequence[tuple], day_pnl_by_champ: Dict[str, Dict[Any, float]],
                                  reject_when: str, tail_q: float, min_keep: int) -> tuple:
    """Apply the shared threshold search to an equally weighted portfolio.

    min_keep applies to both retained and rejected strategies.
    """
    return _best_cvar_threshold(calib, day_pnl_by_champ, reject_when, tail_q, min_keep)


def _validation_decisions(recs: Sequence[dict], day_pnl_by_champ: Dict[str, Dict[Any, float]],
                          reject_when: str, tail_q: float, min_calib: int,
                          min_participation: float = 0.1, lofo_fold: Optional[Dict[tuple, Any]] = None,
                          *, purge_related: bool = True, split_audit: Optional[list] = None) -> tuple:
    """Fit thresholds on other original folds after excluding related strategies.

    If supplied, lofo_fold must agree with each record's original fold.
    """
    if lofo_fold is not None and any(lofo_fold.get(_uid(r)) != r["fold"] for r in recs):
        raise ValueError("Validation folds must retain their original identity")
    rows = {_uid(record): record for record in recs}
    decided: Dict[tuple, bool] = {}
    n_eval = 0
    for split in original_fold_purge_plan(recs, enabled=purge_related):
        others = [(rows[uid]["statistic"], uid) for uid in split["fit_ids"]
                  if _is_num(rows[uid]["statistic"])]
        detail = {"fold": split["fold"], "n_fit": len(others),
                  "n_validation": len(split["validation_ids"]), "n_purged": len(split["purged_ids"]),
                  "evaluated": False}
        if split_audit is not None:
            split_audit.append(detail)
        if len(others) < min_calib:
            detail["reason"] = "insufficient_reference_after_purge"
            continue
        need = _participation_need(len(others), min_participation)
        # Pass only fitting ledgers to the threshold search.
        fitting_pnl = {uid: day_pnl_by_champ.get(uid, {}) for _, uid in others}
        thr, _gain = _best_cvar_threshold_meanbook(others, fitting_pnl, reject_when, tail_q, need)
        if thr is None:
            detail["reason"] = "no_admissible_reference_threshold"
            continue
        detail["evaluated"] = True
        n_eval += 1
        for uid in split["validation_ids"]:
            if _is_num(rows[uid]["statistic"]):
                decided[uid] = bool(_rejects(rows[uid]["statistic"], float(thr), reject_when))
    return decided, n_eval


def _confusion(decided: Dict[str, bool], win: Dict[str, Optional[bool]], *,
               neutral: Sequence[Any] = ()) -> Dict[str, Any]:
    removed = [c for c, rm in decided.items() if rm]
    n_win = sum(1 for w in win.values() if w is True)
    n_los = sum(1 for w in win.values() if w is False)
    winners_removed = sum(1 for c in removed if win.get(c) is True)
    losers_removed = sum(1 for c in removed if win.get(c) is False)
    neutral_ids = set(neutral) & set(decided)
    neutral_removed = sum(c in neutral_ids for c in removed)
    return {
        "n_validated": len(decided), "n_removed": len(removed),
        "n_dev_winners": n_win, "n_dev_losers": n_los,
        "n_dev_neutral": len(neutral_ids), "neutral_removed": neutral_removed,
        "neutral_definition": "exactly zero held-out net; counted as neither winner nor loser",
        "losers_removed": losers_removed, "winners_removed": winners_removed,
        # No valid days is unlabeled; zero net P&L is neutral.
        "n_removed_unlabeled": len(removed) - winners_removed - losers_removed - neutral_removed,
        # Winner rejection is undefined when there are no winners.
        "winner_rejection_rate": (winners_removed / n_win) if n_win > 0 else None,
        "loser_removal_rate": (losers_removed / n_los) if n_los > 0 else None,
        "winner_rejection_rate_reason": (None if n_win > 0 else "no dev winners to estimate this rate"),
    }


def _paired_block_ci(diffs: Sequence[float], *, n_boot: int = 2000, alpha: float = 0.1,
                     block: Optional[int] = None) -> Optional[list]:
    """Estimate a confidence interval for paired mean daily differences.

    Use a fixed seed and return the union of independent-day and moving-block
    bootstrap intervals. Blocks have length approximately sqrt(n). Return None
    when fewer than two days are available.
    """
    import numpy as np
    d = np.asarray([float(x) for x in diffs], float)
    n = d.size
    if n < 2:
        return None
    rng = np.random.default_rng(0)                        # A fixed seed makes saved reference intervals reproducible.
    # Independent-day bootstrap interval.
    day_means = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    # Moving-block interval retains dependence within contiguous runs.
    L = int(block) if block else max(1, int(round(n ** 0.5)))
    L = min(L, n)
    n_blocks = -(-n // L)
    starts = rng.integers(0, n - L + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(n_boot, n_blocks * L)[:, :n]
    block_means = d[idx].mean(axis=1)
    lo_d, hi_d = np.quantile(day_means, [alpha / 2, 1 - alpha / 2])
    lo_b, hi_b = np.quantile(block_means, [alpha / 2, 1 - alpha / 2])
    # Take the union so the reported interval is no narrower than either estimate.
    return [float(min(lo_d, lo_b)), float(max(hi_d, hi_b))]


def _portfolio_metrics(decided: Dict[str, bool], day_pnl_by_champ: Dict[str, Dict[Any, float]],
                       tail_q: float) -> Dict[str, Any]:
    """Compare retained and complete portfolios on the fixed validation calendar.

    Strategies receive equal weight. Higher CVaR indicates a less severe loss tail.
    """
    all_ids = list(decided.keys())
    kept_ids = [c for c, rm in decided.items() if not rm]
    days = sorted({d for c in all_ids for d in day_pnl_by_champ.get(c, {})},
                  key=lambda x: (type(x).__name__, str(x)))
    if not days:
        return {"cvar_all": None, "cvar_kept": None, "cvar_improvement": None, "return_retained": None}
    from potofgold.multi_agent.analysis.court_statistics import portfolio_series
    s_all = portfolio_series(day_pnl_by_champ, all_ids, days, mode="mean")
    s_kept = portfolio_series(day_pnl_by_champ, kept_ids, days, mode="mean")
    cvar_all, cvar_kept = float(cvar_left(s_all, tail_q)), float(cvar_left(s_kept, tail_q))
    mean_all = sum(s_all) / len(s_all)
    mean_kept = sum(s_kept) / len(s_kept)
    ci = _paired_block_ci([k - a for k, a in zip(s_kept, s_all)])
    return {"cvar_all": cvar_all, "cvar_kept": cvar_kept, "cvar_improvement": cvar_kept - cvar_all,
            "mean_all": mean_all, "mean_kept": mean_kept,
            "return_change": mean_kept - mean_all,
            "return_change_ci90": ci,
            "return_change_ci_excludes_0": (ci is not None and (ci[0] > 0 or ci[1] < 0)),
            # Support requires the return-change interval to be strictly positive.
            "return_change_ci_positive": (ci is not None and ci[0] > 0),
            # A retention ratio is interpretable only when the baseline mean is positive.
            "return_retained": (mean_kept / mean_all) if mean_all > 0 else None}


def _status(conf: Dict[str, Any], port: Dict[str, Any], *, min_removed: int, eps: float = 1e-9,
            support_margin: float = 0.0) -> str:
    if conf["n_removed"] < min_removed or conf["n_validated"] == 0:
        return "insufficient_evidence"
    impr = port.get("cvar_improvement")
    rc = port.get("return_change")
    wrr, lrr = conf["winner_rejection_rate"], conf["loser_removal_rate"]
    if impr is not None and impr < -eps:
        return "contrary_evidence"
    if rc is not None and rc < -eps:
        return "contrary_evidence"
    if wrr is not None and lrr is not None and wrr > lrr:
        return "contrary_evidence"
    if conf["n_dev_winners"] == 0:
        return "insufficient_evidence"
    # Require better tail risk, stronger loser rejection and a positive return-change interval.
    ci_established = bool(port.get("return_change_ci_positive"))
    if (impr is not None and impr > eps and lrr is not None and wrr is not None and (lrr - wrr) > support_margin
            and rc is not None and rc >= -eps and ci_established):
        return "supported_on_dev_validation"
    return "no_relationship_detected"


def build_dev_evidence(diagnostic: str, records: Sequence[dict], *, reject_when: str, version: str,
                       dev_manifest_id: str, scope: str = "all", tail_q: float = 0.05,
                       min_calib: int = 10, min_removed: int = 1, regroup_near_dups: bool = True,
                       corr_threshold: float = 0.99, min_overlap: int = 10,
                       partial: bool = False, impl_version: Optional[str] = None,
                       min_participation: float = 0.1,
                       share_pooled_arm_decisions: bool = False) -> Dict[str, Any]:
    """Build one diagnostic reference from development strategy records.

    Records contain cid, arm, fold, statistic and daily P&L. Report pooled and
    per-condition results and hash the record to create its evidence_id.

    regroup_near_dups enables exclusion of related strategies from fitting,
    without moving their original folds. Executable genome identity and any
    related_groups must be supplied independently of validation outcomes.
    Legacy cid-only records cannot establish complete identity protection.
    corr_threshold and min_overlap are retained but unused.
    """
    recs = [r for r in records if _is_num(r.get("statistic"))]
    day_pnl = {_uid(r): {str(k): float(v) for k, v in (r.get("day_pnl") or {}).items() if _is_num(v)} for r in recs}

    def _decide(rs: Sequence[dict]):
        dpc = {_uid(r): day_pnl[_uid(r)] for r in rs}
        splits: list = []
        decided, n_folds_eval = _validation_decisions(rs, dpc, reject_when, tail_q, min_calib,
                                                     min_participation=min_participation,
                                                     purge_related=regroup_near_dups, split_audit=splits)
        return dpc, decided, n_folds_eval, splits

    def _assemble(rs: Sequence[dict], dpc, decided, n_folds_eval, splits) -> Dict[str, Any]:
        audit = {"n_near_dup_regrouped": 0,
                 "n_duplicate_fit_exclusions": sum(split["n_purged"] for split in splits),
                 "validation_splits": splits}
        win = {_uid(r): _winner(dpc[_uid(r)]) for r in rs if _uid(r) in decided}
        neutral = [uid for uid in decided if dpc.get(uid) and sum(dpc[uid].values()) == 0]
        conf = _confusion(decided, win, neutral=neutral)
        if not decided or n_folds_eval < 2:
            return {"status": "insufficient_evidence", "n_folds_evaluated": n_folds_eval,
                    "n_applicable": len(rs), **audit,
                    "confusion": conf, "portfolio": {}}
        port = _portfolio_metrics(decided, dpc, tail_q)
        return {"status": _status(conf, port, min_removed=min_removed), "n_folds_evaluated": n_folds_eval,
                "n_applicable": len(rs), **audit, "confusion": conf, "portfolio": port}

    def _block(rs: Sequence[dict]) -> Dict[str, Any]:
        return _assemble(rs, *_decide(rs))

    arms = sorted({r.get("arm") for r in recs if r.get("arm") is not None})
    if recs and share_pooled_arm_decisions:
        # Slice one pooled procedure by condition; do not refit a cutoff for each slice.
        p_dpc, p_decided, p_nfe, p_splits = _decide(recs)
        overall = _assemble(recs, p_dpc, p_decided, p_nfe, p_splits)
        per_arm = {}
        for arm in arms:
            rs_arm = [r for r in recs if r.get("arm") == arm]
            uids = {_uid(r) for r in rs_arm}
            d_arm = {u: d for u, d in p_decided.items() if u in uids}
            dpc_arm = {u: v for u, v in p_dpc.items() if u in uids}
            nfe_arm = len({r.get("fold") for r in rs_arm if _uid(r) in d_arm})
            block = _assemble(rs_arm, dpc_arm, d_arm, nfe_arm, p_splits)
            block["decisions_sliced_from_pooled"] = True
            per_arm[arm] = block
    else:
        overall = _block(recs) if recs else {"status": "out_of_scope", "n_applicable": 0,
                                             "reason": "no champion produced an applicable statistic on dev"}
        per_arm = {arm: _block([r for r in recs if r.get("arm") == arm]) for arm in arms}

    rec = {
        "diagnostic": diagnostic, "scope": scope, "version": version, "dev_manifest_id": dev_manifest_id,
        "reject_when": reject_when,
        "reference_rule": (f"On DEV VALIDATION, flag a champion when its {diagnostic} statistic is "
                           f"'{reject_when}' the leave-one-fold-out-fitted threshold (fit on the OTHER dev folds, "
                           f"applied to the excluded original fold after purging its related strategies from fitting), "
                           f"using the recorded shared calibration fit policy. "
                           f"Counts below are REFERENCE-RULE outcomes on held-out "
                           f"dev, NOT predictions of court behaviour."),
        "winner_definition": "held-out net mean PnL > 0",
        "loser_definition": "held-out net mean PnL < 0; exactly zero is neutral",
        "portfolio_convention": "equal-weight over responsible champions (scorer convention), CVaR left-tail",
        "contract": "development outcomes informed this record; the evaluated champion's outcomes are hidden",
        "near_dup_policy": original_fold_policy(recs, enabled=regroup_near_dups),
        "config": {"tail_q": tail_q, "min_calib": min_calib, "min_removed": min_removed,
                   "fit_policy": threshold_fit_policy(min_participation=min_participation,
                                                       min_calib_n=min_calib, portfolio_mode="mean")},
        "impl_version": impl_version,                    # Diagnostic implementation version, independent of the package release.
        "partial": bool(partial),                       # Incomplete builds cannot supply reference support.
        "support_basis": ("day-block bootstrap CI on the equal-weight return change; a 'supported' status is a "
                          "DESCRIPTIVE positive effect unless per-arm/overall return_change_ci_excludes_0 is true"),
        "overall": overall, "per_arm": per_arm,
    }
    # Hash results and configuration together to identify this reference.
    rec["evidence_id"] = hashlib.sha1(
        json.dumps({k: rec[k] for k in ("diagnostic", "scope", "version", "dev_manifest_id", "reject_when",
                                        "impl_version", "partial", "near_dup_policy", "config", "overall", "per_arm")},
                   sort_keys=True, default=str).encode()).hexdigest()[:16]
    return rec


class EvidenceStore:
    """Index reference records by diagnostic, scope, version, population and reference role."""

    def __init__(self, records: Sequence[Dict[str, Any]] = ()):
        self._by_key: Dict[tuple, Dict[str, Any]] = {}
        for r in records:
            self.add(r)

    @staticmethod
    def _key(diagnostic, scope, version, dev_manifest_id, reference_role=None) -> tuple:
        # Include the reference role so primary and secondary records can coexist.
        return (str(diagnostic), str(scope), str(version), str(dev_manifest_id), str(reference_role or ""))

    def add(self, rec: Dict[str, Any]) -> None:
        self._by_key[self._key(rec["diagnostic"], rec.get("scope", "all"), rec["version"], rec["dev_manifest_id"],
                               rec.get("reference_role"))] = rec

    def get(self, diagnostic, scope, version, dev_manifest_id, reference_role=None) -> Optional[Dict[str, Any]]:
        """Find the requested reference role.

        Without a role, prefer the primary record, then a legacy record without a
        role, then secondary context.
        """
        if reference_role is not None:
            return self._by_key.get(self._key(diagnostic, scope, version, dev_manifest_id, reference_role))
        for role in ("primary", "", "secondary_context"):
            rec = self._by_key.get(self._key(diagnostic, scope, version, dev_manifest_id, role))
            if rec is not None:
                return rec
        return None

    def __len__(self) -> int:
        return len(self._by_key)

    @property
    def records(self) -> list:
        return list(self._by_key.values())

    def dev_manifest_ids(self) -> set:
        """Return the development population IDs represented in the store."""
        return {str(r.get("dev_manifest_id")) for r in self._by_key.values()}

    def save(self, path: str) -> None:
        from pathlib import Path
        Path(path).write_text(json.dumps({"records": list(self._by_key.values())}, indent=0, default=str))

    @classmethod
    def load(cls, path: str) -> "EvidenceStore":
        from pathlib import Path
        try:
            d = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            return cls()
        return cls(d.get("records") or [])


EVIDENCE_LOOKUP_TOOL = {
    "name": "diagnostic_evidence_lookup",
    "description": ("Look up frozen development evidence for an available measured result_id. The response separates "
                    "winner/loser discrimination, return effect, tail effect and winner preservation. For dimensional "
                    "references, dev_status summarizes discrimination only; read each other dimension and its "
                    "uncertainty separately. Unresolved means uncertain, not an established absence of relationship. "
                    "Historical composites remain in the store audit. Read the explicit count denominators and "
                    "reference accounting: these CVaR effects do not estimate the graduated allocation policy. "
                    "Supply the result_id of a test already measured or supplied in this session. "
                    "The fitted rule and development distribution help interpret the observation; neither the "
                    "reference nor an adverse observation is a verdict or proof of future loss."),
    "input_schema": {"type": "object", "required": ["result_id"], "additionalProperties": False,
                     "properties": {"result_id": {"type": "string",
                                    "description": "result_id of a measured result available in this session"}}},
}


class UnknownMeasurement(KeyError):
    """An identifier that does not refer to a test executed in this assessment.

    Distinct from KeyError raised while reading a malformed reference record.
    """


def make_evidence_resolver(result_index: Dict[str, Any], *, arm: Optional[str], version: str,
                           dev_manifest_id: str, scope: str = "all",
                           statistic_name: Optional[str] = None, reject_when: Optional[str] = None):
    """Bind executed results to reference lookups.

    Fix the condition, scope, implementation and development population on the
    server. Agents supply only a result ID. Resolve its registered primary
    statistic from the executed measurement.
    """
    def _attr(rec, key):
        return rec.get(key) if isinstance(rec, dict) else getattr(rec, key, None)

    def resolver(measurement_id: str) -> Dict[str, Any]:
        rec = result_index.get(str(measurement_id))
        if rec is None:
            raise UnknownMeasurement(f"unknown result_id {measurement_id!r} — look up only a test you ran this session")
        diagnostic = _attr(rec, "test_id") or _attr(rec, "diagnostic")
        stat = _attr(rec, "statistic")
        ms = _attr(rec, "measured_statistics")
        if ms is not None:
            primary = statistic_name
            if primary is None:
                from potofgold.multi_agent.diagnostics.vulnerability_catalogue import REGISTRY
                entry = REGISTRY.get(diagnostic)
                primary = entry.statistic_name if entry is not None else None
            stat = ms.get(primary) if isinstance(ms, Mapping) and primary is not None else None
        resolved = {"diagnostic": diagnostic, "statistic": stat, "scope": scope,
                    "version": version, "dev_manifest_id": dev_manifest_id, "arm": arm,
                    # Use the measurement's implementation version when checking compatibility.
                    "impl_version": _attr(rec, "test_version") or _attr(rec, "impl_version")}
        if reject_when is not None:
            resolved["reject_when"] = reject_when
        return resolved
    return resolver


def _lookup_reference(m: Mapping[str, Any], store: EvidenceStore, *,
                      allow_partial: bool, require_compatible: bool, reference_role=None) -> Dict[str, Any]:
    """Check measurement definition and scope against the reference.

    Primary records can support a finding; secondary records supply context.
    """
    measurement_id = m.get("measurement_id")
    rec = store.get(m["diagnostic"], m.get("scope", "all"), m["version"], m["dev_manifest_id"],
                    reference_role=reference_role)
    if rec is None:
        return {"measurement_id": measurement_id, "status": "out_of_scope",
                "reason": "no frozen dev evidence for this (diagnostic, scope, version, dev_manifest)"}
    if rec.get("partial") and not allow_partial:          # Partial builds cannot supply reference support.
        return {"measurement_id": measurement_id, "status": "out_of_scope",
                "reason": "partial/smoke evidence artifact — not admissible as final reference evidence"}
    impl = m.get("impl_version")
    if require_compatible:
        if not current_fold_policy(rec.get("near_dup_policy")):
            return {"measurement_id": measurement_id, "status": "out_of_scope",
                    "reason": "obsolete or incomplete original-fold validation policy; rebuild development evidence"}
        def _nonempty_string(value):
            return isinstance(value, str) and bool(value.strip())

        if not _nonempty_string(impl) or not _nonempty_string(rec.get("impl_version")):
            return {"measurement_id": measurement_id, "status": "out_of_scope",
                    "reason": "measurement and evidence require nonempty impl_version metadata"}
        if not _nonempty_string(rec.get("evidence_id")):
            return {"measurement_id": measurement_id, "status": "out_of_scope",
                    "reason": "frozen dev evidence requires a nonempty evidence_id"}
        if m.get("reject_when") is not None and m["reject_when"] != rec.get("reject_when"):
            return {"measurement_id": measurement_id, "status": "out_of_scope",
                    "reason": "reject_when mismatch between measurement definition and frozen evidence"}
    if impl is not None and rec.get("impl_version") is not None and str(impl) != str(rec["impl_version"]):
        return {"measurement_id": measurement_id, "status": "out_of_scope",
                "reason": f"impl_version mismatch: measurement {impl} vs evidence {rec['impl_version']}"}
    arm = m.get("arm")
    if arm is not None:                                    # Use the matching condition's results.
        arm_blocks = rec.get("per_arm")
        if arm_blocks is not None and not isinstance(arm_blocks, Mapping):
            raise TypeError("development evidence per_arm must be a mapping")
        block = (arm_blocks or {}).get(arm)
        if block is not None and not isinstance(block, Mapping):
            raise TypeError("development evidence arm block must be a mapping")
        status = block["status"] if block else "out_of_scope"
    else:
        block, status = rec.get("overall"), rec["overall"]["status"]
    if block is not None:
        if not isinstance(block, Mapping):
            raise TypeError("development evidence block must be a mapping")
        for field in ("confusion", "portfolio"):
            if block.get(field) is not None and not isinstance(block[field], Mapping):
                raise TypeError(f"development evidence {field} must be a mapping")
    return {"measurement_id": measurement_id, "diagnostic": m["diagnostic"], "statistic": m.get("statistic"),
            "status": status, "block": block, "reference_rule": rec.get("reference_rule"),
            "support_basis": rec.get("support_basis"), "partial": rec.get("partial"),
            "reference_role": rec.get("reference_role"), "evidence": rec}


def diagnostic_reference_lookup(diagnostic: str, store: EvidenceStore, *, scope: str, version: str,
                                dev_manifest_id: str, arm: str, impl_version: str,
                                reject_when: str, reference_role=None) -> Dict[str, Any]:
    """Read compatible development findings before a test runs.

    This supplies context, including contrary or insufficient findings. A
    rejection still requires an executed, reliable and adverse measurement.
    """
    if not arm or not version or not dev_manifest_id:
        return {"diagnostic": diagnostic, "status": "out_of_scope",
                "reason": "missing_development_evidence_identity_or_arm"}
    result = _lookup_reference(
        {"diagnostic": diagnostic, "scope": scope, "version": version, "dev_manifest_id": dev_manifest_id,
         "arm": arm, "impl_version": impl_version, "reject_when": reject_when},
        store, allow_partial=False, require_compatible=True, reference_role=reference_role)
    return {k: v for k, v in result.items() if k not in ("measurement_id", "statistic")}


def diagnostic_evidence_lookup(measurement_id: str, store: EvidenceStore, resolver,
                               *, allow_partial: bool = False,
                               require_compatible: bool = False) -> Dict[str, Any]:
    """Check an executed measurement against its development reference.

    Require a finite numeric primary statistic for strict admission and use
    the matching strategy condition. Preserve legacy handling when
    implementation metadata is absent.
    """
    m = dict(resolver(measurement_id), measurement_id=measurement_id)
    statistic = m.get("statistic")
    if require_compatible and (not _is_num(statistic) or not math.isfinite(statistic)):
        return {"measurement_id": measurement_id, "status": "out_of_scope",
                "reason": "measurement must be a finite nonboolean primary statistic"}
    return _lookup_reference(m, store, allow_partial=allow_partial, require_compatible=require_compatible)
