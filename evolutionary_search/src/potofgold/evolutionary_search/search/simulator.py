"""Evaluate rule trees and simulate intraday option trades with proxy prices."""

from __future__ import annotations
from .normalization import TERMINAL_NORM_STATS
import math
import os
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .grammar import FuncNode, GType, Node, TermNode, to_str
from .normalization import NORMALIZED_TERMINALS
from .records import BacktestResult, Trade
from .wing_iv import correct_put_wing_skew

_WING_IV_FIX_ENABLED = os.environ.get("GP_WING_IV_FIX", "1") != "0"
MAX_SPREAD_WIDTH = 40.0
LAG_MAX = 30


def _covered_gap_width(active_legs, strikes) -> float:
    w = 0.0
    for ci in range(len(active_legs)):
        if active_legs[ci].qty_sign < 0:
            oti = active_legs[ci].option_type
            gaps = [
                abs(strikes[ci] - strikes[cj])
                for cj in range(len(active_legs))
                if active_legs[cj].qty_sign > 0
                and active_legs[cj].option_type == oti
                and (strikes[cj] != strikes[ci])
            ]
            if gaps:
                w = max(w, max(gaps))
    return w


CONCENTRATION_CAP_FRAC = float(os.environ.get("GP_CONCENTRATION_CAP", "0.04"))
_SIZING_CREDIT_FILL = float(os.environ.get("GP_SIZING_CREDIT_FILL", "0.25"))
_DEBIT_UW_MARK_DECAY = float(os.environ.get("GP_DEBIT_UW_MARK_DECAY", "1.2"))
_MARK_IV_VIX_MULT = float(os.environ.get("GP_MARK_IV_VIX_MULT", "4"))


def _apply_mark_iv_cap(iv, bar_grid, vix_i, mult):
    if mult <= 0 or vix_i is None or (not vix_i > 0.01):
        return (iv, bar_grid)
    corrupt = mult * vix_i
    if iv > corrupt:
        iv = vix_i
    if bar_grid is not None:
        bar_grid = np.where(bar_grid > corrupt, vix_i, bar_grid)
    return (iv, bar_grid)


def _repair_scalar_iv_vs_vix(iv_arr, vix_frac_arr, mult):
    if mult <= 0 or vix_frac_arr is None:
        return iv_arr
    corrupt = mult * vix_frac_arr
    repairable = (vix_frac_arr > 0.01) & (iv_arr > corrupt)
    return np.where(repairable, vix_frac_arr, iv_arr)


_MIN_SPREAD_COST_PER_LEG = 0.1


def _clamp_credit_fill_factor(base_factor: float, regime_mult: float) -> float:
    return min(base_factor * regime_mult, 1.0)


_MAX_SORTINO = 10.0
_MIN_DOWNSIDE_OBS = 5
_DOWNSIDE_DEV_STD_FRACTION = 1.0 / 3.0
_MAX_HOLD_VESTIGIAL_PENALTY = 1.0
STOP_LOSS_EXECUTION_DISCOUNT = 0.85
STOP_LOSS_FILL_MARKUP = 1.1
_MAX_STOP_SLIPPAGE = 1.5 * 1.5
_DEBIT_STOP_SHAPES = frozenset({"long_strangle"})
EOD_FORCE_CLOSE_MTC = 25.0
ENTRY_CUTOFF_MTC = 30.0
UNREALIZED_PROFIT_PCT = "UnrealizedProfitPct"
MAX_FAVORABLE_EXCURSION = "MaxFavorableExcursion"
MAX_ADVERSE_EXCURSION = "MaxAdverseExcursion"
PROFIT_GIVEBACK = "ProfitGiveback"
_PATH_STATE_TERMINALS = (
    MAX_FAVORABLE_EXCURSION,
    MAX_ADVERSE_EXCURSION,
    PROFIT_GIVEBACK,
)
_UPP_DENOM_FLOOR = 1e-06


def _load_pca_bases_for_vectorized():
    raise ValueError(
        "Embedding evaluation requires saved training-fold readout components"
    )


def _project_emb_pc(
    v: np.ndarray, mean_: np.ndarray, components_k: np.ndarray, pc_std_k: float
) -> np.ndarray:
    is_zero = ~np.any(v, axis=1)
    centered = v - mean_[np.newaxis, :]
    raw = centered @ components_k
    raw[is_zero] = 0.0
    return raw / float(pc_std_k) if pc_std_k > 1e-12 else raw


def _safe_shift(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return arr
    result = np.zeros_like(arr)
    result[k:] = arr[:-k]
    return result


def _session_lag_daily(
    values: np.ndarray, dates: np.ndarray, prior_value: "Optional[float]" = None
) -> np.ndarray:
    out = values.copy()
    n = len(values)
    if n == 0:
        return out
    day_start = 0
    prev_day_val = prior_value
    while day_start < n:
        d = dates[day_start]
        day_end = day_start
        while day_end < n and dates[day_end] == d:
            day_end += 1
        if prev_day_val is not None:
            out[day_start:day_end] = prev_day_val
        prev_day_val = values[day_end - 1]
        day_start = day_end
    return out


class VectorizedTreeEvaluator:

    def __init__(
        self,
        terminal_data: Dict[str, np.ndarray],
        n_bars: int,
        day_boundary_mask: Optional[np.ndarray] = None,
        within_day_pos: Optional[np.ndarray] = None,
        fold_recenter_stats: Optional[Dict] = None,
        fold_id: Optional[str] = None,
        pca_bases: Optional[Dict] = None,
        emb_proj_cache: Optional[Dict] = None,
    ):
        self.terminal_data = terminal_data
        self.n_bars = n_bars
        self._fold_recenter_stats = fold_recenter_stats
        self._fold_id = fold_id
        self._pca_bases = pca_bases
        self._emb_proj_cache = emb_proj_cache
        if day_boundary_mask is not None:
            self._not_day_start = day_boundary_mask
        else:
            self._not_day_start = np.ones(n_bars, dtype=np.float64)
            self._not_day_start[0] = 0.0
        if within_day_pos is not None:
            self._within_day_pos = within_day_pos
        else:
            self._within_day_pos = np.arange(n_bars, dtype=np.int64)

    def evaluate(self, node: Node) -> np.ndarray:
        if isinstance(node, TermNode):
            return self._eval_terminal(node)
        return self._eval_function(node)

    def _eval_terminal(self, node: TermNode) -> np.ndarray:
        if node.ret_type == GType.REAL:
            if node.value is not None:
                return np.full(self.n_bars, float(node.value), dtype=np.float64)
            arr = self.terminal_data.get(node.name)
            if arr is not None:
                return arr.astype(np.float64)
            if not hasattr(self, "_warned_missing"):
                self._warned_missing = set()
            if node.name not in self._warned_missing:
                self._warned_missing.add(node.name)
                import warnings

                warnings.warn(
                    f"Terminal '{node.name}' not in terminal_data — returning zeros. GP trees referencing this terminal will see constant 0.0.",
                    stacklevel=2,
                )
            return np.zeros(self.n_bars, dtype=np.float64)
        if node.ret_type.name.startswith("EMB_"):
            if node.name.startswith("PROTO_"):
                return self._resolve_proto(node.name)
            arr = self.terminal_data.get(node.name)
            if arr is not None and arr.ndim == 2:
                return arr.astype(np.float64)
            if not hasattr(self, "_warned_missing_emb"):
                self._warned_missing_emb = set()
            if node.name not in self._warned_missing_emb:
                self._warned_missing_emb.add(node.name)
                import warnings

                warnings.warn(
                    f"Embedding terminal '{node.name}' is missing from terminal_data; "
                    "returning zero vectors. Projections of this input will be constant. "
                    "Check data.TYPED_VECTOR_COLUMNS and the prepared Parquet columns.",
                    stacklevel=2,
                )
            return np.zeros((self.n_bars, 384), dtype=np.float64)
        if node.ret_type in (GType.INT, GType.BANK_IDX):
            return np.full(
                self.n_bars,
                int(node.value) if node.value is not None else 1,
                dtype=np.float64,
            )
        if node.ret_type == GType.SIDE:
            return np.full(self.n_bars, node.value, dtype=object)
        if node.ret_type == GType.REGIME:
            return np.full(self.n_bars, node.value, dtype=object)
        if node.ret_type == GType.SHAPE_ID:
            return np.full(self.n_bars, node.value, dtype=object)
        return np.zeros(self.n_bars, dtype=np.float64)

    def _eval_function(self, node: FuncNode) -> np.ndarray:
        name = node.name
        ch = node.children
        if name == "Lag":
            lag = int(np.clip(self.evaluate(ch[1]).flat[0], 0, LAG_MAX))
            arr = self.evaluate(ch[0])
            if arr.ndim == 1:
                shifted = _safe_shift(arr, lag)
                shifted[self._within_day_pos < lag] = 0.0
                return shifted
            return arr
        if name == "Delta":
            lag = int(np.clip(self.evaluate(ch[1]).flat[0], 1, LAG_MAX))
            arr = self.evaluate(ch[0])
            if arr.ndim == 1:
                shifted = _safe_shift(arr, lag)
                mask = self._within_day_pos >= lag
                return np.where(mask, arr - shifted, 0.0)
            return np.zeros(self.n_bars, dtype=np.float64)
        if name == "CrossAbove":
            a = self.evaluate(ch[0]).astype(np.float64)
            b = self.evaluate(ch[1]).astype(np.float64)
            prev_a = _safe_shift(a, 1)
            prev_b = _safe_shift(b, 1)
            signal = ((a > b) & (prev_a <= prev_b)).astype(np.float64)
            signal *= self._not_day_start
            return signal
        if name == "CrossBelow":
            a = self.evaluate(ch[0]).astype(np.float64)
            b = self.evaluate(ch[1]).astype(np.float64)
            prev_a = _safe_shift(a, 1)
            prev_b = _safe_shift(b, 1)
            signal = ((a < b) & (prev_a >= prev_b)).astype(np.float64)
            signal *= self._not_day_start
            return signal
        if name == "GT":
            return (
                self.evaluate(ch[0]).astype(np.float64)
                > self.evaluate(ch[1]).astype(np.float64)
            ).astype(np.float64)
        if name == "LT":
            return (
                self.evaluate(ch[0]).astype(np.float64)
                < self.evaluate(ch[1]).astype(np.float64)
            ).astype(np.float64)
        if name == "AND":
            return (
                (self.evaluate(ch[0]).astype(np.float64) > 0.5)
                & (self.evaluate(ch[1]).astype(np.float64) > 0.5)
            ).astype(np.float64)
        if name == "OR":
            return (
                (self.evaluate(ch[0]).astype(np.float64) > 0.5)
                | (self.evaluate(ch[1]).astype(np.float64) > 0.5)
            ).astype(np.float64)
        if name == "NOT":
            return (self.evaluate(ch[0]).astype(np.float64) <= 0.5).astype(np.float64)
        if name == "Add":
            return self.evaluate(ch[0]).astype(np.float64) + self.evaluate(
                ch[1]
            ).astype(np.float64)
        if name == "Sub":
            return self.evaluate(ch[0]).astype(np.float64) - self.evaluate(
                ch[1]
            ).astype(np.float64)
        if name == "Mul":
            return self.evaluate(ch[0]).astype(np.float64) * self.evaluate(
                ch[1]
            ).astype(np.float64)
        if name == "Div":
            a = self.evaluate(ch[0]).astype(np.float64)
            b = self.evaluate(ch[1]).astype(np.float64)
            return a / np.sqrt(1.0 + b * b)
        if name == "Sqrt":
            return np.sqrt(np.abs(self.evaluate(ch[0]).astype(np.float64)))
        if name == "Abs":
            return np.abs(self.evaluate(ch[0]).astype(np.float64))
        if name in ("WeightedSum3", "WeightedSum5"):
            _n = len(ch) // 2
            acc = self.evaluate(ch[_n]).astype(np.float64) * self.evaluate(
                ch[0]
            ).astype(np.float64)
            for _k in range(1, _n):
                acc = acc + self.evaluate(ch[_n + _k]).astype(
                    np.float64
                ) * self.evaluate(ch[_k]).astype(np.float64)
            return acc
        if name in ("IfThenElse", "IfSide", "IfThenElseShape"):
            cond = self.evaluate(ch[0]).astype(np.float64) > 0.5
            then_val = self.evaluate(ch[1])
            else_val = self.evaluate(ch[2])
            return np.where(cond, then_val, else_val)
        if name.startswith("EmbNorm_"):
            v = self.evaluate(ch[0])
            if v.ndim == 2:
                return np.linalg.norm(v, axis=1)
            return np.zeros(self.n_bars, dtype=np.float64)
        if name.startswith("EmbCos_"):
            va = self.evaluate(ch[0])
            vb = self.evaluate(ch[1])
            if va.ndim == 2 and vb.ndim == 2:
                dot = np.sum(va * vb, axis=1)
                na = np.linalg.norm(va, axis=1)
                nb = np.linalg.norm(vb, axis=1)
                denom = na * nb
                denom = np.where(denom < 1e-08, 1.0, denom)
                return dot / denom
            return np.zeros(self.n_bars, dtype=np.float64)
        if name.startswith("EmbSub_"):
            va = self.evaluate(ch[0])
            vb = self.evaluate(ch[1])
            if va.ndim == 2 and vb.ndim == 2:
                return va - vb
            return np.zeros((self.n_bars, 384), dtype=np.float64)
        if name.startswith("EmbLag_"):
            v = self.evaluate(ch[0])
            lag = max(0, min(int(self.evaluate(ch[1]).flat[0]), LAG_MAX))
            if v.ndim == 2 and lag > 0:
                shifted = np.zeros_like(v)
                shifted[lag:] = v[:-lag]
                mask = self._within_day_pos >= lag
                shifted[~mask] = 0.0
                return shifted
            return v if v.ndim == 2 else np.zeros((self.n_bars, 384), dtype=np.float64)
        if name.startswith("EmbProj_"):
            suffix = name[len("EmbProj_") :]
            _dot = suffix.rfind("_")
            try:
                k = int(suffix[_dot + 1 :])
                group = suffix[:_dot]
            except (ValueError, IndexError):
                return np.zeros(self.n_bars, dtype=np.float64)
            ch0 = ch[0]
            if (
                self._emb_proj_cache is not None
                and isinstance(ch0, TermNode)
                and (ch0.defn.name == group)
                and ((group, k) in self._emb_proj_cache)
            ):
                mahal = self._emb_proj_cache[group, k]
                return self._apply_emb_fold_recenter(mahal, group, k)
            v = self.evaluate(ch0)
            if v.ndim != 2:
                return np.zeros(self.n_bars, dtype=np.float64)
            bases = (
                self._pca_bases
                if self._pca_bases is not None
                else _load_pca_bases_for_vectorized()
            )
            if bases is None or group not in bases:
                return np.zeros(self.n_bars, dtype=np.float64)
            g = bases[group]
            components = g["components"]
            if k >= components.shape[0]:
                return np.zeros(self.n_bars, dtype=np.float64)
            mean_ = g["mean_"]
            pc_std = g["pc_std_"]
            mahal = _project_emb_pc(v, mean_, components[k], float(pc_std[k]))
            return self._apply_emb_fold_recenter(mahal, group, k)
        if name.startswith("EmbDir_"):
            group = name[len("EmbDir_") :]
            bases = (
                self._pca_bases
                if self._pca_bases is not None
                else _load_pca_bases_for_vectorized()
            )
            if bases is None or group not in bases:
                return np.zeros(self.n_bars, dtype=np.float64)
            g = bases[group]
            dir_bank = g.get("dir_bank_")
            if dir_bank is None:
                return np.zeros(self.n_bars, dtype=np.float64)
            b = int(self.evaluate(ch[1]).flat[0]) % int(dir_bank.shape[0])
            ch0 = ch[0]
            if (
                self._emb_proj_cache is not None
                and isinstance(ch0, TermNode)
                and (ch0.defn.name == group)
                and ((group, "dir", b) in self._emb_proj_cache)
            ):
                return self._apply_emb_fold_recenter(
                    self._emb_proj_cache[group, "dir", b], group, b
                )
            v = self.evaluate(ch0)
            if v.ndim != 2:
                return np.zeros(self.n_bars, dtype=np.float64)
            dir_mean = g.get("dir_mean_", g["mean_"])
            dir_scale = g["dir_scale_"]
            mahal = _project_emb_pc(v, dir_mean, dir_bank[b], float(dir_scale[b]))
            return self._apply_emb_fold_recenter(mahal, group, b)
        if name == "InRegime":
            regime_arr = self.terminal_data.get(
                "PredRegime", np.zeros(self.n_bars, dtype=np.float64)
            )
            target = self.evaluate(node.children[0])
            return (regime_arr == target).astype(np.float64)
        if name == "RegimeIs":
            left = self.evaluate(node.children[0])
            right = self.evaluate(node.children[1])
            return (left == right).astype(np.float64)
        warnings.warn(
            f"VectorizedTreeEvaluator: unknown function '{name}', returning zeros",
            stacklevel=2,
        )
        return np.zeros(self.n_bars, dtype=np.float64)

    def _resolve_proto(self, name: str) -> np.ndarray:
        suffix = name[len("PROTO_") :]
        _dot = suffix.rfind("_")
        try:
            p = int(suffix[_dot + 1 :])
            group = suffix[:_dot]
        except (ValueError, IndexError):
            return np.zeros((self.n_bars, 384), dtype=np.float64)
        bases = (
            self._pca_bases
            if self._pca_bases is not None
            else _load_pca_bases_for_vectorized()
        )
        if bases is None or group not in bases:
            return np.zeros((self.n_bars, 384), dtype=np.float64)
        proto = bases[group].get("proto_bank_")
        if proto is None or p >= proto.shape[0]:
            return np.zeros((self.n_bars, 384), dtype=np.float64)
        tiled = np.tile(proto[p].astype(np.float64), (self.n_bars, 1))
        emb = self.terminal_data.get(group)
        if (
            emb is not None
            and getattr(emb, "ndim", 0) == 2
            and (emb.shape[0] == self.n_bars)
        ):
            tiled[~np.any(emb, axis=1)] = 0.0
        return tiled

    def _apply_emb_fold_recenter(
        self, mahal: np.ndarray, group: str, k: int
    ) -> np.ndarray:
        if hasattr(self, "_fold_recenter_stats") and self._fold_recenter_stats:
            fold_id = getattr(self, "_fold_id", None)
            if not fold_id or fold_id not in self._fold_recenter_stats:
                raise RuntimeError(
                    f"Fold recentering statistics do not contain fold_id={fold_id!r}; "
                    f"available folds: {sorted(self._fold_recenter_stats.keys())}. "
                    "Supply statistics fitted on the same training partition and "
                    "indexed by this run's fold identifier."
                )
            grp_stats = self._fold_recenter_stats[fold_id].get(group)
            if grp_stats and k < len(grp_stats["mean"]):
                mahal = (mahal - float(grp_stats["mean"][k])) / float(
                    grp_stats["std"][k]
                )
        return mahal


def tree_references_position_state(node: Node) -> bool:
    if node is None:
        return False
    if isinstance(node, TermNode):
        return node.name == UNREALIZED_PROFIT_PCT or node.name in _PATH_STATE_TERMINALS
    return any((tree_references_position_state(c) for c in node.children))


tree_references_upp = tree_references_position_state
_EMB_OP_PREFIXES = ("EmbProj_", "EmbDir_", "EmbCos_", "EmbSub_", "EmbNorm_", "EmbLag_")
_EMB_EXIT_KEY = "__embexit__:"


def _is_emb_op(node) -> bool:
    return isinstance(node, FuncNode) and node.name.startswith(_EMB_OP_PREFIXES)


def _collect_emb_op_nodes(tree, out=None) -> Dict:
    if out is None:
        out = {}
    if tree is None:
        return out
    if _is_emb_op(tree):
        out[to_str(tree)] = tree
        return out
    for c in getattr(tree, "children", None) or ():
        _collect_emb_op_nodes(c, out)
    return out


def _build_exit_terminal_data(terminal_data, exit_tree, evaluator, uses_upp):
    if not uses_upp:
        return terminal_data
    emb_ops = _collect_emb_op_nodes(exit_tree)
    if not emb_ops:
        return terminal_data
    out = dict(terminal_data)
    for k, node in emb_ops.items():
        out[_EMB_EXIT_KEY + k] = evaluator.evaluate(node).astype(np.float64)
    return out


def _eval_node_scalar(
    node: Node,
    i: int,
    terminal_data: Dict[str, np.ndarray],
    within_day_pos: np.ndarray,
    not_day_start: np.ndarray,
    upp_value: float,
    upp_hist: List[float],
) -> float:
    if isinstance(node, TermNode):
        if node.name == UNREALIZED_PROFIT_PCT:
            return float(upp_value)
        if node.name in _PATH_STATE_TERMINALS:
            if not upp_hist:
                return 0.0
            if node.name == MAX_FAVORABLE_EXCURSION:
                return float(max(upp_hist))
            if node.name == MAX_ADVERSE_EXCURSION:
                return float(min(upp_hist))
            return float(max(upp_hist) - upp_value)
        if node.ret_type == GType.REAL:
            if node.value is not None:
                return float(node.value)
            arr = terminal_data.get(node.name)
            if arr is not None and arr.ndim == 1:
                return float(arr[i])
            return 0.0
        if node.ret_type in (GType.INT, GType.BANK_IDX):
            return float(int(node.value)) if node.value is not None else 1.0
        if node.ret_type in (GType.SIDE, GType.REGIME):
            return node.value
        return 0.0
    name = node.name
    ch = node.children

    def _lagged_scalar(child: Node, lag: int) -> float:
        if lag <= 0:
            return _eval_node_scalar(
                child,
                i,
                terminal_data,
                within_day_pos,
                not_day_start,
                upp_value,
                upp_hist,
            )
        if within_day_pos[i] < lag:
            return 0.0
        if isinstance(child, TermNode) and child.name == UNREALIZED_PROFIT_PCT:
            if lag < len(upp_hist):
                return float(upp_hist[-1 - lag])
            return 0.0
        if isinstance(child, TermNode) and child.name in _PATH_STATE_TERMINALS:
            _n = len(upp_hist) - lag
            if _n <= 0:
                return 0.0
            _win = upp_hist[:_n]
            if child.name == MAX_FAVORABLE_EXCURSION:
                return float(max(_win))
            if child.name == MAX_ADVERSE_EXCURSION:
                return float(min(_win))
            return float(max(_win) - _win[-1])
        if isinstance(child, TermNode) and child.ret_type == GType.REAL:
            if child.value is not None:
                return float(child.value)
            arr = terminal_data.get(child.name)
            if arr is not None and arr.ndim == 1:
                return float(arr[i - lag])
            return 0.0
        if i - lag >= 0:
            return _eval_node_scalar(
                child,
                i - lag,
                terminal_data,
                within_day_pos,
                not_day_start,
                0.0,
                upp_hist,
            )
        return 0.0

    if name == "Lag":
        lag = int(
            np.clip(
                _eval_node_scalar(
                    ch[1],
                    i,
                    terminal_data,
                    within_day_pos,
                    not_day_start,
                    upp_value,
                    upp_hist,
                ),
                0,
                30,
            )
        )
        return _lagged_scalar(ch[0], lag)
    if name == "Delta":
        lag = int(
            np.clip(
                _eval_node_scalar(
                    ch[1],
                    i,
                    terminal_data,
                    within_day_pos,
                    not_day_start,
                    upp_value,
                    upp_hist,
                ),
                1,
                30,
            )
        )
        if within_day_pos[i] < lag:
            return 0.0
        cur = _eval_node_scalar(
            ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
        )
        return cur - _lagged_scalar(ch[0], lag)
    if name in ("CrossAbove", "CrossBelow"):
        a = _eval_node_scalar(
            ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
        )
        b = _eval_node_scalar(
            ch[1], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
        )
        prev_a = _lagged_scalar(ch[0], 1)
        prev_b = _lagged_scalar(ch[1], 1)
        if not_day_start[i] <= 0.0:
            return 0.0
        if name == "CrossAbove":
            return 1.0 if a > b and prev_a <= prev_b else 0.0
        return 1.0 if a < b and prev_a >= prev_b else 0.0
    if name == "GT":
        return (
            1.0
            if _eval_node_scalar(
                ch[0],
                i,
                terminal_data,
                within_day_pos,
                not_day_start,
                upp_value,
                upp_hist,
            )
            > _eval_node_scalar(
                ch[1],
                i,
                terminal_data,
                within_day_pos,
                not_day_start,
                upp_value,
                upp_hist,
            )
            else 0.0
        )
    if name == "LT":
        return (
            1.0
            if _eval_node_scalar(
                ch[0],
                i,
                terminal_data,
                within_day_pos,
                not_day_start,
                upp_value,
                upp_hist,
            )
            < _eval_node_scalar(
                ch[1],
                i,
                terminal_data,
                within_day_pos,
                not_day_start,
                upp_value,
                upp_hist,
            )
            else 0.0
        )
    if name == "AND":
        return (
            1.0
            if _eval_node_scalar(
                ch[0],
                i,
                terminal_data,
                within_day_pos,
                not_day_start,
                upp_value,
                upp_hist,
            )
            > 0.5
            and _eval_node_scalar(
                ch[1],
                i,
                terminal_data,
                within_day_pos,
                not_day_start,
                upp_value,
                upp_hist,
            )
            > 0.5
            else 0.0
        )
    if name == "OR":
        return (
            1.0
            if _eval_node_scalar(
                ch[0],
                i,
                terminal_data,
                within_day_pos,
                not_day_start,
                upp_value,
                upp_hist,
            )
            > 0.5
            or _eval_node_scalar(
                ch[1],
                i,
                terminal_data,
                within_day_pos,
                not_day_start,
                upp_value,
                upp_hist,
            )
            > 0.5
            else 0.0
        )
    if name == "NOT":
        return (
            1.0
            if _eval_node_scalar(
                ch[0],
                i,
                terminal_data,
                within_day_pos,
                not_day_start,
                upp_value,
                upp_hist,
            )
            <= 0.5
            else 0.0
        )
    if name == "Add":
        return _eval_node_scalar(
            ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
        ) + _eval_node_scalar(
            ch[1], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
        )
    if name == "Sub":
        return _eval_node_scalar(
            ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
        ) - _eval_node_scalar(
            ch[1], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
        )
    if name == "Mul":
        return _eval_node_scalar(
            ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
        ) * _eval_node_scalar(
            ch[1], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
        )
    if name == "Div":
        a = _eval_node_scalar(
            ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
        )
        b = _eval_node_scalar(
            ch[1], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
        )
        return a / math.sqrt(1.0 + b * b)
    if name == "Sqrt":
        return math.sqrt(
            abs(
                _eval_node_scalar(
                    ch[0],
                    i,
                    terminal_data,
                    within_day_pos,
                    not_day_start,
                    upp_value,
                    upp_hist,
                )
            )
        )
    if name == "Abs":
        return abs(
            _eval_node_scalar(
                ch[0],
                i,
                terminal_data,
                within_day_pos,
                not_day_start,
                upp_value,
                upp_hist,
            )
        )
    if name in ("WeightedSum3", "WeightedSum5"):

        def _e(_c):
            return _eval_node_scalar(
                _c, i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
            )

        _n = len(ch) // 2
        return sum((_e(ch[_n + _k]) * _e(ch[_k]) for _k in range(_n)))
    if name in ("IfThenElse", "IfSide"):
        cond = (
            _eval_node_scalar(
                ch[0],
                i,
                terminal_data,
                within_day_pos,
                not_day_start,
                upp_value,
                upp_hist,
            )
            > 0.5
        )
        branch = ch[1] if cond else ch[2]
        return _eval_node_scalar(
            branch, i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
        )
    if name == "InRegime":
        regime_arr = terminal_data.get("PredRegime")
        cur_regime = (
            float(regime_arr[i])
            if regime_arr is not None and regime_arr.ndim == 1
            else 0.0
        )
        target = _eval_node_scalar(
            ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
        )
        target_v = target.value if hasattr(target, "value") else target
        return 1.0 if cur_regime == target_v else 0.0
    if name == "RegimeIs":
        left = _eval_node_scalar(
            ch[0], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
        )
        right = _eval_node_scalar(
            ch[1], i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
        )
        return 1.0 if left == right else 0.0
    if _is_emb_op(node):
        _arr = terminal_data.get(_EMB_EXIT_KEY + to_str(node))
        if _arr is not None and getattr(_arr, "ndim", 0) == 1:
            return float(_arr[i])
    return 0.0


def _exit_signal_at_bar(
    exit_tree: Node,
    i: int,
    terminal_data: Dict[str, np.ndarray],
    within_day_pos: np.ndarray,
    not_day_start: np.ndarray,
    upp_value: float,
    upp_hist: List[float],
) -> bool:
    v = _eval_node_scalar(
        exit_tree, i, terminal_data, within_day_pos, not_day_start, upp_value, upp_hist
    )
    if isinstance(v, float) and math.isnan(v):
        return False
    return float(v) > 0.5


_TERMINAL_VALID_FROM = {
    "RealizedVol30m": 1,
    "RealizedVol30m_5m": 5,
    "ATM_IV_5m": 6,
    "DeltaSpread1": 3,
    "RawSpread_5m": 6,
    "ATM_IV": 2,
    "RawSpread": 2,
}


def prepare_terminal_data(
    data: pd.DataFrame,
    normalize_terminals: bool = True,
    norm_stats_override: "Optional[Dict[str, Tuple[float, float, str]]]" = None,
    lag_daily_vix: bool = True,
    vix_prior: "Optional[Dict[str, float]]" = None,
) -> Dict[str, np.ndarray]:
    terminal_data: Dict[str, np.ndarray] = {}
    from .data import TYPED_VECTOR_COLUMNS

    _vixspot_orig = (
        data["VIXSpot"].values.astype(np.float64).copy()
        if "VIXSpot" in data.columns
        else None
    )
    if lag_daily_vix and "date" in data.columns:
        _dates_lag = data["date"].values
        _to_lag = [c for c in ("VIXSpot", "VIXTermSlope") if c in data.columns]
        if _to_lag:
            data = data.copy()
            for _c in _to_lag:
                _prior = vix_prior.get(_c) if vix_prior else None
                data[_c] = _session_lag_daily(
                    data[_c].values.astype(np.float64), _dates_lag, prior_value=_prior
                )

    def _get_norm(name: str) -> tuple:
        if norm_stats_override and name in norm_stats_override:
            return norm_stats_override[name]
        return TERMINAL_NORM_STATS.get(name, (0.0, 1.0, "standard"))

    for col in data.columns:
        if col in ("date", "window_idx", "bar_position"):
            continue
        if col in TYPED_VECTOR_COLUMNS:
            sample = data[col].iloc[0]
            if isinstance(sample, (list, np.ndarray)) and len(sample) > 1:
                vec_arr = np.array(data[col].tolist(), dtype=np.float32)
                terminal_data[col] = vec_arr
            continue
        try:
            arr = data[col].values.astype(np.float64)
        except (ValueError, TypeError):
            continue
        if col == "ATM_IV":
            zero_mask = arr == 0.0
            if zero_mask.any() and "date" in data.columns:
                dates = data["date"].values
                for j in range(len(arr)):
                    if zero_mask[j] and j > 0 and (str(dates[j]) == str(dates[j - 1])):
                        arr[j] = arr[j - 1]
        if normalize_terminals and col in NORMALIZED_TERMINALS:
            (center, scale, _) = _get_norm(col)
            if scale > 1e-12:
                arr = (arr - center) / scale
                arr = np.clip(arr, -5.0, 5.0)
        terminal_data[col] = arr
    if "BarOfDay" not in terminal_data and "MinutesToClose" in data.columns:
        mtc = data["MinutesToClose"].values.astype(np.float64)
        if "date" in data.columns and len(data) > 2:
            dates = data["date"].values
            _cadence = 1.0
            for _j in range(1, min(len(data), 100)):
                if str(dates[_j]) == str(dates[_j - 1]):
                    _diff = abs(mtc[_j - 1] - mtc[_j])
                    if 0.1 < _diff < 100:
                        _cadence = _diff
                        break
            if _cadence not in (1.0, 5.0, 15.0, 30.0):
                import warnings

                warnings.warn(
                    f"Unusual bar cadence detected: {_cadence} min. Expected 1/5/15/30. BarOfDay may be miscalibrated.",
                    stacklevel=2,
                )
        else:
            _cadence = 5.0
        if len(mtc) == 0 or np.all(np.isnan(mtc)):
            _session_len = 390.0
        else:
            _session_len = float(np.nanmax(mtc))
        if _session_len < 100:
            _session_len = 390.0
        bar_of_day = (_session_len - mtc) / _cadence
        if normalize_terminals and "BarOfDay" in NORMALIZED_TERMINALS:
            (center, scale, _) = _get_norm("BarOfDay")
            if scale > 1e-12:
                bar_of_day = (bar_of_day - center) / scale
        terminal_data["BarOfDay"] = bar_of_day
    if "MinutesToClose" in data.columns:
        raw_mtc = data["MinutesToClose"].values.astype(np.float64)
        theta_urgency = 1.0 / np.sqrt(np.maximum(raw_mtc, 1.0))
        if normalize_terminals and "ThetaUrgency" in NORMALIZED_TERMINALS:
            (center, scale, _) = _get_norm("ThetaUrgency")
            if scale > 1e-12:
                theta_urgency = (theta_urgency - center) / scale
        terminal_data["ThetaUrgency"] = theta_urgency
    for base_col in ("ATM_IV", "RealizedVol30m", "RawSpread"):
        smooth_name = f"{base_col}_5m"
        if smooth_name not in terminal_data and base_col in data.columns:
            raw_vals = data[base_col].values.astype(np.float64).copy()
            if base_col == "ATM_IV" and "date" in data.columns:
                _dates_ff = data["date"].values
                for _j in range(len(raw_vals)):
                    if (
                        raw_vals[_j] < 1e-06
                        and _j > 0
                        and (str(_dates_ff[_j]) == str(_dates_ff[_j - 1]))
                    ):
                        raw_vals[_j] = raw_vals[_j - 1]
            smoothed = np.zeros_like(raw_vals)
            if "date" in data.columns:
                dates = data["date"].values
                day_start = 0
                for j in range(len(raw_vals)):
                    if j > 0 and str(dates[j]) != str(dates[j - 1]):
                        day_start = j
                    window_start = max(day_start, j - 4)
                    smoothed[j] = np.mean(raw_vals[window_start : j + 1])
            else:
                for j in range(len(raw_vals)):
                    window_start = max(0, j - 4)
                    smoothed[j] = np.mean(raw_vals[window_start : j + 1])
            if normalize_terminals and smooth_name in NORMALIZED_TERMINALS:
                (center, scale, _) = _get_norm(smooth_name)
                if scale > 1e-12:
                    smoothed = (smoothed - center) / scale
                    smoothed = np.clip(smoothed, -5.0, 5.0)
            terminal_data[smooth_name] = smoothed
    if (
        "PutCallSkew10" not in terminal_data
        and "IV_10Dp" in data.columns
        and ("IV_10Dc" in data.columns)
        and ("bar_position" in data.columns)
    ):
        _iv10p = data["IV_10Dp"].values.astype(np.float64)
        _iv10c = data["IV_10Dc"].values.astype(np.float64)
        _bar_pos = data["bar_position"].values.astype(np.float64)
        _skew10 = np.where(
            (_bar_pos < 270) & (_iv10p > 0.001) & (_iv10c > 0.001), _iv10p - _iv10c, 0.0
        )
        if normalize_terminals and "PutCallSkew10" in NORMALIZED_TERMINALS:
            (center, scale, _) = _get_norm("PutCallSkew10")
            if scale > 1e-12:
                _skew10 = (_skew10 - center) / scale
                _skew10 = np.clip(_skew10, -5.0, 5.0)
        terminal_data["PutCallSkew10"] = _skew10
    if "VIXChange" not in terminal_data and "VIXSpot" in data.columns:
        vix = data["VIXSpot"].values.astype(np.float64)
        vix_change = np.zeros_like(vix)
        if "date" in data.columns:
            dates = data["date"].values
            prev_day_vix = vix[0]
            for j in range(len(vix)):
                if j > 0 and str(dates[j]) != str(dates[j - 1]):
                    prev_day_vix = vix[j - 1]
                vix_change[j] = vix[j] - prev_day_vix
        if normalize_terminals and "VIXChange" in NORMALIZED_TERMINALS:
            (center, scale, _) = _get_norm("VIXChange")
            if scale > 1e-12:
                vix_change = (vix_change - center) / scale
                vix_change = np.clip(vix_change, -5.0, 5.0)
        terminal_data["VIXChange"] = vix_change
    if "date" in data.columns:
        dates = data["date"].values
        _unique_dates = []
        _day_starts = []
        _prev_d = None
        for j in range(len(dates)):
            _d = str(dates[j])
            if _d != _prev_d:
                _unique_dates.append(_d)
                _day_starts.append(j)
                _prev_d = _d
        _n_days = len(_unique_dates)
        price_col = next(
            (c for c in ("SPXClose", "spx_close", "close") if c in data.columns), None
        )
        _day_closes = None
        _daily_rets = None
        if price_col:
            prices = data[price_col].values.astype(np.float64)
            _mtc_all = (
                data["MinutesToClose"].values.astype(np.float64)
                if "MinutesToClose" in data.columns
                else None
            )

            def _day_close_idx(d):
                d_start = _day_starts[d]
                d_end = _day_starts[d + 1] if d < _n_days - 1 else len(prices)
                if _mtc_all is not None and d_end > d_start:
                    seg = _mtc_all[d_start:d_end]
                    j = int(np.argmin(np.abs(seg - 15.0)))
                    return d_start + j
                return d_end - 1

            _day_closes = np.array([prices[_day_close_idx(d)] for d in range(_n_days)])
            _daily_rets = np.diff(np.log(np.maximum(_day_closes, 1.0)))
        if _day_closes is not None and "SPXReturn3d" not in terminal_data:
            spx_ret3 = np.zeros(len(prices))
            for d in range(_n_days):
                d_start = _day_starts[d]
                d_end = _day_starts[d + 1] if d < _n_days - 1 else len(prices)
                if d >= 4 and _day_closes[d - 4] > 0:
                    val = (_day_closes[d - 1] - _day_closes[d - 4]) / _day_closes[d - 4]
                else:
                    val = 0.0
                spx_ret3[d_start:d_end] = val
            if normalize_terminals and "SPXReturn3d" in NORMALIZED_TERMINALS:
                (center, scale, _) = _get_norm("SPXReturn3d")
                if scale > 1e-12:
                    spx_ret3 = (spx_ret3 - center) / scale
                    spx_ret3 = np.clip(spx_ret3, -5.0, 5.0)
            terminal_data["SPXReturn3d"] = spx_ret3
        if "VIXSpot" in data.columns and "VIXMean5d" not in terminal_data:
            vix = (
                _vixspot_orig
                if _vixspot_orig is not None
                else data["VIXSpot"].values.astype(np.float64)
            )
            _day_vix = np.array([vix[_day_starts[d]] for d in range(_n_days)])
            vix_mean5 = np.zeros(len(vix))
            for d in range(_n_days):
                d_start = _day_starts[d]
                d_end = _day_starts[d + 1] if d < _n_days - 1 else len(vix)
                if d >= 5:
                    window = _day_vix[d - 5 : d]
                    vix_mean5[d_start:d_end] = float(np.mean(window))
                else:
                    vix_mean5[d_start:d_end] = _day_vix[0]
            if normalize_terminals and "VIXMean5d" in NORMALIZED_TERMINALS:
                (center, scale, _) = _get_norm("VIXMean5d")
                if scale > 1e-12:
                    vix_mean5 = (vix_mean5 - center) / scale
                    vix_mean5 = np.clip(vix_mean5, -5.0, 5.0)
            terminal_data["VIXMean5d"] = vix_mean5
        _rv5_raw = None
        if _daily_rets is not None and "RV5d" not in terminal_data:
            rv5 = np.zeros(len(prices))
            _rv5_raw = np.zeros(_n_days)
            for d in range(_n_days):
                d_start = _day_starts[d]
                d_end = _day_starts[d + 1] if d < _n_days - 1 else len(prices)
                if d >= 6:
                    window = _daily_rets[d - 6 : d - 1]
                    raw_val = float(np.std(window) * np.sqrt(252))
                else:
                    raw_val = 0.15
                _rv5_raw[d] = raw_val
                rv5[d_start:d_end] = raw_val
            if normalize_terminals and "RV5d" in NORMALIZED_TERMINALS:
                (center, scale, _) = _get_norm("RV5d")
                if scale > 1e-12:
                    rv5 = (rv5 - center) / scale
                    rv5 = np.clip(rv5, -5.0, 5.0)
            terminal_data["RV5d"] = rv5
        if (
            "ATM_IV" in data.columns
            and _rv5_raw is not None
            and ("IVRVGap5d" not in terminal_data)
        ):
            iv = data["ATM_IV"].values.astype(np.float64)
            _IV_SETTLE = 6
            _day_iv = np.array(
                [
                    iv[
                        min(
                            _day_starts[d] + _IV_SETTLE,
                            _day_starts[d + 1] - 1 if d < _n_days - 1 else len(iv) - 1,
                        )
                    ]
                    for d in range(_n_days)
                ]
            )
            ivrvgap = np.zeros(len(iv))
            for d in range(_n_days):
                d_start = _day_starts[d]
                d_end = _day_starts[d + 1] if d < _n_days - 1 else len(iv)
                if d >= 5:
                    iv_window = _day_iv[d - 5 : d]
                    ivrvgap[d_start:d_end] = float(np.mean(iv_window)) - _rv5_raw[d]
                else:
                    ivrvgap[d_start:d_end] = 0.0
            if normalize_terminals and "IVRVGap5d" in NORMALIZED_TERMINALS:
                (center, scale, _) = _get_norm("IVRVGap5d")
                if scale > 1e-12:
                    ivrvgap = (ivrvgap - center) / scale
                    ivrvgap = np.clip(ivrvgap, -5.0, 5.0)
            terminal_data["IVRVGap5d"] = ivrvgap
    if "SessionReturn" not in terminal_data:
        price_col = next(
            (c for c in ("SPXClose", "spx_close", "close") if c in data.columns), None
        )
        if price_col and "date" in data.columns:
            prices = data[price_col].values.astype(np.float64)
            dates = data["date"].values
            session_return = np.zeros(len(prices), dtype=np.float64)
            session_open = prices[0]
            prev_date = str(dates[0])
            for j in range(len(prices)):
                cur_date = str(dates[j])
                if cur_date != prev_date:
                    session_open = prices[j]
                    prev_date = cur_date
                if session_open > 0:
                    session_return[j] = (prices[j] - session_open) / session_open
            if normalize_terminals and "SessionReturn" in NORMALIZED_TERMINALS:
                (center, scale, _) = _get_norm("SessionReturn")
                if scale > 1e-12:
                    session_return = (session_return - center) / scale
                    session_return = np.clip(session_return, -5.0, 5.0)
            terminal_data["SessionReturn"] = session_return
    if "PredRegime" in terminal_data:
        pred_regime = terminal_data["PredRegime"]
        for ind_name, threshold in [
            ("RegimeAboveLow", 1.0),
            ("RegimeIsHigh", 2.0),
            ("RegimeIsPremium", 3.0),
        ]:
            if ind_name not in terminal_data:
                raw_ind = (pred_regime >= threshold).astype(np.float64)
                if normalize_terminals and ind_name in TERMINAL_NORM_STATS:
                    (center, scale, _) = _get_norm(ind_name)
                    if scale > 1e-12:
                        raw_ind = (raw_ind - center) / scale
                terminal_data[ind_name] = raw_ind
        terminal_data["PredRegime"] = pred_regime.astype(np.int32)
    if UNREALIZED_PROFIT_PCT not in terminal_data:
        terminal_data[UNREALIZED_PROFIT_PCT] = np.zeros(len(data), dtype=np.float64)
    for _pname in _PATH_STATE_TERMINALS:
        if _pname not in terminal_data:
            terminal_data[_pname] = np.zeros(len(data), dtype=np.float64)
    if "date" in data.columns and _TERMINAL_VALID_FROM:
        _dates_nz = data["date"].values
        _ds_nz = np.r_[0, np.where(_dates_nz[1:] != _dates_nz[:-1])[0] + 1]
        _de_nz = np.r_[_ds_nz[1:], len(data)]
        for _term, _valid_from in _TERMINAL_VALID_FROM.items():
            _arr = terminal_data.get(_term)
            if not (
                isinstance(_arr, np.ndarray)
                and _arr.ndim == 1
                and (_arr.shape[0] == len(data))
            ):
                continue
            if normalize_terminals and _term in NORMALIZED_TERMINALS:
                _neutral = 0.0
            else:
                (_c, _s, _) = _get_norm(_term)
                _neutral = float(_c)
            for _s0, _e0 in zip(_ds_nz, _de_nz):
                _arr[_s0 : min(_s0 + int(_valid_from), _e0)] = _neutral
    return terminal_data


def _template_to_shape(template):
    if getattr(template, "delta_fixed", False):
        return "iron_butterfly"
    n = len(template.legs)
    if n == 4:
        return "iron_condor"
    if n == 2:
        return (
            "put_vertical" if template.legs[0].option_type == "put" else "call_vertical"
        )
    return None


def _compute_dynamic_legs(template, delta_value: float, width_value=None, shape=None):
    if template.delta_range is None:
        return None
    (min_d, max_d) = template.delta_range
    if width_value is not None or shape is not None:
        from .structure import build_structure, SHAPES

        short_delta = min_d + (delta_value if delta_value is not None else 0.5) * (
            max_d - min_d
        )
        (wmin, wmax) = (0.05, 0.4)
        wing_offset = wmin + (
            float(width_value) if width_value is not None else 0.5
        ) * (wmax - wmin)
        sh = shape or _template_to_shape(template)
        if sh in SHAPES:
            return [tuple(leg) for leg in build_structure(sh, short_delta, wing_offset)]
    if template.delta_fixed:
        wing_delta = min_d + delta_value * (max_d - min_d)
        wing_delta = max(0.05, wing_delta)
        return [
            ("call", +0.5, -1, 1),
            ("call", +wing_delta, +1, 1),
            ("put", -0.5, -1, 1),
            ("put", -wing_delta, +1, 1),
        ]
    short_delta = min_d + delta_value * (max_d - min_d)
    wing_offset = template.wing_offset or 0.15
    long_delta = max(short_delta - wing_offset, 0.05)
    n_legs = len(template.legs)
    if n_legs == 4:
        return [
            ("call", +short_delta, -1, 1),
            ("call", +long_delta, +1, 1),
            ("put", -short_delta, -1, 1),
            ("put", -long_delta, +1, 1),
        ]
    elif n_legs == 2:
        leg0 = template.legs[0]
        leg1 = template.legs[1]
        sign = -1 if leg0.option_type == "put" else +1
        return [
            (leg0.option_type, sign * short_delta, leg0.qty_sign, leg0.ratio),
            (leg1.option_type, sign * long_delta, leg1.qty_sign, leg1.ratio),
        ]
    elif n_legs == 3:
        leg0_type = template.legs[0].option_type
        sign = -1 if leg0_type == "put" else +1
        lower_wing_delta = short_delta + wing_offset
        upper_wing_delta = max(short_delta - 0.15, 0.05)
        return [
            (leg0_type, sign * lower_wing_delta, +1, 1),
            (leg0_type, sign * short_delta, -1, 2),
            (leg0_type, sign * upper_wing_delta, +1, 1),
        ]
    return None


_CALENDAR_TAU_ON = os.environ.get("GP_CALENDAR_TAU", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_BASE_CREDIT_FACTORS_LEGACY = {
    "bull_put_credit": 0.83,
    "bear_call_credit": 0.83,
    "iron_condor": 0.85,
    "iron_butterfly": 0.76,
}
_BASE_CREDIT_FACTORS_TAU365 = {
    "bull_put_credit": 0.844,
    "bear_call_credit": 0.882,
    "iron_condor": 0.871,
    "iron_butterfly": 1.104,
}
_BASE_CREDIT_FACTORS = (
    _BASE_CREDIT_FACTORS_TAU365 if _CALENDAR_TAU_ON else _BASE_CREDIT_FACTORS_LEGACY
)
_GROSS_REALISM_FACTORS_LEGACY = {
    "bull_put_credit": 0.71,
    "bear_call_credit": 0.72,
    "iron_condor": 0.7,
    "iron_butterfly": 0.65,
    "long_strangle": 1.0,
}
_GROSS_REALISM_FACTORS_TAU365 = {
    "bull_put_credit": 0.76,
    "bear_call_credit": 0.76,
    "iron_condor": 0.76,
    "iron_butterfly": 0.78,
    "long_strangle": 0.845,
}
_GROSS_REALISM_FACTORS = (
    _GROSS_REALISM_FACTORS_TAU365 if _CALENDAR_TAU_ON else _GROSS_REALISM_FACTORS_LEGACY
)
_SHAPE_FACTOR_KEY = {
    "put_vertical": "bull_put_credit",
    "call_vertical": "bear_call_credit",
    "iron_condor": "iron_condor",
    "iron_butterfly": "iron_butterfly",
    "long_strangle": "long_strangle",
}
_CREDIT_FACTOR_EXEMPT_SHAPES = frozenset({"long_strangle"})


def _validate_factor_coverage():
    which = "TAU365" if _CALENDAR_TAU_ON else "LEGACY"
    for shp, key in _SHAPE_FACTOR_KEY.items():
        if shp not in _CREDIT_FACTOR_EXEMPT_SHAPES and key not in _BASE_CREDIT_FACTORS:
            raise AssertionError(
                f"shape {shp!r} -> credit key {key!r} UNMAPPED in _BASE_CREDIT_FACTORS_{which}"
            )
        if key not in _GROSS_REALISM_FACTORS:
            raise AssertionError(
                f"shape {shp!r} -> gross key {key!r} UNMAPPED in _GROSS_REALISM_FACTORS_{which}"
            )


_validate_factor_coverage()


def _resolve_factor_key(shape, template_name: str) -> str:
    base = (
        template_name.replace("_standard", "")
        .replace("_narrow", "")
        .replace("_wide", "")
    )
    return _SHAPE_FACTOR_KEY.get(shape, base) if shape is not None else base


def _regime_credit_mult(shape, iv: float) -> float:
    if shape in _DEBIT_STOP_SHAPES:
        return 1.0
    if iv < 0.2:
        return 1.0
    if iv < 0.3:
        return 0.85
    return 0.7


def _debit_slippage_for(n_legs: int) -> float:
    return 1.05 if n_legs >= 3 else 1.1


def _sigmoid_gene_map(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.where(np.isnan(arr), 0.0, arr)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-arr))


def prepare_static_arrays(
    data: pd.DataFrame, default_iv: float = 0.2
) -> Dict[str, np.ndarray]:
    n_bars = len(data)
    cols = data.columns
    if "date" in cols:
        date_arr = data["date"].values
        codes = pd.factorize(date_arr, sort=False)[0]
        boundary = np.empty(n_bars, dtype=bool)
        boundary[0] = True
        boundary[1:] = codes[1:] != codes[:-1]
        day_mask = np.where(boundary, 0.0, 1.0)
        _idx = np.arange(n_bars)
        within_day_pos = (
            _idx - np.maximum.accumulate(np.where(boundary, _idx, 0))
        ).astype(np.int64)
    else:
        date_arr = None
        day_mask = np.ones(n_bars, dtype=np.float64)
        within_day_pos = np.arange(n_bars, dtype=np.int64)
    price_col = next(
        (c for c in ("SPXClose", "spx_close", "close") if c in cols), "SPXClose"
    )
    iv_col = next((c for c in ("ATM_IV", "atm_iv", "ATMIV") if c in cols), None)
    spot_arr = (
        data[price_col].values.astype(np.float64)
        if price_col in cols
        else np.zeros(n_bars)
    )
    iv_arr = (
        data[iv_col].values.astype(np.float64)
        if iv_col and iv_col in cols
        else np.full(n_bars, default_iv)
    )
    mtc_arr = (
        data["MinutesToClose"].values.astype(np.float64)
        if "MinutesToClose" in cols
        else np.full(n_bars, 60.0)
    )
    spread_arr = (
        data["RawSpread"].values.astype(np.float64)
        if "RawSpread" in cols
        else np.full(n_bars, 0.015)
    )
    _last_valid_iv = default_iv
    for _i in range(n_bars):
        if iv_arr[_i] > 0.001:
            _last_valid_iv = iv_arr[_i]
        else:
            iv_arr[_i] = _last_valid_iv
    iv_arr = _repair_scalar_iv_vs_vix(
        iv_arr,
        (
            data["VIXSpot"].values.astype(np.float64) / 100.0
            if "VIXSpot" in cols
            else None
        ),
        _MARK_IV_VIX_MULT,
    )
    iv_arr = np.clip(iv_arr, 0.0, 1.0)
    if all((c in cols for c in GRID_IV_COLUMNS)):
        grid_iv_matrix = np.column_stack(
            [data[c].values.astype(np.float64) for c in GRID_IV_COLUMNS]
        )
        if _WING_IV_FIX_ENABLED:
            grid_iv_matrix = correct_put_wing_skew(grid_iv_matrix)
    else:
        grid_iv_matrix = None
    return {
        "spot_arr": spot_arr,
        "iv_arr": iv_arr,
        "mtc_arr": mtc_arr,
        "spread_arr": spread_arr,
        "date_arr": date_arr,
        "grid_iv_matrix": grid_iv_matrix,
        "day_mask": day_mask,
        "within_day_pos": within_day_pos,
    }


import weakref as _weakref

_STATIC_ARRAYS_CACHE: Dict[int, tuple] = {}
_STATIC_ARRAYS_CACHE_ENABLED = True


def _get_static_arrays(data, default_iv):
    key = id(data)
    ent = _STATIC_ARRAYS_CACHE.get(key)
    if ent is not None and ent[0]() is data:
        return ent[1]
    bundle = prepare_static_arrays(data, default_iv)
    try:
        _STATIC_ARRAYS_CACHE[key] = (_weakref.ref(data), bundle)
    except TypeError:
        pass
    return bundle


def vectorized_backtest(
    entry_tree: Node,
    exit_tree: Node,
    size_tree: Node,
    data: pd.DataFrame,
    template,
    delta_tree: Optional[Node] = None,
    width_tree: Optional[Node] = None,
    shape: Optional[str] = None,
    shape_tree: Optional[Node] = None,
    cost_multiplier: float = 1.0,
    terminal_data: Optional[Dict[str, np.ndarray]] = None,
    static_arrays: Optional[Dict[str, np.ndarray]] = None,
    notional: float = 1000.0,
    fee_per_leg: float = 2.5,
    warmup_bars: int = 15,
    min_bars_in_trade: int = 15,
    max_bars_in_trade: int = 330,
    default_iv: float = 0.2,
    stop_loss_credit_multiple: float = 2.5,
    fold_recenter_stats: Optional[Dict] = None,
    fold_id: Optional[str] = None,
    pca_bases: Optional[Dict] = None,
    emb_proj_cache: Optional[Dict] = None,
    _override_entry_signals: Optional[np.ndarray] = None,
    _override_exit_signals: Optional[np.ndarray] = None,
    entry_decision_mask: Optional[np.ndarray] = None,
    entry_floor_within_day: Optional[int] = None,
    _sizing_log: Optional[list] = None,
    _mark_log: Optional[list] = None,
) -> BacktestResult:
    from .records import MultiLegOptionsBacktester, SimpleBacktester
    from .grammar import Side as GSide

    n_bars = len(data)
    if terminal_data is None:
        terminal_data = prepare_terminal_data(data)
    if static_arrays is None and _STATIC_ARRAYS_CACHE_ENABLED:
        static_arrays = _get_static_arrays(data, default_iv)
    if static_arrays is not None:
        day_mask = static_arrays["day_mask"]
        within_day_pos = static_arrays["within_day_pos"]
    else:
        day_mask = np.ones(n_bars, dtype=np.float64)
        within_day_pos = np.zeros(n_bars, dtype=np.int64)
        if "date" in data.columns:
            dates = data["date"].values
            day_mask[0] = 0.0
            pos = 0
            for i in range(1, n_bars):
                if str(dates[i]) != str(dates[i - 1]):
                    day_mask[i] = 0.0
                    pos = 0
                else:
                    pos += 1
                within_day_pos[i] = pos
        else:
            within_day_pos = np.arange(n_bars, dtype=np.int64)
    evaluator = VectorizedTreeEvaluator(
        terminal_data,
        n_bars,
        day_boundary_mask=day_mask,
        within_day_pos=within_day_pos,
        fold_recenter_stats=fold_recenter_stats,
        fold_id=fold_id,
        pca_bases=pca_bases,
        emb_proj_cache=emb_proj_cache,
    )
    if _override_entry_signals is not None:
        entry_signals = _override_entry_signals.astype(np.float64)
    elif entry_tree is not None:
        entry_signals = evaluator.evaluate(entry_tree).astype(np.float64)
    else:
        entry_signals = np.zeros(n_bars, dtype=np.float64)
    if _override_exit_signals is not None:
        exit_signals = _override_exit_signals.astype(np.float64)
    elif exit_tree is not None:
        exit_signals = evaluator.evaluate(exit_tree).astype(np.float64)
    else:
        exit_signals = np.zeros(n_bars, dtype=np.float64)
    _exit_uses_upp = (
        _override_exit_signals is None
        and exit_tree is not None
        and tree_references_upp(exit_tree)
    )
    _exit_terminal_data = _build_exit_terminal_data(
        terminal_data, exit_tree, evaluator, _exit_uses_upp
    )
    size_signals = evaluator.evaluate(size_tree).astype(np.float64)
    entry_signals = np.where(np.isnan(entry_signals), 0.0, entry_signals)
    exit_signals = np.where(np.isnan(exit_signals), 0.0, exit_signals)
    size_signals = np.where(np.isnan(size_signals), 0.0, size_signals)
    if entry_decision_mask is not None:
        entry_signals = entry_signals * np.asarray(
            entry_decision_mask, dtype=np.float64
        )
    if entry_floor_within_day is not None:
        if "date" not in getattr(data, "columns", ()):
            raise ValueError(
                "entry_floor_within_day requires a per-day 'date' column: without it `within_day_pos` is the GLOBAL index (np.arange fallback) and the floor would gate globally — the very day-0-only bug this per-day floor fixes. Production fold frames always carry 'date'."
            )
        entry_signals = entry_signals * (
            within_day_pos >= int(entry_floor_within_day)
        ).astype(np.float64)
    size_signals = np.clip(size_signals, 0.0, 1.0)
    width_signals = (
        _sigmoid_gene_map(evaluator.evaluate(width_tree))
        if width_tree is not None
        else None
    )
    delta_signals = (
        _sigmoid_gene_map(evaluator.evaluate(delta_tree))
        if delta_tree is not None
        else None
    )
    shape_signals = evaluator.evaluate(shape_tree) if shape_tree is not None else None
    price_col = next(
        (c for c in ("SPXClose", "spx_close", "close") if c in data.columns), "SPXClose"
    )
    iv_col = next((c for c in ("ATM_IV", "atm_iv", "ATMIV") if c in data.columns), None)
    if static_arrays is not None:
        spot_arr = static_arrays["spot_arr"]
        iv_arr = static_arrays["iv_arr"]
        mtc_arr = static_arrays["mtc_arr"]
        spread_arr = static_arrays["spread_arr"]
        date_arr = static_arrays["date_arr"]
        grid_iv_matrix = static_arrays["grid_iv_matrix"]
    else:
        spot_arr = (
            data[price_col].values.astype(np.float64)
            if price_col in data.columns
            else np.zeros(n_bars)
        )
        iv_arr = (
            data[iv_col].values.astype(np.float64)
            if iv_col and iv_col in data.columns
            else np.full(n_bars, default_iv)
        )
        mtc_arr = (
            data["MinutesToClose"].values.astype(np.float64)
            if "MinutesToClose" in data.columns
            else np.full(n_bars, 60.0)
        )
        spread_arr = (
            data["RawSpread"].values.astype(np.float64)
            if "RawSpread" in data.columns
            else np.full(n_bars, 0.015)
        )
        _last_valid_iv = default_iv
        for _i in range(n_bars):
            if iv_arr[_i] > 0.001:
                _last_valid_iv = iv_arr[_i]
            else:
                iv_arr[_i] = _last_valid_iv
        iv_arr = _repair_scalar_iv_vs_vix(
            iv_arr,
            (
                data["VIXSpot"].values.astype(np.float64) / 100.0
                if "VIXSpot" in data.columns
                else None
            ),
            _MARK_IV_VIX_MULT,
        )
        iv_arr = np.clip(iv_arr, 0.0, 1.0)
        date_arr = data["date"].values if "date" in data.columns else None
        _has_grid_ivs = all((c in data.columns for c in GRID_IV_COLUMNS))
        if _has_grid_ivs:
            grid_iv_matrix = np.column_stack(
                [data[c].values.astype(np.float64) for c in GRID_IV_COLUMNS]
            )
            if _WING_IV_FIX_ENABLED:
                grid_iv_matrix = correct_put_wing_skew(grid_iv_matrix)
        else:
            grid_iv_matrix = None
    _mark_vix_arr = (
        data["VIXSpot"].values.astype(np.float64) / 100.0
        if _MARK_IV_VIX_MULT > 0 and "VIXSpot" in data.columns
        else None
    )
    bar_returns = np.zeros(n_bars)
    trades: List[Trade] = []
    equity = 0.0
    _ruined = False
    in_position = False
    entry_bar = 0
    entry_spot = 0.0
    prev_position_value = 0.0
    prev_spot = 0.0
    strikes: List[float] = []
    entry_net_value = 0.0
    upp_hist: List[float] = []
    raw_entry_val = 0.0
    entry_n_contracts = 0
    entry_cost = 0.0
    _entry_available = notional
    entry_gross = 0.0
    has_legs = hasattr(template, "legs") and template.legs
    active_legs = template.legs
    _entry_shape = None
    MAX_TRADES_PER_DAY = 8
    trades_today = 0
    pending_entry = False
    pending_size = 0.0
    _prev_date = None
    for i in range(n_bars):
        spot = spot_arr[i]
        iv = max(iv_arr[i], 0.01)
        mtc = mtc_arr[i]
        _bar_grid = grid_iv_matrix[i] if grid_iv_matrix is not None else None
        if _mark_vix_arr is not None:
            (iv, _bar_grid) = _apply_mark_iv_cap(
                iv, _bar_grid, _mark_vix_arr[i], _MARK_IV_VIX_MULT
            )
        (_bar_rho, _bar_beta) = (0.0, 0.0)
        _surface_computed = False
        if date_arr is not None:
            _cur_date = str(date_arr[i])
            if _prev_date is not None and _cur_date != _prev_date:
                trades_today = 0
                pending_entry = False
                if in_position:
                    prev_s = spot_arr[i - 1]
                    if np.isnan(prev_s) or prev_s <= 0:
                        prev_s = spot_arr[max(0, i - 2)] if i >= 2 else spot_arr[0]
                        if np.isnan(prev_s) or prev_s <= 0:
                            prev_s = 5000.0
                    prev_iv = max(iv_arr[i - 1], 0.01)
                    if has_legs:
                        _prev_grid = (
                            grid_iv_matrix[i - 1]
                            if grid_iv_matrix is not None
                            else None
                        )
                        if _prev_grid is not None:
                            (_prev_rho, _prev_beta) = _estimate_surface_params(
                                _prev_grid, prev_iv
                            )
                        else:
                            (_prev_rho, _prev_beta) = (0.0, 0.0)
                        close_val = _net_pos_value(
                            active_legs,
                            strikes,
                            prev_s,
                            0.0,
                            prev_iv,
                            grid_ivs=_prev_grid,
                            rho_t=_prev_rho,
                            beta_t=_prev_beta,
                        )
                        exit_cost = (
                            _settlement_exit_cost(
                                active_legs, strikes, prev_s, prev_iv, fee_per_leg
                            )
                            * cost_multiplier
                        )
                        pnl = (
                            close_val - entry_net_value - entry_cost - exit_cost
                        ) * entry_n_contracts
                        final_m2m = (
                            close_val - prev_position_value
                        ) * entry_n_contracts
                        bar_returns[i - 1] += final_m2m / notional
                        bar_returns[i - 1] -= exit_cost * entry_n_contracts / notional
                    else:
                        pnl = (prev_s - entry_spot) * entry_n_contracts
                        bar_returns[i - 1] += (
                            (prev_s - prev_spot) * entry_n_contracts / notional
                        )
                    if pnl < -_entry_available:
                        bar_returns[i - 1] += (-_entry_available - pnl) / notional
                        pnl = -_entry_available
                    equity += pnl / notional
                    if equity <= -1.0:
                        _ruined = True
                    trades.append(
                        Trade(
                            entry_bar,
                            i - 1,
                            GSide.NEUTRAL,
                            entry_spot,
                            prev_s,
                            pnl,
                            i - 1 - entry_bar,
                            exit_reason="session",
                            shape=_entry_shape,
                            entry_n_contracts=float(entry_n_contracts),
                            sizing_diag=None,
                        )
                    )
                    in_position = False
            _prev_date = _cur_date
        if i < warmup_bars:
            continue
        if spot <= 0 or np.isnan(spot):
            continue
        if in_position:
            bars_held = i - entry_bar
            _mark_is_debit = _entry_shape in _DEBIT_STOP_SHAPES
            if has_legs:
                if not _surface_computed and _bar_grid is not None:
                    (_bar_rho, _bar_beta) = _estimate_surface_params(_bar_grid, iv)
                    _surface_computed = True
                curr_val = _debit_uw_mark_decay(
                    _net_pos_value(
                        active_legs,
                        strikes,
                        spot,
                        mtc,
                        iv,
                        grid_ivs=_bar_grid,
                        rho_t=_bar_rho,
                        beta_t=_bar_beta,
                    ),
                    raw_entry_val,
                    _mark_is_debit,
                )
                bar_pnl = (curr_val - prev_position_value) * entry_n_contracts
                prev_position_value = curr_val
                if _mark_log is not None:
                    _mark_log.append(
                        (
                            str(date_arr[entry_bar]),
                            int(entry_bar),
                            int(bars_held),
                            float(
                                (curr_val - raw_entry_val - entry_cost)
                                * entry_n_contracts
                                * 100.0
                            ),
                            _entry_shape,
                            float(entry_n_contracts),
                        )
                    )
            else:
                bar_pnl = (spot - prev_spot) * entry_n_contracts
                prev_spot = spot
            bar_returns[i] += bar_pnl / notional
            if _exit_uses_upp:
                if has_legs and abs(entry_net_value) > _UPP_DENOM_FLOOR:
                    upp_now = (curr_val - entry_net_value) / abs(entry_net_value)
                else:
                    upp_now = 0.0
                upp_hist.append(upp_now if upp_now == upp_now else 0.0)
            _exit_by_stop = False
            _exit_by_time_decay = False
            if (
                has_legs
                and stop_loss_credit_multiple
                and (stop_loss_credit_multiple > 0)
            ):
                unrealised = curr_val - raw_entry_val - entry_cost
                if _entry_shape is None:
                    _use_credit_stop = template.is_credit and entry_net_value < -0.01
                    _use_debit_stop = not template.is_credit and entry_net_value > 0.01
                else:
                    _shape_is_debit = _entry_shape in _DEBIT_STOP_SHAPES
                    _use_credit_stop = not _shape_is_debit and entry_net_value < -0.01
                    _use_debit_stop = _shape_is_debit or entry_net_value > 0.01
                if _use_credit_stop:
                    _base_mult = (
                        stop_loss_credit_multiple * STOP_LOSS_EXECUTION_DISCOUNT
                    )
                    _time_factor = 0.75 + 0.5 * min(mtc / 400.0, 1.0)
                    _time_adjusted_mult = _base_mult * _time_factor
                    credit_received = abs(entry_net_value)
                    if unrealised < -_time_adjusted_mult * credit_received:
                        _exit_by_stop = True
                elif _use_debit_stop:
                    _debit_risk_basis = max(entry_gross, entry_net_value)
                    if unrealised < -0.8 * _debit_risk_basis:
                        _exit_by_stop = True
                    _has_ratio = any((l.ratio > 1 for l in active_legs))
                    _debit_underwater_bars = (
                        240 if len(active_legs) >= 3 or _has_ratio else 40
                    )
                    if bars_held >= _debit_underwater_bars and unrealised < 0:
                        _exit_by_time_decay = True
            if _exit_uses_upp:
                _exit_fired = _exit_signal_at_bar(
                    exit_tree,
                    i,
                    _exit_terminal_data,
                    within_day_pos,
                    day_mask,
                    upp_hist[-1] if upp_hist else 0.0,
                    upp_hist,
                )
            else:
                _exit_fired = exit_signals[i] > 0.5
            _exit_by_signal = bars_held >= min_bars_in_trade and _exit_fired
            _exit_by_max_hold = bars_held >= max_bars_in_trade
            _exit_by_eod = mtc <= EOD_FORCE_CLOSE_MTC
            should_exit = (
                _exit_by_stop
                or _exit_by_time_decay
                or _exit_by_signal
                or _exit_by_max_hold
                or _exit_by_eod
            )
            if should_exit:
                if _exit_by_stop:
                    _reason = "stop_loss"
                elif _exit_by_time_decay:
                    _reason = "time_decay"
                elif _exit_by_signal:
                    _reason = "signal"
                elif _exit_by_max_hold:
                    _reason = "max_hold"
                else:
                    _reason = "eod"
                if has_legs:
                    exit_cost = (
                        _entry_costs(
                            active_legs,
                            strikes,
                            spot,
                            mtc,
                            iv,
                            fee_per_leg,
                            atm_spread=spread_arr[i],
                            grid_ivs=_bar_grid,
                            rho_t=_bar_rho,
                            beta_t=_bar_beta,
                        )
                        * cost_multiplier
                    )
                    if _reason == "stop_loss":
                        hours_rem = max(mtc / 60.0, 0.0)
                        _sl_slippage = 1.2 + 0.3 * (1.0 - min(hours_rem / 6.5, 1.0))
                        if iv >= 0.3:
                            _sl_slippage *= 1.5
                        elif iv >= 0.2:
                            _sl_slippage *= 1.25
                        exit_cost *= _sl_slippage
                else:
                    exit_cost = 0.0
                if has_legs:
                    close_val = _debit_uw_mark_decay(
                        _net_pos_value(
                            active_legs,
                            strikes,
                            spot,
                            mtc,
                            iv,
                            grid_ivs=_bar_grid,
                            rho_t=_bar_rho,
                            beta_t=_bar_beta,
                        ),
                        raw_entry_val,
                        _mark_is_debit,
                    )
                    pnl = (
                        close_val - entry_net_value - entry_cost - exit_cost
                    ) * entry_n_contracts
                else:
                    pnl = (spot - entry_spot) * entry_n_contracts
                if _reason == "stop_loss" and pnl < 0.0:
                    pnl *= STOP_LOSS_FILL_MARKUP
                    _markup_delta = pnl - pnl / STOP_LOSS_FILL_MARKUP
                    bar_returns[i] += _markup_delta / notional
                bar_returns[i] -= exit_cost * entry_n_contracts / notional
                if pnl < -_entry_available:
                    bar_returns[i] += (-_entry_available - pnl) / notional
                    pnl = -_entry_available
                equity += pnl / notional
                if equity <= -1.0:
                    _ruined = True
                trades.append(
                    Trade(
                        entry_bar,
                        i,
                        GSide.NEUTRAL,
                        entry_spot,
                        spot,
                        pnl,
                        bars_held,
                        exit_reason=_reason,
                        shape=_entry_shape,
                        entry_n_contracts=float(entry_n_contracts),
                        sizing_diag=None,
                    )
                )
                in_position = False
        else:
            if mtc <= ENTRY_CUTOFF_MTC:
                pending_entry = False
                continue
            if _ruined:
                pending_entry = False
                continue
            if trades_today >= MAX_TRADES_PER_DAY:
                pending_entry = False
                continue
            if pending_entry:
                size = pending_size
                pending_entry = False
            elif entry_signals[i] > 0.5:
                pending_entry = True
                pending_size = max(0.0, min(1.0, size_signals[i]))
                continue
            else:
                continue
            if size > 1e-06 and has_legs:
                if template.delta_range is not None and (
                    delta_signals is not None
                    or width_signals is not None
                    or shape is not None
                    or (shape_signals is not None)
                ):
                    _dv = delta_signals[i] if delta_signals is not None else None
                    _wv = width_signals[i] if width_signals is not None else None
                    _sh = str(shape_signals[i]) if shape_signals is not None else shape
                    _dyn = _compute_dynamic_legs(
                        template, _dv, width_value=_wv, shape=_sh
                    )
                    if _dyn is not None:
                        from .templates import Leg as _Leg

                        active_legs = tuple(
                            (
                                _Leg(ot, d, qty_sign=qs, ratio=r)
                                for (ot, d, qs, r) in _dyn
                            )
                        )
                        _entry_shape = _sh
                    else:
                        active_legs = template.legs
                        _entry_shape = _sh
                else:
                    active_legs = template.legs
                    _entry_shape = None
                _eff_factor_key = _resolve_factor_key(_entry_shape, template.name)
                _eff_credit_factor = _BASE_CREDIT_FACTORS.get(_eff_factor_key, 0.8)
                _eff_gross_realism = _GROSS_REALISM_FACTORS.get(_eff_factor_key, 1.0)
                strikes = [
                    MultiLegOptionsBacktester._delta_to_strike(
                        spot, leg.delta_target, iv, mtc
                    )
                    for leg in active_legs
                ]
                if _entry_shape == "iron_butterfly":
                    strikes = MultiLegOptionsBacktester._butterfly_strikes(
                        strikes, active_legs, spot
                    )
                else:
                    strikes = MultiLegOptionsBacktester._enforce_strike_separation(
                        strikes, active_legs
                    )
                if _covered_gap_width(active_legs, strikes) > MAX_SPREAD_WIDTH:
                    pending_entry = False
                    continue
                if not _surface_computed and _bar_grid is not None:
                    (_bar_rho, _bar_beta) = _estimate_surface_params(_bar_grid, iv)
                    _surface_computed = True
                raw_entry_val = _net_pos_value(
                    active_legs,
                    strikes,
                    spot,
                    mtc,
                    iv,
                    grid_ivs=_bar_grid,
                    rho_t=_bar_rho,
                    beta_t=_bar_beta,
                )
                if raw_entry_val < -0.01:
                    call_credit = 0.0
                    put_credit = 0.0
                    for leg, strike_k in zip(active_legs, strikes):
                        if leg.qty_sign >= 0:
                            continue
                        if _bar_grid is not None:
                            leg_iv = _empirical_iv(_bar_grid, leg.delta_target, iv)
                            lv = _edgeworth_option_value(
                                spot,
                                strike_k,
                                mtc,
                                leg_iv,
                                leg.option_type == "call",
                                _bar_rho,
                                _bar_beta,
                            )
                        else:
                            leg_iv = _skew_iv(iv, spot, strike_k, mtc)
                            lv = _option_value(
                                spot, strike_k, mtc, leg_iv, leg.option_type == "call"
                            )
                        short_val = abs(leg.qty_sign * leg.ratio * lv)
                        if leg.option_type == "call":
                            call_credit += short_val
                        else:
                            put_credit += short_val
                    _regime_mult = _regime_credit_mult(_entry_shape, iv)
                    _credit_factor = _clamp_credit_fill_factor(
                        _eff_credit_factor, _regime_mult
                    )
                    haircut_credit = (call_credit + put_credit) * _credit_factor
                    if call_credit + put_credit > 0.01:
                        haircut_ratio = haircut_credit / (call_credit + put_credit)
                    else:
                        haircut_ratio = 0.85
                    entry_net_value = raw_entry_val * haircut_ratio
                elif raw_entry_val > 0.01:
                    _debit_slippage = _debit_slippage_for(len(active_legs))
                    entry_net_value = raw_entry_val * _debit_slippage
                else:
                    entry_net_value = raw_entry_val
                entry_gross = _gross_pos_value(
                    active_legs,
                    strikes,
                    spot,
                    mtc,
                    iv,
                    grid_ivs=_bar_grid,
                    rho_t=_bar_rho,
                    beta_t=_bar_beta,
                )
                if template.is_credit:
                    abs_val = max(entry_gross * _eff_gross_realism, 2.0)
                else:
                    abs_val = max(
                        entry_gross * _eff_gross_realism, abs(entry_net_value), 2.0
                    )
                _total_ratio = sum((l.ratio for l in active_legs))
                _has_ratio_leg = any((l.ratio > 1 for l in active_legs))
                if _has_ratio_leg:
                    abs_val *= _total_ratio / 2.0
                gross_contracts = int(notional * size / abs_val)
                entry_n_contracts = min(gross_contracts, int(notional / 2.0))
                if _has_ratio_leg:
                    _max_risk = notional * 0.05
                    if len(strikes) >= 2:
                        _max_width = max(
                            (
                                abs(strikes[j] - strikes[j + 1])
                                for j in range(len(strikes) - 1)
                            )
                        )
                        _max_by_risk = max(1, int(_max_risk / max(_max_width, 1.0)))
                        entry_n_contracts = min(entry_n_contracts, _max_by_risk)
                if entry_n_contracts < 1:
                    entry_n_contracts = 1
                if entry_n_contracts * abs_val > 2 * notional:
                    entry_n_contracts = max(1, int(2 * notional / abs_val))
                if len(strikes) >= 2:
                    if len(strikes) == 4:
                        max_spread_width = max(
                            abs(strikes[0] - strikes[1]), abs(strikes[2] - strikes[3])
                        )
                    elif len(strikes) == 3 and (not template.is_credit):
                        max_spread_width = max(
                            (
                                abs(strikes[j] - strikes[j + 1])
                                for j in range(len(strikes) - 1)
                            )
                        )
                    else:
                        max_spread_width = max(
                            (
                                abs(strikes[j] - strikes[j + 1])
                                for j in range(len(strikes) - 1)
                            )
                        )
                    _sizing_net = _sizing_credit_basis(raw_entry_val, entry_net_value)
                    margin_per_contract = _margin_per_contract(
                        _sizing_net,
                        max_spread_width,
                        _structural_loss_floor(active_legs, strikes),
                    )
                    _cost_pc = (
                        _entry_costs(
                            active_legs,
                            strikes,
                            spot,
                            mtc,
                            iv,
                            fee_per_leg,
                            atm_spread=spread_arr[i],
                            grid_ivs=_bar_grid,
                            rho_t=_bar_rho,
                            beta_t=_bar_beta,
                        )
                        * cost_multiplier
                    )
                    _worst_loss_pc = max(
                        margin_per_contract + _cost_pc * (1.0 + _MAX_STOP_SLIPPAGE),
                        1e-06,
                    )
                    available_capital = notional * max(1.0 + equity, 0.0)
                    if _worst_loss_pc * entry_n_contracts > available_capital:
                        safe_contracts = int(available_capital / _worst_loss_pc)
                        if safe_contracts < 1:
                            continue
                        entry_n_contracts = safe_contracts
                    _max_conc_contracts = max(
                        1,
                        int(
                            CONCENTRATION_CAP_FRAC * available_capital / _worst_loss_pc
                        ),
                    )
                    entry_n_contracts = min(entry_n_contracts, _max_conc_contracts)
                _entry_is_credit = raw_entry_val < -0.01
                entry_cost = (
                    _entry_costs(
                        active_legs,
                        strikes,
                        spot,
                        mtc,
                        iv,
                        fee_per_leg,
                        atm_spread=spread_arr[i],
                        grid_ivs=_bar_grid,
                        rho_t=_bar_rho,
                        beta_t=_bar_beta,
                        skip_spread_crossing=_entry_is_credit,
                    )
                    * cost_multiplier
                )
                in_position = True
                entry_bar = i
                entry_spot = spot
                _entry_available = notional * max(1.0 + equity, 0.0)
                upp_hist.clear()
                if _sizing_log is not None:
                    _sizing_log.append(
                        {
                            "entry_bar": int(i),
                            "n_contracts": int(entry_n_contracts),
                            "gross": float(entry_gross),
                            "net": float(entry_net_value),
                            "raw_net": float(raw_entry_val),
                            "entry_cost": float(entry_cost),
                            "entry_available": float(_entry_available),
                            "equity_before": float(equity),
                            "strikes": [float(_s) for _s in strikes],
                            "legs": [
                                (
                                    str(_lg.option_type),
                                    float(_lg.delta_target),
                                    int(_lg.qty_sign),
                                    float(_lg.ratio),
                                )
                                for _lg in active_legs
                            ],
                            "leg_vals": [
                                float(
                                    _net_pos_value(
                                        [_lg],
                                        [_sk],
                                        spot,
                                        mtc,
                                        iv,
                                        grid_ivs=_bar_grid,
                                        rho_t=_bar_rho,
                                        beta_t=_bar_beta,
                                    )
                                )
                                for (_lg, _sk) in zip(active_legs, strikes)
                            ],
                        }
                    )
                prev_position_value = raw_entry_val
                trades_today += 1
                bar_returns[i] -= entry_cost * entry_n_contracts / notional
                bar_returns[i] += (
                    (raw_entry_val - entry_net_value) * entry_n_contracts / notional
                )
            elif size > 1e-06:
                entry_n_contracts = 1
                entry_cost = 0.0
                in_position = True
                entry_bar = i
                entry_spot = spot
                prev_spot = spot
                upp_hist.clear()
                trades_today += 1
    if in_position:
        last_spot = spot_arr[-1]
        last_iv = max(iv_arr[-1], 0.01)
        if has_legs:
            _last_grid = grid_iv_matrix[-1] if grid_iv_matrix is not None else None
            if _last_grid is not None:
                (_last_rho, _last_beta) = _estimate_surface_params(_last_grid, last_iv)
            else:
                (_last_rho, _last_beta) = (0.0, 0.0)
            close_val = _net_pos_value(
                active_legs,
                strikes,
                last_spot,
                0.0,
                last_iv,
                grid_ivs=_last_grid,
                rho_t=_last_rho,
                beta_t=_last_beta,
            )
            exit_cost = (
                _settlement_exit_cost(
                    active_legs, strikes, last_spot, last_iv, fee_per_leg
                )
                * cost_multiplier
            )
            pnl = (
                close_val - entry_net_value - entry_cost - exit_cost
            ) * entry_n_contracts
            final_m2m = (close_val - prev_position_value) * entry_n_contracts
            bar_returns[-1] += final_m2m / notional
            bar_returns[-1] -= exit_cost * entry_n_contracts / notional
        else:
            pnl = (last_spot - entry_spot) * entry_n_contracts
        if pnl < -_entry_available:
            bar_returns[-1] += (-_entry_available - pnl) / notional
            pnl = -_entry_available
        equity += pnl / notional
        trades.append(
            Trade(
                entry_bar,
                n_bars - 1,
                GSide.NEUTRAL,
                entry_spot,
                last_spot,
                pnl,
                n_bars - 1 - entry_bar,
                exit_reason="end_of_data",
                shape=_entry_shape,
                entry_n_contracts=float(entry_n_contracts),
                sizing_diag=None,
            )
        )
    bpd = SimpleBacktester._derive_bars_per_day(data)
    _n_days = (
        data["date"].nunique() if "date" in data.columns else max(1, len(data) // bpd)
    )
    daily_returns = []
    daily_arr = np.array([], dtype=np.float64)
    if date_arr is not None and len(bar_returns) > 0:
        unique_dates = []
        daily_returns = []
        day_start = 0
        for i in range(1, n_bars):
            if str(date_arr[i]) != str(date_arr[i - 1]):
                daily_returns.append(float(np.sum(bar_returns[day_start:i])))
                unique_dates.append(str(date_arr[day_start]))
                day_start = i
        daily_returns.append(float(np.sum(bar_returns[day_start:])))
        daily_arr = np.array(daily_returns, dtype=np.float64)
        annualization = math.sqrt(252.0)
        returns_std = np.std(daily_arr) if len(daily_arr) > 1 else 1e-09
        sharpe = float(np.mean(daily_arr) / max(returns_std, 1e-09) * annualization)
        downside = daily_arr[daily_arr < 0]
        _mean_daily = float(np.mean(daily_arr))
        _overall_std = float(np.std(daily_arr)) if len(daily_arr) > 1 else 0.0
        if len(downside) >= _MIN_DOWNSIDE_OBS:
            dd_dev = float(np.sqrt(np.mean(downside**2)))
            sortino = float(
                np.clip(
                    _mean_daily / max(dd_dev, 1e-09) * annualization,
                    -_MAX_SORTINO,
                    _MAX_SORTINO,
                )
            )
        else:
            if len(downside) > 0:
                _emp_dd = float(np.sqrt(np.mean(downside**2)))
            else:
                _emp_dd = 0.0
            _dd_floor = _DOWNSIDE_DEV_STD_FRACTION * _overall_std
            dd_dev = max(_emp_dd, _dd_floor, 1e-09)
            sortino = float(
                np.clip(
                    _mean_daily / dd_dev * annualization, -_MAX_SORTINO, _MAX_SORTINO
                )
            )
    else:
        annualization = math.sqrt(252 * bpd) if bpd > 0 else 1.0
        returns_std = np.std(bar_returns) if len(bar_returns) > 1 else 1e-09
        sharpe = float(np.mean(bar_returns) / max(returns_std, 1e-09) * annualization)
        sortino = SimpleBacktester._sortino(bar_returns, bars_per_day=bpd)
    from scipy.stats import skew as _skew_fn, kurtosis as _kurtosis_fn

    _return_skew = 0.0
    _return_kurtosis = 0.0
    if len(daily_arr) > 3:
        _return_skew = float(_skew_fn(daily_arr))
        _return_kurtosis = float(_kurtosis_fn(daily_arr))
    cum_returns = np.cumsum(bar_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = running_max - cum_returns
    max_dd_uncapped = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    max_dd = min(max_dd_uncapped, 1.0) if notional > 0 else 0.0
    if notional <= 0:
        max_dd_uncapped = 0.0
    pnls = [t.pnl for t in trades]
    wins = sum((1 for p in pnls if p > 0))
    win_rate = wins / len(trades) if trades else 0.0
    _gross_profit = sum((p for p in pnls if p > 0))
    _gross_loss = -sum((p for p in pnls if p < 0))
    profit_factor = (
        _gross_profit / _gross_loss
        if _gross_loss > 1e-09
        else 10.0 if _gross_profit > 0 else 0.0
    )
    signal_exits = sum((1 for t in trades if t.exit_reason == "signal"))
    exit_util = signal_exits / len(trades) if trades else 0.0
    if pnls and len(pnls) >= 10:
        losses = [p for p in pnls if p < 0]
        if losses:
            avg_loss = sum(losses) / len(losses)
            worst_loss = min(losses)
            if avg_loss < -1e-06 and worst_loss < 5.0 * avg_loss:
                sharpe *= 0.5
                sortino *= 0.5
    _raw_sharpe = sharpe
    if trades and len(trades) >= 5:
        min_bar_exits = sum(
            (
                1
                for t in trades
                if t.exit_bar - t.entry_bar <= min_bars_in_trade + 1
                and t.exit_reason == "signal"
            )
        )
        min_bar_frac = min_bar_exits / len(trades)
        if min_bar_frac > 0.8 and _raw_sharpe <= 0.0:
            sharpe = -1000000.0
            sortino = -1000000.0
        early_stop_exits = sum(
            (
                1
                for t in trades
                if t.exit_bar - t.entry_bar <= min_bars_in_trade + 1
                and t.exit_reason == "stop_loss"
            )
        )
        early_stop_frac = early_stop_exits / len(trades)
        if early_stop_frac > 0.8 and _raw_sharpe <= 0.0:
            sharpe = -1000000.0
            sortino = -1000000.0
        max_hold_exits = sum(
            (1 for t in trades if t.exit_bar - t.entry_bar >= max_bars_in_trade - 1)
        )
        max_hold_frac = max_hold_exits / len(trades)
        if max_hold_frac > 0.9 and _raw_sharpe <= 0.0:
            sharpe -= _MAX_HOLD_VESTIGIAL_PENALTY
            sortino -= _MAX_HOLD_VESTIGIAL_PENALTY
        _signal_exit_trades = [t for t in trades if t.exit_reason == "signal"]
        if len(_signal_exit_trades) >= 5:
            _exit_bars_in_day = []
            for t in _signal_exit_trades:
                _exit_bars_in_day.append(
                    evaluator._within_day_pos[min(t.exit_bar, n_bars - 1)]
                )
            _exit_std = float(np.std(_exit_bars_in_day))
            if _exit_std < 3.0 and _raw_sharpe <= 0.0:
                sharpe = -1000000.0
                sortino = -1000000.0
    _eligible_mask = evaluator._within_day_pos >= warmup_bars
    _n_eligible = int(_eligible_mask.sum())
    _n_entry_fires = (
        int((entry_signals[_eligible_mask] > 0.5).sum()) if _n_eligible > 0 else 0
    )
    _fire_rate = _n_entry_fires / max(_n_eligible, 1)
    _in_pos_mask = np.zeros(n_bars, dtype=bool)
    for _t in trades:
        _in_pos_mask[_t.entry_bar + 1 : min(_t.exit_bar + 1, n_bars)] = True
    _flat_elig_mask = _eligible_mask & ~_in_pos_mask
    _n_flat_elig = int(_flat_elig_mask.sum())
    _n_flat_fire = (
        int((entry_signals[_flat_elig_mask] > 0.5).sum()) if _n_flat_elig > 0 else 0
    )
    _fire_rate_flat = _n_flat_fire / max(_n_flat_elig, 1)
    _cond_sharpe_gap = 0.0
    if date_arr is not None and len(daily_returns) > 5:
        _day_had_entry = []
        _ds = 0
        _wdp = evaluator._within_day_pos
        for _di in range(1, n_bars):
            if str(date_arr[_di]) != str(date_arr[_di - 1]):
                _eligible = _wdp[_ds:_di] >= warmup_bars
                _day_had_entry.append(
                    bool(np.any(entry_signals[_ds:_di][_eligible] > 0.5))
                    if _eligible.any()
                    else False
                )
                _ds = _di
        _eligible = _wdp[_ds:] >= warmup_bars
        _day_had_entry.append(
            bool(np.any(entry_signals[_ds:][_eligible] > 0.5))
            if _eligible.any()
            else False
        )
        _day_flags = np.array(_day_had_entry, dtype=bool)
        if 3 < _day_flags.sum() < len(_day_flags) - 3:
            _entry_day_returns = daily_arr[_day_flags]
            _ann = math.sqrt(252.0)
            _entry_std = float(np.std(_entry_day_returns))
            _entry_sharpe = (
                float(np.mean(_entry_day_returns) / max(_entry_std, 1e-09) * _ann)
                if _entry_std > 1e-09
                else 0.0
            )
            _entry_mean = float(np.mean(_entry_day_returns))
            _all_mean = float(np.mean(daily_arr))
            if _entry_mean < _all_mean:
                _entry_sharpe = min(_entry_sharpe, _raw_sharpe)
            _cond_sharpe_gap = _entry_sharpe - _raw_sharpe
    return BacktestResult(
        returns=bar_returns,
        trades=trades,
        equity_curve=cum_returns,
        max_drawdown=max_dd,
        sharpe=sharpe,
        sortino=sortino,
        total_trades=len(trades),
        win_rate=win_rate,
        exit_utilization=exit_util,
        n_days=_n_days,
        entry_fire_rate=_fire_rate,
        entry_fire_rate_flat=_fire_rate_flat,
        conditional_sharpe_gap=_cond_sharpe_gap,
        avg_position_size=(
            float(np.mean([size_signals[max(t.entry_bar - 1, 0)] for t in trades]))
            if trades
            else 0.5
        ),
        return_skew=_return_skew,
        return_kurtosis=_return_kurtosis,
        max_drawdown_uncapped=max_dd_uncapped,
        profit_factor=profit_factor,
    )


_SQRT2 = math.sqrt(2.0)
_TAU_YEAR_MIN = (365.0 if _CALENDAR_TAU_ON else 252.0) * 390.0
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
GRID_IV_COLUMNS: tuple = (
    "IV_5Dp",
    "IV_10Dp",
    "IV_25Dp",
    "IV_40Dp",
    "IV_ATMp",
    "IV_ATM",
    "IV_ATMc",
    "IV_40Dc",
    "IV_25Dc",
    "IV_10Dc",
    "IV_5Dc",
)
_GRID_DELTAS = np.array(
    [-0.5, -0.4, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.4, 0.5], dtype=np.float64
)


def _prepare_empirical_grid(grid_ivs: np.ndarray, atm_iv: float):
    if grid_ivs is None or len(grid_ivs) != 11:
        return None
    _any_nonzero = False
    for _v in grid_ivs:
        if _v > 0.001:
            _any_nonzero = True
            break
    if not _any_nonzero:
        return None
    atm_val = grid_ivs[5]
    atmc_val = grid_ivs[6]
    if atm_val > 0.001 and atmc_val > 0.001:
        atm_avg = 0.5 * (atm_val + atmc_val)
    elif atm_val > 0.001:
        atm_avg = atm_val
    elif atmc_val > 0.001:
        atm_avg = atmc_val
    else:
        atm_avg = atm_iv
    ivs_10 = [
        grid_ivs[4],
        grid_ivs[3],
        grid_ivs[2],
        grid_ivs[1],
        grid_ivs[0],
        grid_ivs[10],
        grid_ivs[9],
        grid_ivs[8],
        grid_ivs[7],
        atm_avg,
    ]
    for _k in range(10):
        if ivs_10[_k] < 0.001:
            ivs_10[_k] = atm_iv
    _stale_floor = 0.3 * atm_iv
    for _k in range(10):
        if ivs_10[_k] < _stale_floor:
            ivs_10[_k] = atm_iv
    _sorted_iv = sorted((_v for _v in ivs_10 if _v > 0.001))
    if _sorted_iv:
        _central_iv = _sorted_iv[len(_sorted_iv) // 2]
        _min_iv = _sorted_iv[0]
        _iv_ceiling = min(1.0, max(3.0 * _central_iv, 0.3), 2.5 * _min_iv)
        for _k in range(10):
            if ivs_10[_k] > _iv_ceiling:
                ivs_10[_k] = _iv_ceiling
    return ivs_10


def _interp_iv_prepared(prepared, delta_target: float, atm_iv: float) -> float:
    if prepared is None:
        return max(atm_iv, 0.01)
    delta_clamped = max(-0.5, min(0.5, delta_target))
    result = float(np.interp(delta_clamped, _GRID_DELTAS, prepared))
    return max(result, 0.01)


def _empirical_iv(grid_ivs: np.ndarray, delta_target: float, atm_iv: float) -> float:
    return _interp_iv_prepared(
        _prepare_empirical_grid(grid_ivs, atm_iv), delta_target, atm_iv
    )


def _phi(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _estimate_surface_params(grid_ivs: np.ndarray, atm_iv: float) -> tuple:
    if grid_ivs is None or len(grid_ivs) != 11 or atm_iv < 0.001:
        return (0.0, 0.0)
    iv_atmc = grid_ivs[6]
    iv_atmp = grid_ivs[4]
    if iv_atmc > 0.001 and iv_atmp > 0.001:
        rho_t = (iv_atmc - iv_atmp) / (2.0 * atm_iv)
    else:
        rho_t = 0.0
    iv_25dc = grid_ivs[8]
    iv_25dp = grid_ivs[2]
    iv_atm = grid_ivs[5]
    if iv_25dc > 0.001 and iv_25dp > 0.001 and (iv_atm > 0.001):
        beta_t = (iv_25dc + iv_25dp - 2.0 * iv_atm) / (0.0625 * atm_iv)
    else:
        beta_t = 0.0
    return (rho_t, beta_t)


def _edgeworth_option_value(
    spot: float,
    strike: float,
    mtc: float,
    iv: float,
    is_call: bool,
    rho_t: float = 0.0,
    beta_t: float = 0.0,
) -> float:
    if abs(rho_t) < 1e-10 and abs(beta_t) < 1e-10 or mtc <= 0:
        return _option_value(spot, strike, mtc, iv, is_call)
    bs_call = _option_value(spot, strike, mtc, iv, True)
    bs_price = bs_call
    if spot <= 0 or strike <= 0 or iv < 0.001 or (bs_price < 1e-10):
        return _option_value(spot, strike, mtc, iv, is_call)
    tau = mtc / _TAU_YEAR_MIN
    sqrt_tau = math.sqrt(tau)
    sigma_sqrt_tau = iv * sqrt_tau
    if sigma_sqrt_tau < 1e-12:
        return bs_price
    d = (strike - spot) / (spot * sigma_sqrt_tau)
    phi_d = _phi(d)
    rho_capped = max(-0.2, min(0.2, rho_t))
    beta_capped = max(-10.0, min(10.0, beta_t))
    skew_correction = -(d * d - 1.0) * phi_d / 6.0
    kurt_correction = (d * d * d - 3.0 * d) * phi_d / 24.0
    correction = 1.0 + rho_capped * skew_correction + beta_capped * kurt_correction
    correction = max(0.75, min(1.3, correction))
    _call_intrinsic = max(spot - strike, 0.0)
    _corrected_tv = max(bs_price - _call_intrinsic, 0.0) * correction
    if is_call:
        return _call_intrinsic + _corrected_tv
    return max(strike - spot, 0.0) + _corrected_tv


def _option_value(
    spot: float, strike: float, mtc: float, iv: float, is_call: bool
) -> float:
    if spot <= 0 or strike <= 0:
        return 0.0
    if mtc <= 0:
        return max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
    tau = mtc / _TAU_YEAR_MIN
    sigma_sqrt_tau = iv * math.sqrt(tau)
    if sigma_sqrt_tau < 1e-12:
        return max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
    d1 = (
        math.log(spot / strike) + 0.5 * sigma_sqrt_tau * sigma_sqrt_tau
    ) / sigma_sqrt_tau
    d2 = d1 - sigma_sqrt_tau
    if is_call:
        return spot * 0.5 * math.erfc(-d1 / _SQRT2) - strike * 0.5 * math.erfc(
            -d2 / _SQRT2
        )
    else:
        return strike * 0.5 * math.erfc(d2 / _SQRT2) - spot * 0.5 * math.erfc(
            d1 / _SQRT2
        )


def _skew_iv(
    atm_iv: float,
    spot: float,
    strike: float,
    mtc: float = 60.0,
    skew_slope: float = -0.15,
) -> float:
    if spot <= 0 or strike <= 0 or atm_iv < 0.001:
        return max(atm_iv, 0.01)
    tau = max(mtc, 1.0) / _TAU_YEAR_MIN
    sigma_sqrt_tau = atm_iv * math.sqrt(tau)
    if sigma_sqrt_tau < 1e-09:
        return atm_iv
    if mtc > 240:
        effective_slope = skew_slope * 0.7
    elif mtc > 120:
        effective_slope = skew_slope
    else:
        effective_slope = skew_slope * 1.3
    tau_ref = 60.0 / _TAU_YEAR_MIN
    tau_damping = min(1.0, math.sqrt(tau / tau_ref))
    d_std = (strike - spot) / (spot * sigma_sqrt_tau)
    d_std *= tau_damping
    iv = atm_iv * (1.0 + effective_slope * d_std)
    if strike > spot:
        iv = max(iv, atm_iv * 0.92)
    return min(max(iv, atm_iv * 0.5, 0.01), atm_iv * 3.0)


def _structural_loss_floor(legs, strikes) -> float:
    if not legs:
        return 0.0
    paired = list(zip(legs, strikes))
    call_net = sum(
        (l.qty_sign * l.ratio for (l, _) in paired if l.option_type == "call")
    )
    if call_net < 0:
        return float("-inf")
    ks = [float(k) for (_, k) in paired]
    breakpoints = [0.0] + ks + [2.0 * max(ks)] if ks else [0.0]
    worst = 0.0
    for s in breakpoints:
        v = 0.0
        for leg, k in paired:
            intrinsic = (
                max(s - k, 0.0) if leg.option_type == "call" else max(k - s, 0.0)
            )
            v += leg.qty_sign * leg.ratio * intrinsic
        if v < worst:
            worst = v
    return worst


def _structural_gain_ceiling(legs, strikes) -> float:
    if not legs:
        return 0.0
    paired = list(zip(legs, strikes))
    call_net = sum(
        (l.qty_sign * l.ratio for (l, _) in paired if l.option_type == "call")
    )
    if call_net > 0:
        return float("inf")
    ks = [float(k) for (_, k) in paired]
    breakpoints = [0.0] + ks + [2.0 * max(ks)] if ks else [0.0]
    best = 0.0
    for s in breakpoints:
        v = 0.0
        for leg, k in paired:
            intrinsic = (
                max(s - k, 0.0) if leg.option_type == "call" else max(k - s, 0.0)
            )
            v += leg.qty_sign * leg.ratio * intrinsic
        if v > best:
            best = v
    return best


def _net_pos_value(
    legs, strikes, spot, mtc, iv, grid_ivs=None, rho_t: float = 0.0, beta_t: float = 0.0
) -> float:
    total = 0.0
    _leg_vals = []
    _prep_grid = _prepare_empirical_grid(grid_ivs, iv) if grid_ivs is not None else None
    for leg, strike in zip(legs, strikes):
        if grid_ivs is not None:
            leg_iv = _interp_iv_prepared(_prep_grid, leg.delta_target, iv)
        else:
            leg_iv = _skew_iv(iv, spot, strike, mtc)
        if grid_ivs is not None:
            val = _edgeworth_option_value(
                spot, strike, mtc, leg_iv, leg.option_type == "call", rho_t, beta_t
            )
        else:
            val = _option_value(spot, strike, mtc, leg_iv, leg.option_type == "call")
        _leg_cap = spot if leg.option_type == "call" else strike
        val = min(max(val, 0.0), _leg_cap)
        _leg_vals.append((leg, strike, val))
        total += leg.qty_sign * leg.ratio * val
    _used = [False] * len(_leg_vals)
    for _a in range(len(_leg_vals)):
        if _used[_a]:
            continue
        (leg_a, strike_a, val_a) = _leg_vals[_a]
        for _b in range(_a + 1, len(_leg_vals)):
            if _used[_b]:
                continue
            (leg_b, strike_b, val_b) = _leg_vals[_b]
            if (
                leg_a.option_type == leg_b.option_type
                and leg_a.qty_sign == -leg_b.qty_sign
                and (leg_a.ratio == leg_b.ratio)
                and (abs(strike_a - strike_b) > 1e-09)
            ):
                r = leg_a.ratio
                width = abs(strike_a - strike_b)
                pair_contrib = leg_a.qty_sign * r * val_a + leg_b.qty_sign * r * val_b
                if leg_a.qty_sign == 1:
                    (long_strike, short_strike) = (strike_a, strike_b)
                else:
                    (long_strike, short_strike) = (strike_b, strike_a)
                if leg_a.option_type == "call":
                    is_debit_vertical = long_strike < short_strike
                else:
                    is_debit_vertical = long_strike > short_strike
                bound = width * r
                if is_debit_vertical:
                    (lo, hi) = (0.0, bound)
                else:
                    (lo, hi) = (-bound, 0.0)
                clamped = min(max(pair_contrib, lo), hi)
                total += clamped - pair_contrib
                _used[_a] = True
                _used[_b] = True
                break
    _floor = _structural_loss_floor(legs, strikes)
    if total < _floor:
        total = _floor
    _ceiling = _structural_gain_ceiling(legs, strikes)
    if total > _ceiling:
        total = _ceiling
    return total


def _sizing_credit_basis(raw_entry_val: float, entry_net_value: float) -> float:
    if raw_entry_val < -0.01:
        return min(raw_entry_val * _SIZING_CREDIT_FILL, -0.011)
    return entry_net_value


def _debit_uw_mark_decay(
    pos_value: float, raw_entry_val: float, is_debit: bool
) -> float:
    if is_debit:
        excess = pos_value - raw_entry_val
        if excess < 0.0:
            return raw_entry_val + excess / _DEBIT_UW_MARK_DECAY
    return pos_value


def _margin_per_contract(
    entry_net_value: float, max_spread_width: float, structural_floor: float
) -> float:
    if entry_net_value < -0.01:
        loss_basis = max(max_spread_width, -structural_floor)
        return max(0.0, loss_basis - abs(entry_net_value))
    return max(0.0, -structural_floor + entry_net_value)


def _gross_pos_value(
    legs, strikes, spot, mtc, iv, grid_ivs=None, rho_t: float = 0.0, beta_t: float = 0.0
) -> float:
    total = 0.0
    _prep_grid = _prepare_empirical_grid(grid_ivs, iv) if grid_ivs is not None else None
    for leg, strike in zip(legs, strikes):
        if grid_ivs is not None:
            leg_iv = _interp_iv_prepared(_prep_grid, leg.delta_target, iv)
        else:
            leg_iv = _skew_iv(iv, spot, strike, mtc)
        if grid_ivs is not None:
            val = _edgeworth_option_value(
                spot, strike, mtc, leg_iv, leg.option_type == "call", rho_t, beta_t
            )
        else:
            val = _option_value(spot, strike, mtc, leg_iv, leg.option_type == "call")
        total += abs(leg.qty_sign * leg.ratio * val)
    return total


_SPREAD_MULT_TABLE = [
    (0.5, 1.0),
    (0.4, 1.2),
    (0.35, 1.4),
    (0.25, 1.8),
    (0.2, 2.0),
    (0.15, 2.5),
    (0.1, 3.0),
    (0.05, 4.0),
]


def _spread_multiplier(abs_delta: float) -> float:
    if abs_delta >= 0.5:
        return 1.0
    if abs_delta <= 0.05:
        return 4.0
    for i in range(len(_SPREAD_MULT_TABLE) - 1):
        (d_hi, m_hi) = _SPREAD_MULT_TABLE[i]
        (d_lo, m_lo) = _SPREAD_MULT_TABLE[i + 1]
        if d_lo <= abs_delta <= d_hi:
            frac = (abs_delta - d_lo) / (d_hi - d_lo) if d_hi > d_lo else 0.0
            return m_lo + frac * (m_hi - m_lo)
    return 1.0


def _entry_costs(
    legs,
    strikes,
    spot,
    mtc,
    iv,
    fee_per_leg,
    atm_spread: float = 0.015,
    grid_ivs=None,
    rho_t: float = 0.0,
    beta_t: float = 0.0,
    skip_spread_crossing: bool = False,
) -> float:
    cost = 0.0
    if atm_spread <= 0.001:
        atm_spread = 0.015
    if mtc > 360:
        tod_mult = 1.5
    elif mtc > 300:
        tod_mult = 1.2
    elif mtc <= 15:
        tod_mult = 2.5
    elif mtc <= 30:
        tod_mult = 2.0
    elif mtc <= 60:
        tod_mult = 1.3
    else:
        tod_mult = 1.0
    _prep_grid = _prepare_empirical_grid(grid_ivs, iv) if grid_ivs is not None else None
    for leg, strike in zip(legs, strikes):
        cost += fee_per_leg / 100.0 * leg.ratio
        abs_delta = abs(leg.delta_target)
        rel_spread = atm_spread * _spread_multiplier(abs_delta) * tod_mult
        if grid_ivs is not None:
            leg_iv = _interp_iv_prepared(_prep_grid, leg.delta_target, iv)
            leg_value = _edgeworth_option_value(
                spot, strike, mtc, leg_iv, leg.option_type == "call", rho_t, beta_t
            )
        else:
            leg_iv = _skew_iv(iv, spot, strike, mtc)
            leg_value = _option_value(
                spot, strike, mtc, leg_iv, leg.option_type == "call"
            )
        if leg.qty_sign < 0:
            _spread_frac = 0.75 if leg.option_type == "call" else 0.65
        else:
            _spread_frac = 0.5
        _intrinsic = (
            max(spot - strike, 0.0)
            if leg.option_type == "call"
            else max(strike - spot, 0.0)
        )
        _extrinsic = max(leg_value - _intrinsic, 0.0)
        if not skip_spread_crossing:
            cost += (
                max(
                    _spread_frac * rel_spread * _extrinsic,
                    _MIN_SPREAD_COST_PER_LEG * tod_mult,
                )
                * leg.ratio
            )
        cost += 0.02 * leg.ratio
    return cost


def _settlement_exit_cost(legs, strikes, spot, iv, fee_per_leg: float = 2.5) -> float:
    cost = 0.0
    for leg, strike in zip(legs, strikes):
        is_call = leg.option_type == "call"
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        if intrinsic > 0.01:
            cost += fee_per_leg / 100.0 * leg.ratio
        cost += 0.02 * leg.ratio
    return cost
