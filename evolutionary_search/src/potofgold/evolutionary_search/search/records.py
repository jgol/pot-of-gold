"""Trade records and numerical helpers shared by the execution simulator."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from .grammar import Side


def _norm_ppf(p: float) -> float:
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    if p < 0.5:
        t = math.sqrt(-2.0 * math.log(p))
        (c0, c1, c2) = (2.515517, 0.802853, 0.010328)
        (d1, d2, d3) = (1.432788, 0.189269, 0.001308)
        return -(
            t
            - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
        )
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        (c0, c1, c2) = (2.515517, 0.802853, 0.010328)
        (d1, d2, d3) = (1.432788, 0.189269, 0.001308)
        return t - (c0 + c1 * t + c2 * t * t) / (
            1.0 + d1 * t + d2 * t * t + d3 * t * t * t
        )


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    side: Side
    entry_price: float
    exit_price: float
    pnl: float
    bars_held: int
    exit_reason: str = "unknown"
    shape: Optional[str] = None
    entry_n_contracts: float = 0.0
    sizing_diag: Optional[dict] = None


@dataclass
class BacktestResult:
    returns: np.ndarray
    trades: List[Trade]
    equity_curve: np.ndarray
    max_drawdown: float
    sharpe: float
    sortino: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    n_days: int = 1
    exit_utilization: float = 0.0
    entry_fire_rate: float = 0.0
    entry_fire_rate_flat: float = 0.0
    conditional_sharpe_gap: float = 0.0
    avg_position_size: float = 0.5
    return_skew: float = 0.0
    return_kurtosis: float = 0.0
    max_drawdown_uncapped: float = 0.0
    profit_factor: float = 0.0

    @property
    def psr(self) -> float:
        from scipy.stats import norm as _norm_dist, skew as _skew, kurtosis as _kurt

        if self.n_days < 5 or len(self.returns) == 0:
            return 0.0
        sr = self.sharpe
        n = self.n_days
        daily = self.returns
        if hasattr(self, "_daily_returns") and self._daily_returns is not None:
            daily = self._daily_returns
        elif len(self.returns) > n * 2:
            chunk = len(self.returns) // max(n, 1)
            if chunk > 1:
                trimmed = self.returns[: chunk * n]
                daily = trimmed.reshape(n, chunk).sum(axis=1)
        s = float(_skew(daily)) if len(daily) > 2 else 0.0
        k = float(_kurt(daily)) if len(daily) > 3 else 0.0
        se_sr = math.sqrt((1.0 - s * sr + (k + 2.0) / 4.0 * sr**2) / max(n, 1))
        if se_sr < 1e-10:
            return 1.0 if sr > 0 else 0.0
        return float(_norm_dist.cdf(sr / se_sr))


class SimpleBacktester:

    @staticmethod
    def _sortino(returns: np.ndarray, bars_per_day: int) -> float:
        if len(returns) == 0:
            return 0.0
        mean_ret = float(np.mean(returns))
        downside = returns[returns < 0]
        if len(downside) == 0:
            return 10.0 if mean_ret > 1e-10 else 0.0
        downside_dev = float(np.sqrt(np.mean(downside**2)))
        if downside_dev < 1e-10:
            return 0.0
        return float(mean_ret / downside_dev * math.sqrt(252 * bars_per_day))

    @staticmethod
    def _derive_bars_per_day(data: pd.DataFrame) -> int:
        if "date" not in data.columns:
            return 12
        per_date = data.groupby("date").size()
        if len(per_date) == 0:
            return 12
        bpd = int(per_date.median())
        if not 10 <= bpd <= 405:
            raise RuntimeError(
                f"Derived bars_per_day={bpd} is outside the supported [10, 405] range. "
                f"Session-length counts: {per_date.value_counts().head(3).to_dict()}. "
                "Check the prepared Parquet's date and bar columns before computing Sharpe."
            )
        return bpd


class MultiLegOptionsBacktester:

    @staticmethod
    def _delta_to_strike(
        spot: float,
        delta_target: float,
        iv: float,
        minutes_to_expiry: float,
        skew_slope_override: float = -0.15,
    ) -> float:
        from .simulator import _skew_iv, _TAU_YEAR_MIN

        if iv <= 0 or minutes_to_expiry <= 0:
            raw = spot * (1.0 - delta_target * 0.01)
        else:
            tau = minutes_to_expiry / _TAU_YEAR_MIN
            is_call = delta_target > 0
            sigma_sqrt_tau = iv * math.sqrt(max(tau, 1e-12))
            if is_call:
                d_clamped = max(min(delta_target, 0.999), 0.001)
                d1 = _norm_ppf(d_clamped)
            else:
                d_clamped = max(min(1.0 + delta_target, 0.999), 0.001)
                d1 = _norm_ppf(d_clamped)
            raw = spot * math.exp(-d1 * sigma_sqrt_tau + 0.5 * sigma_sqrt_tau**2)
            _SQRT2 = math.sqrt(2.0)
            _INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
            for _ in range(5):
                skew_vol = _skew_iv(
                    iv, spot, raw, minutes_to_expiry, skew_slope=skew_slope_override
                )
                s_sqrt_t = skew_vol * math.sqrt(max(tau, 1e-12))
                if s_sqrt_t < 1e-09:
                    break
                d1_skew = (
                    math.log(spot / max(raw, 1.0)) + 0.5 * s_sqrt_t**2
                ) / s_sqrt_t
                cdf_d1 = 0.5 * math.erfc(-d1_skew / _SQRT2)
                if is_call:
                    actual_delta = cdf_d1
                else:
                    actual_delta = cdf_d1 - 1.0
                err = actual_delta - delta_target
                if abs(err) < 0.001:
                    break
                pdf_d1 = _INV_SQRT_2PI * math.exp(-0.5 * d1_skew * d1_skew)
                d_delta_dK = -pdf_d1 / (max(raw, 1.0) * s_sqrt_t)
                if abs(d_delta_dK) < 1e-12:
                    break
                raw = raw - err / d_delta_dK
                raw = max(raw, spot * 0.7)
                raw = min(raw, spot * 1.3)
        return round(raw / 5.0) * 5.0

    @staticmethod
    def _enforce_strike_separation(
        strikes: List[float], legs: tuple, min_sep: float = 5.0
    ) -> List[float]:
        if len(strikes) <= 1:
            return strikes
        n = len(strikes)
        if len(legs) >= n:
            indexed = sorted(
                range(n), key=lambda j: (strikes[j], -legs[j].delta_target)
            )
        else:
            indexed = sorted(range(n), key=lambda j: (strikes[j], j))
        sorted_strikes = [strikes[j] for j in indexed]
        for i in range(1, n):
            if sorted_strikes[i] - sorted_strikes[i - 1] < min_sep - 0.01:
                sorted_strikes[i] = sorted_strikes[i - 1] + min_sep
            sorted_strikes[i] = round(sorted_strikes[i] / 5.0) * 5.0
            if sorted_strikes[i] - sorted_strikes[i - 1] < min_sep - 0.01:
                sorted_strikes[i] = sorted_strikes[i - 1] + min_sep
                sorted_strikes[i] = math.ceil(sorted_strikes[i] / 5.0) * 5.0
        result = [0.0] * n
        for i, orig_idx in enumerate(indexed):
            result[orig_idx] = sorted_strikes[i]
        return result

    @staticmethod
    def _butterfly_strikes(
        strikes: List[float], legs: tuple, spot: float, min_sep: float = 5.0
    ) -> List[float]:
        body_k = round(spot / 5.0) * 5.0
        out = list(strikes)
        for i, leg in enumerate(legs):
            if leg.qty_sign < 0:
                out[i] = body_k
            elif leg.option_type == "call":
                out[i] = max(round(out[i] / 5.0) * 5.0, body_k + min_sep)
            else:
                out[i] = min(round(out[i] / 5.0) * 5.0, body_k - min_sep)
        return out
