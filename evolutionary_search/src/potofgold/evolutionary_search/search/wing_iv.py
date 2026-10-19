"""Extrapolate put-wing implied volatility from the observed ATM and 40-delta IVs."""

from __future__ import annotations
import numpy as np

(_I_5DP, _I_10DP, _I_25DP, _I_40DP, _I_ATMP) = (0, 1, 2, 3, 4)
_ATMP_DELTA = 0.5
_MIN_ANCHOR_IV = 0.02


def correct_put_wing_skew(grid_ivs):
    g = np.array(grid_ivs, dtype=np.float64)
    if g.ndim == 1:
        return _correct_row(g)
    if g.ndim == 2 and g.shape[1] == 11:
        return np.stack([_correct_row(r) for r in g])
    raise ValueError(f"expected length-11 grid or (N,11) array, got shape {g.shape}")


def _correct_row(g):
    out = g.copy()
    (iv_40dp, iv_atmp) = (g[_I_40DP], g[_I_ATMP])
    if iv_40dp < _MIN_ANCHOR_IV or iv_atmp < _MIN_ANCHOR_IV or iv_40dp <= iv_atmp:
        return out
    slope = (iv_40dp - iv_atmp) / (_ATMP_DELTA - 0.4)
    for idx, d in ((_I_25DP, 0.25), (_I_10DP, 0.1), (_I_5DP, 0.05)):
        extrap = iv_40dp + slope * (0.4 - d)
        out[idx] = max(g[idx], extrap)
    return out
