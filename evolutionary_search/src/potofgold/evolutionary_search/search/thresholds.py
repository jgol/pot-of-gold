"""Snap rule thresholds to a fixed grid or training-derived quantile landmarks."""

import os
from typing import Sequence, Tuple
import numpy as np

DEFAULT_GRID: Tuple[float, ...] = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
DEFAULT_QUANTILES: Tuple[float, ...] = (0.05, 0.16, 0.5, 0.84, 0.95)


def grid_enabled() -> bool:
    return os.environ.get("GP_THRESHOLD_GRID", "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )


def landmark_enabled() -> bool:
    return os.environ.get("GP_THRESHOLD_LANDMARK", "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )


def nearest_grid(x, grid: Sequence[float] = DEFAULT_GRID) -> float:
    xf = float(x)
    if not np.isfinite(xf):
        return xf
    g = np.asarray(grid, dtype=float)
    d = np.abs(g - xf)
    order = np.lexsort((np.abs(g), d))
    return float(g[order[0]])


def build_landmark_grid(
    train_values, quantiles: Sequence[float] = DEFAULT_QUANTILES
) -> Tuple[float, ...]:
    v = np.asarray(train_values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0,)
    qs = [float(round(q, 6)) for q in np.quantile(v, list(quantiles))]
    return tuple(sorted(set([0.0] + qs)))
