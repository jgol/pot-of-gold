"""Option-leg structures and names recovered from evolved shape expressions."""

from __future__ import annotations
import os
import re
from typing import List, Tuple

SHAPES = (
    "put_vertical",
    "call_vertical",
    "iron_condor",
    "iron_butterfly",
    "long_strangle",
)
_SHAPE_TOKEN_RE = re.compile("Shape_([A-Za-z_]+?)(?=\\s*[(),]|$)")


def allowed_shapes() -> Tuple[str, ...]:
    raw = os.environ.get("GP_SHAPES_ALLOWED", "").strip()
    if not raw:
        return SHAPES
    want = {s.strip() for s in raw.split(",") if s.strip()}
    filt = tuple((s for s in SHAPES if s in want))
    return filt if filt else SHAPES


def resolve_constant_shape(shape_tree_sexpr):
    if not shape_tree_sexpr:
        return None
    names = set(_SHAPE_TOKEN_RE.findall(shape_tree_sexpr))
    shapes = {n for n in names if n in SHAPES}
    return next(iter(shapes)) if len(shapes) == 1 else None


def routed_shape(shape_tree_sexpr, fallback=None):
    if not shape_tree_sexpr:
        return fallback
    tokens = set(_SHAPE_TOKEN_RE.findall(str(shape_tree_sexpr)))
    if not tokens:
        return fallback
    if len(tokens) >= 2:
        return "router(multi)"
    only = next(iter(tokens))
    return only if only in SHAPES else fallback


DELTA_FLOOR = 0.05
Leg = Tuple[str, float, int, int]


def build_structure(shape: str, short_delta: float, wing_offset: float) -> List[Leg]:
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}; expected one of {SHAPES}")
    (sd, w) = (float(short_delta), float(wing_offset))
    long_mag = max(sd - w, DELTA_FLOOR)
    if (
        shape in ("put_vertical", "call_vertical", "iron_condor")
        and sd - long_mag < DELTA_FLOOR - 1e-09
    ):
        raise ValueError(
            f"short_delta={sd:.3f} too close to wing (long |delta|={long_mag:.3f}); need a short−long delta gap ≥ {DELTA_FLOOR} for a price-space defined-risk {shape}"
        )
    if shape == "put_vertical":
        return [("put", -sd, -1, 1), ("put", -long_mag, +1, 1)]
    if shape == "call_vertical":
        return [("call", +sd, -1, 1), ("call", +long_mag, +1, 1)]
    if shape == "iron_condor":
        return [
            ("call", +sd, -1, 1),
            ("call", +long_mag, +1, 1),
            ("put", -sd, -1, 1),
            ("put", -long_mag, +1, 1),
        ]
    if shape == "long_strangle":
        return [("call", +sd, +1, 1), ("put", -sd, +1, 1)]
    wing_mag = max(0.5 - w, DELTA_FLOOR)
    return [
        ("call", +0.5, -1, 1),
        ("call", +wing_mag, +1, 1),
        ("put", -0.5, -1, 1),
        ("put", -wing_mag, +1, 1),
    ]
