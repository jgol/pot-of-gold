"""Option structures and initial rule trees used to seed strategy evolution."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from .grammar import GType, Node, FuncNode, TermDef, TermNode, GT, LT, AND


@dataclass(frozen=True)
class Leg:
    option_type: Literal["call", "put"]
    delta_target: float
    qty_sign: int
    ratio: int = 1

    def __post_init__(self):
        if self.option_type == "call" and self.delta_target < 0:
            raise ValueError(
                f"call leg has negative delta {self.delta_target}; call deltas must be positive"
            )
        if self.option_type == "put" and self.delta_target > 0:
            raise ValueError(
                f"put leg has positive delta {self.delta_target}; put deltas must be negative"
            )
        if self.qty_sign not in (-1, 1):
            raise ValueError(f"qty_sign must be -1 or +1, got {self.qty_sign}")
        if self.ratio < 1:
            raise ValueError(f"ratio must be >=1, got {self.ratio}")


@dataclass
class Template:
    name: str
    description: str
    direction: Literal["neutral", "bullish", "bearish"]
    legs: Tuple[Leg, ...]
    entry_seed: Node
    exit_seed: Node
    size_seed: Node
    is_credit: bool = False
    citations: Tuple[str, ...] = ()
    delta_range: Optional[Tuple[float, float]] = None
    wing_offset: Optional[float] = None
    delta_fixed: bool = False
    delta_seed: Optional[Node] = None
    stop_seed: Optional[float] = None

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    def __post_init__(self):
        if self.direction not in ("neutral", "bullish", "bearish"):
            raise ValueError(
                f"direction must be neutral/bullish/bearish, got {self.direction}"
            )
        if not self.legs:
            raise ValueError(
                f"template {self.name} has no option legs; at least two are required"
            )
        if len(self.legs) < 2:
            raise ValueError(
                f"template {self.name} has {len(self.legs)} option leg(s); at least two are required"
            )
        if self.entry_seed.ret_type != GType.BOOL:
            raise ValueError(
                f"template {self.name} entry_seed has ret_type {self.entry_seed.ret_type}, expected BOOL"
            )
        if self.exit_seed.ret_type != GType.BOOL:
            raise ValueError(
                f"template {self.name} exit_seed has ret_type {self.exit_seed.ret_type}, expected BOOL"
            )
        if self.size_seed.ret_type != GType.REAL:
            raise ValueError(
                f"template {self.name} size_seed has ret_type {self.size_seed.ret_type}, expected REAL"
            )
        if self.delta_seed is not None and self.delta_seed.ret_type != GType.REAL:
            raise ValueError(
                f"template {self.name} delta_seed has ret_type {self.delta_seed.ret_type}, expected REAL"
            )


def _real_term(name: str, value=None) -> TermNode:
    return TermNode(defn=TermDef(name, GType.REAL), value=value)


def _ephemeral_real(value: float) -> TermNode:
    return TermNode(defn=TermDef("EphReal", GType.REAL), value=value)


def _gt(lhs: Node, rhs: Node) -> FuncNode:
    return FuncNode(defn=GT, children=[lhs, rhs])


def _lt(lhs: Node, rhs: Node) -> FuncNode:
    return FuncNode(defn=LT, children=[lhs, rhs])


def _and(a: Node, b: Node) -> FuncNode:
    return FuncNode(defn=AND, children=[a, b])


_VRP_IV_THRESHOLD_NORM: float = 1.0
_VRP_GAP_THRESHOLD_NORM: float = 0.5


def _vrp_compound_entry(
    iv_threshold_norm: float = _VRP_IV_THRESHOLD_NORM,
    gap_threshold_norm: float = _VRP_GAP_THRESHOLD_NORM,
) -> FuncNode:
    _iv_eph = _ephemeral_real(iv_threshold_norm)
    _iv_eph.fold_normalized = True
    _gap_eph = _ephemeral_real(gap_threshold_norm)
    _gap_eph.fold_normalized = True
    return _and(
        _gt(_real_term("ATM_IV"), _iv_eph), _gt(_real_term("IVRVGap5d"), _gap_eph)
    )


def _exit_hold_to_expiry() -> FuncNode:
    return _lt(_real_term("MinutesToClose"), _ephemeral_real(-5.0))


def iron_condor_base() -> Template:
    return Template(
        name="iron_condor",
        description="4-leg neutral credit; GP-evolved strike width",
        direction="neutral",
        legs=(
            Leg("call", +0.25, qty_sign=-1),
            Leg("call", +0.1, qty_sign=+1),
            Leg("put", -0.25, qty_sign=-1),
            Leg("put", -0.1, qty_sign=+1),
        ),
        entry_seed=_vrp_compound_entry(),
        exit_seed=_exit_hold_to_expiry(),
        size_seed=_ephemeral_real(0.5),
        is_credit=True,
        delta_range=(0.15, 0.4),
        wing_offset=0.15,
        delta_seed=_ephemeral_real(0.4),
        stop_seed=0.0,
    )


def base_template_by_name(name: str) -> Template:
    if name != "iron_condor":
        raise ValueError(
            f"Unsupported template {name!r}; the experiment uses iron_condor with evolved structures"
        )
    return iron_condor_base()
