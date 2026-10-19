"""Define typed rule trees, terminals, primitives and genome serialization."""

from __future__ import annotations
import os
import random
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class GType(Enum):
    REAL = auto()
    BOOL = auto()
    REGIME = auto()
    SIDE = auto()
    INT = auto()
    BANK_IDX = auto()
    SHAPE_ID = auto()
    EMB_SHARED = auto()
    EMB_GRID = auto()
    EMB_STRIKE = auto()
    EMB_SPX = auto()
    EMB_VIX = auto()
    EMB_FLOW_RAW = auto()
    EMB_FLOW_AGG = auto()
    EMB_GRID_PUT = auto()
    EMB_GRID_ATM = auto()
    EMB_GRID_CALL = auto()
    EMB_VAR_ATM_IV = auto()
    EMB_VAR_ATM_SPREAD = auto()
    EMB_VAR_SPX_RET = auto()


EMB_TYPES: Tuple["GType", ...] = (
    GType.EMB_SHARED,
    GType.EMB_GRID,
    GType.EMB_STRIKE,
    GType.EMB_SPX,
    GType.EMB_VIX,
    GType.EMB_FLOW_RAW,
    GType.EMB_FLOW_AGG,
    GType.EMB_GRID_PUT,
    GType.EMB_GRID_ATM,
    GType.EMB_GRID_CALL,
    GType.EMB_VAR_ATM_IV,
    GType.EMB_VAR_ATM_SPREAD,
    GType.EMB_VAR_SPX_RET,
)


class Regime(IntEnum):
    LOW_VOL = 0
    MID_VOL = 1
    HIGH_VOL_STRESSED = 2
    HIGH_VOL_PREMIUM = 3


class Side(Enum):
    CALL = 0
    PUT = 1
    NEUTRAL = 2


@dataclass(frozen=True)
class FuncDef:
    name: str
    arg_types: Tuple[GType, ...]
    ret_type: GType


@dataclass(frozen=True)
class TermDef:
    name: str
    ret_type: GType
    value: Any = None
    sampler: Optional[Callable[[], Any]] = field(
        default=None, hash=False, compare=False
    )
    bank_hi: Optional[int] = field(default=None, hash=False, compare=False)

    def sample_value(self) -> Any:
        return self.sampler() if self.sampler is not None else self.value


@dataclass
class FuncNode:
    defn: FuncDef
    children: List[Node] = field(default_factory=list)

    @property
    def ret_type(self) -> GType:
        return self.defn.ret_type

    @property
    def name(self) -> str:
        return self.defn.name


@dataclass
class TermNode:
    defn: TermDef
    value: Any = None

    def __post_init__(self):
        if self.value is None:
            self.value = self.defn.sample_value()

    @property
    def ret_type(self) -> GType:
        return self.defn.ret_type

    @property
    def name(self) -> str:
        return self.defn.name


Node = FuncNode | TermNode


def depth(node: Node) -> int:
    if isinstance(node, TermNode):
        return 0
    if isinstance(node, FuncNode) and node.defn.name.startswith("EmbProj_"):
        if len(node.children) == 1 and isinstance(node.children[0], TermNode):
            return 1
    return 1 + max((depth(c) for c in node.children), default=0)


def node_count(node: Node) -> int:
    if isinstance(node, TermNode):
        return 1
    if isinstance(node, FuncNode) and node.defn.name.startswith("EmbProj_"):
        if len(node.children) == 1 and isinstance(node.children[0], TermNode):
            return 1
    if isinstance(node, FuncNode) and node.defn.name.startswith("EmbDir_"):
        if (
            len(node.children) == 2
            and isinstance(node.children[0], TermNode)
            and isinstance(node.children[1], TermNode)
        ):
            return 1
    if isinstance(node, FuncNode) and node.defn.name.startswith("EmbCos_"):
        if (
            len(node.children) == 2
            and isinstance(node.children[0], TermNode)
            and isinstance(node.children[1], TermNode)
            and node.children[1].name.startswith("PROTO_")
            and (
                os.environ.get("GP_LEAN_PROTO_NONLINEAR", "0")
                not in ("0", "false", "False")
            )
        ):
            return 1
    return 1 + sum((node_count(c) for c in node.children))


def all_nodes_indexed(
    node: Node, _d: int = 0, _p: Tuple[int, ...] = ()
) -> List[Tuple[Node, int, Tuple[int, ...]]]:
    out: List[Tuple[Node, int, Tuple[int, ...]]] = [(node, _d, _p)]
    if isinstance(node, FuncNode):
        for i, ch in enumerate(node.children):
            out.extend(all_nodes_indexed(ch, _d + 1, _p + (i,)))
    return out


def to_str(node: Node) -> str:
    if isinstance(node, TermNode):
        v = node.value
        if isinstance(v, Enum):
            return v.name
        if isinstance(v, (int, float)):
            return f"{node.name}({v})"
        return node.name
    args = ", ".join((to_str(c) for c in node.children))
    return f"{node.name}({args})"


_COMMUTATIVE_OP_NAMES = frozenset({"AND", "OR", "Add", "Mul", "RegimeIs"})


def _is_commutative_func(node: "FuncNode") -> bool:
    if len(node.children) != 2:
        return False
    name = node.defn.name
    return name in _COMMUTATIVE_OP_NAMES or name.startswith("EmbCos_")


def canonical_key(node: Node, _eph_round: Optional[int] = None) -> str:
    if isinstance(node, TermNode):
        v = node.value
        if isinstance(v, Enum):
            return v.name
        if isinstance(v, float) and _eph_round is not None and (node.name == "EphReal"):
            return f"{node.name}({round(v, _eph_round)})"
        if isinstance(v, (int, float)):
            return f"{node.name}({v})"
        return node.name
    child_keys = [canonical_key(c, _eph_round) for c in node.children]
    name = node.defn.name
    if name == "LT" and len(child_keys) == 2:
        return f"GT({child_keys[1]}, {child_keys[0]})"
    if _is_commutative_func(node):
        child_keys = sorted(child_keys)
    return f"{name}({', '.join(child_keys)})"


def fold_degenerate_shape_router(tree: Optional[Node]) -> Optional[Node]:
    if tree is None:
        return None
    if not isinstance(tree, FuncNode):
        return tree
    kids = [fold_degenerate_shape_router(c) for c in tree.children]
    if tree.name == "IfThenElseShape" and len(kids) == 3:
        (_then, _else) = (kids[1], kids[2])
        if canonical_key(_then) == canonical_key(_else):
            return _then
    return replace(tree, children=kids)


_TIME_ONLY_TERMINALS = frozenset({"MinutesToClose", "BarOfDay", "ThetaUrgency"})
_DATA_QUALITY_TERMINALS = frozenset({"GridReliability"})
_POSITION_STATE_TERMINALS = frozenset(
    {
        "UnrealizedProfitPct",
        "MaxFavorableExcursion",
        "MaxAdverseExcursion",
        "ProfitGiveback",
    }
)
_DAY_CONSTANT_TERMINALS = frozenset(
    {
        "SPXReturn3d",
        "VIXMean5d",
        "RV5d",
        "IVRVGap5d",
        "OvernightGap",
        "VIXSpot",
        "VIXTermSlope",
        "VIXChange",
        "CalDayOfWeek",
        "CalIsOpexWeek",
        "CalTurnOfMonth",
        "CalDaysToFomc",
    }
)
_ENTRY_EXCLUDE_DAY_CONSTANT = os.environ.get(
    "GP_ENTRY_EXCLUDE_DAY_CONSTANT", ""
).strip().lower() in ("1", "true", "yes", "on")
_COMPARISON_OPS = frozenset({"GT", "LT", "CrossAbove", "CrossBelow"})
_ENTRY_KNIFE_EDGE_OPS = frozenset({"CrossAbove", "CrossBelow"})
_ENTRY_HIGH_CAP_OPS = frozenset({"WeightedSum3", "WeightedSum5"})
_ENTRY_ROBUST_EXCLUDED_OPS = _ENTRY_KNIFE_EDGE_OPS | _ENTRY_HIGH_CAP_OPS
_ENTRY_CONJUNCTION_OPS = frozenset({"AND", "OR"})


def _entry_robust_on() -> bool:
    return os.environ.get("GP_ENTRY_ROBUST", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _entry_max_conjunctions() -> Optional[int]:
    raw = os.environ.get("GP_ENTRY_MAX_CONJ", "").strip()
    if not raw:
        return 2
    try:
        v = int(raw)
    except ValueError:
        return 2
    return None if v < 0 else v


def _count_func_ops(node: "Node", names: frozenset) -> int:
    if not isinstance(node, FuncNode):
        return 0
    here = 1 if node.name in names else 0
    return here + sum((_count_func_ops(c, names) for c in node.children))


def _uses_excluded_entry_op(node: "Node") -> bool:
    if _count_func_ops(node, _ENTRY_ROBUST_EXCLUDED_OPS) > 0:
        return True
    max_conj = _entry_max_conjunctions()
    if (
        max_conj is not None
        and _count_func_ops(node, _ENTRY_CONJUNCTION_OPS) > max_conj
    ):
        return True
    return False


def _bare_terminal_name(node) -> Optional[str]:
    if node is None:
        return None
    if getattr(node, "children", None):
        return None
    name = getattr(node, "name", None)
    if name in (None, "EphReal", "EphInt", "EmbBankIdx"):
        return None
    return name


def snap_tree_thresholds(root: "Node", grid=None, grid_map=None) -> "Node":
    from .thresholds import DEFAULT_GRID, grid_enabled, landmark_enabled, nearest_grid

    if root is None or not grid_enabled():
        return root
    g = DEFAULT_GRID if grid is None else grid
    _use_map = grid_map if landmark_enabled() else None

    def _snap_child(child, siblings):
        if (
            getattr(child, "name", None) != "EphReal"
            or getattr(child, "value", None) is None
        ):
            return
        target = g
        if _use_map is not None:
            for sib in siblings:
                sname = _bare_terminal_name(sib)
                if sname is not None and sname in _use_map:
                    target = _use_map[sname]
                    break
        child.value = nearest_grid(child.value, target)

    def _rec(node):
        if node is None:
            return
        children = list(getattr(node, "children", None) or [])
        if getattr(node, "name", None) in _COMPARISON_OPS:
            for i, c in enumerate(children):
                _snap_child(c, [s for (j, s) in enumerate(children) if j != i])
        for c in children:
            _rec(c)

    _rec(root)
    return root


def bias_comparison_data_terminals(
    node: "Node", p_snap: float = 0.5, grid=None
) -> "Node":
    from .thresholds import DEFAULT_GRID, grid_enabled

    if node is None or not grid_enabled():
        return node
    _ensure_lookups()
    eph_def = _TERM_LOOKUP.get("EphReal")
    if eph_def is None:
        return node
    g = list(DEFAULT_GRID if grid is None else grid)

    def _is_bare_data_term(c) -> bool:
        return not (getattr(c, "children", None) or []) and getattr(
            c, "name", None
        ) not in (None, "EphReal", "EphInt", "EmbBankIdx")

    def _rec(n):
        if n is None:
            return
        children = getattr(n, "children", None)
        if (
            children
            and getattr(n, "name", None) in _COMPARISON_OPS
            and (len(children) >= 2)
        ):
            if (
                _is_bare_data_term(children[0])
                and _is_bare_data_term(children[1])
                and (random.random() < p_snap)
            ):
                children[1] = TermNode(defn=eph_def, value=float(random.choice(g)))
        for c in getattr(n, "children", None) or []:
            _rec(c)

    _rec(node)
    return node


def forbid_noncarrier_comparison_terminals(node: "Node", carriers, grid=None) -> "Node":
    if node is None or not carriers:
        return node
    _ensure_lookups()
    eph_def = _TERM_LOOKUP.get("EphReal")
    if eph_def is None:
        return node
    allow = set(carriers)

    def _forbidden(c) -> bool:
        return (
            not (getattr(c, "children", None) or [])
            and getattr(c, "name", None)
            not in (None, "EphReal", "EphInt", "EmbBankIdx")
            and (getattr(c, "name", None) not in allow)
        )

    def _rec(n):
        if n is None:
            return
        children = getattr(n, "children", None)
        if children and getattr(n, "name", None) in _COMPARISON_OPS:
            for i, c in enumerate(children):
                if _forbidden(c):
                    children[i] = TermNode(defn=eph_def, value=0.0)
        for c in getattr(n, "children", None) or []:
            _rec(c)

    _rec(node)
    return node


def _leaf_terminal_names(node: Node) -> set:
    if isinstance(node, TermNode):
        return (
            {node.name}
            if node.name not in ("EphReal", "EphInt", "EmbBankIdx")
            else set()
        )
    return set().union(*(_leaf_terminal_names(c) for c in node.children))


def _is_time_vs_market(node: FuncNode) -> bool:
    if node.name not in _COMPARISON_OPS or len(node.children) < 2:
        return False
    left = _leaf_terminal_names(node.children[0])
    right = _leaf_terminal_names(node.children[1])
    left_all_time = left.issubset(_TIME_ONLY_TERMINALS) if left else False
    right_all_time = right.issubset(_TIME_ONLY_TERMINALS) if right else False
    left_all_eph = not left
    right_all_eph = not right
    if left_all_time and right_all_eph or (right_all_time and left_all_eph):
        return True
    for terms, other in [(left, right), (right, left)]:
        if not terms:
            continue
        other_all_eph = not other
        if not other_all_eph:
            continue
        time_frac = len(terms & _TIME_ONLY_TERMINALS) / len(terms)
        if time_frac >= 0.5 and len(terms & _TIME_ONLY_TERMINALS) >= 1:
            return True
    if not left or not right:
        return False
    if left_all_time == right_all_time:
        return False
    return True


def _count_ite_nodes(node: Node) -> int:
    if not isinstance(node, FuncNode):
        return 0
    count = 1 if node.name == "IfThenElse" else 0
    return count + sum((_count_ite_nodes(c) for c in node.children))


def _evaluates_to_constant(node: Node) -> bool:
    if isinstance(node, TermNode):
        if node.name in ("EphReal", "EphInt", "EmbBankIdx"):
            return True
        return node.name.startswith("PROTO_") and os.environ.get(
            "GP_LEAN_PROTO_NONLINEAR", "0"
        ) not in ("0", "false", "False")
    if isinstance(node, FuncNode):
        return all((_evaluates_to_constant(c) for c in node.children))
    return False


def simplify_tree(node: Node) -> Node:
    if isinstance(node, TermNode):
        return node
    simplified_children = [simplify_tree(c) for c in node.children]
    node = FuncNode(defn=node.defn, children=simplified_children)
    if node.name == "NOT" and len(simplified_children) == 1:
        child = simplified_children[0]
        if (
            isinstance(child, FuncNode)
            and child.name == "NOT"
            and (len(child.children) == 1)
        ):
            return child.children[0]
        if (
            isinstance(child, FuncNode)
            and child.name == "LT"
            and (len(child.children) == 2)
        ):
            return FuncNode(defn=GT, children=[child.children[1], child.children[0]])
        if (
            isinstance(child, FuncNode)
            and child.name == "GT"
            and (len(child.children) == 2)
        ):
            return FuncNode(defn=LT, children=[child.children[1], child.children[0]])
    return node


def is_structurally_degenerate(
    node: Node, max_ite: int = 2, _root: bool = True
) -> bool:
    if not isinstance(node, FuncNode):
        return False
    if _count_ite_nodes(node) > max_ite:
        return True
    if node.name in _COMPARISON_OPS and len(node.children) >= 2:
        if to_str(node.children[0]) == to_str(node.children[1]):
            return True
        if _root and _is_time_vs_market(node):
            return True
        if _evaluates_to_constant(node.children[0]) and _evaluates_to_constant(
            node.children[1]
        ):
            return True
        if node.name in ("CrossAbove", "CrossBelow"):
            if _evaluates_to_constant(node.children[0]) or _evaluates_to_constant(
                node.children[1]
            ):
                return True
    if node.name == "Sub" and len(node.children) >= 2:
        if to_str(node.children[0]) == to_str(node.children[1]):
            return True
    if (
        node.name.startswith("EmbCos_")
        and len(node.children) >= 2
        and (
            os.environ.get("GP_LEAN_PROTO_NONLINEAR", "0")
            not in ("0", "false", "False")
        )
    ):
        if to_str(node.children[0]) == to_str(node.children[1]):
            return True
    if _root and (
        node.name == "NOT"
        and len(node.children) == 1
        and isinstance(node.children[0], FuncNode)
        and (node.children[0].name in ("CrossAbove", "CrossBelow"))
    ):
        return True
    if node.name in ("AND", "OR") and len(node.children) == 2:
        for a, b in [
            (node.children[0], node.children[1]),
            (node.children[1], node.children[0]),
        ]:
            if (
                isinstance(a, FuncNode)
                and a.name == "NOT"
                and (len(a.children) == 1)
                and (to_str(a.children[0]) == to_str(b))
            ):
                return True
    if _root:
        all_terms = _leaf_terminal_names(node)
        if all_terms & _DATA_QUALITY_TERMINALS:
            return True
        if all_terms:
            time_frac = len(all_terms & _TIME_ONLY_TERMINALS) / len(all_terms)
            if time_frac >= 0.8:
                return True
    return any(
        (
            is_structurally_degenerate(c, max_ite=max_ite, _root=False)
            for c in node.children
            if isinstance(c, FuncNode)
        )
    )


def _entry_exclude_proto_const_on() -> bool:
    return os.environ.get("GP_ENTRY_EXCLUDE_PROTO_CONST", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _reads_proto_only_const(node: "Node") -> bool:
    names = _leaf_terminal_names(node)
    return any((str(n).startswith("PROTO_") for n in names)) and (
        not any((str(n).startswith("EMB_") for n in names))
    )


def _entry_compares_live_emb_vs_proto_const(node: "Node") -> bool:
    if isinstance(node, FuncNode):
        if node.name in _COMPARISON_OPS and len(node.children) == 2:
            (a, b) = node.children
            a_live = any((str(n).startswith("EMB_") for n in _leaf_terminal_names(a)))
            b_live = any((str(n).startswith("EMB_") for n in _leaf_terminal_names(b)))
            if (
                a_live
                and _reads_proto_only_const(b)
                or (b_live and _reads_proto_only_const(a))
            ):
                return True
        return any((_entry_compares_live_emb_vs_proto_const(c) for c in node.children))
    return False


def has_illegal_position_state(node: "Node", slot: str) -> bool:
    if slot not in ("entry", "delta"):
        return False
    leaves = _leaf_terminal_names(node)
    if leaves & _POSITION_STATE_TERMINALS:
        return True
    if (
        slot == "entry"
        and _ENTRY_EXCLUDE_DAY_CONSTANT
        and leaves
        and (leaves <= _DAY_CONSTANT_TERMINALS)
    ):
        return True
    if slot == "entry" and _entry_robust_on() and _uses_excluded_entry_op(node):
        return True
    if (
        slot == "entry"
        and _entry_exclude_proto_const_on()
        and _entry_compares_live_emb_vs_proto_const(node)
    ):
        return True
    return False


def _build_from_sexpr_lookups() -> Tuple[Dict[str, "FuncDef"], Dict[str, "TermDef"]]:
    func_lookup: Dict[str, FuncDef] = {}
    for fd in [
        GT,
        LT,
        AND,
        OR,
        NOT,
        ADD,
        SUB,
        MUL,
        DIV,
        SQRT,
        ABS,
        WEIGHTED_SUM3,
        WEIGHTED_SUM5,
        LAG,
        DELTA,
        CROSS_ABOVE,
        CROSS_BELOW,
        IN_REGIME,
        REGIME_IS,
        IF_REAL,
        IF_SIDE,
        IF_SHAPE,
    ]:
        func_lookup[fd.name] = fd
    for fd in EMB_OPERATORS:
        func_lookup[fd.name] = fd
    term_lookup: Dict[str, TermDef] = {}
    for td in build_terminal_set():
        term_lookup[td.name] = td
    for td in build_emb_lean_terminal_set():
        term_lookup.setdefault(td.name, td)
    for td in build_scalar_only_terminal_set():
        term_lookup.setdefault(td.name, td)
    for r in Regime:
        term_lookup[r.name] = TermDef(r.name, GType.REGIME, value=r)
    for s in Side:
        term_lookup[s.name] = TermDef(s.name, GType.SIDE, value=s)
    if "AbstainFlag" not in term_lookup:
        term_lookup["AbstainFlag"] = TermDef("AbstainFlag", GType.REAL)
    return (func_lookup, term_lookup)


_FUNC_LOOKUP: Dict[str, "FuncDef"] = {}
_TERM_LOOKUP: Dict[str, "TermDef"] = {}


def _ensure_lookups() -> None:
    global _FUNC_LOOKUP, _TERM_LOOKUP
    if not _FUNC_LOOKUP:
        (_FUNC_LOOKUP, _TERM_LOOKUP) = _build_from_sexpr_lookups()


def _tokenize_sexpr(s: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in " \t\n\r":
            i += 1
        elif c in "(,)":
            tokens.append(c)
            i += 1
        elif c == "-" and i + 1 < len(s) and (s[i + 1].isdigit() or s[i + 1] == "."):
            j = i + 1
            while j < len(s) and (
                s[j].isdigit()
                or s[j] == "."
                or s[j] == "e"
                or (s[j] == "E")
                or (s[j] == "-" and j > 0 and (s[j - 1] in "eE"))
            ):
                j += 1
            tokens.append(s[i:j])
            i = j
        elif c.isdigit() or c == ".":
            j = i
            while j < len(s) and (
                s[j].isdigit()
                or s[j] == "."
                or s[j] == "e"
                or (s[j] == "E")
                or (s[j] == "-" and j > 0 and (s[j - 1] in "eE"))
            ):
                j += 1
            tokens.append(s[i:j])
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            raise ValueError(f"Unexpected character {c!r} at position {i} in: {s!r}")
    return tokens


def from_sexpr(s: str) -> Node:
    _ensure_lookups()
    tokens = _tokenize_sexpr(s)
    (node, pos) = _parse_sexpr(tokens, 0)
    if pos != len(tokens):
        raise ValueError(
            f"Unexpected tokens after position {pos}: {' '.join(tokens[pos:pos + 5])}... in: {s!r}"
        )
    return node


def _parse_sexpr(tokens: List[str], pos: int) -> Tuple[Node, int]:
    if pos >= len(tokens):
        raise ValueError("Unexpected end of expression")
    name = tokens[pos]
    if pos + 1 < len(tokens) and tokens[pos + 1] == "(":
        if name in _FUNC_LOOKUP:
            fd = _FUNC_LOOKUP[name]
            pos += 2
            children: List[Node] = []
            while pos < len(tokens) and tokens[pos] != ")":
                if tokens[pos] == ",":
                    pos += 1
                    continue
                (child, pos) = _parse_sexpr(tokens, pos)
                children.append(child)
            if pos >= len(tokens):
                raise ValueError(f"Unbalanced parentheses in s-expression for '{name}'")
            pos += 1
            return (FuncNode(defn=fd, children=children), pos)
        if name in _TERM_LOOKUP:
            td = _TERM_LOOKUP[name]
            pos += 2
            val_str = tokens[pos]
            pos += 1
            if pos < len(tokens) and tokens[pos] == ")":
                pos += 1
            if td.ret_type in (GType.INT, GType.BANK_IDX):
                value = int(val_str)
            else:
                value = float(val_str)
            return (TermNode(defn=td, value=value), pos)
        raise ValueError(f"Unknown function/terminal name: {name!r}")
    if name in _TERM_LOOKUP:
        td = _TERM_LOOKUP[name]
        return (TermNode(defn=td, value=td.value), pos + 1)
    try:
        if "." in name:
            return (TermNode(defn=_TERM_LOOKUP["EphReal"], value=float(name)), pos + 1)
        return (TermNode(defn=_TERM_LOOKUP["EphInt"], value=int(name)), pos + 1)
    except (ValueError, KeyError):
        pass
    raise ValueError(f"Unknown token: {name!r} at position {pos}")


def deep_copy(node: Node) -> Node:
    if isinstance(node, TermNode):
        return TermNode(defn=node.defn, value=node.value)
    return FuncNode(defn=node.defn, children=[deep_copy(c) for c in node.children])


def _replace_at_path(root: Node, path: Tuple[int, ...], repl: Node) -> Node:
    if not path:
        return repl
    assert isinstance(root, FuncNode)
    (idx, rest) = (path[0], path[1:])
    kids = list(root.children)
    kids[idx] = _replace_at_path(root.children[idx], rest, repl)
    return FuncNode(defn=root.defn, children=kids)


def _node_at_path(root: Node, path: Tuple[int, ...]) -> Node:
    n = root
    for i in path:
        assert isinstance(n, FuncNode)
        n = n.children[i]
    return n


def _path_is_emblag_first_child(root: Node, path: Tuple[int, ...]) -> bool:
    if not path or path[-1] != 0:
        return False
    parent = _node_at_path(root, path[:-1])
    return isinstance(parent, FuncNode) and is_emblag_funcdef(parent.defn)


def _crossover_would_violate_emblag(
    root: Node, path: Tuple[int, ...], repl: Node
) -> bool:
    if _path_is_emblag_first_child(root, path) and (not isinstance(repl, TermNode)):
        return True

    def _bad(n: Node) -> bool:
        if isinstance(n, TermNode):
            return False
        if is_emblag_funcdef(n.defn) and (not isinstance(n.children[0], TermNode)):
            return True
        return any((_bad(c) for c in n.children))

    return _bad(repl)


GT = FuncDef("GT", (GType.REAL, GType.REAL), GType.BOOL)
LT = FuncDef("LT", (GType.REAL, GType.REAL), GType.BOOL)
AND = FuncDef("AND", (GType.BOOL, GType.BOOL), GType.BOOL)
OR = FuncDef("OR", (GType.BOOL, GType.BOOL), GType.BOOL)
NOT = FuncDef("NOT", (GType.BOOL,), GType.BOOL)
ADD = FuncDef("Add", (GType.REAL, GType.REAL), GType.REAL)
SUB = FuncDef("Sub", (GType.REAL, GType.REAL), GType.REAL)
MUL = FuncDef("Mul", (GType.REAL, GType.REAL), GType.REAL)
DIV = FuncDef("Div", (GType.REAL, GType.REAL), GType.REAL)
SQRT = FuncDef("Sqrt", (GType.REAL,), GType.REAL)
ABS = FuncDef("Abs", (GType.REAL,), GType.REAL)
LAG = FuncDef("Lag", (GType.REAL, GType.INT), GType.REAL)
DELTA = FuncDef("Delta", (GType.REAL, GType.INT), GType.REAL)
CROSS_ABOVE = FuncDef("CrossAbove", (GType.REAL, GType.REAL), GType.BOOL)
CROSS_BELOW = FuncDef("CrossBelow", (GType.REAL, GType.REAL), GType.BOOL)
IN_REGIME = FuncDef("InRegime", (GType.REGIME,), GType.BOOL)
REGIME_IS = FuncDef("RegimeIs", (GType.REGIME, GType.REGIME), GType.BOOL)
IF_REAL = FuncDef("IfThenElse", (GType.BOOL, GType.REAL, GType.REAL), GType.REAL)
IF_SIDE = FuncDef("IfSide", (GType.BOOL, GType.SIDE, GType.SIDE), GType.SIDE)
from .structure import SHAPES as _SHAPE_MENU
from .structure import allowed_shapes as _allowed_shapes

SHAPE_TERMINALS: List[TermDef] = [
    TermDef(f"Shape_{s}", GType.SHAPE_ID, value=s) for s in _SHAPE_MENU
]
IF_SHAPE = FuncDef(
    "IfThenElseShape", (GType.BOOL, GType.SHAPE_ID, GType.SHAPE_ID), GType.SHAPE_ID
)


def _shape_router_disabled() -> bool:
    return os.environ.get("GP_DISABLE_SHAPE_ROUTER", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _active_shape_terminals() -> List[TermDef]:
    _allow = set(_allowed_shapes())
    return [t for t in SHAPE_TERMINALS if t.value in _allow]


EMB_OPERATORS: List[FuncDef] = []
_EMBLAG_FUNCDEFS: set[FuncDef] = set()
for _t in EMB_TYPES:
    _n = _t.name
    EMB_OPERATORS.append(FuncDef(f"EmbNorm_{_n}", (_t,), GType.REAL))
    EMB_OPERATORS.append(FuncDef(f"EmbCos_{_n}", (_t, _t), GType.REAL))
    EMB_OPERATORS.append(FuncDef(f"EmbSub_{_n}", (_t, _t), _t))
    _emblag = FuncDef(f"EmbLag_{_n}", (_t, GType.INT), _t)
    EMB_OPERATORS.append(_emblag)
    _EMBLAG_FUNCDEFS.add(_emblag)
    EMB_OPERATORS.append(FuncDef(f"EmbDir_{_n}", (_t, GType.BANK_IDX), GType.REAL))
from .readouts import PROJECTION_COMPONENT_COUNTS as _PROJECTION_COMPONENT_COUNTS

SERIALIZED_PROJECTION_COUNTS: Dict[str, int] = dict(_PROJECTION_COMPONENT_COUNTS)
from .readouts import B_MAX

PROTO_P_PER_GROUP = 2
_TOTAL_EMBPROJ_OPERATORS_V9 = sum(SERIALIZED_PROJECTION_COUNTS.values())
assert (
    _TOTAL_EMBPROJ_OPERATORS_V9 == 135
), f"SERIALIZED_PROJECTION_COUNTS sums to {_TOTAL_EMBPROJ_OPERATORS_V9}, expected 135 (84 group-pooled + 15 per-variate R2 + 36 R1 grid sub-pools)"
_TOTAL_EMBPROJ_OPERATORS_LEAN = sum(
    (min(3, _k) for _k in SERIALIZED_PROJECTION_COUNTS.values())
)
assert (
    _TOTAL_EMBPROJ_OPERATORS_LEAN <= 39
), f"lean EmbProj budget {_TOTAL_EMBPROJ_OPERATORS_LEAN} exceeds 39 (top-3 x 13 groups)"
for _t in EMB_TYPES:
    _n = _t.name
    if _n not in SERIALIZED_PROJECTION_COUNTS:
        raise RuntimeError(
            f"Projection component counts are missing an entry for {_n!r}. Got keys: {sorted(SERIALIZED_PROJECTION_COUNTS)}. Update the readout component counts."
        )
    _k_for_group = SERIALIZED_PROJECTION_COUNTS[_n]
    for _k in range(_k_for_group):
        EMB_OPERATORS.append(FuncDef(f"EmbProj_{_n}_{_k}", (_t,), GType.REAL))


def is_emblag_funcdef(fd: FuncDef) -> bool:
    return fd in _EMBLAG_FUNCDEFS


WEIGHTED_SUM3 = FuncDef("WeightedSum3", (GType.REAL,) * 6, GType.REAL)
WEIGHTED_SUM5 = FuncDef("WeightedSum5", (GType.REAL,) * 10, GType.REAL)
ALL_FUNCTIONS: List[FuncDef] = [
    GT,
    LT,
    AND,
    OR,
    NOT,
    ADD,
    SUB,
    MUL,
    DIV,
    SQRT,
    ABS,
    WEIGHTED_SUM3,
    WEIGHTED_SUM5,
    LAG,
    DELTA,
    CROSS_ABOVE,
    CROSS_BELOW,
    IN_REGIME,
    REGIME_IS,
    IF_REAL,
    IF_SIDE,
    IF_SHAPE,
] + EMB_OPERATORS
OPERATOR_CLASSES: Dict[str, Tuple[FuncDef, ...]] = {
    "arithmetic": (ADD, SUB, MUL, DIV, SQRT, ABS),
    "comparison": (GT, LT),
    "logical": (AND, OR, NOT),
    "temporal": (LAG, DELTA, CROSS_ABOVE, CROSS_BELOW),
    "regime": (IN_REGIME, REGIME_IS),
    "conditional": (IF_REAL, IF_SIDE),
    "shape_selector": (IF_SHAPE,),
    "linear_combo": (WEIGHTED_SUM3, WEIGHTED_SUM5),
    "emb_norm": tuple((f for f in EMB_OPERATORS if f.name.startswith("EmbNorm_"))),
    "emb_cos": tuple((f for f in EMB_OPERATORS if f.name.startswith("EmbCos_"))),
    "emb_sub": tuple((f for f in EMB_OPERATORS if f.name.startswith("EmbSub_"))),
    "emb_lag": tuple((f for f in EMB_OPERATORS if f.name.startswith("EmbLag_"))),
    "emb_proj": tuple((f for f in EMB_OPERATORS if f.name.startswith("EmbProj_"))),
    "emb_dir": tuple((f for f in EMB_OPERATORS if f.name.startswith("EmbDir_"))),
}
_FUNC_TO_CLASS: Dict[FuncDef, str] = {}
for _cls_name, _cls_funcs in OPERATOR_CLASSES.items():
    for _f in _cls_funcs:
        _FUNC_TO_CLASS[_f] = _cls_name
_unclassified = [f.name for f in ALL_FUNCTIONS if f not in _FUNC_TO_CLASS]
assert (
    not _unclassified
), f"F2 violation: {len(_unclassified)} ALL_FUNCTIONS not in any OPERATOR_CLASSES bucket: {_unclassified[:5]}... Add them to OPERATOR_CLASSES."
_LAG_CHOICES = (1, 3, 5, 10, 15, 30)


def build_terminal_set() -> List[TermDef]:
    (T, R) = (TermDef, GType.REAL)
    terms: List[TermDef] = []
    for emb_t in EMB_TYPES:
        terms.append(T(emb_t.name, emb_t))
    terms += [T("PredRV15", R), T("PredRV30", R), T("PredSpread", R)]
    terms += [T("PredGammaAccel", R), T("PredSmileConvexity", R)]
    terms += [T("PredJump", R), T("PredFlowToxicity", R)]
    terms += [T("RegimeAboveLow", R), T("RegimeIsHigh", R), T("RegimeIsPremium", R)]
    terms += [
        T("DeltaSpread1", R),
        T("DeltaSpread5", R),
        T("RawSpread", R),
        T("ATM_IV", R),
        T("VIXSpot", R),
        T("VIXTermSlope", R),
        T("RealizedVol30m", R),
    ]
    terms += [T("MinutesToClose", R)]
    terms += [T("BarOfDay", R)]
    terms += [T("SessionReturn", R)]
    terms += [T("ATM_IV_5m", R), T("RealizedVol30m_5m", R), T("RawSpread_5m", R)]
    terms += [T("VIXChange", R)]
    terms += [T("PutCallSkew", R), T("SessionPosition", R)]
    terms += [T("PutCallSkew10", R)]
    terms += [T("OvernightGap", R)]
    terms += [T("GridReliability", R)]
    terms += [T("ThetaUrgency", R)]
    terms += [T("SPXReturn3d", R), T("VIXMean5d", R)]
    terms += [T("RV5d", R), T("IVRVGap5d", R)]
    terms += [T("UnrealizedProfitPct", R)]
    terms += [
        T("MaxFavorableExcursion", R),
        T("MaxAdverseExcursion", R),
        T("ProfitGiveback", R),
    ]
    terms.append(T("EphReal", R, sampler=lambda: round(random.uniform(-3, 3), 4)))
    terms.append(T("EphInt", GType.INT, sampler=lambda: random.choice(_LAG_CHOICES)))
    terms.append(
        T("EmbBankIdx", GType.BANK_IDX, sampler=lambda: random.randrange(B_MAX))
    )
    for r in Regime:
        terms.append(T(r.name, GType.REGIME, value=r))
    for s in Side:
        terms.append(T(s.name, GType.SIDE, value=s))
    terms += list(_active_shape_terminals())
    return terms


def build_scalar_only_terminal_set() -> List[TermDef]:
    (T, R) = (TermDef, GType.REAL)
    terms: List[TermDef] = []
    terms += [
        T("DeltaSpread1", R),
        T("DeltaSpread5", R),
        T("RawSpread", R),
        T("ATM_IV", R),
        T("VIXSpot", R),
        T("VIXTermSlope", R),
        T("RealizedVol30m", R),
    ]
    terms += [T("MinutesToClose", R)]
    terms += [T("BarOfDay", R)]
    terms += [T("SessionReturn", R)]
    terms += [T("ATM_IV_5m", R), T("RealizedVol30m_5m", R), T("RawSpread_5m", R)]
    terms += [T("VIXChange", R)]
    terms += [T("PutCallSkew", R)]
    terms += [T("PutCallSkew10", R)]
    terms += [T("SessionPosition", R)]
    terms += [T("OvernightGap", R)]
    if os.environ.get("GP_DISABLE_CALENDAR", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        terms += [
            T("CalDayOfWeek", R),
            T("CalIsOpexWeek", R),
            T("CalTurnOfMonth", R),
            T("CalDaysToFomc", R),
        ]
    terms += [T("NetGammaProxy", R)]
    terms += [T("ThetaUrgency", R)]
    terms += [T("SPXReturn3d", R), T("VIXMean5d", R)]
    terms += [T("RV5d", R), T("IVRVGap5d", R)]
    terms += [T("UnrealizedProfitPct", R)]
    terms += [
        T("MaxFavorableExcursion", R),
        T("MaxAdverseExcursion", R),
        T("ProfitGiveback", R),
    ]
    terms.append(T("EphReal", R, sampler=lambda: round(random.uniform(-3, 3), 4)))
    terms.append(T("EphInt", GType.INT, sampler=lambda: random.choice(_LAG_CHOICES)))
    for s in Side:
        terms.append(T(s.name, GType.SIDE, value=s))
    return terms


def build_emb_only_terminal_set() -> List[TermDef]:
    (T, R) = (TermDef, GType.REAL)
    terms: List[TermDef] = []
    for grp in (
        "EMB_SHARED",
        "EMB_GRID",
        "EMB_STRIKE",
        "EMB_SPX",
        "EMB_VIX",
        "EMB_FLOW_RAW",
        "EMB_FLOW_AGG",
        "EMB_GRID_PUT",
        "EMB_GRID_ATM",
        "EMB_GRID_CALL",
        "EMB_VAR_ATM_IV",
        "EMB_VAR_ATM_SPREAD",
        "EMB_VAR_SPX_RET",
    ):
        terms.append(T(grp, GType[grp]))
    terms += [T("MinutesToClose", R), T("BarOfDay", R), T("SessionReturn", R)]
    terms += [T("ThetaUrgency", R)]
    terms += [T("SPXReturn3d", R), T("VIXMean5d", R), T("RV5d", R), T("IVRVGap5d", R)]
    terms += [T("UnrealizedProfitPct", R)]
    terms += [
        T("MaxFavorableExcursion", R),
        T("MaxAdverseExcursion", R),
        T("ProfitGiveback", R),
    ]
    terms.append(T("EphReal", R, sampler=lambda: round(random.uniform(-3, 3), 4)))
    terms.append(T("EphInt", GType.INT, sampler=lambda: random.choice(_LAG_CHOICES)))
    for s in Side:
        terms.append(T(s.name, GType.SIDE, value=s))
    return terms


_SCALAR_ONLY_FORBIDDEN_FUNCS: Tuple[FuncDef, ...] = (IN_REGIME, REGIME_IS)
SCALAR_ONLY_FUNCTIONS: List[FuncDef] = [
    f
    for f in ALL_FUNCTIONS
    if f not in EMB_OPERATORS and f not in _SCALAR_ONLY_FORBIDDEN_FUNCS
]


def build_emb_lean_terminal_set(bank_cap: Optional[int] = None) -> List[TermDef]:
    terms = list(build_emb_only_terminal_set())
    for grp in (
        "EMB_SHARED",
        "EMB_GRID",
        "EMB_STRIKE",
        "EMB_SPX",
        "EMB_VIX",
        "EMB_FLOW_RAW",
        "EMB_FLOW_AGG",
        "EMB_GRID_PUT",
        "EMB_GRID_ATM",
        "EMB_GRID_CALL",
        "EMB_VAR_ATM_IV",
        "EMB_VAR_ATM_SPREAD",
        "EMB_VAR_SPX_RET",
    ):
        for _p in range(PROTO_P_PER_GROUP):
            terms.append(TermDef(f"PROTO_{grp}_{_p}", GType[grp]))
    _bank_hi = (
        int(B_MAX) if bank_cap is None else max(1, min(int(B_MAX), int(bank_cap)))
    )
    terms.append(
        TermDef(
            "EmbBankIdx",
            GType.BANK_IDX,
            sampler=lambda: random.randrange(_bank_hi),
            bank_hi=_bank_hi,
        )
    )
    return terms


_UNSUPERVISED_EMB_OPS = frozenset(
    f
    for f in EMB_OPERATORS
    if f.name.startswith(("EmbNorm_", "EmbCos_", "EmbSub_", "EmbLag_"))
)


def build_emb_lean_functions() -> List[FuncDef]:
    import os

    _drop_unsup = os.environ.get("GP_LEAN_DROP_UNSUP", "1") not in (
        "0",
        "false",
        "False",
    )
    result: List[FuncDef] = []
    for f in ALL_FUNCTIONS:
        if f in (IN_REGIME, REGIME_IS):
            continue
        if _drop_unsup and f in _UNSUPERVISED_EMB_OPS:
            continue
        if f.name.startswith("EmbProj_"):
            group = f.name[len("EmbProj_") :].rsplit("_", 1)[0]
            k = int(f.name.rsplit("_", 1)[1])
            if k >= min(3, SERIALIZED_PROJECTION_COUNTS.get(group, 0)):
                continue
        result.append(f)
    return result


class Grammar:

    def __init__(
        self,
        *,
        functions: Sequence[FuncDef] | None = None,
        terminals: Sequence[TermDef] | None = None,
        max_depth: int = 6,
        max_nodes: int = 25,
    ):
        self.functions = list(functions or ALL_FUNCTIONS)
        self.terminals = list(terminals or build_terminal_set())
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        if _shape_router_disabled():
            self.functions = [f for f in self.functions if f.name != "IfThenElseShape"]
        elif IF_SHAPE not in self.functions:
            self.functions.append(IF_SHAPE)
        _have_terms = {t.name for t in self.terminals}
        self.terminals.extend(
            (t for t in _active_shape_terminals() if t.name not in _have_terms)
        )
        self._funcs_by_ret: Dict[GType, List[FuncDef]] = {}
        for f in self.functions:
            self._funcs_by_ret.setdefault(f.ret_type, []).append(f)
        self._terms_by_ret: Dict[GType, List[TermDef]] = {}
        for t in self.terminals:
            self._terms_by_ret.setdefault(t.ret_type, []).append(t)
        self._emb_proj_leaf_pairs: List[Tuple[FuncDef, TermDef]] = []
        for _f in self.functions:
            if _f.name.startswith("EmbProj_") and _f.arg_types:
                _raw = [
                    t
                    for t in self._terms_by_ret.get(_f.arg_types[0], [])
                    if not t.name.startswith("PROTO_")
                ]
                if _raw:
                    self._emb_proj_leaf_pairs.append((_f, _raw[0]))
        self._emb_dir_leaf_pairs: List[Tuple[FuncDef, TermDef]] = []
        if self._has_term(GType.BANK_IDX):
            for _f in self.functions:
                if _f.name.startswith("EmbDir_") and _f.arg_types:
                    _raw = [
                        t
                        for t in self._terms_by_ret.get(_f.arg_types[0], [])
                        if not t.name.startswith("PROTO_")
                    ]
                    if _raw:
                        self._emb_dir_leaf_pairs.append((_f, _raw[0]))
        self._emb_proto_leaf_pairs: List[Tuple[FuncDef, TermDef, TermDef]] = []
        if os.environ.get("GP_LEAN_PROTO_NONLINEAR", "0") not in (
            "0",
            "false",
            "False",
        ):
            _emb_ops_by_name = {f.name: f for f in EMB_OPERATORS}
            for _grp_t, _terms in self._terms_by_ret.items():
                _protos = [t for t in _terms if t.name.startswith("PROTO_")]
                if not _protos:
                    continue
                _cos = _emb_ops_by_name.get(f"EmbCos_{_grp_t.name}")
                if _cos is None:
                    continue
                _raw = next(
                    (t for t in _terms if not t.name.startswith("PROTO_")), None
                )
                if _raw is None:
                    continue
                for _p_term in _protos:
                    self._emb_proto_leaf_pairs.append((_cos, _raw, _p_term))
        self._has_emb_leaf = bool(
            self._emb_proj_leaf_pairs
            or self._emb_dir_leaf_pairs
            or self._emb_proto_leaf_pairs
        )
        self._emb_leaf_frac = float(os.environ.get("GP_EMB_LEAF_FRAC", "0.5"))
        self._min_depth = self._compute_min_depths()

    def _random_real_leaf(self, ret: GType) -> Node:
        if (
            ret == GType.REAL
            and self._has_emb_leaf
            and (random.random() < self._emb_leaf_frac)
        ):
            n_proj = len(self._emb_proj_leaf_pairs)
            n_dir = len(self._emb_dir_leaf_pairs)
            idx = random.randrange(n_proj + n_dir + len(self._emb_proto_leaf_pairs))
            if idx < n_proj:
                (fd, term) = self._emb_proj_leaf_pairs[idx]
                return FuncNode(defn=fd, children=[TermNode(defn=term)])
            if idx < n_proj + n_dir:
                (fd, term) = self._emb_dir_leaf_pairs[idx - n_proj]
                return FuncNode(
                    defn=fd,
                    children=[TermNode(defn=term), self._random_term(GType.BANK_IDX)],
                )
            (fd, emb_term, proto_term) = self._emb_proto_leaf_pairs[
                idx - n_proj - n_dir
            ]
            return FuncNode(
                defn=fd, children=[TermNode(defn=emb_term), TermNode(defn=proto_term)]
            )
        return self._random_term(ret)

    def _compute_min_depths(self) -> Dict[GType, int]:
        known: Dict[GType, int] = {t: 0 for t in self._terms_by_ret}
        changed = True
        while changed:
            changed = False
            for f in self.functions:
                if all((at in known for at in f.arg_types)):
                    cost = 1 + max((known[at] for at in f.arg_types), default=0)
                    if f.ret_type not in known or cost < known[f.ret_type]:
                        known[f.ret_type] = cost
                        changed = True
        return known

    def _has_term(self, t: GType) -> bool:
        return bool(self._terms_by_ret.get(t))

    def _random_term(self, t: GType) -> TermNode:
        return TermNode(defn=random.choice(self._terms_by_ret[t]))

    def _feasible_func(self, ret: GType, budget: int) -> Optional[FuncDef]:
        by_class: Dict[str, List[FuncDef]] = {}
        for f in self._funcs_by_ret.get(ret, []):
            if not all((self._min_depth.get(a, 99) < budget for a in f.arg_types)):
                continue
            if is_emblag_funcdef(f) and (not self._has_term(f.arg_types[0])):
                continue
            cls = _FUNC_TO_CLASS.get(f)
            if cls is None:
                cls = "_unclassified"
            by_class.setdefault(cls, []).append(f)
        if not by_class:
            return None
        chosen_class = random.choice(sorted(by_class.keys()))
        return random.choice(by_class[chosen_class])

    def _make_children_for(
        self, fd: FuncDef, md: int, gen_fn: Callable[[GType, int], Node]
    ) -> List[Node]:
        if is_emblag_funcdef(fd):
            first = self._random_term(fd.arg_types[0])
            rest = [gen_fn(a, md - 1) for a in fd.arg_types[1:]]
            return [first] + rest
        children = [gen_fn(a, md - 1) for a in fd.arg_types]
        if fd.name.startswith("EmbSub_") or fd.name.startswith("EmbCos_"):
            if (
                len(children) == 2
                and isinstance(children[0], TermNode)
                and isinstance(children[1], TermNode)
                and (children[0].name == children[1].name)
            ):
                emb_type = fd.arg_types[1]
                found_different = False
                for _ in range(5):
                    alt = self._random_term(emb_type)
                    if alt.name != children[0].name:
                        children[1] = alt
                        found_different = True
                        break
                if not found_different:
                    emb_lag_fd = next(
                        (
                            f
                            for f in self.functions
                            if f.name == f"EmbLag_{children[0].name}"
                        ),
                        None,
                    )
                    if emb_lag_fd is not None:
                        lag_term = TermNode(
                            defn=TermDef("EphInt", GType.INT),
                            value=random.choice(_LAG_CHOICES),
                        )
                        children[1] = FuncNode(
                            defn=emb_lag_fd,
                            children=[self._random_term(emb_type), lag_term],
                        )
        return children

    def grow(self, ret: GType, max_d: int | None = None) -> Node:
        md = max_d if max_d is not None else self.max_depth
        can_term = self._has_term(ret)
        fd = self._feasible_func(ret, md) if md > 0 else None
        if fd is None:
            return (
                self._random_real_leaf(ret) if can_term else self._force_func(ret, md)
            )
        if can_term and random.random() < 0.5:
            return self._random_real_leaf(ret)
        return FuncNode(defn=fd, children=self._make_children_for(fd, md, self.grow))

    def full(self, ret: GType, max_d: int | None = None) -> Node:
        md = max_d if max_d is not None else self.max_depth
        fd = self._feasible_func(ret, md) if md > 0 else None
        if fd is None:
            return (
                self._random_real_leaf(ret)
                if self._has_term(ret)
                else self._force_func(ret, md)
            )
        return FuncNode(defn=fd, children=self._make_children_for(fd, md, self.full))

    def _force_func(self, ret: GType, md: int) -> Node:
        extra = self._min_depth.get(ret, 1)
        effective_md = max(md, extra) + 1
        fd = self._feasible_func(ret, effective_md)
        if fd is None:
            raise ValueError(f"Cannot produce type {ret}")
        return FuncNode(
            defn=fd, children=self._make_children_for(fd, effective_md, self.grow)
        )

    def _gen_capped(self, ret: GType, md: int) -> Node:
        for _ in range(10):
            t = (self.full if random.random() < 0.5 else self.grow)(ret, md)
            if node_count(t) <= self.max_nodes and depth(t) <= self.max_depth:
                return t
        for _fb_attempt in range(5):
            _fb_depth = max(self._min_depth.get(ret, 0), 2) if _fb_attempt < 3 else 1
            fallback = self.grow(ret, max_d=_fb_depth)
            if (
                depth(fallback) <= self.max_depth
                and node_count(fallback) <= self.max_nodes
            ):
                return fallback
        return self._random_real_leaf(ret) if self._has_term(ret) else fallback

    def crossover(self, pa: Node, pb: Node) -> Tuple[Node, Node]:
        na = all_nodes_indexed(pa)
        nb = all_nodes_indexed(pb)
        b_idx: Dict[GType, List[Tuple[Node, Tuple[int, ...]]]] = {}
        for n, _, p in nb:
            b_idx.setdefault(n.ret_type, []).append((n, p))
        random.shuffle(na)
        for node_a, _, pa_path in na:
            matches = b_idx.get(node_a.ret_type)
            if not matches:
                continue
            _shuffled_matches = list(matches)
            random.shuffle(_shuffled_matches)
            for node_b, pb_path in _shuffled_matches[:5]:
                if _crossover_would_violate_emblag(
                    pa, pa_path, node_b
                ) or _crossover_would_violate_emblag(pb, pb_path, node_a):
                    continue
                c1 = _replace_at_path(deep_copy(pa), pa_path, deep_copy(node_b))
                c2 = _replace_at_path(deep_copy(pb), pb_path, deep_copy(node_a))
                if (
                    depth(c1) <= self.max_depth
                    and depth(c2) <= self.max_depth
                    and (node_count(c1) <= self.max_nodes)
                    and (node_count(c2) <= self.max_nodes)
                ):
                    return (c1, c2)
        return (deep_copy(pa), deep_copy(pb))

    def point_mutate(self, tree: Node, p: float = 0.1) -> Node:
        return self._pm(deep_copy(tree), p, parent_fd=None, arg_idx=None)

    def _pm(
        self, n: Node, p: float, parent_fd: Optional[FuncDef], arg_idx: Optional[int]
    ) -> Node:
        in_emblag_first_slot = (
            parent_fd is not None and arg_idx == 0 and is_emblag_funcdef(parent_fd)
        )
        if isinstance(n, TermNode):
            if random.random() >= p or not self._has_term(n.ret_type):
                return n
            if n.name == "EphReal" and random.random() < 0.7:
                new_val = n.value + random.gauss(0, 0.3)
                new_val = max(-3.0, min(3.0, new_val))
                return TermNode(defn=n.defn, value=round(new_val, 4))
            if n.name == "EmbBankIdx" and random.random() < 0.7:
                _hi = getattr(n.defn, "bank_hi", None) or B_MAX
                new_val = max(
                    0, min(int(_hi) - 1, int(n.value) + random.choice((-1, 1)))
                )
                return TermNode(defn=n.defn, value=new_val)
            return self._random_real_leaf(n.ret_type)
        if random.random() < p and (not in_emblag_first_slot):
            fd = self._feasible_func(n.ret_type, 3)
            if fd:
                return FuncNode(
                    defn=fd, children=self._make_children_for(fd, 3, self.grow)
                )
        n.children = [
            self._pm(c, p, parent_fd=n.defn, arg_idx=i)
            for (i, c) in enumerate(n.children)
        ]
        return n

    def subtree_mutate(self, tree: Node, max_d: int = 3) -> Node:
        ns = all_nodes_indexed(tree)
        cands = [(n, p) for (n, _, p) in ns if p]
        if not cands:
            return self.grow(tree.ret_type, max_d=max_d)
        (tgt, path) = random.choice(cands)
        if _path_is_emblag_first_child(tree, path):
            repl = self._random_term(tgt.ret_type)
        else:
            repl = self.grow(tgt.ret_type, max_d)
        new = _replace_at_path(deep_copy(tree), path, repl)
        return new if node_count(new) <= self.max_nodes else deep_copy(tree)
