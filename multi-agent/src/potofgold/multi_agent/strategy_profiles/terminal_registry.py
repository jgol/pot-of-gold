"""Descriptions and units for GP variables, operators, embedding readouts and structures."""
import re
from typing import Optional, Sequence

from potofgold.multi_agent.strategy_profiles import terminal_meanings as _tm

# Reuse the variable descriptions shown in strategy profiles.
TERMINAL_MEANINGS = {k: (v[1], v[2]) for k, v in _tm.TERMINALS.items()}
POSITION_STATE_MEANINGS = {k: (v[1], v[2]) for k, v in _tm.POSITION_STATE.items()}

OPERATOR_GLOSS = {
    "LT": "a < b", "GT": "a > b", "LTE": "a <= b", "GTE": "a >= b", "EQ": "a == b",
    "AND": "both", "OR": "either", "NOT": "negation",
    "Add": "a + b", "Sub": "a - b", "Mul": "a * b", "Div": "a / b",
    "Abs": "absolute value", "Neg": "negate", "Min": "minimum", "Max": "maximum", "Sqrt": "square root",
    "Delta": "change in X over N bars (temporal difference)",
    "CrossAbove": "fires the bar a rises above b", "CrossBelow": "fires the bar a falls below b",
    "IfThenElse": "a if cond else b", "IfThenElseShape": "per-bar shape router (cond ? shapeA : shapeB)",
    "IfSide": "branch by option side (cond ? sideA : sideB)",
    "Lag": "the value n bars ago",
    "WeightedSum3": "weighted sum of 3 inputs", "WeightedSum5": "weighted sum of 5 inputs",
    "InRegime": "true if the current vol-regime is in the given set", "RegimeIs": "current vol-regime == X",
}

REGIME_LITERALS = {k: v[1] for k, v in _tm.REGIME_LITERALS.items()}
SIDE_LITERALS = {k: v[1] for k, v in _tm.SIDE_LITERALS.items()}

CONST_GLOSS = {
    "EphReal": "an evolved real constant — a threshold, in the normalized (z) units of the terminal it is compared against",
    "EphInt": "an evolved integer constant (e.g. a lag or count)",
}

EMB_FAMILY_GLOSS = {
    "EMB_FLOW_AGG": "aggregated order-flow embedding",
    "EMB_FLOW_RAW": "raw order-flow embedding",
    "EMB_GRID": "full option-grid embedding",
    "EMB_GRID_ATM": "ATM-region option-grid embedding",
    "EMB_GRID_CALL": "call-side option-grid embedding",
    "EMB_GRID_PUT": "put-side option-grid embedding",
    "EMB_SHARED": "shared/global market-state embedding",
    "EMB_SPX": "SPX price/return embedding",
    "EMB_STRIKE": "per-strike embedding",
    "EMB_VAR_ATM_IV": "ATM-IV variation embedding",
    "EMB_VAR_ATM_SPREAD": "ATM-spread variation embedding",
    "EMB_VAR_SPX_RET": "SPX-return variation embedding",
    "EMB_VIX": "VIX / vol-term embedding",
}

_EMB_OP_GLOSS = {
    "EmbProj": "learned projection of the {fam} (unitless)",
    # EmbDir is a centered continuous projection onto a direction-bank row.
    "EmbDir": "centered continuous projection of the {fam} onto a GP-selected supervised direction-bank row (signed, unitless)",
    "EmbCos": "cosine similarity of the {fam} to a prototype (-1..1)",
    "EmbNorm": "magnitude (vector length) of the {fam}",
    "EmbSub": "difference of two {fam} vectors",
    "EmbLag": "the {fam} lagged n bars",
    "PROTO": "frozen prototype centroid of the {fam} (the reference EmbCos compares against)",
}

SHAPE_GLOSS = {
    "Shape_iron_butterfly": "short iron butterfly (defined-risk, short ATM + wings)",
    "Shape_iron_condor": "short iron condor (defined-risk, short OTM strangle + wings)",
    "Shape_long_strangle": "long strangle (long OTM call+put, debit)",
    "Shape_call_vertical": "call vertical spread (defined-risk)",
    "Shape_put_vertical": "put vertical spread (defined-risk)",
    "Shape_put_ratio_backspread": "put ratio backspread (net-long puts, undefined-risk tail)",
    "Shape_call_ratio_backspread": "call ratio backspread (net-long calls, undefined-risk tail)",
}

_EMB_TOKEN = re.compile(r"^(EmbProj|EmbDir|EmbCos|EmbNorm|EmbSub|EmbLag|PROTO)_(EMB_[A-Z_]+?)(?:_(\d+))?$")


def _emb_describe(token: str, basis_targets: Sequence[str] = ()) -> Optional[str]:
    # Keep basis_targets in the signature; the case builder adds the target list.
    if token == "EmbBankIdx":
        return ("an evolved index selecting WHICH supervised direction-bank row EmbDir projects onto "
                "(not a signal itself, and not an encoder/instance selector)")
    if token in EMB_FAMILY_GLOSS:
        return EMB_FAMILY_GLOSS[token]
    m = _EMB_TOKEN.match(token)
    if not m:
        return None
    op, fam, _k = m.group(1), m.group(2), m.group(3)
    if fam not in EMB_FAMILY_GLOSS:
        return None
    return _EMB_OP_GLOSS[op].format(fam=EMB_FAMILY_GLOSS[fam])


def describe(token: str, basis_targets: Sequence[str] = ()) -> Optional[str]:
    """Return the description of a registered token, or None if it is unknown."""
    if token in TERMINAL_MEANINGS:
        meaning, unit = TERMINAL_MEANINGS[token]
        return f"{meaning} [{unit}]"
    if token in POSITION_STATE_MEANINGS:
        meaning, unit = POSITION_STATE_MEANINGS[token]
        return f"{meaning} [{unit}]"
    if token in OPERATOR_GLOSS:
        return OPERATOR_GLOSS[token]
    if token in CONST_GLOSS:
        return CONST_GLOSS[token]
    if token in SHAPE_GLOSS:
        return SHAPE_GLOSS[token]
    if token in REGIME_LITERALS:
        return REGIME_LITERALS[token]
    if token in SIDE_LITERALS:
        return SIDE_LITERALS[token]
    return _emb_describe(token, basis_targets)


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def identifiers(*trees: str) -> set:
    out = set()
    for t in trees:
        out.update(_IDENT.findall(str(t or "")))
    return out


def assert_covers(tokens: Sequence[str]) -> bool:
    """Raise an error for any token without a registered description."""
    missing = sorted(t for t in set(tokens) if describe(t) is None)
    if missing:
        raise ValueError(f"terminal_registry: {len(missing)} identifier(s) have no description "
                         f"(add one line each): {missing}")
    return True
