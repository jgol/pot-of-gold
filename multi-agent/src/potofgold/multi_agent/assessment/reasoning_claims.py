"""Check specific scenario claims against the supplied numerical measurements."""
from __future__ import annotations

import re
from collections.abc import Iterator

_SKIP_FIELDS = frozenset({
    "prosecutor_quote", "prosecution_rationale", "judge_quote", "decisive_metric_values",
    "risk_citations", "metric_values", "citation_values", "admission_notes",
})
_SPLIT = re.compile(r"(?<=[.!?])\s+(?!\d)|\n+|;|\bbut\b", re.I)
_SCENARIOS = r"(?:all|every)\s+(?:(?:of|the|tested|training|cost|boundary|stress)\s+){0,6}scenarios?"
_LOSSES = r"(?:los(?:e|es|ing) money|(?:net|pnl|profit)\s+(?:is |are |remains? )?negative|(?:is |are )?loss[- ]making)"
_ALL_LOSE = re.compile(rf"(?:{_LOSSES}.{{0,45}}?{_SCENARIOS}|{_SCENARIOS}.{{0,45}}?{_LOSSES})", re.I)
_ONLY_BASELINE = re.compile(
    r"\bnet\s+(?:is |remains? )?positive\s+only\s+(?:at|for|in)\s+"
    r"(?:(?:the|exact|selected|cost|boundary)\s+){0,4}baseline(?:\s+cost/boundary)?", re.I)
_SAME_INTERVENTION = re.compile(
    r"\bshare\s+(?:the\s+)?same\s+(?:entry[- ]rule|entry[- ]threshold|boundary)\s+intervention", re.I)
_QUALIFIED = re.compile(
    r"\b(?:if|hypothetically|not|never|cannot|can't|doesn't|don't|"
    r"incorrect|inaccurate|overstated|unsupported|unjustified|reject(?:s|ed)?|deny|denies)\b", re.I)
_CRITICISM_AFTER = re.compile(
    r"^[\s\"'’”)]*(?:(?:is|was|would be)\s+)?(?:incorrect|inaccurate|false|overstated|unsupported|unjustified)\b", re.I)


def _asserted(clause, match):
    # Scope negation to the statement being checked.
    return not (_QUALIFIED.search(clause[max(0, match.start() - 90):match.start()])
                or _CRITICISM_AFTER.search(clause[match.end():]))


def _prose_strings(value: object, path: str = "output") -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            name = str(key)
            if name in _SKIP_FIELDS or name.endswith(("_id", "_ids")):
                continue
            yield from _prose_strings(child, f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _prose_strings(child, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def validate_scenario_claims(output: dict, scenario_context: list[dict]) -> None:
    """Reject universal claims contradicted by an admitted scenario measurement.

    Quoted, qualified and hypothetical statements remain part of agent reasoning.
    """
    if not scenario_context:
        return
    for path, prose in _prose_strings(output):
        for clause in _SPLIT.split(prose):
            universal = _ALL_LOSE.search(clause)
            baseline_only = _ONLY_BASELINE.search(clause)
            if (universal or baseline_only) and _asserted(clause, universal or baseline_only):
                scope = (universal or baseline_only).group().lower()
                cost, boundary = "cost" in scope, "boundary" in scope
                families = ({"cost"} if cost and not boundary else {"boundary"} if boundary and not cost
                            else {"cost", "boundary"})
                for group in scenario_context:
                    if group["family"] not in families:
                        continue
                    for scenario in group["scenarios"]:
                        net = scenario.get("net")
                        if net and (net["value"] >= 0 if universal else net["value"] > 0):
                            raise ValueError(
                                f"SCENARIO_CONTRADICTION: {path}: {net['exhibit_id']} = {net['value']} "
                                f"in the {group['family']} grid contradicts {repr((universal or baseline_only).group())}. "
                                "Limit the statement to the scenarios it describes and acknowledge the counterexample. "
                                "This correction requires no particular allocation or forecast.")
            same_intervention = _SAME_INTERVENTION.search(clause)
            if ("cost" in clause.lower() and "boundary" in clause.lower()
                    and same_intervention and _asserted(clause, same_intervention)
                    and {g["family"] for g in scenario_context} == {"cost", "boundary"}):
                raise ValueError(
                    f"SCENARIO_CONTRADICTION: {path}: cost multipliers and entry-boundary offsets are different "
                    "interventions on shared training data. Boundary stability and boundary economics share "
                    "the entry intervention. Shared data do not make the tests statistically independent.")
