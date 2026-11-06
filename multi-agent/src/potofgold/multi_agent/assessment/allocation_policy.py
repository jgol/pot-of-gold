"""Percentage recommendations shared by the court and single-model comparator."""

import math
from statistics import median

PERCENTAGE = "percentage_v1"
MATERIAL_SPREAD_PP = 25.0


def check_policy(policy):
    if policy != PERCENTAGE:
        raise ValueError(f"Unknown allocation policy: {policy!r}")
    return policy


def valid_percent(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 100)


def percent_action(value):
    if not valid_percent(value):
        raise ValueError("allocation_percent must be a finite number from 0 to 100")
    return "FULL" if value == 100 else "CASH" if value == 0 else "PARTIAL"


def percentage_vote_issue(value):
    if "allocation_percent" not in value or (value["allocation_percent"] is not None
                                             and not valid_percent(value["allocation_percent"])):
        return "allocation_percent", "Supply a finite percentage from 0 to 100, or null for deliberate abstention."
    if value["allocation_percent"] is not None and not (
            isinstance(value.get("allocation_rationale"), str) and value["allocation_rationale"].strip()):
        return "allocation_rationale", "Explain why this exposure is justified relative to higher and lower exposure."
    return None


def normalize_percentage_vote(value):
    issue = percentage_vote_issue(value)
    missing_reasoning = value.pop("_allocation_reasoning_missing", False)
    if issue or missing_reasoning:
        value.update(allocation_action=None, allocation_status="technical_failure", allocation_multiplier=None)
    elif value["allocation_percent"] is None:
        value.update(allocation_action="ABSTAIN", allocation_status="deliberate_abstention", allocation_multiplier=None)
    else:
        percent = float(value["allocation_percent"])
        value.update(allocation_percent=percent, allocation_action=percent_action(percent),
                     allocation_status="explicit", allocation_multiplier=percent / 100)


def explicit_percentages(panel):
    return [float(j["allocation_percent"]) for j in panel
            if j.get("allocation_status") == "explicit" and valid_percent(j.get("allocation_percent"))]


def aggregate_percentages(panel):
    votes = explicit_percentages(panel)
    fallback = None if len(votes) >= 2 else "fewer_than_two_valid_percentage_votes"
    percent = float(median(votes)) if fallback is None else 100.0
    return {"percent": percent, "multiplier": percent / 100, "action": percent_action(percent),
            "percent_votes": votes, "fallback_reason": fallback,
            "spread_pp": max(votes) - min(votes) if votes else None}
