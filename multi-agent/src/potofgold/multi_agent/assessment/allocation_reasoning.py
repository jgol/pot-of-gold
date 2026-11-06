"""Organize allocation explanations and requests for further measurements."""

from copy import deepcopy
import re
from potofgold.multi_agent.assessment.allocation_policy import PERCENTAGE, check_policy, valid_percent

ALLOCATION_LABELS = {"FULL": "Full", "PARTIAL": "Partial allocation", "CASH": "No position", "ABSTAIN": "Abstain"}


def allocation_label(action, percent=None):
    if action == "PARTIAL" and valid_percent(percent):
        return f"{percent:g}% allocation"
    return ALLOCATION_LABELS.get(action, "Unspecified")


INVESTIGATION_SCHEMA = {'type': 'array',
 'maxItems': 3,
 'items': {'type': 'object',
           'properties': {'question': {'type': 'string'},
                          'test_id': {'type': 'string'},
                          'requested_observation': {'type': 'string',
                                                    'description': 'Exact observation capability needed; use '
                                                                   'an unavailable experiment name when no '
                                                                   'offered tool can supply it.'},
                          'existing_evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
                          'missing_observation': {'type': 'string'},
                          'source_quote': {'type': 'string'},
                          'interpretation_error': {'type': 'string'},
                          'allocation_percent_if_concern_confirmed': {'type': 'number',
                                                                      'minimum': 0,
                                                                      'maximum': 100},
                          'allocation_percent_if_concern_weakened': {'type': 'number',
                                                                     'minimum': 0,
                                                                     'maximum': 100}},
           'required': ['question',
                        'test_id',
                        'requested_observation',
                        'existing_evidence_ids',
                        'missing_observation',
                        'allocation_percent_if_concern_confirmed',
                        'allocation_percent_if_concern_weakened'],
           'additionalProperties': False}}


def investigation_schema(allocation_policy=PERCENTAGE):
    check_policy(allocation_policy)
    return deepcopy(INVESTIGATION_SCHEMA)


def allocation_investigations(panel, session, case, sources, allocation_policy=PERCENTAGE):
    """Route specific unresolved questions to available measurements or reference checks."""
    result = []
    for judge in panel:
        requests = judge.get("allocation_investigations")
        for request in (requests[:3] if isinstance(requests, list) else []):
            if not isinstance(request, dict):
                continue
            item = {k: deepcopy(v) for k, v in request.items()
                    if k in investigation_schema(allocation_policy)["items"]["properties"]}
            item["model"] = judge.get("_judge_model", judge.get("model"))
            actions = tuple(item.get(f"allocation_percent_if_concern_{ending}")
                            for ending in ("confirmed", "weakened"))
            valid_actions = all(valid_percent(a) for a in actions)
            cites = item.get("existing_evidence_ids")
            reason = None
            if not all(isinstance(item.get(k), str) and item[k].strip() for k in
                       ("question", "test_id", "missing_observation")):
                reason = "incomplete_question"
            elif not valid_actions or actions[0] == actions[1]:
                reason = "no_allocation_consequence"
            elif not isinstance(cites, list) or not cites or any(not isinstance(k, str) or k not in case for k in cites):
                reason = "unbound_existing_evidence"
            elif item["test_id"] == "existing_evidence":
                quote = item.get("source_quote", "")
                if (not isinstance(quote, str) or len(quote.strip()) < 20
                        or not any(quote.strip() == text.strip() for text in sources if isinstance(text, str))
                        or not isinstance(item.get("interpretation_error"), str)
                        or not item["interpretation_error"].strip()):
                    reason = "unbound_interpretation_correction"
            else:
                reason = session.question_status(item["test_id"], item.get("requested_observation"))
            item.update(status="actionable" if reason is None else "recorded_limitation", reason=reason)
            result.append(item)
    return result


def allocation_details(panel):
    """Retain each allocation vote with reasons, counterarguments and citations."""
    keys = ("model", "allocation_action", "allocation_percent", "allocation_multiplier", "allocation_status", "allocation_rationale", "allocation_exhibit_ids",
            "allocation_metric_values", "allocation_competing_explanation", "allocation_limitations",
            "prediction_limitations", "allocation_investigations")
    return [{**{k: deepcopy(j.get(k)) for k in keys if k in j},
             **({"allocation_label": allocation_label(j["allocation_action"], j.get("allocation_percent"))} if "allocation_action" in j else {})}
            for j in panel or []]


def sentence_summary(text, *, max_chars=600):
    """Return complete sentences, including a long first sentence when necessary."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", str(text or "").strip())
    chosen = []
    for part in parts:
        if chosen and len(" ".join(chosen + [part])) > max_chars:
            break
        if part:
            chosen.append(part)
    return " ".join(chosen)


def investigation_queue(verdict):
    """Group related investigation requests while retaining their full questions."""
    requests = (getattr(verdict, "disagreement", {}) or {}).get("allocation_investigations", [])
    groups = {}
    for request in requests:
        if request.get("status") == "reviewed":
            continue
        key = (request.get("test_id"), request.get("status"))
        item = groups.setdefault(key, {"item": "allocation_investigation", "test_id": key[0], "status": key[1],
                                     "questions": [], "existing_evidence_ids": []})
        question = {k: request.get(k) for k in ("question", "missing_observation", "reason")}
        for key_percent in ("allocation_percent_if_concern_confirmed", "allocation_percent_if_concern_weakened"):
            if key_percent in request:
                question[key_percent] = request[key_percent]
                item.setdefault("possible_allocation_percentages", [])
                if request[key_percent] not in item["possible_allocation_percentages"]:
                    item["possible_allocation_percentages"].append(request[key_percent])
        if question not in item["questions"]:
            item["questions"].append(question)
        for cite in request.get("existing_evidence_ids", []):
            if cite not in item["existing_evidence_ids"]:
                item["existing_evidence_ids"].append(cite)
    queue = sorted(groups.values(), key=lambda q: (q["status"] != "actionable", str(q["test_id"])))[:3]
    if not queue:
        for point in ((getattr(verdict, "analysis", None) or {}).get("unresolved_points") or [])[:3]:
            queue.append({"item": "unresolved_question", "question": point.get("blocking_point", point.get("question")),
                          "missing_observation": point.get("reason"), "status": "recorded_limitation"})
    allocation = (getattr(verdict, "disagreement", {}) or {}).get("allocation", {})
    reason = getattr(verdict, "allocation_fallback_reason", None) or allocation.get("fallback_reason")
    if reason:
        queue.insert(0, {"item": "allocation_fallback", "reason": reason, "status": "review_action_votes",
                         "status_counts": allocation.get("status_counts", {})})
    return queue[:4]


def finish_investigations(verdict, preloaded_records):
    """Move completed investigations from the pending queue into the audit."""
    original_ids = {r.get("result_id") for r in preloaded_records}
    measured = {r.get("test_id") for r in getattr(verdict, "tool_records", [])
                if r.get("result_id") not in original_ids and r.get("measured_statistics")}
    disagreement = verdict.disagreement
    corrected = bool((verdict.analysis or {}).get("interpretation_correction_bindings"))
    for request in disagreement.get("allocation_investigations", []):
        if request.get("status") == "actionable" and (
                request.get("test_id") in measured or (request.get("test_id") == "existing_evidence" and corrected)):
            request["status"] = "reviewed" if disagreement.get("passes", 1) > 1 else "measured_pending_review"
