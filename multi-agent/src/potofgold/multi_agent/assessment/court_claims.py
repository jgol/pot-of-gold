"""Match defense assessments to the prosecution statements they address.

Quotation matching retains sentence boundaries and qualifications.
"""
from __future__ import annotations
from potofgold.multi_agent.assessment.prompts import DEFENSE_CLAIM_GUIDE

from copy import deepcopy
import re


_ABBREVIATIONS = re.compile(
    r"\b(?:e\.g\.|i\.e\.|vs\.|approx\.|etc\.|dr\.|mr\.|mrs\.|ms\.|prof\.|fig\.|no\.|st\.)"
    r"|\b(?:[A-Za-z]\.){2,}", re.IGNORECASE)
_CLOSERS = '\"\'\u201d\u2019)]}'


def _quote_boundaries(text: str) -> tuple[set[int], set[int]]:
    """Return conservative sentence/paragraph starts and ends in unmodified text."""
    if not text.strip():
        return set(), set()
    starts = {len(text) - len(text.lstrip())}
    ends = {len(text.rstrip())}
    protected = {index for match in _ABBREVIATIONS.finditer(text)
                 for index in range(match.start(), match.end()) if text[index] == "."}
    for index, character in enumerate(text):
        if character not in ".!?" or index in protected:
            continue
        if character == "." and index + 1 < len(text) and text[index + 1].isdigit():
            # Decimal points in numbers do not end a sentence.
            continue
        end = index + 1
        while end < len(text) and text[end] in ".!?" + _CLOSERS:
            end += 1
        if end < len(text) and not text[end].isspace():
            continue
        ends.add(end)
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
        if start < len(text):
            starts.add(start)
    for match in re.finditer(r"\n[ \t\r]*\n", text):
        end = match.start()
        while end and text[end - 1].isspace():
            end -= 1
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        if end:
            ends.add(end)
        if start < len(text):
            starts.add(start)
    return starts, ends


def _quote_span(quote, rationale: str) -> tuple[tuple[int, int] | None, str | None]:
    if not isinstance(quote, str) or not quote or quote != quote.strip():
        return None, "missing_or_malformed_prosecutor_quote"
    if len(quote) < 12 or len(re.findall(r"\w+", quote)) < 3:
        return None, "insubstantial_prosecutor_quote"
    starts, ends = _quote_boundaries(rationale)
    offset = rationale.find(quote)
    if offset < 0:
        return None, "prosecutor_quote_not_verbatim"
    while offset >= 0:
        end = offset + len(quote)
        if offset in starts and end in ends:
            return (offset, end), None
        offset = rationale.find(quote, offset + 1)
    return None, "prosecutor_quote_omits_sentence_context"


def _complete_quote(quote, rationale):
    """Find the complete source statement containing a quoted fragment."""
    if not isinstance(quote, str) or not quote:
        return None
    offset = rationale.find(quote)
    if offset < 0:
        return None
    starts, ends = _quote_boundaries(rationale)
    before = [start for start in starts if start <= offset]
    after = [end for end in ends if end >= offset + len(quote)]
    return rationale[max(before):min(after)] if before and after else None


def bind_defense_claims(defense, prosecution) -> tuple[dict, list[dict]]:
    """Check claim identities and return a corrected defense plus audit records.

    Original invalid assessments stay in the audit. Evidence and scope checks
    are applied separately before judging.
    """
    source = prosecution if isinstance(prosecution, dict) else {}
    rationale = source.get("rationale")
    rationale = rationale if isinstance(rationale, str) else ""
    alleged_class = source.get("alleged_class")
    unavailable = (source.get("brief_status") == "TECHNICAL_FAILURE"
                   or str(alleged_class).upper() == "UNAVAILABLE")
    asserted = (isinstance(alleged_class, str) and bool(alleged_class.strip())
                and alleged_class.strip().upper() != "NONE_FOUND")
    source_position = "PROSECUTION_UNAVAILABLE" if unavailable else "FAULT_ALLEGED" if asserted else "NO_FAULT_ASSERTED"
    submitted = defense if isinstance(defense, dict) else {}
    rows = submitted.get("allegation_assessments", [])
    corrections = []
    if not isinstance(rows, list):
        corrections.append({"assessment_index": None, "reason": "malformed_assessment_list",
                            "submitted_assessments": deepcopy(rows)})
        rows = []
    bound = []
    for index, raw in enumerate(rows):
        row = deepcopy(raw) if isinstance(raw, dict) else {}
        span, reason = _quote_span(row.get("prosecutor_quote"), rationale)
        if unavailable:
            reason = "missing_prosecution_brief"
        if reason is not None:
            corrections.append({"assessment_index": index, "reason": reason,
                                "submitted_assessment": deepcopy(raw)})
            if reason == "prosecutor_quote_omits_sentence_context":
                suggestion = _complete_quote(row.get("prosecutor_quote"), rationale)
                if suggestion:
                    corrections[-1]["suggested_prosecutor_quote"] = suggestion
            bound.append({
                "allegation": "No verified prosecution claim supplied.",
                "prosecutor_quote": "", "claim_scope": "UNKNOWN",
                "classification": "UNRESOLVED", "decisive_exhibit_ids": [],
                "basis": "The submitted quotation does not identify a complete verbatim prosecution claim.",
                "claim_binding": {"status": "UNBOUND", "source_position": source_position},
            })
            continue
        start, end = span
        # Use the prosecution's statement as the judge-visible claim identity.
        assessment = {key: deepcopy(row[key]) for key in (
            "claim_scope", "classification", "decisive_exhibit_ids", "context_exhibit_ids", "basis") if key in row}
        quote = rationale[start:end]
        assessment.update(allegation=quote, prosecutor_quote=quote,
            claim_binding={"status": "BOUND", "source_position": source_position,
                           "span_start": start, "span_end": end})
        bound.append(assessment)
    out = {
        "allegation_assessments": bound,
        "source_position": source_position,
        "prosecution_rationale": rationale,
        "verdict_on_claim": "UNRESOLVED",
        "rebuttal_rationale": "Assess only the bound quoted statements and their supplied source context.",
    }
    if isinstance(submitted.get("retention_case"), dict):
        out["retention_case"] = deepcopy(submitted["retention_case"])
    return out, corrections
