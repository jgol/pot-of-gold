"""Check that cited numerical values match the supplied strategy evidence."""
from __future__ import annotations

from typing import Any, Dict


class MalformedVerdict(ValueError):
    """An assessment with missing, invalid or inconsistent citations."""


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _verify_cites(verdict: dict, cf_dict: dict, priors: dict) -> None:
    """Check citation identities and values against case fields or prior exhibits.

    Values must be numeric and agree within 5% relative or 0.05 absolute error.
    Separate checks exclude Boolean flags from decisive citations.
    """
    allowed: Dict[str, Any] = dict(cf_dict)
    for pr in priors.values():
        ev = pr.get("exhibit_values")
        for k, val in (ev if isinstance(ev, dict) else {}).items():
            allowed.setdefault(k, val)
    dmv = verdict.get("decisive_metric_values")
    if not isinstance(dmv, dict):                    # Malformed mappings are validation failures, not attribute errors.
        raise MalformedVerdict(
            f"decisive_metric_values must be an object {{exhibit_id: value}}, got "
            f"{type(dmv).__name__} {str(dmv)[:80]!r}")
    for k, cited in dmv.items():
        nv = _num(cited)
        if nv is None:
            raise MalformedVerdict(f"decisive cite '{k}'={cited!r} is not numeric — a decisive cite must be a "
                                   f"number/bool exhibit, never the rule text or a categorical label")
        if k not in allowed:
            raise MalformedVerdict(f"decisive cite '{k}' is not a case-file exhibit (fabricated)")
        na = _num(allowed[k])
        if na is None:
            raise MalformedVerdict(f"decisive cite '{k}'={cited} but exhibit {allowed[k]!r} is not a number (fabricated)")
        if abs(nv - na) > max(0.05, 0.05 * abs(na)):     # Allow numerical rounding in the model's response.
            raise MalformedVerdict(f"decisive cite '{k}'={cited} disagrees with exhibit {allowed[k]}")
