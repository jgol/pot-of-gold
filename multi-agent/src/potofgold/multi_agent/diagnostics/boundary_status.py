"""Distinguish unattempted boundary parity checks from measured mismatches."""

PARITY_FAILURES = frozenset({"baseline_entry_signal_timing_mismatch", "zero_offset_entry_signal_mismatch",
                             "zero_offset_trade_replay_mismatch"})
STRUCTURAL_INAPPLICABILITY = frozenset({"nested_comparison_boundaries", "boundary_budget_exceeded",
    "unsupported_comparison_boundary", "no_level_comparison_boundary", "boolean_entry_tree_required",
    "outcome_dependent_entry_operand"})


def parity_presentation(control):
    """Normalize historical default-false metadata without changing the saved record.

    Only known early exits establish that no parity test ran. Unknown historical
    failures remain unspecified rather than being relabeled as a successful test.
    """
    out = dict(control or {})
    if out.get("baseline_parity_ok") is True:
        status = "passed"
    elif out.get("reason") in PARITY_FAILURES:
        status = "failed"
    elif out.get("reason") in STRUCTURAL_INAPPLICABILITY:
        status = "not_tested"
    else:
        status = out.get("baseline_parity_status", "not_recorded")
    out["baseline_parity_status"] = status
    out["baseline_parity_ok"] = {"passed": True, "failed": False}.get(status)
    if status == "not_tested":
        out["baseline_parity_note"] = "The diagnostic stopped before parity testing; no reconstruction mismatch was measured."
    return out
