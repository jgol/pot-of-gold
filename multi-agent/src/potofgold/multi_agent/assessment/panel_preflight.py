"""Prepare and check selected strategies before making model calls."""
from copy import deepcopy

from potofgold.multi_agent.assessment.run_readiness import assert_runner_ready

SCHEMA = "court-panel-preflight-v1"


def check_all(cases, members, factory, check_packet, *, fingerprint):
    if not callable(factory):
        raise ValueError("Preflight requires a catalogue runner factory")
    if not cases:
        raise ValueError("Preflight requires a nonempty selected cohort")
    if getattr(factory, "cohort_scope", None) == "development_only":
        raise ValueError("The saved-development factory cannot prepare a full held-out run; select a training-only "
                         "runner covering the evaluation champions")
    rows = []
    for case in cases:
        runner = None
        try:
            runner = factory(members[case.champion_id])
            assert_runner_ready(runner)
            detail = check_packet(case, runner)
            if not isinstance(detail, dict) or detail.get("ready") is not True:
                raise ValueError("The training packet did not report verified readiness")
            rows.append({"champion_id": case.champion_id, "status": "ready", "measurements": detail})
        except Exception as exc:
            rows.append({"champion_id": case.champion_id, "status": "failed",
                         "error": f"{type(exc).__name__}: {exc}"})
        finally:
            # Reuse the factory's fold cache instead of retaining each strategy's preparation.
            del runner
    return {"schema": SCHEMA, "status": "ready" if all(r["status"] == "ready" for r in rows) else "failed",
            "provider_calls": 0,
            "run_fingerprint": deepcopy(fingerprint),
            "champion_ids": sorted(c.champion_id for c in cases),
            "n_attempted": len(rows), "n_ready": sum(r["status"] == "ready" for r in rows), "members": rows}


def verify_receipt(receipt, cases, fingerprint):
    if not isinstance(receipt, dict):
        raise ValueError("Preflight receipt must be an object")
    ids = sorted(c.champion_id for c in cases)
    rows = receipt.get("members", [])
    if (receipt.get("schema") != SCHEMA or receipt.get("status") != "ready"
            or receipt.get("run_fingerprint") != fingerprint or receipt.get("champion_ids") != ids
            or receipt.get("n_attempted") != len(ids) or receipt.get("n_ready") != len(ids)
            or not ids or len(ids) != len(set(ids)) or not isinstance(rows, list)
            or any(not isinstance(r, dict) or r.get("status") != "ready" for r in rows)
            or sorted(r.get("champion_id", "") for r in rows) != ids):
        raise ValueError("Preflight receipt does not cover this exact cohort, source, inputs and policy; rerun --preflight-only")
