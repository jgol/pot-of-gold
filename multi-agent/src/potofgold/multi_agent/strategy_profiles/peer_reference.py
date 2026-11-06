"""Build fixed peer distributions from training measurements for assessment."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "frozen-cause-peer-reference-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def numeric_case(cf):
    return {k: float(v) for k, v in cf.to_llm_dict().items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}


def make_reference(case_files, *, arm, selection_sha256, source_identity):
    """Freeze numeric training measurements for a specified peer population."""
    ids = [cf.champion_id for cf in case_files]
    if not ids or len(ids) != len(set(ids)) or any(cf.arm != arm for cf in case_files):
        raise ValueError("Peer reference needs unique, nonempty, same-arm membership")
    doc = {"schema": SCHEMA, "arm": arm, "selection_sha256": selection_sha256,
           "source_identity": deepcopy(source_identity),
           "members": {cf.champion_id: numeric_case(cf) for cf in sorted(case_files, key=lambda c: c.champion_id)}}
    doc["content_sha256"] = _digest(doc)
    return doc


def load_reference(path, *, expected_sha256, arm, cases):
    """Verify the reference hash and candidate measurements, then compute peer summaries."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Frozen peer reference file changed")
    doc = json.loads(data)
    if (doc.get("schema") != SCHEMA or doc.get("arm") != arm
            or doc.get("content_sha256") != _digest({k: v for k, v in doc.items() if k != "content_sha256"})
            or not isinstance(doc.get("members"), dict) or not doc["members"]):
        raise ValueError("Invalid frozen peer reference")
    for cf in cases:
        if doc["members"].get(cf.champion_id) != numeric_case(cf):
            raise ValueError("Candidate training exhibits differ from the frozen peer reference: " + cf.champion_id)
    for row in doc["members"].values():
        if not isinstance(row, dict) or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                                           or not math.isfinite(v) for v in row.values()):
            raise ValueError("Peer reference must contain finite numeric cause summaries only")
    from potofgold.multi_agent.assessment.court_compute import build_cohort_stats
    return build_cohort_stats(list(doc["members"].values())), sorted(doc["members"])
