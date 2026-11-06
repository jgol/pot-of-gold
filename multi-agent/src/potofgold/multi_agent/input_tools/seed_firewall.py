"""Assign strategies to the original H3 development and evaluation populations.

The split uses seed-2 strategy IDs and keeps pilot strategies in development.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from importlib.resources import files
from typing import Any, Dict, Optional

MANIFEST_PATH = str(files("potofgold.multi_agent.reference").joinpath("seed_firewall.json"))

_CACHE: Dict[str, Any] = {}


def _assert_manifest_invariants(m: Dict[str, Any]) -> None:
    """Validate the split settings and required pilot exclusions."""
    if m.get("engine_sha") != "b304782":
        raise AssertionError(f"[firewall] manifest engine_sha != b304782: {m.get('engine_sha')!r}")
    if str(m.get("cohort_seed")) != "2":
        raise AssertionError(f"[firewall] manifest cohort_seed != 2: {m.get('cohort_seed')!r}")
    fdi = m.get("forced_dev_ids")
    if not isinstance(fdi, list) or not fdi:
        raise AssertionError("[firewall] forced_dev_ids missing/empty — the exposed champions would leak into test")
    dp = m.get("dev_pct")
    if isinstance(dp, bool) or not isinstance(dp, int) or not (0 < dp < 100):   # Python treats Boolean values as integers.
        raise AssertionError(f"[firewall] dev_pct must be an int in (0,100): {dp!r}")
    if not m.get("hash_salt"):
        raise AssertionError("[firewall] hash_salt missing")


def load_manifest(path: str = MANIFEST_PATH) -> Dict[str, Any]:
    if path not in _CACHE:
        m = json.loads(Path(path).read_text())
        _assert_manifest_invariants(m)
        _CACHE[path] = m
    return copy.deepcopy(_CACHE[path])   # Return a copy so callers cannot alter the cached manifest.


def _load_canonical_ids(m: Dict[str, Any]) -> Optional[set]:
    """Load the original cohort membership from the extracts or bundled ID list."""
    files = m.get("canonical_cohort_files")
    if not files:
        return None
    ids: set = set()
    for f in files:
        if not Path(f).is_file():
            # The bundled IDs replace the original research-extract paths.
            from importlib.resources import files as package_files
            saved = json.loads(package_files("potofgold.multi_agent.reference")
                               .joinpath("canonical_membership.json").read_text())
            if f in saved:
                ids.update(saved[f]["champion_ids"])
                continue
        d = json.loads(Path(f).read_text())
        rows = d if isinstance(d, list) else (d.get("champions") or d.get("members") or [])
        ids |= {str(r["champion_id"]) for r in rows if r.get("champion_id") is not None}
    if not ids:
        raise AssertionError(f"[firewall] canonical_cohort_files {files} yielded 0 champion_ids — refusing to "
                             f"restrict the cohort to nothing (wrong file shape or empty extract)")
    return ids


def firewall_manifest_id(manifest: Optional[Dict[str, Any]] = None) -> str:
    """Hash the split configuration and cohort membership to identify the partition."""
    m = manifest or load_manifest()
    key = {k: m.get(k) for k in ("engine_sha", "cohort_seed", "hash_salt", "dev_pct")}
    key["forced_dev_ids"] = sorted(str(x) for x in (m.get("forced_dev_ids") or []))
    # Hash membership as well as filenames so replacement data changes the split ID.
    try:
        ids = _load_canonical_ids(m)
        key["canonical_membership"] = sorted(ids) if ids else None
    except Exception:                                    # noqa: BLE001
        key["canonical_membership_files"] = sorted(str(x) for x in (m.get("canonical_cohort_files") or []))
    return "fw_" + hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _hash_pct(cid: str, salt: str) -> int:
    """Map a strategy ID to a reproducible bucket in [0, 100) using salted SHA-1."""
    return int(hashlib.sha1(f"{salt}:{cid}".encode()).hexdigest(), 16) % 100


def firewall_role(champion_id: str, manifest: Optional[Dict[str, Any]] = None) -> str:
    """Return development membership for forced IDs or hash buckets below dev_pct."""
    m = manifest or load_manifest()
    cid = str(champion_id)
    if cid in set(m.get("forced_dev_ids") or []):
        return "dev"
    return "dev" if _hash_pct(cid, m["hash_salt"]) < int(m["dev_pct"]) else "test"
