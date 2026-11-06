"""Verify training replay and bind run identity to the exact source and inputs."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from potofgold.multi_agent import runtime_source_paths


def assert_runner_ready(runner):
    """Require a successful training replay before assessing a strategy."""
    if not callable(runner):
        raise RuntimeError("factory returned no callable catalogue runner")
    readiness = getattr(runner, "readiness", None)
    if not isinstance(readiness, dict) or readiness.get("replay_checked") is not True:
        raise RuntimeError("catalogue runner has no verified training replay")
    if readiness.get("ready") is not True or readiness.get("replay_fidelity") != 1.0:
        raise RuntimeError(f"catalogue runner training replay failed: {readiness}")


def _file_hash(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def build_run_fingerprint(*, config, input_paths, repo_root, factory=None):
    """Identify the source, inputs and settings required to resume a run.

    Exclude limits and spending ceilings so an unchanged run can continue with
    a larger budget.
    """
    root = Path(repo_root)
    sources = runtime_source_paths(root)
    if factory is not None:
        try:
            location = inspect.getsourcefile(factory)
            if location:
                path = Path(location)
                key = str(path.relative_to(root) if path.is_relative_to(root) else path)
                sources[key] = path
        except (TypeError, OSError):
            pass
    source_hashes = {name: _file_hash(path) for name, path in sorted(sources.items())}
    result = {"schema_version": 2, "config": config,
            "inputs": {name: _file_hash(path) if path else None for name, path in sorted(input_paths.items())},
            "source_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()}
    hook = getattr(factory, "fingerprint_inputs", None)
    if hook is not None:
        if not callable(hook):
            raise ValueError("Catalogue factory fingerprint_inputs must be callable")
        extra = hook()
        if (not isinstance(extra, dict) or not isinstance(extra.get("files"), dict)
                or not isinstance(extra.get("metadata", {}), dict)):
            raise ValueError("Catalogue factory fingerprint must declare files and metadata")
        hashes = {}
        for name, path in sorted(extra["files"].items()):
            if not path or not Path(path).is_file():
                raise ValueError(f"Catalogue factory fingerprint input is missing: {name}")
            hashes[name] = {"path": str(Path(path).resolve()), "sha256": _file_hash(path)}
        result["factory_inputs"] = {"files": hashes, "metadata": extra.get("metadata", {})}
    return result
