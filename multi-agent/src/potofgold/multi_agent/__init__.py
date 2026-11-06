"""Multi-agent assessment and portfolio evaluation for 0DTE strategy research."""

from importlib.resources import files
from pathlib import Path


def runtime_source_paths(root=None):
    """Files that identify court execution, including the shared GP implementation."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    own = [*root.rglob("*.py"), *(root / "reference").glob("*.json"),
           *(root / "diagnostics").glob("*.csv")]
    shared = Path(str(files("potofgold.evolutionary_search")))
    paths = {str(p.relative_to(root)): p for p in sorted(own)}
    paths.update({"evolutionary_search/" + str(p.relative_to(shared)): p
                  for p in sorted((shared / "search").glob("*.py"))})
    return paths
