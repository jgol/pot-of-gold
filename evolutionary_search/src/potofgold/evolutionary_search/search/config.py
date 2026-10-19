"""Explicit settings for the released five-fold strategy-search experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import os
from pathlib import Path
import sys


# These switches identify the recorded experiment, including settings that were
# absent from its historical Git snapshot. Ambient GP_* variables are ignored.
EXPERIMENT_FLAGS = {
    "GP_MIN_COVERAGE": "0",
    "GP_CALENDAR_TAU": "1",
    "GP_ENTRY_EXCLUDE_DAY_CONSTANT": "1",
    "GP_ONESIDED_TRIM": "0",
    "GP_OBJ1_RAW_MEAN": "1",
    "GP_COVERAGE_MATCHED_TEST": "1",
    "GP_THRESHOLD_GRID": "1",
    "GP_THRESHOLD_LANDMARK": "1",
    "GP_PARSIMONY_TOURNAMENT": "1",
    "GP_GENOME_DEDUP": "1",
    "GP_CIA_K": "2",
    "GP_CIA_CHAMP_BAND": "0.0",
    "GP_LEAN_PROTO_NONLINEAR": "1",
    "GP_MARK_IV_VIX_MULT": "4",
    "GP_SHAPES_ALLOWED": "put_vertical,call_vertical,iron_condor,iron_butterfly,long_strangle",
    "GP_DISABLE_CALENDAR": "1",
    "GP_DISABLE_SHAPE_ROUTER": "1",
    "GP_OBJ1_EXCESS_BASELINE": "1",
    "GP_ENTRY_ROBUST": "1",
    "GP_ENTRY_MAX_CONJ": "2",
    "GP_ENTRY_EXCLUDE_PROTO_CONST": "1",
    "GP_EMB_LEAF_FRAC": "0.5",
    "GP_EMB_GEN0_FLOOR": "0.5",
    "GP_CVAR_TAIL_MAX_AUTO": "1",
    "GP_WING_IV_FIX": "1",
    "GP_CONCENTRATION_CAP": "0.04",
    "GP_SIZING_CREDIT_FILL": "0.25",
    "GP_DEBIT_UW_MARK_DECAY": "1.2",
}


@dataclass(frozen=True)
class SearchConfig:
    population_size: int = 256
    generations: int = 100
    n_folds: int = 5
    split_seed: int = 0
    minimum_trades: int = 20
    entry_start_bar: int = 60
    max_tree_depth: int = 6
    max_tree_nodes: int = 25
    evaluation_workers: int = 1
    save_checkpoints: bool = True
    checkpoint_every: int = 5

    def __post_init__(self):
        positive = (
            "population_size",
            "generations",
            "n_folds",
            "minimum_trades",
            "max_tree_depth",
            "max_tree_nodes",
            "evaluation_workers",
            "checkpoint_every",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.population_size < 2 or self.n_folds < 2:
            raise ValueError("population_size and n_folds must be at least 2")
        for name in ("split_seed", "entry_start_bar"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.save_checkpoints, bool):
            raise ValueError("save_checkpoints must be true or false")

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path | None = None) -> SearchConfig:
    if path is None:
        return SearchConfig()
    import yaml

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Search configuration does not exist: {path}")
    values = yaml.safe_load(path.read_text()) or {}
    if not isinstance(values, dict):
        raise ValueError("Search configuration must be a YAML mapping")
    unknown = set(values) - {f.name for f in fields(SearchConfig)}
    if unknown:
        raise ValueError(f"Unknown search settings: {', '.join(sorted(unknown))}")
    return SearchConfig(**values)


_active_settings = None


def ensure_reproducible_process(module):
    """Start CLI execution with the experiment's hash seed and numerical threads."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = "0"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[name] = "1"
        os.execve(
            sys.executable, [sys.executable, "-m", module, *sys.argv[1:]], environment
        )


def configure_runtime(config: SearchConfig, checkpoint_dir: Path | None = None) -> dict:
    """Configure import-time numerical settings before loading the search modules."""
    global _active_settings
    settings = dict(EXPERIMENT_FLAGS)
    if config.save_checkpoints and checkpoint_dir is not None:
        settings["GP_CHECKPOINT_DIR"] = str(checkpoint_dir.resolve())
        settings["GP_CHECKPOINT_EVERY"] = str(config.checkpoint_every)
    numerical = {k: v for k, v in settings.items() if not k.startswith("GP_CHECKPOINT")}
    if _active_settings is None:
        loaded = [
            name
            for name in sys.modules
            if name.startswith(__package__ + ".")
            and name.rsplit(".", 1)[-1]
            in {"grammar", "simulator", "fitness", "evolution"}
        ]
        if loaded:
            raise RuntimeError(
                "Configure search settings before importing the numerical runtime; use the search or replay command in a fresh process."
            )
    elif _active_settings != numerical:
        raise RuntimeError(
            "A process cannot change numerical search settings after importing the runtime"
        )
    for name in list(os.environ):
        if name.startswith("GP_"):
            del os.environ[name]
    os.environ.update(settings)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    _active_settings = numerical
    return settings
