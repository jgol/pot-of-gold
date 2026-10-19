"""Prepare fold inputs and persist the fitted embedding readouts."""

from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .grammar import (
    build_scalar_only_terminal_set,
    build_emb_lean_terminal_set,
    build_emb_lean_functions,
    SCALAR_ONLY_FUNCTIONS,
    Grammar,
    GType,
    _POSITION_STATE_TERMINALS,
)
from .simulator import prepare_terminal_data, GRID_IV_COLUMNS, _project_emb_pc
from .data import load_minute_parquet, TYPED_VECTOR_COLUMNS
from .readouts import (
    fit_lean_vector_bases_masked,
    assert_basis_fit_is_causal,
    fit_target_residualizer,
    apply_target_residual,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    """Represent unavailable statistics as JSON null without stringifying numbers."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def permute_embeddings(df, seed):
    """Apply the experiment's shared full-row permutation negative control."""
    from .data import PROBE_SCALAR_COLUMNS_V2, REGIME_PROB_COLUMNS

    permutation = np.random.default_rng(int(seed)).permutation(len(df))
    shuffled = df.copy()
    for column in TYPED_VECTOR_COLUMNS + PROBE_SCALAR_COLUMNS_V2 + REGIME_PROB_COLUMNS:
        if column in df:
            shuffled[column] = df[column].values[permutation]
    return shuffled


def load_inputs(path, condition, permutation_seed=None):
    """Load an ordered minute panel without changing its alignment to targets."""
    import pyarrow.parquet as pq

    metadata = pq.read_metadata(path).metadata or {}
    input_metadata = json.loads(
        metadata.get(b"potofgold.evolutionary_search.inputs", b"{}")
    )
    repair = input_metadata.get("iv_repair_mult")
    if repair is not None and float(repair) != 4.0:
        raise ValueError(
            "Prepared data uses a different IV repair threshold; this experiment requires iv_repair_mult=4"
        )
    df, _ = load_minute_parquet(path)
    required = {
        "date",
        "bar_position",
        "SPXClose",
        "ATM_IV",
        "VIXSpot",
        "RawSpread",
        "MinutesToClose",
        "DeltaSpread1",
        "DeltaSpread5",
        "VIXTermSlope",
        "RealizedVol30m",
        *GRID_IV_COLUMNS,
    }
    if condition != "indicators":
        required.update(TYPED_VECTOR_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Prepared data is missing required columns: {', '.join(missing)}"
        )
    if df.empty:
        raise ValueError("Prepared data has no rows")
    df["date"] = df["date"].astype(str)
    index = pd.MultiIndex.from_frame(df[["date", "bar_position"]])
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(
            "Prepared rows must have unique (date, bar_position) keys in chronological order"
        )
    bars = df["bar_position"].to_numpy()
    if (
        not np.all(np.isfinite(bars))
        or np.any(bars < 0)
        or np.any(bars != np.floor(bars))
    ):
        raise ValueError("bar_position must contain nonnegative integers")
    for name in ("SPXClose", "ATM_IV", "VIXSpot", *GRID_IV_COLUMNS):
        values = df[name].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(
                f"Execution column {name} must contain finite positive values"
            )
    if condition != "indicators":
        for name in TYPED_VECTOR_COLUMNS:
            values = np.stack(df[name].to_numpy())
            if values.shape != (len(df), 384) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{name} must contain one finite 384-value vector per minute"
                )
            if float(np.max(np.std(values.astype(np.float64), axis=0))) < 1e-9:
                raise ValueError(
                    f"{name} is constant; provide live encoder representations"
                )
    if permutation_seed is not None:
        df = permute_embeddings(df, permutation_seed)
    df.attrs["input_metadata"] = input_metadata
    df.attrs["prepared_parquet_sha256"] = file_sha256(path)
    # Session-lagged indicators need the complete chronological panel here.
    return df, prepare_terminal_data(df, normalize_terminals=True)


def load_targets(path, df):
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(
            "Embeddings and combined search require --targets pointing to an aligned six-target NPZ"
        )
    with np.load(path, allow_pickle=False) as data:
        if "metadata" in data:
            raw_metadata = data["metadata"]
            encoded = (
                raw_metadata.tobytes().decode("utf-8")
                if raw_metadata.dtype == np.uint8
                else str(raw_metadata.item())
            )
            metadata = json.loads(encoded)
            expected_hash = metadata.get("prepared_parquet_sha256")
            if expected_hash is not None and expected_hash != df.attrs.get(
                "prepared_parquet_sha256"
            ):
                raise ValueError(
                    "Targets were prepared for a different parquet file; regenerate them with the supplied panel"
                )
        required = {"targets", "valid", "date", "bar_position", "target_names"}
        if not required <= set(data.files):
            raise ValueError(
                f"Targets NPZ is missing keys: {sorted(required - set(data.files))}"
            )
        targets = np.asarray(data["targets"], dtype=np.float64)
        valid = np.asarray(data["valid"], dtype=bool)
        if targets.shape != (len(df), 6) or valid.shape != (len(df),):
            raise ValueError(
                f"Expected targets ({len(df)}, 6) and valid ({len(df)},), got {targets.shape} and {valid.shape}"
            )
        if not (
            np.array_equal(data["date"].astype(str), df["date"].to_numpy())
            and np.array_equal(data["bar_position"], df["bar_position"].to_numpy())
        ):
            raise ValueError(
                "Targets date/bar_position keys do not match the prepared data row order"
            )
        primary = int(data["primary_target_idx"]) if "primary_target_idx" in data else 0
        names = [str(x) for x in data["target_names"]]
        if (
            len(names) != 6
            or not 0 <= primary < 6
            or names[primary] != "volatility_difference"
        ):
            raise ValueError(
                "The primary target must be the volatility_difference axis"
            )
        if not np.all(np.isfinite(targets[valid])):
            raise ValueError("Valid target rows contain nonfinite values")
    return targets, valid, primary


def fit_readouts(df, train_mask, test_mask, targets, valid, primary):
    """Fit scalar residualizers and supervised embedding directions on training rows."""
    assert_basis_fit_is_causal(train_mask, test_mask, df)
    surface = _scalar_baseline(df)
    residuals = np.array(targets, dtype=np.float64, copy=True)
    for col in range(residuals.shape[1]):
        fitted = fit_target_residualizer(residuals[:, col], surface, train_mask)
        if fitted is None:
            raise ValueError(
                "Too few finite training rows to fit the priced-scalar residualizer"
            )
        residuals[:, col] = apply_target_residual(residuals[:, col], surface, fitted)
    bases = fit_lean_vector_bases_masked(
        df,
        residuals,
        valid,
        train_mask,
        primary_target=residuals[:, primary],
        primary_target_idx=primary,
    )
    info = {
        "basis_method": "lean",
        "basis_causal": True,
        "n_basis_fit_rows": int(np.sum(train_mask)),
        "n_basis_fit_days": int(df.loc[train_mask, "date"].nunique()),
        "d1_residualize": True,
        "d1_residualize_scope": "all",
        "d1_estimator": "ridge",
        "emb_leaf_frac": "0.5",
        "emb_gen0_floor": "0.5",
        "lean_proto_nonlinear": "1",
        "emb_bank_cap": _min_realized_bank(bases),
        **_lean_k_provenance(bases, 6),
    }
    return bases, info


def make_grammar(condition, config, bases=None):
    """Use identical typed operators for the embedding and combined conditions."""
    if condition == "indicators":
        functions, terminals = SCALAR_ONLY_FUNCTIONS, build_scalar_only_terminal_set()
    else:
        functions = build_emb_lean_functions()
        terminals = build_emb_lean_terminal_set(bank_cap=_min_realized_bank(bases))
        if condition == "combined":

            def unique(items):
                seen = set()
                return [
                    item
                    for item in items
                    if not (item.name in seen or seen.add(item.name))
                ]

            functions = unique(list(SCALAR_ONLY_FUNCTIONS) + list(functions))
            terminals = unique(list(build_scalar_only_terminal_set()) + list(terminals))
    return Grammar(
        functions=functions,
        terminals=terminals,
        max_depth=config.max_tree_depth,
        max_nodes=config.max_tree_nodes,
    )


def validate_terminals(grammar, splits):
    names = [
        term.name
        for term in grammar.terminals
        if term.ret_type == GType.REAL
        and term.value is None
        and term.sampler is None
        and term.name not in _POSITION_STATE_TERMINALS
    ]
    for split, td in splits.items():
        missing = [name for name in names if name not in td]
        constant = [
            name
            for name in names
            if name in td and float(np.nanstd(np.asarray(td[name], dtype=float))) < 1e-9
        ]
        if missing or constant:
            raise ValueError(
                f"{split} has unusable strategy inputs: missing={missing}, constant={constant}"
            )


def save_readouts(bases, path):
    """Store every fitted array without pickle or refitting during replay."""
    arrays, metadata = {}, {}
    for group, values in bases.items():
        if isinstance(values, dict):
            metadata[group] = {}
            for name, value in values.items():
                if isinstance(value, np.ndarray):
                    arrays[f"{group}/{name}"] = value
                else:
                    metadata[group][name] = value
        else:
            metadata[group] = values
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)


def load_readouts(path):
    with np.load(path, allow_pickle=False) as data:
        bases = json.loads(str(data["metadata"]))
        for key in data.files:
            if key != "metadata":
                group, name = key.split("/", 1)
                bases[group][name] = np.array(data[key], copy=True)
    for group in TYPED_VECTOR_COLUMNS:
        if not isinstance(bases.get(group), dict):
            raise ValueError(f"Saved readouts lack required embedding group {group}")
        for name in (
            "components",
            "mean_",
            "pc_std_",
            "dir_bank_",
            "dir_scale_",
            "proto_bank_",
        ):
            if name not in bases[group]:
                raise ValueError(f"Saved readouts lack {group}/{name}")
    return bases


BASELINE_EXCLUDE = frozenset(
    {
        "UnrealizedProfitPct",
        "MaxFavorableExcursion",
        "MaxAdverseExcursion",
        "ProfitGiveback",
        "EphReal",
        "EphInt",
        "CALL",
        "PUT",
        "NEUTRAL",
        "CalDayOfWeek",
        "CalIsOpexWeek",
        "CalTurnOfMonth",
        "CalDaysToFomc",
        "NetGammaProxy",
    }
)


def _scalar_baseline(df: pd.DataFrame) -> np.ndarray:
    """Build the training residualizer's indicator matrix in grammar order."""
    want = [
        t.name
        for t in build_scalar_only_terminal_set()
        if not (
            t.name.startswith("EMB_")
            or t.name.startswith("PROTO")
            or t.name in BASELINE_EXCLUDE
        )
    ]
    pool = prepare_terminal_data(df, normalize_terminals=True)
    cols = []
    seen = set()
    for nm in want:
        if nm in seen:
            continue
        seen.add(nm)
        if nm not in pool or np.ndim(pool[nm]) != 1 or len(pool[nm]) != len(df):
            continue
        vals = np.asarray(pool[nm], dtype=np.float64)
        if not np.any(np.isfinite(vals)):
            continue
        cols.append(vals)
    if not cols:
        raise RuntimeError(
            "No scalar baseline columns (build_scalar_only_terminal_set ∩ available features)."
        )
    return np.nan_to_num(np.column_stack(cols), nan=0.0, posinf=0.0, neginf=0.0)


def _min_realized_bank(bases):
    banks = [
        int(g["dir_bank_"].shape[0])
        for g in bases.values()
        if isinstance(g, dict) and g.get("dir_bank_") is not None
    ]
    return min(banks) if banks else None


def _lean_k_provenance(bases, n_targets):
    kd = {
        col: int(g["k_used"])
        for (col, g) in bases.items()
        if isinstance(g, dict) and "k_used" in g
    }
    er = {
        col: round(float(g["effective_rank"]), 3)
        for (col, g) in bases.items()
        if isinstance(g, dict) and "effective_rank" in g
    }
    return {
        "lean_k_used": kd,
        "lean_eff_rank": er,
        "lean_n_targets": int(n_targets) if n_targets is not None else None,
        "lean_k_min": min(kd.values()) if kd else None,
        "lean_k_max": max(kd.values()) if kd else None,
    }


def _build_emb_proj_cache(td, bases):
    """Compute fixed fold readouts once, shared by every candidate evaluation."""
    cache = {}
    for group, g in bases.items():
        raw = td.get(group)
        if raw is None or getattr(raw, "ndim", 0) != 2:
            continue
        v = raw.astype(np.float64)
        components = g["components"]
        mean_ = g["mean_"]
        pc_std = g["pc_std_"]
        for k in range(components.shape[0]):
            cache[group, k] = _project_emb_pc(v, mean_, components[k], float(pc_std[k]))
        # EmbDir can select any bank direction. Its scaling differs from EmbProj,
        # so compute it with the same arithmetic as the evaluator's fallback.
        dir_bank = g.get("dir_bank_")
        if dir_bank is not None:
            dir_mean = g.get("dir_mean_", mean_)
            for b, direction in enumerate(dir_bank):
                cache[group, "dir", b] = _project_emb_pc(
                    v, dir_mean, direction, float(g["dir_scale_"][b])
                )
    return cache
