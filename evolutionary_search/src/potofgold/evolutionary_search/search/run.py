"""Run one fold of indicators, embeddings, or combined strategy search."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from .config import configure_runtime, ensure_reproducible_process, load_config


def run_fold(args):
    config = load_config(args.config)
    if not 0 <= args.fold < config.n_folds:
        raise ValueError(f"--fold must be between 0 and {config.n_folds - 1}")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if args.permutation_seed is not None and args.condition != "permuted_embeddings":
        raise ValueError("--permutation-seed applies only to permuted_embeddings")
    permutation_seed = (
        (42 if args.permutation_seed is None else args.permutation_seed)
        if args.condition == "permuted_embeddings"
        else None
    )
    if permutation_seed is not None and permutation_seed < 0:
        raise ValueError("--permutation-seed must be nonnegative")
    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Prepared minute data does not exist: {data_path}")
    if args.condition != "indicators" and (
        not args.targets or not Path(args.targets).is_file()
    ):
        raise FileNotFoundError(
            "Embeddings and combined search require an existing --targets NPZ"
        )
    out = Path(args.out).resolve()
    tag = f"fold_{args.fold}"
    result_path = out / f"{tag}.json"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Run result already exists: {result_path}; use --overwrite to replace it"
        )
    configure_runtime(config)

    from . import pipeline
    from .evolution import evolve_capture, build_surviving_set, build_survivor_record
    from .templates import base_template_by_name
    from .validation import kfold_day_split, slice_by_mask

    started = time.monotonic()
    target_hash = pipeline.file_sha256(args.targets) if args.targets else None
    df, td_full = pipeline.load_inputs(data_path, args.condition, permutation_seed)
    data_hash = df.attrs["prepared_parquet_sha256"]
    input_metadata = df.attrs["input_metadata"]
    if df["date"].nunique() < config.n_folds:
        raise ValueError(
            f"Prepared data must contain at least {config.n_folds} distinct trading days"
        )
    train_mask, test_mask = kfold_day_split(
        df, args.fold, config.n_folds, seed=config.split_seed
    )
    train_df, train_td = slice_by_mask(df, td_full, train_mask)
    test_df, test_td = slice_by_mask(df, td_full, test_mask)
    bases, basis_info = None, {}
    train_cache, test_cache = None, None
    readout_artifact = None
    if args.condition != "indicators":
        targets, valid, primary = pipeline.load_targets(args.targets, df)
        bases, basis_info = pipeline.fit_readouts(
            df, train_mask, test_mask, targets, valid, primary
        )
        train_cache = pipeline._build_emb_proj_cache(train_td, bases)
        test_cache = pipeline._build_emb_proj_cache(test_td, bases)
    grammar = pipeline.make_grammar(args.condition, config, bases)
    pipeline.validate_terminals(grammar, {"train": train_td, "test": test_td})
    source_hashes = {
        p.name: pipeline.file_sha256(p)
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    train_days = sorted(train_df["date"].unique().tolist())
    test_days = sorted(test_df["date"].unique().tolist())
    checkpoint_identity = hashlib.sha256(
        json.dumps(
            {
                "data": data_hash,
                "targets": target_hash,
                "config": config.to_dict(),
                "condition": args.condition,
                "seed": args.seed,
                "fold": args.fold,
                "permutation_seed": permutation_seed,
                "source": source_hashes,
                "train_days": train_days,
                "test_days": test_days,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    flags = configure_runtime(config, out / "checkpoints" / checkpoint_identity)
    template = base_template_by_name("iron_condor")
    out.mkdir(parents=True, exist_ok=True)
    if bases is not None:
        readout_path = out / f"{tag}.readouts.npz"
        pipeline.save_readouts(bases, readout_path)
        readout_artifact = {
            "path": readout_path.name,
            "sha256": pipeline.file_sha256(readout_path),
        }
    print(
        f"Searching {args.condition}, seed {args.seed}, fold {args.fold}: "
        f"{config.population_size} candidates × {config.generations} generations",
        flush=True,
    )
    champion = evolve_capture(
        grammar,
        train_df,
        template,
        train_td,
        pop_size=config.population_size,
        n_gen=config.generations,
        seed=args.seed,
        evolve_structure=True,
        gate=False,
        decision_offset=config.entry_start_bar,
        min_trades=config.minimum_trades,
        pca_bases=bases,
        emb_proj_cache=train_cache,
        eval_workers=config.evaluation_workers,
        heartbeat_path=str(out / "logs" / f"{tag}.progress"),
        heartbeat_tag=tag,
    )
    common = dict(
        gate=False,
        decision_offset=config.entry_start_bar,
        min_trades=config.minimum_trades,
        perm_seed=args.fold,
        pca_bases=bases,
        emb_proj_cache=test_cache,
        train_df=train_df,
        train_td=train_td,
        train_emb_proj_cache=train_cache,
    )
    members = build_surviving_set(champion, test_df, template, test_td, **common)
    prefix = f"{args.condition}-s{args.seed}-f{args.fold}"
    for index, member in enumerate(members):
        member["strategy_id"] = f"{prefix}-m{index:03d}"
    result = {
        "schema_version": 1,
        "condition": args.condition,
        "seed": args.seed,
        "synthetic_data": input_metadata.get("synthetic_data"),
        "fold": args.fold,
        "n_folds": config.n_folds,
        "gate": False,
        "tag": tag,
        "pop": config.population_size,
        "gen": config.generations,
        "template": "iron_condor",
        "config": config.to_dict(),
        "train_days": int(train_df["date"].nunique()),
        "test_days": int(test_df["date"].nunique()),
        "train_day_list": train_days,
        "test_day_list": test_days,
        "readouts": readout_artifact,
        "surviving_set": members,
        "provenance": {
            "condition": args.condition,
            "evolution_seed": args.seed,
            "fold_partition_seed": config.split_seed,
            "n_folds": config.n_folds,
            "decision_offset": config.entry_start_bar,
            "min_trades": config.minimum_trades,
            "data": str(data_path),
            "data_sha256": data_hash,
            "targets_sha256": target_hash,
            "effective_flags": flags,
            "input_metadata": input_metadata,
            "shuffle_emb_seed": permutation_seed,
            "python": sys.version.split()[0],
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "libraries": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn", "pymoo")
            },
            "source_sha256": source_hashes,
            "checkpoint_identity": checkpoint_identity,
            **basis_info,
        },
    }
    if champion is None:
        result["champion"] = None
        result["status"] = "no_feasible_strategy"
    else:
        record = build_survivor_record(champion, test_df, template, test_td, **common)
        result.update(record)
        # The selected champion is not necessarily a member of the two-objective front.
        matching = next((m for m in members if m["genome"] == record["genome"]), None)
        result["strategy_id"] = (
            matching["strategy_id"] if matching else f"{prefix}-champion"
        )
        result["status"] = "complete"
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(pipeline.json_safe(result), indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(result_path)
    print(f"Saved {len(members)} surviving strategies to {result_path}", flush=True)
    return result_path


def main(argv=None):
    if argv is None:
        ensure_reproducible_process(__spec__.name)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared minute parquet")
    parser.add_argument(
        "--targets",
        help="Row-aligned six-target NPZ; required for embedding conditions",
    )
    parser.add_argument(
        "--condition",
        choices=("indicators", "embeddings", "combined", "permuted_embeddings"),
        required=True,
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        help="Shared embedding permutation seed; default 42 for permuted_embeddings",
    )
    parser.add_argument(
        "--config", help="YAML settings; defaults to the recorded experiment budget"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument(
        "--out", required=True, help="Run folder for this condition and seed"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_fold(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"Search failed: {exc}\n")


if __name__ == "__main__":
    main()
