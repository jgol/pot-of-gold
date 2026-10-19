"""Replay a saved strategy with its original execution and fold-readout settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    EXPERIMENT_FLAGS,
    SearchConfig,
    configure_runtime,
    ensure_reproducible_process,
)


def _select_record(args):
    path = Path(args.run or args.strategy).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Strategy file does not exist: {path}")
    if args.strategy:
        from ..analysis.archive import load_strategy

        record, fold, all_days = load_strategy(path)
        if args.strategy_id is not None and args.strategy_id != record["strategy_id"]:
            raise ValueError(
                f"Strategy ID {args.strategy_id!r} does not match the saved strategy {record['strategy_id']!r}"
            )
        provenance = fold["provenance"]
        config = SearchConfig(
            n_folds=provenance["n_folds"],
            split_seed=provenance["fold_partition_seed"],
            minimum_trades=provenance["min_trades"],
            entry_start_bar=provenance["decision_offset"],
            save_checkpoints=False,
        )
        return (
            path,
            record,
            {
                "condition": fold["condition"],
                "fold": fold["fold"],
                "config": config.to_dict(),
                "provenance": provenance,
                "test_day_list": fold["test_days"],
                "train_day_list": sorted(set(all_days) - set(fold["test_days"])),
                "readouts": None,
            },
        )
    if not args.strategy_id:
        raise ValueError("--strategy-id is required with --run")
    document = json.loads(path.read_text())
    members = document.get("surviving_set", [])
    record = next(
        (r for r in members if r.get("strategy_id") == args.strategy_id), None
    )
    if document.get("strategy_id") == args.strategy_id:
        record = document
    if record is None:
        raise ValueError(f"Unknown strategy ID: {args.strategy_id}")
    return path, record, document


def replay(args):
    source, record, run = _select_record(args)
    strategy_id = record["strategy_id"]
    output = Path(args.out).resolve()
    protected = {source, Path(args.data).resolve()}
    if args.targets:
        protected.add(Path(args.targets).resolve())
    if run.get("readouts"):
        protected.add((source.parent / run["readouts"]["path"]).resolve())
    if output in protected or (
        args.strategy and output.is_relative_to(source.parents[3])
    ):
        raise ValueError(
            "Replay output must not overwrite an input or saved readout artifact, "
            "or be saved inside the strategy archive"
        )
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Replay output already exists: {output}; use --overwrite to replace it"
        )
    config = SearchConfig(**{**run["config"], "save_checkpoints": False})
    recorded_flags = run["provenance"].get("effective_flags", {})
    for name, value in recorded_flags.items():
        if name.startswith("GP_CHECKPOINT"):
            continue
        if value is not None and EXPERIMENT_FLAGS.get(name) != value:
            raise ValueError(
                f"Saved strategy uses unsupported settings: {name}={value!r}"
            )
    configure_runtime(config)

    import numpy as np
    from . import pipeline
    from .genetics import individual_from_genome_dict
    from .fitness import genome_trades, per_day_sum_pnl
    from .templates import base_template_by_name
    from .validation import slice_by_mask, judge_champion

    condition = run["condition"]
    shuffle_seed = run["provenance"].get("shuffle_emb_seed")
    df, td_full = pipeline.load_inputs(args.data, condition, shuffle_seed)
    data_hash = df.attrs["prepared_parquet_sha256"]
    expected_days = set(run["train_day_list"]) | set(run["test_day_list"])
    actual_days = set(df["date"].unique())
    if args.strategy or args.split == "test":
        if expected_days != actual_days:
            raise ValueError(
                "Replay needs the complete recorded minute panel, including its training and held-out days; the supplied calendar differs"
            )
    if args.run and args.split == "test":
        expected_hash = run["provenance"].get("data_sha256")
        if expected_hash and data_hash != expected_hash:
            raise ValueError(
                "Held-out replay data does not match the run's saved data hash; use --split all for a new evaluation panel"
            )
    bases, origin = None, "not_required"
    if condition != "indicators":
        artifact = run.get("readouts")
        if artifact is not None:
            previous_metadata = run["provenance"].get("input_metadata", {})
            current_metadata = df.attrs.get("input_metadata", {})

            def metadata_value(metadata, key):
                value = metadata.get(key)
                if value is None and isinstance(
                    metadata.get("reused_embedding_metadata"), dict
                ):
                    return metadata_value(metadata["reused_embedding_metadata"], key)
                return value

            for key in (
                "checkpoint_sha256",
                "ssl_features",
                "checkpoint_log1p_cols",
                "raw_log1p_indices",
                "cold_start",
                "iv_repair_before_encoding",
                "iv_repair_mult",
                "preprocessing",
                "pooling_contract",
                "embedding_width",
            ):
                expected = metadata_value(previous_metadata, key)
                if expected is not None and expected != metadata_value(
                    current_metadata, key
                ):
                    raise ValueError(
                        f"Prepared representations differ from the saved strategy's encoder contract: {key}"
                    )
            path = source.parent / artifact["path"]
            if not path.is_file():
                raise FileNotFoundError(f"The run's saved readouts are missing: {path}")
            if pipeline.file_sha256(path) != artifact["sha256"]:
                raise ValueError("Saved readouts do not match their recorded hash")
            bases = pipeline.load_readouts(path)
            origin = "saved_training_fold"
        else:
            if expected_days != actual_days:
                raise ValueError(
                    "Reconstructing archived readouts requires the full original training and held-out panel"
                )
            targets, valid, primary = pipeline.load_targets(args.targets, df)
            train_mask = df["date"].isin(run["train_day_list"]).to_numpy()
            recorded_rows = run["provenance"].get("n_basis_fit_rows")
            if recorded_rows is not None and int(train_mask.sum()) != recorded_rows:
                raise ValueError(
                    "The archived training row count does not match this prepared panel"
                )
            test_mask = ~train_mask
            bases, _ = pipeline.fit_readouts(
                df, train_mask, test_mask, targets, valid, primary
            )
            origin = "reconstructed_from_training_fold"
    mask = (
        df["date"].isin(run["test_day_list"]).to_numpy()
        if args.split == "test"
        else np.ones(len(df), dtype=bool)
    )
    evaluation_df, evaluation_td = slice_by_mask(df, td_full, mask)
    cache = (
        pipeline._build_emb_proj_cache(evaluation_td, bases)
        if bases is not None
        else None
    )
    strategy = individual_from_genome_dict(record["genome"])
    template = base_template_by_name(strategy.template_name)
    trades = genome_trades(
        strategy,
        evaluation_df,
        template,
        evaluation_td,
        warmup_bars=config.entry_start_bar,
        entry_floor_within_day=config.entry_start_bar,
        pca_bases=bases,
        emb_proj_cache=cache,
    )
    daily = per_day_sum_pnl(trades, evaluation_df["date"].to_numpy())
    result = {
        "schema_version": 1,
        "strategy_id": strategy_id,
        "condition": condition,
        "source": str(source),
        "source_sha256": pipeline.file_sha256(source),
        "data_sha256": data_hash,
        "split": args.split,
        "readout_origin": origin,
        "genome": record["genome"],
        "days": sorted(evaluation_df["date"].unique().tolist()),
        "traded_days": list(daily),
        "traded_day_pnl": [float(x) for x in daily.values()],
        "trade_count": len(trades),
        "trades": [
            {
                "entry_date": str(evaluation_df.iloc[t.entry_bar]["date"]),
                "entry_bar": int(evaluation_df.iloc[t.entry_bar]["bar_position"]),
                "exit_date": str(evaluation_df.iloc[t.exit_bar]["date"]),
                "exit_bar": int(evaluation_df.iloc[t.exit_bar]["bar_position"]),
                "pnl": float(t.pnl),
                "shape": t.shape,
                "exit_reason": t.exit_reason,
                "contracts": float(t.entry_n_contracts),
            }
            for t in trades
        ],
    }
    if args.split == "test":
        result["judge"] = judge_champion(
            strategy,
            evaluation_df,
            template,
            evaluation_td,
            perm_seed=run["fold"],
            gate_decisions=False,
            decision_offset=config.entry_start_bar,
            min_trades=config.minimum_trades,
            pca_bases=bases,
            emb_proj_cache=cache,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(pipeline.json_safe(result), indent=2, allow_nan=False) + "\n"
    )
    print(f"Replayed {strategy_id}: {len(trades)} trades; readouts={origin}; {output}")
    return output


def main(argv=None):
    if argv is None:
        ensure_reproducible_process(__spec__.name)
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run", help="New-run fold JSON, alongside its saved readout artifact"
    )
    source.add_argument("--strategy", help="Individual archived strategy JSON")
    parser.add_argument(
        "--strategy-id", help="Required with --run; optional with --strategy"
    )
    parser.add_argument("--data", required=True, help="Prepared minute parquet")
    parser.add_argument(
        "--targets",
        help="Aligned six-target NPZ when reconstructing archived embedding readouts",
    )
    parser.add_argument("--split", choices=("test", "all"), default="test")
    parser.add_argument("--out", required=True, help="Replay output JSON")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.run and not args.strategy_id:
        parser.error("--strategy-id is required with --run")
    try:
        replay(args)
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        KeyError,
    ) as exc:
        parser.exit(2, f"Replay failed: {exc}\n")


if __name__ == "__main__":
    main()
