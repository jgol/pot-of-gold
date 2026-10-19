"""Prepare aligned market/embedding Parquet and six financial targets explicitly."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .market import market_day
from .schema import (EMBEDDING_COLUMNS, HISTORICAL_CHECKPOINT_SHA256, METADATA_KEY,
                     decode_metadata, file_sha256, validate_keys)
from .targets import TARGET_NAMES, day_targets


def read_market(path):
    path = Path(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path, comment="#")
    else:
        raise ValueError("Market input must be .csv or .parquet")
    return validate_keys(frame)


def _embedding_source(path):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    missing = set(("date", "bar_position", *EMBEDDING_COLUMNS)) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Embedding parquet is missing columns: {sorted(missing)}")
    # Only index columns are retained globally; vector row groups are streamed below.
    keys = parquet.read(columns=["date", "bar_position"]).to_pandas()
    ordered = validate_keys(keys, "embeddings")
    if not np.array_equal(keys.astype(str).to_numpy(), ordered.astype(str).to_numpy()):
        raise ValueError("Reused embeddings must be sorted by date and bar_position")
    recorded = parquet.schema_arrow.metadata or {}
    if METADATA_KEY not in recorded:
        raise ValueError("Reused embeddings require preparation metadata linking them to their raw corpus")
    return ordered, json.loads(recorded[METADATA_KEY])


def _read_embedding_day(path, day, expected_bars):
    import pyarrow.parquet as pq
    frame = pq.read_table(path, filters=[("date", "=", day)],
                          columns=["date", "bar_position", *EMBEDDING_COLUMNS]).to_pandas()
    frame = validate_keys(frame, "embeddings")
    if not np.array_equal(frame["bar_position"].to_numpy(), expected_bars):
        raise ValueError(f"Reused embedding keys differ from the requested rows for {day}")
    out = {}
    for name in EMBEDDING_COLUMNS:
        try:
            values = np.stack(frame[name]).astype(np.float32)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid vectors in {name}") from error
        if values.shape != (len(frame), 384) or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite 384-dimensional vectors")
        out[name] = values
    return out


def prepare(data, market, out, targets_out, *, checkpoint=None, without_embeddings=False,
            embeddings=None, preprocessing="auto", cold_start="masked", iv_repair_mult=4.0,
            batch_size=32, device="cpu", max_cores=2, overwrite=False):
    """Build named outputs; no search, basis fitting, training or report generation."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    sources = [Path(data), Path(market)]
    if checkpoint is not None:
        sources.append(Path(checkpoint))
    if embeddings is not None:
        sources.append(Path(embeddings))
    outputs = [Path(out), Path(targets_out)]
    if outputs[0].suffix != ".parquet" or outputs[1].suffix != ".npz":
        raise ValueError("Outputs must be a .parquet input table and a .npz target archive")
    if len({p.resolve() for p in outputs}) != 2 or any(p.resolve() in {s.resolve() for s in sources} for p in outputs):
        raise ValueError("Outputs must be distinct and must not replace an input")
    for p in outputs:
        if p.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {p}; choose a new path or --overwrite")
    if sum((bool(checkpoint), bool(embeddings), bool(without_embeddings))) != 1:
        raise ValueError("Choose exactly one of --checkpoint, --embeddings or --without-embeddings")
    if iv_repair_mult < 0 or not np.isfinite(iv_repair_mult) or max_cores < 1 or batch_size < 1:
        raise ValueError("Invalid IV repair multiplier, core limit or batch size")
    market_frame = read_market(market)
    source_hashes = {str(p.resolve()): file_sha256(p) for p in sources}
    checkpoint_hash = source_hashes.get(str(Path(checkpoint).resolve())) if checkpoint else None
    historical = checkpoint_hash == HISTORICAL_CHECKPOINT_SHA256
    if preprocessing == "auto":
        preprocessing = "historical-gp" if historical else "transformer"
    if preprocessing not in ("historical-gp", "transformer"):
        raise ValueError("Invalid preprocessing profile")
    if historical and preprocessing != "historical-gp":
        raise ValueError("The historical GP checkpoint requires historical-gp preprocessing")
    model = ckpt = None
    if checkpoint:
        import torch
        from potofgold.encoder.inference import load_encoder
        torch.set_num_threads(max_cores)
        model, ckpt = load_encoder(checkpoint, device=device)
    reused_keys, reused_metadata = _embedding_source(embeddings) if embeddings else (None, None)
    if embeddings:
        if reused_metadata.get("corpus_sha256") != source_hashes[str(Path(data).resolve())]:
            raise ValueError("Reused embeddings were prepared from a different raw corpus")
        if reused_metadata.get("iv_repair_mult") != iv_repair_mult or reused_metadata.get("cold_start") != cold_start:
            raise ValueError("Reused embeddings have different IV repair or cold-start settings")
    chunks, valid_chunks, dates, bars_all = [], [], [], []
    repaired_raw = repaired_market = 0
    previous_close = None
    writer = None
    temporary = []
    try:
        for p in outputs:
            p.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=p.parent, prefix=f".{p.stem}-", suffix=p.suffix, delete=False) as f:
                temporary.append(Path(f.name))
        with np.load(data, allow_pickle=False) as corpus:
            manifest = decode_metadata(corpus["_manifest"])
            corpus_dates = sorted(manifest["dates"])
            if len(set(corpus_dates)) != len(corpus_dates):
                raise ValueError("Corpus manifest contains duplicate dates")
            if list(market_frame["date"].drop_duplicates()) != corpus_dates:
                raise ValueError("Market dates must exactly match the corpus manifest")
            if reused_keys is not None and list(reused_keys["date"].drop_duplicates()) != corpus_dates:
                raise ValueError("Reused embedding dates must exactly match the corpus manifest")
            metadata = {
                "schema_version": 1, "synthetic_data": bool(manifest.get("synthetic", False)),
                "source_sha256": source_hashes, "checkpoint_sha256": checkpoint_hash,
                "corpus_sha256": source_hashes[str(Path(data).resolve())],
                "market_sha256": source_hashes[str(Path(market).resolve())],
                "historical_gp_checkpoint": historical, "preprocessing": preprocessing if checkpoint else None,
                "raw_log1p_indices": ([101, 102, 104, 139] if preprocessing == "historical-gp"
                                      else [101, 102, 104, 132, 139]) if checkpoint else None,
                "iv_repair_mult": iv_repair_mult, "iv_repair_before_encoding": bool(checkpoint),
                "cold_start": cold_start if checkpoint or embeddings else None,
                "readout": "13 multilayer pools, 384 dimensions" if not without_embeddings else None,
                "embedding_source": str(Path(embeddings).resolve()) if embeddings else None,
                "reused_embedding_metadata": reused_metadata,
                "scalar_derivation": {"spread_spike_threshold": 0.035, "derived_atm_iv_winsor": 0.8,
                                      "rv30_units": "unannualized standard deviation",
                                      "supplied_market_columns_override_derivation": True},
                "market_overrides": sorted(set(market_frame.columns) - {"date", "bar_position", "SPXClose"}),
                "target_names": list(TARGET_NAMES), "target_horizon": "strictly future through final supplied day bar",
                "target_annualization": "252 * full session bar count; population standard deviation",
                "basis_fitted": False, "normalization": "checkpoint statistics; no fitting" if checkpoint else None,
                "pooling_contract": "v3-147-multilayer-13-v1", "embedding_width": 384 if not without_embeddings else None,
            }
            if ckpt is not None:
                metadata["ssl_features"] = [int(i) for i in ckpt["ssl_features"]]
                metadata["checkpoint_log1p_cols"] = [int(i) for i in ckpt.get("log1p_cols", [])]
            # Index each date once instead of scanning the full panel for every day.
            market_by_day = market_frame.groupby("date", sort=False)
            reused_by_day = reused_keys.groupby("date", sort=False) if reused_keys is not None else None
            for day in corpus_dates:
                raw = np.asarray(corpus[f"d_{day}"])
                supplied = market_by_day.get_group(day).reset_index(drop=True)
                frame, count = market_day(raw, supplied, previous_close=previous_close, iv_repair_mult=iv_repair_mult)
                repaired_market += count
                previous_close = float(frame["SPXClose"].iloc[-1])
                targets, valid = day_targets(frame)
                bars = np.arange(len(frame), dtype=np.int64)
                if checkpoint:
                    from .embeddings import encode_readouts
                    values, bars, count = encode_readouts(model, ckpt, raw, preprocessing=preprocessing,
                                                         cold_start=cold_start, iv_repair_mult=iv_repair_mult,
                                                         batch_size=batch_size)
                    repaired_raw += count
                elif embeddings:
                    available = reused_by_day.get_group(day)["bar_position"].to_numpy()
                    bars = np.arange(0 if cold_start == "masked" else 59, len(frame), dtype=np.int64)
                    if not np.array_equal(available, bars):
                        raise ValueError(f"Reused embeddings missing or extra bars for {day}")
                    values = _read_embedding_day(embeddings, day, bars)
                frame = frame.iloc[bars].copy().reset_index(drop=True)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if not without_embeddings:
                    for name in EMBEDDING_COLUMNS:
                        array = pa.FixedSizeListArray.from_arrays(pa.array(values[name].ravel(), type=pa.float32()), 384)
                        table = table.append_column(name, array)
                schema_metadata = dict(table.schema.metadata or {})
                schema_metadata[METADATA_KEY] = json.dumps(metadata, sort_keys=True).encode("utf-8")
                table = table.replace_schema_metadata(schema_metadata)
                if writer is None:
                    writer = pq.ParquetWriter(temporary[0], table.schema, compression="zstd")
                writer.write_table(table, row_group_size=len(frame))
                chunks.append(targets[bars])
                valid_chunks.append(valid[bars])
                dates.extend([day] * len(bars))
                bars_all.extend(bars)
                print(f"Prepared {day}: {len(frame)} rows", flush=True)
            if writer is None:
                raise ValueError("No days prepared")
            writer.close()
            writer = None
        targets = np.concatenate(chunks)
        valid = np.concatenate(valid_chunks)
        metadata.update(rows=len(targets), days=len(corpus_dates), encoder_iv_cells_repaired=repaired_raw,
                        market_iv_cells_repaired=repaired_market)
        metadata["prepared_parquet_sha256"] = file_sha256(temporary[0])
        np.savez_compressed(temporary[1], targets=targets, valid=valid, date=np.asarray(dates),
                            bar_position=np.asarray(bars_all, dtype=np.int64), target_names=np.asarray(TARGET_NAMES),
                            primary_target_idx=np.asarray(0),
                            metadata=np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8))
        for source, destination in zip(temporary, outputs):
            source.replace(destination)
        return metadata
    finally:
        if writer is not None:
            writer.close()
        for path in temporary:
            path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Shared raw-feature corpus NPZ")
    parser.add_argument("--market", type=Path, required=True, help="Complete CSV/Parquet with date, bar_position, SPXClose")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--embeddings", type=Path, help="Reuse an aligned prepared Parquet's thirteen vectors")
    group.add_argument("--without-embeddings", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--targets-out", type=Path, required=True)
    parser.add_argument("--preprocessing", choices=("auto", "historical-gp", "transformer"), default="auto")
    parser.add_argument("--cold-start", choices=("masked", "omit"), default="masked")
    parser.add_argument("--iv-repair-mult", type=float, default=4.0, help="IV > multiplier * VIX/100 becomes VIX/100; zero disables")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--max-cores", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        metadata = prepare(**vars(args))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"Wrote {args.out} and {args.targets_out}: {metadata['rows']} rows, {metadata['days']} days; "
          f"synthetic={metadata['synthetic_data']}. No search or basis fitting performed.")


if __name__ == "__main__":
    main()
