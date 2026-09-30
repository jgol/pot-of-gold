"""Load the experiment's encoder and export embeddings using its saved preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .checkpoints import load_checkpoint, training_metadata


def load_encoder(path, device="cpu"):
    """Return (encoder in evaluation mode, checkpoint including normalization)."""
    from .model import iTransformerEncoder

    checkpoint = load_checkpoint(path)
    model = iTransformerEncoder(checkpoint["ssl_hyperparams"],
                                n_variates=checkpoint["n_ssl_features"])
    model.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def prepare_day(raw, checkpoint, *, stride=30):
    """Return normalized (windows, ending-bar indices) from one raw daily panel.

    Raw-feature transforms follow the recorded experiment. Normalization is read
    from the checkpoint and is never refitted on the day being encoded.
    """
    from .config import PREPROCESS_LOG1P_INDICES
    from .data import apply_vix_forwardfill, preprocess_day, should_skip_day

    if checkpoint.get("profile", "experiment") != "experiment":
        raise ValueError("Checkpoint preprocessing must match the recorded experiment")
    raw = np.asarray(raw)
    features = checkpoint["ssl_features"]
    seq_len = int(checkpoint["ssl_hyperparams"]["seq_len"])
    if raw.ndim != 2 or raw.shape[1] <= max(features):
        raise ValueError("Expected a raw daily panel with the study's stored feature columns")
    if stride < 1:
        raise ValueError("stride must be positive")
    if len(raw) < seq_len or should_skip_day(raw):
        return np.empty((0, seq_len, len(features)), dtype=np.float32), np.empty(0, dtype=np.int64)
    day = preprocess_day(raw, log1p_indices=tuple(PREPROCESS_LOG1P_INDICES))
    apply_vix_forwardfill(day)
    starts = np.arange(0, len(day) - seq_len + 1, stride)
    windows = np.stack([day[start:start + seq_len, features] for start in starts])
    if checkpoint.get("z_mean") is None or checkpoint.get("z_std") is None:
        raise ValueError("Checkpoint does not contain the fitted normalization statistics")
    # Checkpoints store these positions within the selected feature list, unlike
    # the raw-column indices used by preprocess_day above.
    cols = checkpoint.get("log1p_cols", [])
    if cols:
        windows[:, :, cols] = np.sign(windows[:, :, cols]) * np.log1p(np.abs(windows[:, :, cols]))
    windows = (windows - np.asarray(checkpoint["z_mean"], dtype=np.float32)) / np.asarray(checkpoint["z_std"], dtype=np.float32)
    windows = np.nan_to_num(np.clip(windows, -5.0, 5.0), nan=0.0).astype(np.float32)
    # The ending bar identifies a window independently of its embedding values.
    return windows, starts + seq_len - 1


def encode_day(model, checkpoint, raw, *, readout="pooled", stride=30, batch_size=64):
    """Return (embeddings, ending-bar indices); one output per input window.

    Readouts: pooled (N, 128), variates (N, 147, 128), multilayer (N, 147, 384)
    for the released models. The latter concatenates all three block outputs.
    """
    import torch

    if readout not in ("pooled", "variates", "multilayer"):
        raise ValueError("readout must be pooled, variates or multilayer")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    windows, bars = prepare_day(raw, checkpoint, stride=stride)
    device = next(model.parameters()).device
    hp = checkpoint["ssl_hyperparams"]
    width = hp["d_model"] * (hp["n_layers"] if readout == "multilayer" else 1)
    shape = (0, width) if readout == "pooled" else (0, checkpoint["n_ssl_features"], width)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            tokens = model.encode_variates(torch.from_numpy(windows[start:start + batch_size]).to(device),
                                           multi_layer=readout == "multilayer")
            embedding = tokens.mean(dim=1) if readout == "pooled" else tokens
            chunks.append(embedding.cpu().numpy())
    return (np.concatenate(chunks) if chunks else np.empty(shape, dtype=np.float32)), bars


def export_embeddings(checkpoint_path, data_path, out, *, readout="pooled", stride=30,
                      device="cpu", overwrite=False):
    """Export window readouts and enough identity information to align later evaluation."""
    from . import config

    checkpoint_path, data_path, out = map(Path, (checkpoint_path, data_path, out))
    if out.suffix != ".npz":
        raise ValueError("--out must end with .npz")
    if out.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {out}; choose another path or use --overwrite")
    if out.resolve() in {checkpoint_path.resolve(), data_path.resolve()}:
        raise ValueError("Embedding output must be separate from the checkpoint and input corpus")
    model, checkpoint = load_encoder(checkpoint_path, device)
    # Original checkpoints retain their exact bytes; their saved metadata lives in experiment.json.
    run = checkpoint.get("training_metadata")
    if run is None:
        try:
            run = training_metadata(checkpoint_path)
        except ValueError:
            run = {}  # Inference also supports compatible checkpoints with unknown training provenance.
    if checkpoint.get("profile", run.get("profile", "experiment")) != "experiment":
        raise ValueError("Checkpoint preprocessing must match the recorded experiment")
    split_keys = ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end")
    split_dates = {key: config.EXPERIMENT_PROFILE[key.upper()] for key in split_keys}
    split_dates.update({key: value for key, value in run.get("date_span", {}).items()
                        if key in split_dates})
    chunks, dates, bars = [], [], []
    with np.load(data_path, allow_pickle=False) as corpus:
        manifest = json.loads(bytes(corpus["_manifest"]).decode("utf-8"))
        for day in sorted(manifest["dates"]):
            z, end_bars = encode_day(model, checkpoint, corpus[f"d_{day}"], readout=readout,
                                     stride=stride)
            if len(z):
                chunks.append(z)
                dates.extend([day] * len(z))
                bars.extend(end_bars)
    if not chunks:
        raise ValueError("The corpus contained no usable input windows")
    hp = checkpoint["ssl_hyperparams"]
    # Content hashes bind later evaluation to the exact checkpoint and corpus;
    # dates and ending bars identify rows even if an export is reordered.
    metadata = {"schema_version": 1, "readout": readout, "stride": stride, "profile": "experiment",
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "corpus_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "tokenizer": checkpoint.get("tokenizer", hp.get("tokenizer", run.get("tokenizer"))),
                "seed": run.get("seed", checkpoint.get("seed")),
                "seq_len": int(hp["seq_len"]), "n_features": int(checkpoint["n_ssl_features"]),
                "ssl_features": [int(index) for index in checkpoint["ssl_features"]],
                "split_dates": split_dates, "corpus_splits": manifest.get("splits"),
                "synthetic_data": bool(manifest.get("synthetic", False))}
    embeddings = np.concatenate(chunks)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, embeddings=embeddings, dates=np.asarray(dates),
                        last_bar=np.asarray(bars, dtype=np.int64),
                        metadata=np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8))
    return embeddings.shape, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--readout", choices=("pooled", "variates", "multilayer"), default="pooled")
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--max-cores", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from .train import _cap_cores
    _cap_cores(args.max_cores)
    import torch
    torch.set_num_threads(max(1, args.max_cores))
    try:
        shape, metadata = export_embeddings(args.checkpoint, args.data, args.out, readout=args.readout,
                                            stride=args.stride, device=args.device,
                                            overwrite=args.overwrite)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {args.out}: embeddings {shape}; synthetic input={metadata['synthetic_data']}")


if __name__ == "__main__":
    main()
