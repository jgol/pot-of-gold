"""Inspect raw feature tables, daily corpora, and exported embeddings."""
import argparse
import json
from pathlib import Path

import numpy as np

from ..encoder.config import feature_names


def _panel_preview(panels, manifest, rows):
    names = feature_names()
    # Show named columns across the feature families; the file retains all 248 columns.
    columns = (0, 41, 105, 111, 119, 132)
    lines = ["Sample of 6 named raw columns (248 stored):",
             "date\tbar\t" + "\t".join(names[index] for index in columns)]
    remaining = rows
    for day in manifest["dates"]:
        panel = panels[f"d_{day}"]
        for bar in range(min(len(panel), remaining)):
            lines.append(f"{day}\t{bar}\t" + "\t".join(f"{float(panel[bar, index]):.7g}" for index in columns))
        remaining -= min(len(panel), remaining)
        if remaining == 0:
            break
    return lines


def inspect_file(path, *, rows=5):
    """Return a readable preview of a synthetic CSV, corpus NPZ or embedding NPZ."""
    if rows < 1:
        raise ValueError("--rows must be positive")
    path = Path(path)
    lines = [f"File: {path}"]
    if path.suffix == ".csv":
        from .simulation import read_csv

        panels, manifest = read_csv(path)
        lines.extend([f"Synthetic raw CSV: {len(manifest['dates'])} days × 405 bars × 248 features",
                      f"Dates: {manifest['dates'][0]} to {manifest['dates'][-1]}",
                      "Generator: " + json.dumps(manifest["generator"], sort_keys=True),
                      "Split day counts: " + ", ".join(f"{name}={len(days)}" for name, days in manifest["splits"].items())])
        lines.extend(_panel_preview(panels, manifest, rows))
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            if "embeddings" in arrays:
                embeddings = arrays["embeddings"]
                dates, bars = arrays["dates"], arrays["last_bar"]
                if (embeddings.ndim not in (2, 3) or dates.shape != (len(embeddings),)
                        or bars.shape != (len(embeddings),)):
                    raise ValueError("Embedding rows, dates and last_bar must be aligned")
                metadata = json.loads(bytes(arrays["metadata"]).decode("utf-8"))
                for key in arrays.files:
                    lines.append(f"{key}: shape={arrays[key].shape}, dtype={arrays[key].dtype}")
                lines.extend([f"Readout: {metadata.get('readout', 'unknown')}; stride={metadata.get('stride', 'unknown')}",
                              f"Model: {metadata.get('tokenizer', 'unknown')}; seed={metadata.get('seed', 'unknown')}; "
                              f"profile={metadata.get('profile', 'unknown')}",
                              f"Synthetic input: {metadata.get('synthetic_data', False)}",
                              f"Checkpoint SHA-256: {metadata.get('checkpoint_sha256', 'unknown')}",
                              f"Corpus SHA-256: {metadata.get('corpus_sha256', 'unknown')}"])
                if len(dates):
                    lines.append(f"Dates: {min(dates.tolist())} to {max(dates.tolist())}")
                label = "first variate, " if embeddings.ndim == 3 else ""
                shown = min(5, embeddings.shape[-1])
                lines.extend([f"Sample: {label}first {shown} embedding dimensions",
                              "date\tlast_bar\t" + "\t".join(f"z{i:03d}" for i in range(shown))])
                for i in range(min(rows, len(embeddings))):
                    values = embeddings[i, 0] if embeddings.ndim == 3 else embeddings[i]
                    lines.append(f"{dates[i]}\t{bars[i]}\t" + "\t".join(f"{float(value):.7g}" for value in values[:shown]))
            elif "_manifest" in arrays:
                manifest = json.loads(bytes(arrays["_manifest"]).decode("utf-8"))
                days = manifest["dates"]
                if not days:
                    raise ValueError("Corpus contains no dates")
                lines.extend([f"Corpus: {len(days)} days; synthetic={manifest.get('synthetic', False)}",
                              f"Dates: {min(days)} to {max(days)}",
                              f"_manifest: shape={arrays['_manifest'].shape}, dtype={arrays['_manifest'].dtype}",
                              "Daily array shapes/dtypes: " + ", ".join(sorted({
                                  f"{arrays[f'd_{day}'].shape} {arrays[f'd_{day}'].dtype}" for day in days})),
                              f"Daily keys: d_{days[0]} through d_{days[-1]}"])
                lines.extend(_panel_preview(arrays, manifest, rows))
            else:
                raise ValueError("Expected a corpus NPZ or an embedding NPZ")
    else:
        raise ValueError("--input must be a .csv or .npz file")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Inspect raw feature tables, corpus NPZ files, or exported embeddings.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()
    try:
        print(inspect_file(args.input, rows=args.rows))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
