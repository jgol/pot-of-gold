"""Generate a reproducible demonstration corpus without using market observations.

The panel has the experiment's shape and feature order. Shared persistent factors,
heavy-tailed innovations and trailing aggregates give it temporal and cross-variate
structure. Values are illustrative; this is not an options pricing simulator.
"""
import argparse
import csv
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np

from ..encoder.config import EXPERIMENT_PROFILE, feature_names

N_BARS = 405
N_VARIATES = 248
CSV_FORMAT = "potofgold.synthetic.csv.v1"


def _output_path(out, suffix, overwrite):
    out = Path(out)
    if out.suffix != suffix:
        raise ValueError(f"The output must have a {suffix} extension")
    if out.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {out}; choose another path or use --overwrite")
    return out


def _dates(start, end, count):
    days = np.arange(np.datetime64(start), np.datetime64(end) + 1)
    days = days[np.is_busday(days)]
    if not 1 <= count <= len(days):
        raise ValueError(f"Requested {count} days; {start}..{end} has {len(days)} weekdays")
    return days[np.linspace(0, len(days) - 1, count, dtype=int)].astype(str).tolist()


def _rolling_mean(x, width):
    c = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    ends = np.arange(1, len(x) + 1)
    starts = np.maximum(ends - width, 0)
    return (c[ends] - c[starts]) / (ends - starts)


def _day(rng):
    factors = np.empty((N_BARS, 3))
    factors[0] = rng.normal(size=3)
    innovations = rng.standard_t(6, size=(N_BARS, 3)) * 0.18
    for bar in range(1, N_BARS):
        factors[bar] = factors[bar - 1] * [0.96, 0.90, 0.75] + innovations[bar]
    noise = rng.normal(size=(N_BARS, N_VARIATES))
    x = np.zeros((N_BARS, N_VARIATES), dtype=np.float64)
    remaining = np.maximum(390 - np.arange(N_BARS), 0)
    returns = 0.00015 * (factors[:, 0] + noise[:, 105])
    session_return = np.cumsum(returns)
    iv = np.clip(0.19 + rng.normal(0, 0.02) + 0.035 * factors[:, 1], 0.06, 0.65)
    for grid in range(11):
        start = grid * 8
        distance = (grid - 5) / 5
        x[:, start] = np.maximum(0.01, (1 + iv * 15) * np.sqrt((remaining + 1) / 390)
                                  + distance * session_return * 100 + 0.1 * noise[:, start])
        x[:, start + 1] = np.maximum(0.01, iv + 0.025 * distance ** 2 + 0.006 * noise[:, start + 1])
        x[:, start + 2] = -0.05 * (1 + np.abs(factors[:, 1]) + 0.2 * np.abs(noise[:, start + 2]))
        x[:, start + 3] = 0.01 * np.exp(-distance ** 2) * (1 + np.abs(noise[:, start + 3]))
        x[:, start + 4] = np.maximum(1, 100 + 20 * factors[:, 2] + 15 * noise[:, start + 4])
        x[:, start + 5] = np.tanh(factors[:, 0] + 0.4 * noise[:, start + 5])
        x[:, start + 6] = np.maximum(0.01, 0.04 + iv * 0.1 + 0.01 * noise[:, start + 6])
        x[:, start + 7] = np.diff(x[:, start + 6], prepend=x[0, start + 6])
    x[:, 88:99] = 1.0
    x[:, 99:105] = factors @ rng.normal(size=(3, 6)) + 0.3 * noise[:, 99:105]
    x[:, 105] = session_return
    x[:, 106] = np.abs(returns) + 0.0001 * np.abs(noise[:, 106])
    x[:, 107] = returns
    for column, width in [(108, 5), (109, 15), (110, 30)]:
        x[:, column] = session_return - np.concatenate((np.zeros(width), session_return[:-width]))
    for column, width in [(111, 15), (112, 30), (113, 60)]:
        x[:, column] = np.sqrt(_rolling_mean(x[:, 106] ** 2, width) * 252 * 390 / (4 * np.log(2)))
    x[:, 114] = remaining
    x[:, 115] = x[:, 106] / np.maximum(_rolling_mean(x[:, 106], 30), 1e-8)
    x[:, 116] = session_return - _rolling_mean(session_return, 30)
    x[:, 117] = np.expm1(session_return)
    x[:, 118] = np.maximum(0.1, 1 + 0.3 * factors[:, 2] + 0.1 * noise[:, 118])
    levels = rng.uniform(14, 28) + np.arange(4) * rng.normal(0.6, 0.2)
    x[:, 119:123] = levels
    x[:, 123:125] = levels[0] + np.array([1.0, 2.0]) + 0.2 * factors[:, 1, None]
    x[:, 125] = x[:, 120] / x[:, 119]
    x[:, 126] = x[:, 119] / x[:, 121]
    x[:, 127] = x[:, 119] / x[:, 122]
    x[:, 128] = x[:, 123] - x[:, 124]
    x[:, 129] = (x[:, 123] - x[:, 119]) / x[:, 119]
    x[:, 130] = (levels[3] - levels[0]) / 171
    x[:, 131:140] = factors @ rng.normal(size=(3, 9)) + 0.35 * noise[:, 131:140]
    x[:, 131] = np.exp(np.clip(x[:, 131], -3, 3))
    for col in (132, 137):
        x[:, col] = np.sign(x[:, col]) * np.log1p(np.abs(x[:, col]))
    x[:, 138] = np.sign(returns)
    x[:, 139] = np.diff(x[:, 135], prepend=x[0, 135])
    for source in range(131, 140):
        x[:, source + 10] = _rolling_mean(x[:, source], 3)
        x[:, source + 19] = _rolling_mean(x[:, source], 15)
    # Column 140 and the unused 159..247 block stay zero, matching their exclusion.
    return x.astype(np.float16)


def _generate(seed, train_days, val_days, test_days):
    splits = {name: _dates(EXPERIMENT_PROFILE[f"{name.upper()}_START"], EXPERIMENT_PROFILE[f"{name.upper()}_END"], count)
              for name, count in [("train", train_days), ("val", val_days), ("test", test_days)]}
    dates = sorted(sum(splits.values(), []))
    rng = np.random.default_rng(seed)
    data = {f"d_{day}": _day(rng) for day in dates}
    manifest = {
        "dates": dates, "n_bars": N_BARS,
        "n_variates": N_VARIATES, "synthetic": True, "splits": splits,
        "generator": {"name": "potofgold.sim_tools.simulation", "version": 1, "seed": seed},
        "description": "Simulated feature panel; no real market data or empirical performance claims.",
    }
    return data, manifest


def _write_corpus(out, panels, manifest):
    data = {key: panel.astype(np.float16) for key, panel in panels.items()}
    data["_manifest"] = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)


def generate_corpus(out, *, seed=0, train_days=16, val_days=8, test_days=4, overwrite=False):
    """Generate a reproducible corpus directly as daily NPZ arrays."""
    out = _output_path(out, ".npz", overwrite)
    data, manifest = _generate(seed, train_days, val_days, test_days)
    _write_corpus(out, data, manifest)
    return manifest


def generate_csv(out, *, seed=0, train_days=16, val_days=8, test_days=4, overwrite=False):
    """Write readable raw values, with generator metadata in the first comment line."""
    out = _output_path(out, ".csv", overwrite)
    data, manifest = _generate(seed, train_days, val_days, test_days)
    manifest = dict(manifest, table_format=CSV_FORMAT)
    names = ["date", "bar", *feature_names()]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        stream.write("# " + json.dumps(manifest, sort_keys=True) + "\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(names)
        for day in manifest["dates"]:
            # Python floats preserve every represented float16 value on a CSV round trip.
            for bar, values in enumerate(data[f"d_{day}"].tolist()):
                writer.writerow([day, bar, *values])
    return manifest


def read_csv(path):
    """Validate and read a complete raw synthetic table; do not preprocess its values."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        first = stream.readline()
        if not first.startswith("# "):
            raise ValueError("CSV must start with its '# {JSON}' generator metadata line")
        manifest = json.loads(first[2:])
        if (not isinstance(manifest, dict) or manifest.get("table_format") != CSV_FORMAT
                or manifest.get("synthetic") is not True
                or manifest.get("n_bars") != N_BARS or manifest.get("n_variates") != N_VARIATES):
            raise ValueError("CSV metadata does not describe the supported synthetic panel format")
        generator = manifest.get("generator", {})
        if (not isinstance(generator, dict)
                or generator.get("name") != "potofgold.sim_tools.simulation"
                or generator.get("version") != 1 or type(generator.get("seed")) is not int):
            raise ValueError("CSV metadata must retain the generator name, version and integer seed")
        days = manifest.get("dates")
        if (not isinstance(days, list) or not days or not all(isinstance(day, str) for day in days)
                or days != sorted(set(days))):
            raise ValueError("CSV metadata dates must be a nonempty sorted list without duplicates")
        for day in days:
            if date.fromisoformat(day).isoformat() != day:
                raise ValueError(f"Date must use YYYY-MM-DD: {day}")
        splits = manifest.get("splits", {})
        if not isinstance(splits, dict) or set(splits) != {"train", "val", "test"}:
            raise ValueError("CSV metadata must identify train, val and test dates")
        split_days = []
        for name, assigned in splits.items():
            if (not isinstance(assigned, list) or not assigned
                    or not all(isinstance(day, str) for day in assigned)
                    or assigned != sorted(set(assigned))):
                raise ValueError(f"Invalid {name} split dates")
            start, end = (EXPERIMENT_PROFILE[f"{name.upper()}_{bound}"] for bound in ("START", "END"))
            if any(not start <= day <= end for day in assigned):
                raise ValueError(f"Dates outside the experiment's {name} range")
            split_days.extend(assigned)
        if sorted(split_days) != days:
            raise ValueError("Split dates must cover each CSV date exactly once")
        reader = csv.reader(stream)
        if next(reader, None) != ["date", "bar", *feature_names()]:
            raise ValueError("CSV header must contain date, bar and the 248 named raw features in order")
        panels = {f"d_{day}": np.empty((N_BARS, N_VARIATES), dtype=np.float64) for day in days}
        seen = {day: np.zeros(N_BARS, dtype=bool) for day in days}
        for line, row in enumerate(reader, start=3):
            if len(row) != N_VARIATES + 2:
                raise ValueError(f"CSV line {line}: expected date, bar and 248 feature values")
            day = row[0]
            if day not in seen:
                raise ValueError(f"CSV line {line}: date is absent from the metadata: {day}")
            try:
                bar = int(row[1])
                values = np.asarray(row[2:], dtype=np.float64)
            except ValueError as error:
                raise ValueError(f"CSV line {line}: bar and features must be numeric") from error
            if not 0 <= bar < N_BARS or seen[day][bar]:
                raise ValueError(f"CSV line {line}: duplicate or out-of-range bar {bar} for {day}")
            if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo(np.float16).max):
                raise ValueError(f"CSV line {line}: features must be finite and within the float16 range")
            panels[f"d_{day}"][bar] = values
            seen[day][bar] = True
        if any(not filled.all() for filled in seen.values()):
            raise ValueError("Every date must have exactly 405 bars numbered 0 through 404")
    return panels, manifest


def pack_csv(source, out, *, overwrite=False):
    """Package validated rows as float16 daily arrays, without normalization or targets."""
    out = _output_path(out, ".npz", overwrite)
    panels, manifest = read_csv(source)
    manifest = dict(manifest, source_csv_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
    _write_corpus(out, panels, manifest)
    return manifest


def pack_main():
    parser = argparse.ArgumentParser(description="Package an inspected synthetic CSV as a corpus NPZ.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        manifest = pack_csv(args.input, args.out, overwrite=args.overwrite)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {args.out}: {len(manifest['dates'])} days × {N_BARS} bars × {N_VARIATES} raw "
          "float16 features. No normalization or targets added.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/synthetic/data.csv"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-days", type=int, default=16)
    parser.add_argument("--val-days", type=int, default=8)
    parser.add_argument("--test-days", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        manifest = generate_csv(args.out, seed=args.seed, train_days=args.train_days,
                                val_days=args.val_days, test_days=args.test_days,
                                overwrite=args.overwrite)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {args.out}: {len(manifest['dates'])} simulated days, "
          f"{N_BARS} bars × {N_VARIATES} columns. Inspect this CSV, then run potofgold-pack-data.")


if __name__ == "__main__":
    main()
