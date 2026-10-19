"""Generate inspectable features and aligned prices from the shared synthetic process."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from potofgold.encoder.config import feature_names
from potofgold.sim_tools.simulation import _generate, CSV_FORMAT
from ..input_tools.market import market_day


def generate(out, *, days=30, seed=0, overwrite=False):
    """Write features.csv and market.csv; packing and embedding export are separate.

    The same shared factors and return path drive both files. Adaptations make
    units explicit for the GP: unannualized Parkinson volatility, a 405-bar
    settlement clock, and fractional bid/ask spreads. This is an illustrative
    feature process, not a calibrated market or option-price simulator.
    """
    import pandas as pd
    if days < 3:
        raise ValueError("--days must be at least 3 (one date in each shared date split)")
    out = Path(out)
    outputs = [out / "features.csv", out / "market.csv"]
    if not overwrite:
        for path in outputs:
            if path.exists():
                raise FileExistsError(f"Output exists: {path}; choose a new directory or --overwrite")
    n_train = max(1, int(days * 0.6))
    n_val = max(1, int(days * 0.25))
    n_test = days - n_train - n_val
    panels, manifest = _generate(seed, n_train, n_val, n_test)
    manifest = dict(manifest, table_format=CSV_FORMAT,
                    execution_demo={
                        "adapter": "potofgold.evolutionary_search.sim_tools.simulation", "version": 1,
                        "shared_generator": manifest["generator"], "initial_spot": 5000.0,
                        "clock": "405 supplied minute bars, MinutesToClose=404-bar_position",
                        "raw_realized_volatility": "unannualized Parkinson standard deviation",
                        "spread": "fractional quote width; shared synthetic spread multiplied by 0.2",
                        "pricing": "search simulator derives option marks from the same IV grid and spot",
                    })
    out.mkdir(parents=True, exist_ok=True)
    market_frames = []
    previous_close = None
    for day in manifest["dates"]:
        panel = panels[f"d_{day}"].astype(np.float32)
        # Reuse the shared realization, changing declared units and clock only.
        panel[:, 111:114] /= np.sqrt(252.0 * 390.0)
        panel[:, 114] = np.arange(len(panel) - 1, -1, -1)
        panel[:, 6:88:8] *= 0.2
        panel[:, 7:88:8] *= 0.2
        panel = panel.astype(np.float16)
        panels[f"d_{day}"] = panel
        # Reuse the day's shared return factor for a varying overnight move;
        # no independent price process is introduced beside the feature panel.
        opening = 5000.0 if previous_close is None else previous_close * np.exp(12.0 * float(panel[0, 107]))
        price = opening * np.exp(panel[:, 105].astype(np.float64))
        supplied = pd.DataFrame({"date": day, "bar_position": np.arange(len(panel)), "SPXClose": price})
        frame, _ = market_day(panel, supplied, previous_close=previous_close, iv_repair_mult=4.0)
        market_frames.append(frame)
        previous_close = float(price[-1])
    with outputs[0].open("w", newline="", encoding="utf-8") as stream:
        stream.write("# " + json.dumps(manifest, sort_keys=True) + "\n")
        writer = csv.writer(stream)
        writer.writerow(["date", "bar", *feature_names()])
        for day in manifest["dates"]:
            for bar, values in enumerate(panels[f"d_{day}"].tolist()):
                writer.writerow([day, bar, *values])
    market = pd.concat(market_frames, ignore_index=True)
    with outputs[1].open("w", newline="", encoding="utf-8") as stream:
        stream.write("# " + json.dumps({"synthetic": True, "seed": seed,
                                        "execution_demo": manifest["execution_demo"]}, sort_keys=True) + "\n")
        market.to_csv(stream, index=False)
    return {"days": days, "rows": len(market), "features": str(outputs[0]), "market": str(outputs[1])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Directory for features.csv and market.csv")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = generate(**vars(args))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {result['features']} and {result['market']}: {result['days']} synthetic days, "
          f"{result['rows']} rows. Inspect, then pack features with potofgold-pack-data.")


if __name__ == "__main__":
    main()
