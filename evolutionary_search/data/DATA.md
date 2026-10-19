# Evolutionary search inputs

Genetic programming (GP) search consumes a minute-level Parquet table and a
row-aligned NumPy archive (NPZ) of six financial targets. Preparation uses a
frozen encoder; supervised embedding readouts are fitted later within each
search fold. Licensed market observations are not distributed.

## Generate a demonstration

From `evolutionary_search/`, after installing the repository:

```bash
potofgold-search-generate-data --out runs/demo --days 30 --seed 0
potofgold-pack-data --input runs/demo/features.csv --out runs/demo/corpus.npz
potofgold-search-prepare --data runs/demo/corpus.npz --market runs/demo/market.csv \
  --checkpoint checkpoints/encoder_convolutional_gp_seed42.pt \
  --out runs/demo/prepared.parquet --targets-out runs/demo/targets.npz
```

The first command writes readable `features.csv` and `market.csv`. Generation
reuses `potofgold.sim_tools.simulation` and its shared factors, noise and return
path. The market prices follow that same return path from an initial spot of
5000; subsequent opening gaps use the shared first-bar return factor. For this
execution demonstration, the adapter converts realized volatility to
unannualized standard deviation, scales fractional spreads by 0.2, and supplies
a 405-bar settlement clock. These adaptations are recorded in the CSV metadata.
This process is illustrative and is not calibrated to market option prices.

Packing remains the existing transformer's separate command: it validates and
stores 405 bars × 248 raw float16 features per day. It does not normalize data.
The same format works with the existing transformer training and inspection
commands. Synthetic measurements do not reproduce the archived experiments.

For indicator-only search, replace `--checkpoint ...` with
`--without-embeddings`; this path does not import torch. To reuse vectors from
an earlier prepared Parquet, use `--embeddings runs/demo/prepared.parquet` and
new output names. Reuse validates all thirteen vectors, row keys, the corpus
hash, and IV repair/cold-start settings. Outputs are not replaced unless
`--overwrite` is supplied.

## Real inputs

`--data` accepts the existing daily NPZ contract: a JSON `_manifest` byte array
with ordered `dates`, and `d_YYYY-MM-DD` arrays. At least the original 159 raw
positions must be present; the released corpus layout has 248. The encoder
selects the same 147 features. See the [transformer data description](../../transformer/data/DATA.md).

`--market` accepts CSV or Parquet with `date` (`YYYY-MM-DD`), integer
`bar_position`, and positive absolute `SPXClose`. Every corpus day must have
every bar, starting at zero, with no duplicate keys. Relative price moves are
checked against raw column 105 to reject inconsistent joins. Absolute prices
cannot be inferred from a return-only feature panel.

The preparer derives the remaining scalars from the same raw day. Supplying
named scalar columns in `--market` preserves those values before IV repair;
this supports existing reconciled historical inputs. No embedding is accepted
through the market file: use `--embeddings` explicitly.

| Prepared input | Meaning / source |
| --- | --- |
| `ATM_IV` | Annualized fractional volatility; derived from raw 41, capped at 0.8 unless supplied |
| `IV_5Dp` … `IV_5Dc` | Eleven annualized fractional IVs, raw `1 + 8k`; original grid order |
| `RawSpread`, `DeltaSpread1`, `DeltaSpread5` | Fractional spread and trailing within-day differences; derived raw 46 spikes above 0.035 carry earlier quotes |
| `MinutesToClose` | Nonnegative minutes remaining, raw 114; raw opening-zero artifact repaired using bar 1 |
| `VIXSpot`, `VIXTermSlope` | VIX in volatility points; slope is VIX9D minus VIX |
| `RealizedVol30m` | Unannualized Parkinson standard deviation, raw 112; partial trailing window during zero warmup |
| `PutCallSkew` | 25-delta put IV minus call IV |
| `SessionPosition`, `OvernightGap` | Running price-range position and opening gap |
| `NetGammaProxy` | Raw 137, already signed-log compressed; corrupt-ATM-IV bars neutralized |
| `GridReliability` | Fraction of positive raw reliability markers 88–98 |

The execution engine derives strikes and option marks from spot, time, spreads
and the IV grid. The synthetic command does not implement a separate pricer.

## Embeddings and preprocessing

Preparation loads the shared `potofgold.encoder.inference.load_encoder`
strictly. It encodes and immediately pools each batch, without storing the
147-vector intermediate for the full corpus. Each final `EMB_*` value is a
float32 vector of length 384: three normalized 128-dimensional block outputs
concatenated in layer order. The transformer's default pooled 128-vector is
not this GP input.

| Pooled embedding | Encoder token positions, half-open |
| --- | --- |
| `EMB_SHARED` | Mean of all 147 |
| `EMB_GRID`, `EMB_STRIKE`, `EMB_SPX` | 0:88, 88:94, 94:108 |
| `EMB_VIX`, `EMB_FLOW_RAW`, `EMB_FLOW_AGG` | 108:120, 120:129, 129:147 |
| `EMB_GRID_PUT`, `EMB_GRID_ATM`, `EMB_GRID_CALL` | 0:32, 32:56, 56:88 |
| `EMB_VAR_ATM_IV`, `EMB_VAR_ATM_SPREAD`, `EMB_VAR_SPX_RET` | Tokens mapped from raw feature indices 41, 46, 105 |

The historical checkpoint hash automatically selects `historical-gp`
preprocessing: raw signed-log transforms on 101, 102, 104 and 139. Other
checkpoints use `transformer`, which also transforms 132 as recorded in the
tokenizer study. Both reuse shared gamma clipping, settlement-tail filling and
VIX forward filling, then use saved checkpoint log transforms and z-score
statistics and clip to [-5, 5]. No normalization is fitted during preparation.
The GP scalar terminals separately use the fixed constants in
[search/normalization.py](../src/potofgold/evolutionary_search/search/normalization.py);
these constants are not estimated anew for each fold.

By default an IV above `4 × VIXSpot/100` is replaced with `VIXSpot/100`, before
encoder input selection and in market/scalar targets. `--iv-repair-mult 0`
disables that recorded repair. `--cold-start masked` includes bars 0–58 using
right-aligned observed prefixes, the saved mask-fill value, and the model's
mask indicator. `--cold-start omit` begins at bar 59. Targets are computed from
the complete market session before this filtering. Batch size and core count
are explicit options (`--batch-size 32 --max-cores 2` by default).

## Targets and provenance

The target archive contains `targets` (N, 6 float32), `valid` (N bool),
`date`, `bar_position`, `target_names`, `primary_target_idx` (0), and JSON-byte
`metadata`. Row keys exactly match the Parquet. Labels use strictly future bars
through the final supplied row of the same day; the last five rows are invalid.
Invalid targets are zero placeholders and must be excluded using `valid`.

The six targets are `volatility_difference`, `volatility_scaled_return`,
`upward_excursion`, `downward_excursion`, `time_to_upward_extremum`, and
`time_to_downward_extremum`. The first four reuse the transformer analysis
formulas: population return standard deviation multiplied by
`sqrt(252 × full day bar count)`, minus anchor IV; IV-scaled forward return;
and the two excursions.
The last two are first-extremum positions divided by the remaining bar count.
Targets remain unscaled and unresidualized. All supervised basis fitting and
residualization happen within each search fold's training rows.

The Parquet schema metadata key `potofgold.evolutionary_search.inputs` records
source/checkpoint hashes, feature order, preprocessing, IV repair, cold starts,
market overrides and whether inputs are synthetic. Target metadata also binds
the exact prepared Parquet hash and records repair counts. No sidecar file is
needed. The outputs alone do not establish historical reproduction.

## Historical checkpoint identity

`checkpoints/encoder_convolutional_gp_seed42.pt` preserves the original GP
encoder checkpoint bytes:

```text
SHA-256 93a428ebe0fd8bbb29f9adbed31470046037d155389a6fd28fc00aaf7d62a6ca
```

Its recorded objective predicts a residual future representation. It uses seed
42, 100 epochs, 147 variables,
60-bar windows, width 128 and three layers. Recorded train dates are
2022-09-19–2024-09-30; validation dates are 2024-10-08–2025-01-31, with five
embargo days. Original training source revision:
`e09136a3728293dc4b190e1c78343fc95af65236`.

This differs from the transformer's released seed-4395 convolutional checkpoint
(`7b9127e6897e33b2c6f44691ed9de5e80918ed67976a2fce36159451cb1d0ba7`),
trained through 2025-09-30. That checkpoint supports new downstream runs but
cannot substitute for the historical GP encoder when reproducing archived runs.

The encoder's recorded training period overlaps the GP evaluation calendar.
The GP folds hold out days from readout fitting and strategy evolution, but
they do not establish that those days were unseen by the encoder.

Archived GP records identify an IV-repaired minute Parquet input whose market
columns were reconciled while preserving the embedding columns. The recorded
embedding provenance identifies the historical checkpoint hash above. The
final reconciled input was unavailable for direct hash verification when the
archive was prepared for release. Exact trade replay therefore remains
unverified; it requires the original input panel, aligned targets and recorded
search and execution settings. Recalculating metrics from the released daily
outcomes does not require that panel.
