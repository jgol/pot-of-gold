# Genetic programming for 0DTE options strategies

This module evaluates whether learned market representations improve the
discovery of zero-days-to-expiration (0DTE) options strategies. Strongly typed
genetic programming (GP) evolves rules for entry, exit, position size, option
delta, spread width and structure. A frozen transformer encoder supplies
**embeddings**: numerical vectors representing recent market observations.

The [published results report](results/report.html) contains the original
experiment's measurements, definitions, tables and reproducible Seaborn plots.
Download the HTML file and open it in a browser; it includes its figures and
requires no server. This directory also contains the search implementation,
archived strategies, saved outcomes and commands for reproduction.

## Study design

The experiment compares four input conditions under the same search settings:

| Condition | Search inputs | Search seeds × folds |
| --- | --- | --- |
| Indicators | Scalar market indicators | 5 × 5 |
| Embeddings | Encoder vectors accessed through fitted projections and reference prototypes | 5 × 5 |
| Combined | Both scalar indicators and embedding readouts | 5 × 5 |
| Permuted embeddings | Embedding rows shuffled relative to market observations, as a negative control | 1 × 5 |

The original configuration uses **256 individuals and 100 generations** per
run. The calendar contains **882 dates, from September 19, 2022, through
April 9, 2026**. All bars from a day stay together in one of five folds; four
folds train the search and the remaining fold evaluates frozen strategies.
These grouped folds do not represent chronological walk-forward deployment.

Embedding readouts and threshold landmarks are fitted on each fold's training
rows. The encoder is frozen, and scalar normalization uses the released fixed
constants. The encoder's own training period overlaps the GP evaluation
calendar; holding out a GP fold does not mean its dates were unseen during
encoder training. [Data and preprocessing details](data/DATA.md) identify the
checkpoint, training dates and input requirements.

The search uses NSGA-III multiobjective selection and an options pricing and
execution proxy. Allowed structures are put verticals, call verticals, iron
condors, iron butterflies and long strangles. Reported profit and loss (P&L)
uses the proxy's native units; it is not a dollar account return.

The archive retains **1,999 saved front members across 80 runs**, plus **14
champions outside those fronts**. Losing strategies remain included. Discovery
yield is the fraction of eligible front members with positive training and
held-out mean P&L. Eligibility requires held-out trading activity and a finite
training mean. This descriptive measure is separate from the search's champion
acceptance tests. Extra champions are excluded from front-based denominators.

## Reproduction scope

| Task | What is required |
| --- | --- |
| Read the published report | The self-contained `results/report.html` |
| Recalculate all report metrics and redraw its plots | Released strategies, `results/outcomes.npz` and the installed package |
| Execute the full workflow on synthetic data | The commands below and the released encoder checkpoint |
| Replay original trades or rerun the original search | Matching licensed market inputs, aligned targets and the recorded settings |

Original fitted embedding readout banks were not saved. Archived-strategy
replay reconstructs them from the original training inputs. Those licensed
inputs cannot be redistributed, so the released files support recalculating
the archived measurements but do not establish an exact historical rerun.
Synthetic data demonstrates execution and does not reproduce the original
experiment or establish live trading performance.

## Original experiment assets

These files are preserved in the repository. Reproduction writes new outputs
under `runs/`.

| File | Contents |
| --- | --- |
| [strategies/](strategies/) | One JSON file per strategy, grouped by condition, seed and fold |
| [strategies/manifest.json](strategies/manifest.json) | Required archive index: fold paths, complete calendar, strategy counts, outcome checksum and source provenance |
| `results/outcomes.npz` | Archived training outcomes and dated out-of-sample outcomes, execution masks and always-enter reference outcomes |
| [results/report.html](results/report.html) | Published report with tables and embedded Seaborn plots |
| `checkpoints/encoder_convolutional_gp_seed42.pt` | Original encoder used for the GP input pipeline; bytes preserved |
| [configs/experiment.yaml](configs/experiment.yaml) | Original search budget and fold settings |

The strategy folders follow this structure:

```text
strategies/
  manifest.json
  embeddings/
    seed_0/
      fold_0/
        fold.json
        embeddings-s0-f0-m000.json
        embeddings-s0-f0-m001.json
```

Each strategy file contains its entry, exit, size, delta, width and shape rules,
parameters, identifiers and recorded measurements. For example, open
[embeddings-s0-f0-m000.json](strategies/embeddings/seed_0/fold_0/embeddings-s0-f0-m000.json).
The adjacent `fold.json` holds shared dates, settings and the strategy file list.
The preserved `strategy_id` connects each file to its row in `results/outcomes.npz`.

Metric recalculation reads `strategies/manifest.json` to locate the folds and
check the calendar, counts and outcome checksum.
Archived-strategy replay uses it to recover the full calendar and identify the
training days. New search runs save their own fold records; they do not regenerate
this required archive index.

This GP checkpoint is distinct from the seed-4395 checkpoint in `transformer/`.
See [data/DATA.md](data/DATA.md) for its input and preprocessing requirements.

## Installation

Use Python **3.12**. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[search,reports]'
python -m pip check
cd evolutionary_search
```

Installation and archived-report reproduction were verified from a clean copy
of the committed files on macOS (Apple silicon), using Python 3.12. All
dependencies are public PyPI packages; no private package feed is required.

If a configured package mirror returns an HTTP 404, retry the install from the
repository root with `--no-cache-dir`. On networks with direct PyPI access, add
`--index-url https://pypi.org/simple` to bypass the mirror.

Run the following commands from `evolutionary_search/`. Installation includes
the shared transformer package and the evolutionary search module.

## View or reproduce the original report

Open the [published report](results/report.html) to read the results. To
recalculate its metrics from the saved strategy outcomes and generate a new report:

```bash
potofgold-search-evaluate --strategies strategies \
  --outcomes results/outcomes.npz --out runs/original/metrics.json

potofgold-search-report --results runs/original/metrics.json \
  --out runs/original/report.html
```

The evaluator generates `runs/original/metrics.json` from the archive. This JSON
records calculated results, resampling settings, source checksums and calculation
code hashes. The renderer uses it to generate `runs/original/report.html` with
embedded Seaborn plots. Both reproduction outputs are Git-ignored; the published
report remains available at `results/report.html`.

Discovery comparisons use 2,000 independent strategy-record bootstrap
resamples. Portfolio comparisons use 10,000 circular block resamples of five
dates, preserving within-block dependence. Both methods report 95% confidence
intervals; their assumptions and seeds are recorded in the generated metrics.
Nontrading days carry zero P&L, with an execution mask distinguishing them from
traded days that broke even. The report defines the probabilistic Sharpe ratio
(PSR), conditional value at risk (CVaR) and the probability of backtest
overfitting (PBO) diagnostic. PBO is conditional on the saved fronts; the archive
does not contain the complete history of candidates evaluated during evolution.

## Demonstration workflow

The following steps generate inputs, search for strategies, replay a saved
strategy and produce a new report.

### 1. Generate synthetic features and market inputs

```bash
potofgold-search-generate-data --out runs/demo --days 30 --seed 0
potofgold-inspect --input runs/demo/features.csv --rows 5
```

This creates inspectable `features.csv` and `market.csv` tables without licensed
observations. They demonstrate the pipeline; their performance is not evidence
about live trading or the original experiment.

### 2. Pack the encoder features

```bash
potofgold-pack-data --input runs/demo/features.csv --out runs/demo/corpus.npz
potofgold-inspect --input runs/demo/corpus.npz --rows 5
```

The NumPy archive (NPZ) stores the daily feature matrices and their manifest.
Packing does not train an encoder or run a search.

### 3. Prepare the GP panel and targets

```bash
potofgold-search-prepare --data runs/demo/corpus.npz \
  --market runs/demo/market.csv \
  --checkpoint checkpoints/encoder_convolutional_gp_seed42.pt \
  --out runs/demo/prepared.parquet --targets-out runs/demo/targets.npz
```

Preparation applies the saved encoder and pools its multilayer output into
**13 vectors of 384 dimensions per bar**. It writes these alongside the market
inputs in Parquet, a tabular file format, and generates `targets.npz` alongside it.

The [target builder](src/potofgold/evolutionary_search/input_tools/targets.py)
calculates six future-outcome labels: realized minus implied volatility,
volatility-scaled return, upward and downward price excursions, and the times to
the future high and low. During search, training-fold labels are used to fit the
embedding projection directions and reference prototypes accessed by `EmbProj`,
`EmbDir` and `PROTO` expressions. Preparation creates the labels; search fits the
readouts and evolves strategies.

Inspect a saved vector without printing the whole panel:

```bash
python - <<'PY'
import pandas as pd
panel = pd.read_parquet("runs/demo/prepared.parquet", columns=["date", "bar_position", "EMB_SHARED"])
row = panel.iloc[0]
print(row["date"], "bar", row["bar_position"])
print("Vector length:", len(row["EMB_SHARED"]), "First values:", row["EMB_SHARED"][:5])
PY
```

The same preparation command accepts correctly formatted real data. To use an
encoder trained through the transformer workflow, supply its checkpoint path;
this produces inputs for a new experiment. An indicators-only run can omit
embedding inference with `--without-embeddings`.

### 4. Run each search condition separately

The short configuration is for checking the workflow. Run one or more conditions:

```bash
potofgold-search --data runs/demo/prepared.parquet --targets runs/demo/targets.npz \
  --condition indicators --config configs/demo.yaml --seed 0 --fold 0 \
  --out runs/demo/indicators

potofgold-search --data runs/demo/prepared.parquet --targets runs/demo/targets.npz \
  --condition embeddings --config configs/demo.yaml --seed 0 --fold 0 \
  --out runs/demo/embeddings

potofgold-search --data runs/demo/prepared.parquet --targets runs/demo/targets.npz \
  --condition combined --config configs/demo.yaml --seed 0 --fold 0 \
  --out runs/demo/combined
```

Each run writes files to its `--out` directory:

| Generated file | Contents |
| --- | --- |
| `fold_<N>.json` | Strategy front, champion, held-out daily outcomes, training/test dates and run settings |
| `fold_<N>.readouts.npz` | Fitted embedding directions and prototypes for embedding-based conditions |

For example, open `runs/demo/embeddings/fold_0.json` to inspect the entry, exit,
size, delta, width and shape trees. These new-run files are evaluated with `--runs`
in step 6; the published archive is evaluated with `--strategies` and `--outcomes`.
One demonstration fold exercises the workflow but does not reproduce the
five-fold experiment.

For a full run, use `configs/experiment.yaml`, repeat `--fold` for 0 through 4,
and use a separate output directory per condition and search seed. The recorded
seeds are 0 through 4. The control uses `--condition permuted_embeddings --seed 1`
and a permutation seed of 42: one shared permutation of the embedding rows is
applied before fitting readouts. It preserves alignment between the 13 embedding
groups while breaking their alignment with the market observations.

### 5. Replay a saved strategy

Use a `strategy_id` from the run's `surviving_set`. For example:

```bash
potofgold-search-replay --run runs/demo/embeddings/fold_0.json \
  --strategy-id embeddings-s0-f0-m000 --data runs/demo/prepared.parquet \
  --out runs/demo/replay.json
```

The default replays its held-out days using the saved training-fold readouts.
The JSON contains individual trades and daily outcomes. `--split all` evaluates
the frozen strategy on all supplied days and is intended for a new input panel;
it does not make those days out of sample automatically.

For an archived strategy, supply its individual file directly:

```bash
potofgold-search-replay \
  --strategy strategies/embeddings/seed_0/fold_0/embeddings-s0-f0-m000.json \
  --data /path/to/original/prepared.parquet \
  --targets /path/to/original/targets.npz \
  --out runs/original/embeddings-s0-f0-m000-replay.json
```

The strategy ID is read from that file. Keep it in its condition/seed/fold folder
so replay can find `fold.json` and `manifest.json`. Supply the complete original
minute panel and aligned targets.
Its missing readout coefficients are reconstructed from the recorded training
fold, and the output identifies that reconstruction. The original input panel
must match the archived dates and preprocessing; a matching date range alone
does not establish identical numerical inputs.

### 6. Calculate metrics and render a report

After running any one or more of the conditions above:

```bash
potofgold-search-evaluate --runs runs/demo --out runs/demo/metrics.json
potofgold-search-report --results runs/demo/metrics.json --out runs/demo/report.html
```

The report supports a single condition. Comparisons appear when the required
conditions are present. Small demonstration samples may not support some risk
statistics; unavailable measurements are identified explicitly.

The commands above write working files under the Git-ignored `runs/` directory.
Reports embed their plots rather than writing separate image files. Use a new
output path or supply `--overwrite` to replace a previous output deliberately.

## Source layout

| Under `src/potofgold/evolutionary_search/` | Responsibility |
| --- | --- |
| `search/` | Typed grammar, genetic operations, NSGA-III selection, proxy backtesting, fitness, fold fitting and replay |
| `input_tools/` | Market-input validation, encoder readout pooling and aligned financial targets |
| `analysis/` | Saved-outcome loading, metrics, confidence intervals and HTML reporting |
| `sim_tools/` | Synthetic-data generation only |

[Repository overview](../README.md) · [Transformer](../transformer/README.md) ·
[Data requirements](data/DATA.md) · [Citation](../CITATION.cff) · [License](../LICENSE)
