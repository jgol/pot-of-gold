# Transformer

Self-supervised iTransformer experiments comparing linear, patch and convolutional
tokenizers on 0DTE options time series. The original model, learning objective and
released checkpoints are retained.

This guide exposes every stage: **generate features → pack data → train → export
embeddings → evaluate → render a report**. Each command reads named inputs and
writes named outputs. It does not launch later stages.

## Original experiment and reproduction requirements

**Recomputing the original experimental measurements requires the original
processed market dataset and the corresponding trained encoder checkpoints,
using the evaluation code in this repository.** The licensed dataset cannot be
redistributed and is not included. Checkpoints alone are insufficient.

| Original artifact | Purpose |
| --- | --- |
| `checkpoints/encoder_linear.pt` | Original linear-tokenizer encoder, seed 4395 |
| `checkpoints/encoder_patch.pt` | Original patch-tokenizer encoder, seed 4395 |
| `checkpoints/encoder_convolutional.pt` | Original convolutional-tokenizer encoder, seed 4395 |
| [results/experiment.json](results/experiment.json) | Archived measurements across five seeds, training summaries and corpus identification |
| [results/report.html](results/report.html) | Readable report generated from those archived measurements |

The checkpoint files and archived numerical results are preserved. Rebuilding the
report from `experiment.json` displays measurements already calculated; it does
not rerun their evaluation. Reevaluating seed 4395 requires the original corpus
and the three released checkpoints. Recomputing the complete five-seed comparison
requires the original corpus and all 15 original checkpoints, including the 12
not distributed here. Synthetic data produces new measurements.

## Install

Use Python **3.12**. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[reports]'
cd transformer
```

Run the remaining commands from `transformer/`. Dependencies and command entry
points are in [pyproject.toml](../pyproject.toml). The `reports` extra supplies
Seaborn and plotting dependencies; installation without it supports training,
embedding export and evaluation.

## 1. Generate and inspect synthetic features

```bash
potofgold-generate-data --out runs/demo/features.csv --seed 0
potofgold-inspect --input runs/demo/features.csv --rows 5
```

The CSV contains a metadata comment followed by `date`, `bar` and 248 named raw
feature columns. It can be inspected as text or opened as a table; with pandas,
use `pd.read_csv(path, comment="#")`. The feature order matches the experiment's
stored panel. The encoder selects 147 of those columns. See the
[original-data description](data/DATA.md) for feature groups and the column-name dictionary.

The default example creates 16 training days, eight validation days and four test
days, each with 405 minute bars. Set `--train-days`, `--val-days` and `--test-days`
to generate a larger example. The existing split dates and five-day embargo still
apply; eight validation days leave three after that embargo.

Synthetic features use persistent shared factors, heavy-tailed noise and trailing
aggregates. They use no licensed observations and are not a calibrated options
simulator. Their purpose is to make the complete workflow runnable and inspectable.

## 2. Pack the data into NPZ

```bash
potofgold-pack-data --input runs/demo/features.csv --out runs/demo/corpus.npz
potofgold-inspect --input runs/demo/corpus.npz --rows 5
```

**NPZ is a NumPy archive of named arrays.** Here it stores one `(405, 248)` array
per day and a manifest recording dates, splits and the synthetic generator.
Packing validates and stores the table as `float16`; it does not normalize the
features, train a model or construct prediction targets. The
[data contract](data/DATA.md) describes the exact format.

## 3. Train one or more encoders separately

Choose any one or more of the commands below. A single encoder is fully supported
through evaluation and report rendering. All three commands use the same data and
configuration, changing only the tokenizer.
These short examples train for **two epochs**; the original experiment used **100**.
All commands use the v3 feature layout and the experiment’s recorded date splits
and preprocessing settings in [encoder/config.py](src/potofgold/encoder/config.py).

```bash
potofgold-train --config configs/tokenizer.yaml --tokenizer linear \
  --data runs/demo/corpus.npz --seed 4395 --epochs 2 \
  --device cpu --max-cores 4 --out runs/demo/encoder_linear.pt

potofgold-train --config configs/tokenizer.yaml --tokenizer patch \
  --data runs/demo/corpus.npz --seed 4395 --epochs 2 \
  --device cpu --max-cores 4 --out runs/demo/encoder_patch.pt

potofgold-train --config configs/tokenizer.yaml --tokenizer convolutional \
  --data runs/demo/corpus.npz --seed 4395 --epochs 2 \
  --device cpu --max-cores 4 --out runs/demo/encoder_convolutional.pt
```

Each checkpoint includes the selected encoder weights, prediction/adversarial
heads, training normalization and training metadata. The original architecture
uses 60-bar windows, 147 variables, width 128, three attention blocks and four
heads. A short synthetic run may not meet the study's representation criteria.

The student predicts a residualized future representation from an EMA teacher,
with variance/covariance penalties, linear decorrelation and a scalar adversary.
The original implementation residualizes teacher targets within both training
and validation batches. Its saved validation loss is a combined, batch-dependent
objective. Original summaries do not contain per-epoch histories or a separate
validation prediction loss. Training resumption is not supported.

## 4. Export and inspect embeddings

Run the export command for each encoder you trained. If you chose only linear,
run only the first export command and the inspection command below. For another
tokenizer, inspect its corresponding embedding file instead.

```bash
potofgold-embed --checkpoint runs/demo/encoder_linear.pt \
  --data runs/demo/corpus.npz --out runs/demo/embeddings_linear.npz

potofgold-embed --checkpoint runs/demo/encoder_patch.pt \
  --data runs/demo/corpus.npz --out runs/demo/embeddings_patch.npz

potofgold-embed --checkpoint runs/demo/encoder_convolutional.pt \
  --data runs/demo/corpus.npz --out runs/demo/embeddings_convolutional.npz

potofgold-inspect --input runs/demo/embeddings_linear.npz --rows 5
```

Each file contains `embeddings`, `dates`, `last_bar` and `metadata`. Dates and
zero-based ending-bar indices identify the input window for every embedding.
The default stride is 30 bars, giving 12 windows per complete day; the default
28-day example therefore produces 336 rows. Evaluation applies the split,
embargo and target-validity rules, so its counts can be smaller.

| `--readout` | Output per window |
| --- | --- |
| `pooled` (default) | One 128-dimensional vector from final-layer mean pooling |
| `variates` | 147 vectors, each 128-dimensional |
| `multilayer` | 147 vectors, each 384-dimensional, concatenating three block outputs |

The experiment's representation metrics use `pooled` embeddings. The coordinates
are learned numerical features; individual dimensions do not have assigned
financial meanings.

## 5. Compute metrics from the saved embeddings

**If you trained and exported only the linear encoder**, compute its geometry and
prediction metrics with:

```bash
potofgold-evaluate --embeddings runs/demo/embeddings_linear.npz \
  --data runs/demo/corpus.npz --checkpoints runs/demo/encoder_linear.pt \
  --indicators --bootstrap 1000 --out runs/demo/metrics_linear.json
```

The optional `--indicators` flag adds an original-target indicator comparison and
works with a single encoder. Omit it if you only want geometry and residual-target
prediction metrics. For a single patch or convolutional encoder, replace `linear`
in the filenames with `patch` or `convolutional`, respectively, in evaluation and reporting.

For representation geometry alone, without the corpus or checkpoint:

```bash
potofgold-evaluate --embeddings runs/demo/embeddings_linear.npz \
  --out runs/demo/geometry.json
```

Optionally, **after training and exporting all three encoders**, run the full
three-encoder comparison, including residual-target prediction and an
original-target indicator comparison:

```bash
potofgold-evaluate \
  --embeddings runs/demo/embeddings_linear.npz runs/demo/embeddings_patch.npz runs/demo/embeddings_convolutional.npz \
  --data runs/demo/corpus.npz \
  --checkpoints runs/demo/encoder_linear.pt runs/demo/encoder_patch.pt runs/demo/encoder_convolutional.pt \
  --indicators --bootstrap 1000 --out runs/demo/metrics.json
```

Evaluation consumes the exported embeddings; it does not retrain or run the
encoder again. Prediction metrics also require the corpus to reconstruct targets
and the checkpoints to supply the saved preprocessing metadata. Regression probes
are fitted on training observations and scored on held-out observations.

Full prediction evaluation uses the study's `pooled`, stride-30 exports. Other
readouts and strides can be inspected and evaluated for geometry; geometry of
variate or multilayer exports uses their mean over variables.

Metrics include effective rank (RankMe), participation ratio, RankIC and R².
Residual-target probes and original-target indicator comparisons are reported
separately. Confidence intervals identify their resampling method; a single
training seed does not provide uncertainty across training seeds. Synthetic
scores are demonstration results, not evidence about real-market performance.

## 6. Render a clean metrics report

Use the results file produced by the evaluation command you ran in step 5.
For the single-linear metrics:

```bash
potofgold-report --results runs/demo/metrics_linear.json --out runs/demo/report_linear.html
```

For the optional three-encoder comparison:

```bash
potofgold-report --results runs/demo/metrics.json --out runs/demo/report.html
```

For geometry alone:

```bash
potofgold-report --results runs/demo/geometry.json --out runs/demo/report_geometry.html
```

Rendering reads an existing results file; it does not create `metrics.json` from
`geometry.json`. If you have only `geometry.json`, use the geometry report command
above or run the single-linear prediction evaluation first.

Open the HTML in a browser. It contains metric tables and embedded Seaborn plots,
with no separate image files or external assets. Report rendering does not
recompute metrics or train models.

All outputs from the example remain under `runs/demo/`, which Git ignores.
Commands that write run outputs refuse to replace them unless `--overwrite` is
supplied. To repeat an experiment, use a new run directory or explicitly overwrite.

## Analyze the original experiment

Regenerate the archived report without private data:

```bash
potofgold-report --results results/experiment.json --out runs/original/report.html
```

To recompute measurements when the original corpus is available, repeat steps
4–6 with `--data /path/to/original/corpus.npz` and the released checkpoints under
`checkpoints/`, writing exports and metrics to `runs/original/`. For example:

```bash
potofgold-embed --checkpoint checkpoints/encoder_convolutional.pt \
  --data /path/to/original/corpus.npz --out runs/original/embeddings_convolutional.npz

potofgold-evaluate --embeddings runs/original/embeddings_convolutional.npz \
  --data /path/to/original/corpus.npz --checkpoints checkpoints/encoder_convolutional.pt \
  --bootstrap 1000 --out runs/original/metrics.json

potofgold-report --results runs/original/metrics.json --out runs/original/convolutional_report.html
```

For multiple seeds, export each checkpoint to its own embedding file and supply
all embedding and checkpoint paths to the evaluator. The original seeds were
**0, 42, 4395, 5040 and 6118**. Newly trained copies are new runs; they are not the
missing original checkpoints. Saved historical confidence intervals retain the
methods used in their original evaluations. For the archived resampling settings,
use `--bootstrap 1000 --bootstrap-seed 42`; when adding `--indicators`, also use
`--indicator-bootstrap 10000 --indicator-bootstrap-seed 20260916`. These separate
settings reproduce the original residual-probe and indicator-comparison resampling
configurations. Across-seed summaries use Student-t intervals; relative RankMe
comparisons match models by seed. The settings are recorded with new metrics.

Checkpoint hashes are recorded in `indicator_comparison.checkpoint_sha256` inside
`results/experiment.json`. The released files are base64-encoded PyTorch
checkpoints; use `potofgold.encoder.checkpoints.load_checkpoint` or
`potofgold.encoder.inference.load_encoder` from Python rather than `torch.load(path)`.
Original checkpoint metadata is normalized in memory when read; the released
files retain their original bytes. Only deserialize trusted checkpoints.

## Use the original encoder downstream

A future GP module can load `transformer/checkpoints/encoder_convolutional.pt` directly,
without copying or retraining it. To export its multilayer readout at every
eligible bar, run from this module directory:

```bash
potofgold-embed --checkpoint checkpoints/encoder_convolutional.pt \
  --data /path/to/corpus.npz --readout multilayer --stride 1 \
  --out runs/downstream/embeddings_convolutional.npz
```

This yields 147 vectors of 384 dimensions per eligible window. The GP module's
pooling into 13 readouts happens downstream and is not part of encoder export.

## Source

| Location under `src/potofgold/` | Purpose |
| --- | --- |
| `encoder/` | Model, training, preprocessing, checkpoint loading and embedding export |
| `analysis/` | Financial targets, target normalization and residualization, evaluation, metrics and HTML reports |
| `sim_tools/` | Synthetic-data generation, CSV-to-NPZ packing and optional data/embedding previews |

With real data already in the [documented NPZ format](data/DATA.md), skip steps
1–2 and supply its path to training or embedding export. Training and evaluation
read daily arrays directly from the NPZ through `encoder/data.py`. Training and
analysis do not import the simulation module. The feature dictionary and preprocessing
in `encoder/` describe the original inputs and apply to real data too.

The model and learning objective are in `encoder/model.py` and
`encoder/objective.py`; configuration is in `configs/tokenizer.yaml`.
Financial targets are in `analysis/forward_targets.py`, and their normalization
and residualization are in `analysis/target_processing.py`. Shared date embargo
and indicator extraction helpers remain in `encoder/forward_split.py`.

[Repository overview](../README.md) · [Original data](data/DATA.md) ·
[Citation](../CITATION.cff) · [AI assistance](../AI_DISCLOSURE.md) · [License](../LICENSE)
