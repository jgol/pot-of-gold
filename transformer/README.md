# Transformer

This experiment compares linear, patch and convolutional tokenizers in a
self-supervised iTransformer trained on 0DTE options time series.

The workflow is **generate data → pack → train → export embeddings → evaluate →
report**. Each command runs one step.

## What's included

The repository includes all **15 original checkpoints**: three tokenizers, each
trained with seeds **0, 42, 4395, 5040 and 6118**.

| Artifact | Contents |
| --- | --- |
| [checkpoints/](checkpoints/) | Trained encoders and their preprocessing statistics |
| [results/experiment.json](results/experiment.json) | Per-seed measurements, training summaries and file hashes |
| [results/report.html](results/report.html) | Tables and plots from the saved measurements |

Checkpoint names follow `encoder_<tokenizer>_s<seed>.pt`. Seed 4395 uses the
shorter `encoder_<tokenizer>.pt` names. All files match the recorded SHA-256 hashes.

You can recalculate the summary statistics and regenerate the report from the
saved measurements. Running the original models on their data requires the
licensed market dataset, which cannot be redistributed. The synthetic example
below lets you try the workflow with generated data; its results are separate
from the study's findings.

## Install

Use Python **3.12**. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[reports]'
cd transformer
```

Run the commands below from `transformer/`. The `reports` extra installs Seaborn
and the other plotting dependencies.

## 1. Generate and inspect synthetic features

```bash
potofgold-generate-data --out runs/demo/features.csv --seed 0
potofgold-inspect --input runs/demo/features.csv --rows 5
```

The CSV has `date`, `bar` and 248 feature columns; the encoder uses 147 of them.
See the [data guide](data/DATA.md) for column definitions. To read it with pandas,
use `pd.read_csv(path, comment="#")`.

The default is 16 training days, eight validation days and four test days, with
405 minute bars per day. Adjust these with `--train-days`, `--val-days` and
`--test-days`. The five-day embargo leaves three validation days for evaluation.

## 2. Pack the data into NPZ

```bash
potofgold-pack-data --input runs/demo/features.csv --out runs/demo/corpus.npz
potofgold-inspect --input runs/demo/corpus.npz --rows 5
```

NPZ is NumPy's array archive format. This file holds one `(405, 248)` array per
day, stored as `float16`, plus dates and split information. If your own data
already follows the [NPZ format](data/DATA.md), start at step 3.

## 3. Train encoders

Run whichever encoders you want to compare. These examples use the same settings
apart from the tokenizer and train for **two epochs**; the study used **100**.

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

The model uses 60-bar windows, 147 variables, width 128, three attention blocks
and four heads. Checkpoints save the weights, prediction and adversarial heads,
normalization statistics and training metadata. Training cannot be resumed.

The [learning objective](src/potofgold/encoder/objective.py) combines prediction
of a residualized future teacher representation with variance, covariance,
decorrelation and adversarial terms. Teacher targets are residualized within
both training and validation batches. Saved validation losses combine these
terms; the archive has no per-epoch history or separate prediction loss.

## 4. Export and inspect embeddings

Export each encoder you trained. If you trained only linear, use the first
export command and the inspection command.

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
zero-based ending bars identify each window. The default 30-bar stride gives
12 windows per day, or 336 rows for this example. Evaluation counts can be
smaller after applying date splits, the embargo and target-validity checks.

| `--readout` | Output per window |
| --- | --- |
| `pooled` (default) | One 128-dimensional vector from final-layer mean pooling |
| `variates` | 147 vectors, each 128-dimensional |
| `multilayer` | 147 vectors, each 384-dimensional, concatenating three block outputs |

The study uses `pooled` embeddings. Their dimensions are learned features with
no predefined financial meaning.

## 5. Evaluate embeddings

For the linear encoder:

```bash
potofgold-evaluate --embeddings runs/demo/embeddings_linear.npz \
  --data runs/demo/corpus.npz --checkpoints runs/demo/encoder_linear.pt \
  --indicators --bootstrap 1000 --out runs/demo/metrics_linear.json
```

This computes representation geometry and residual-target prediction scores.
`--indicators` also compares indicators alone with indicators plus embeddings
on the original targets. For another encoder, replace `linear` in the filenames.

For geometry alone, the embedding file is enough:

```bash
potofgold-evaluate --embeddings runs/demo/embeddings_linear.npz \
  --out runs/demo/geometry.json
```

If you trained and exported all three encoders, compare them with:

```bash
potofgold-evaluate \
  --embeddings runs/demo/embeddings_linear.npz runs/demo/embeddings_patch.npz runs/demo/embeddings_convolutional.npz \
  --data runs/demo/corpus.npz \
  --checkpoints runs/demo/encoder_linear.pt runs/demo/encoder_patch.pt runs/demo/encoder_convolutional.pt \
  --indicators --bootstrap 1000 --out runs/demo/metrics.json
```

Evaluation fits regression probes on training observations and scores them on
held-out observations, using saved embeddings and checkpoint preprocessing.
The encoders stay fixed.

Prediction evaluation requires `pooled` embeddings at stride 30. Geometry also
supports other readouts and strides; variate and multilayer outputs are averaged
over variables.

Metrics include effective rank (RankMe), participation ratio, RankIC and R².
Reports distinguish intervals across evaluation days from intervals across
training seeds. Estimating seed variation requires more than one training seed.

## 6. Create a report

Choose the JSON file you produced in step 5. For linear:

```bash
potofgold-report --results runs/demo/metrics_linear.json --out runs/demo/report_linear.html
```

For the three-encoder comparison:

```bash
potofgold-report --results runs/demo/metrics.json --out runs/demo/report.html
```

For geometry alone:

```bash
potofgold-report --results runs/demo/geometry.json --out runs/demo/report_geometry.html
```

Open the HTML in a browser to see tables and embedded Seaborn plots. This step
displays the saved metrics. Outputs go under `runs/`, which Git ignores; use a
new directory or `--overwrite` to repeat a command.

## Analyze the original experiment

To recalculate the effective-rank and participation-ratio summaries and the
convolutional 5% milestone from saved measurements:

```bash
potofgold-evaluate --archive results/experiment.json \
  --out runs/original/representation_metrics.json

potofgold-report --results runs/original/representation_metrics.json \
  --out runs/original/representation_report.html
```

The input arrays are `rankme.per_seed` and `pr.per_seed` under
`across_seed_results.per_tokenizer.<tokenizer>.windows.test` in
`results/experiment.json`, ordered by `across_seed_results.seeds`.

The script calculates means and 95% Student-t intervals across seeds. For the
milestone, it calculates relative RankMe improvement within each matched seed,
then checks whether the lower confidence bound exceeds 5%. It also summarizes
training and validation measurements and records input and code hashes.
This step uses the archived measurements and needs no market data.

To regenerate the full report from its recorded values:

```bash
potofgold-report --results results/experiment.json --out runs/original/report.html
```

With the original licensed corpus, you can recompute measurements from the
checkpoints. For convolutional seed 4395:

```bash
potofgold-embed --checkpoint checkpoints/encoder_convolutional.pt \
  --data /path/to/original/corpus.npz --out runs/original/embeddings_convolutional.npz

potofgold-evaluate --embeddings runs/original/embeddings_convolutional.npz \
  --data /path/to/original/corpus.npz --checkpoints checkpoints/encoder_convolutional.pt \
  --bootstrap 1000 --out runs/original/metrics.json

potofgold-report --results runs/original/metrics.json --out runs/original/convolutional_report.html
```

For **all 15 checkpoints**:

```bash
for checkpoint in checkpoints/encoder_*.pt; do
  checkpoint_name=${checkpoint##*/}
  potofgold-embed --checkpoint "$checkpoint" \
    --data /path/to/original/corpus.npz \
    --out "runs/original/all_seeds/${checkpoint_name%.pt}.npz"
done

potofgold-evaluate --embeddings runs/original/all_seeds/encoder_*.npz \
  --data /path/to/original/corpus.npz --checkpoints checkpoints/encoder_*.pt \
  --bootstrap 1000 --bootstrap-seed 42 \
  --out runs/original/all_seeds/metrics.json

potofgold-report --results runs/original/all_seeds/metrics.json \
  --out runs/original/all_seeds/report.html
```

The commands use the original day-bootstrap settings: 1,000 resamples with seed
42. To include the original-target indicator comparison, add `--indicators
--indicator-bootstrap 10000 --indicator-bootstrap-seed 20260916`.

Checkpoint hashes are in `indicator_comparison.checkpoint_sha256`. The files
contain base64-encoded PyTorch checkpoints. To load them in Python, use
`potofgold.encoder.checkpoints.load_checkpoint` or
`potofgold.encoder.inference.load_encoder`.

## Use the original encoder downstream

The [evolutionary search module](../evolutionary_search/README.md) prepares the
13 GP readouts directly. Its original experiment used the separate
`evolutionary_search/checkpoints/encoder_convolutional_gp_seed42.pt` checkpoint.

To inspect this encoder's full multilayer output:

```bash
potofgold-embed --checkpoint checkpoints/encoder_convolutional.pt \
  --data /path/to/corpus.npz --readout multilayer --stride 1 \
  --out runs/downstream/embeddings_convolutional.npz
```

This saves 147 vectors of 384 dimensions per window and can produce a large
file. Use the GP module's preparation command for its original preprocessing
and pooled readouts.

## Source

| Location under `src/potofgold/` | Purpose |
| --- | --- |
| `encoder/` | Model, training, preprocessing, checkpoint loading and embedding export |
| `analysis/` | Financial targets, regression probes, metrics and reports |
| `sim_tools/` | Synthetic data, CSV-to-NPZ packing and previews |

Start with [model.py](src/potofgold/encoder/model.py),
[objective.py](src/potofgold/encoder/objective.py) and
[tokenizer.yaml](configs/tokenizer.yaml) for the architecture and training setup.
Date splits and preprocessing are in [config.py](src/potofgold/encoder/config.py).

[Repository overview](../README.md) · [Original data](data/DATA.md) ·
[Citation](../CITATION.cff) · [AI assistance](../AI_DISCLOSURE.md) · [License](../LICENSE)
