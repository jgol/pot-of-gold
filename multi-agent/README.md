# Multi-agent assessment of 0DTE strategies

H3 tests whether a court of LLM agents reduces portfolio drawdown and conditional
value at risk (CVaR) compared with a single LLM, deploying all strategies, and
assigning the court's allocation sizes by training P&L rank.

The [published report](results/report.html) contains the findings, tables and
Seaborn plots. Download it and open it in a browser. The saved decisions and
outcomes are sufficient to recalculate the report.

## Agents and prompts

The court runs one round of prosecution and defense, followed by three judges
who assign risk scores. The court and single LLM receive the same strategy
profiles, training measurements and reference rule book, with access to the
same diagnostic tools.

| File | Purpose |
| --- | --- |
| [prompts.py](src/potofgold/multi_agent/assessment/prompts.py) | Prompts for every court role and the single LLM |
| [court.py](src/potofgold/multi_agent/assessment/court.py) | Court procedure and aggregation of judge assessments |
| [single_llm.py](src/potofgold/multi_agent/assessment/single_llm.py) | Single-LLM assessment |
| [run.py](src/potofgold/multi_agent/assessment/run.py) | Command-line execution of both assessments |

## Setup and report reproduction

Use Python 3.12. Install from the repository root, then enter this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[court,reports]'
python -m pip check
cd multi-agent
```

Run the remaining commands from `multi-agent/`. To recalculate the results and
generate the report:

```bash
potofgold-court-evaluate --output runs/reproduction/metrics.json
potofgold-court-report --metrics runs/reproduction/metrics.json \
  --output runs/reproduction/report.html
```

The report evaluates 57 strategies across 882 held-out dates using 8,000 paired
bootstrap samples. Settings are in [configs/experiment.yaml](configs/experiment.yaml).
Allocation is zero for `WONT_GENERALIZE`; otherwise it is
`(100 - risk_score) / 100`, using the judges' median score for the court.
The separate `allocation_percent` field records the model's recommendation;
report sizing uses this risk-score formula.

## Run on your own data

Supply strategy records containing genomes, folds, replay settings and training
P&L, along with the matching market inputs. File formats are described in the
[data guide](data/DATA.md#inputs-for-new-court-runs). Preparation replays each
strategy and checks its training P&L against the saved record.

For embedding strategies:

```bash
export CAT_RUNNER_DATA=/path/to/market.parquet
export CAT_RUNNER_TARGETS=/path/to/targets.npz
export CAT_RUNNER_VIX_STATE=/path/to/vix_state_1030.csv
export CAT_RUNNER_NFOLDS=5

potofgold-court-prepare --champions /path/to/champions.json \
  --output runs/new/exhibits.json
```

The example below uses arm `C` for embeddings. For indicators, use arm `A` and
set `CAT_RUNNER_SCALAR_DATA` to the market parquet file. Combined strategies use
arm `AUC`.

Set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` for model calls. Roles and models are
configured in [configs/court.yaml](configs/court.yaml). Check the inputs, then
run the court and single LLM:

```bash
court_args=(
  --config configs/court.yaml --arm C
  --champions /path/to/champions.json --exhibits runs/new/exhibits.json
  --catalogue-runner-factory potofgold.multi_agent.input_tools.prepare:factory
  --evidence-store data/evidence_store.json
  --out runs/new/court.json --max-cost-usd 10
)
potofgold-court "${court_args[@]}" --preflight-only \
  --preflight-out runs/new/preflight.json
potofgold-court "${court_args[@]}" --preflight-receipt runs/new/preflight.json
```

Evaluate the decisions against held-out daily P&L supplied in the
[outcome format](data/DATA.md#strategy-and-outcome-alignment):

```bash
potofgold-court-evaluate --outcomes /path/to/outcomes.npz \
  --court-decisions runs/new/court.json \
  --single-decisions runs/new/court_baseline.json \
  --output runs/new/metrics.json
potofgold-court-report --metrics runs/new/metrics.json \
  --output runs/new/report.html
```

The assessment writes `court.json`, `court_baseline.json` and `court_audit.json`
under `runs/new/`. Individual agent calls are logged in `runs/agent_audit/`.

## Repository contents

| Directory | Contents |
| --- | --- |
| [results/](results/) | Original court and single-LLM decisions, transcripts, P&L arrays and report |
| [data/](data/) | Strategy cohort, archive checksums, development rule book and input guide |
| [configs/](configs/) | Court settings and report statistics |
| `runs/` | Generated assessments and reports; excluded from version control |

Within [src/potofgold/multi_agent/](src/potofgold/multi_agent/):

| Directory | Purpose |
| --- | --- |
| `assessment/` | Agents, prompts, model calls and response checks |
| `input_tools/` | Market-data preparation and development/test separation |
| `strategy_profiles/` | Strategy descriptions, measurements and reference comparisons |
| `diagnostics/` | Training behavior, cost sensitivity and stability measurements |
| `analysis/` | Portfolio allocation, statistical comparisons and report generation |
| `reference/` | Original development/test membership |

Strategy evaluation uses the shared
[evolutionary search engine](../evolutionary_search/src/potofgold/evolutionary_search/search/).
The [source manifest](src/potofgold/multi_agent/source_manifest.json) records file
origins, checksums and changes made for this release. The
[data guide](data/DATA.md) describes the original experiment inputs and how the
archive was assembled.
