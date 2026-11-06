# H3 archive

The report uses the saved strategy outcomes and model decisions in `../results/`.
New assessments also require the matching market inputs.

## Strategy and outcome alignment

`cohort.json` contains 58 strategies, in the same order as `champion_ids` in
`../results/outcomes.npz`. The evaluator excludes the one record marked
`excluded`, checking that its original court record also reports a degraded
panel. The other 57 strategies are identical across all four allocation policies.

| Array | Shape | Meaning |
| --- | --- | --- |
| `champion_ids` | 58 | IDs joining outcomes, cohort records and decisions |
| `dates` | 882 | Sorted union of the original held-out fold calendars |
| `held_out_pnl` | 882 × 58 | Actual traded-day P&L, net of the source replay's costs; zero elsewhere |
| `held_out_mask` | 882 × 58 | Whether that date belongs to the strategy's held-out fold |
| `train_pnl` | Flat array | Original finite training traded-day P&L values, concatenated by strategy |
| `train_offsets` | 59 | Start/end offsets into `train_pnl` for each strategy |

Use `numpy.load(path, allow_pickle=False)` to inspect the arrays. Training values
retain their original order and are summed to rank the matched-allocation
comparator. They have no date labels. Each policy's daily portfolio is the sum
of allocation × held-out strategy P&L, including days with no trades.

## Original sources

The archive comes from three v12 batches: `paid90_batch01_C20`,
`paid90_batch02_A20`, and `paid90_batch03_mixed18`. Their original arm names are
`A` (indicators), `C` (embeddings), and `AUC` (combined). Decisions and transcripts
are preserved as JSON records; repetitive machine-specific batch headers are
represented by the portable batch metadata in `manifest.json`.

Strategies were joined to the original `sealed_b304782` ledgers using the full
genome excluding fitness, plus fold. The original loader checked the code SHA,
seed and condition and required a complete match. All 58 strategies matched.
Actual traded P&L was zero-filled onto each parent fold's calendar. The manifest
records hashes for the original inputs, join/analysis code, original report and
released files. Paths under `source_sha256` describe historical source locations;
reproduction does not need those directories.

The original report leaves the provenance of the arm-A source
`condition_a_ms_noratio_partA_s2_DIRTY_routers` unresolved.

## Decisions and transcripts

[court.json](../results/court.json) and [single_llm.json](../results/single_llm.json)
contain the original verdict records, alongside the outcomes and report in
`../results/`. [transcripts.json.gz](../results/transcripts.json.gz) contains a
`records` list with one `verdict` and one
`single_llm_baseline` audit record per strategy. Read it with `gzip.open(path,
'rt')` and `json.load`. Court audits retain the prosecution, defense, final
panel, optional analysis, submitted evidence, provider metadata and usage.

All five batch configurations record:

| Role | Recorded model |
| --- | --- |
| Prosecutor | `gpt-6-astra` |
| Defender | `claude-opus-4-8` |
| Judges | `claude-opus-4-8`, `gpt-5.6-terra`, `gpt-6-astra` |
| Optional analyst in the archived experiment | `gpt-5.6-terra` |
| Single LLM | `gpt-6-astra` |

Per-call metadata in the transcripts records the actual provider/model used,
including failed attempts and fallbacks. The single-model metadata identifies
OpenAI and `gpt-6-astra` for all 58 assessments.

Generation used v12's `percentage_v1` allocation task. The published report
assigns zero to `WONT_GENERALIZE` and otherwise uses `(100 - risk_score) / 100`,
taking the judges' median risk score for the court. The separate generated
allocation percentages remain in the archive but do not determine report sizing.

## Inputs for new court runs

The original development rule book is [evidence_store.json](evidence_store.json).
It supplies the reference findings and fitted reference rules used by both
assessment methods, and matches its SHA-256 checksum in the original v12 release.
The risk-scoring workflow uses these rules alongside verified training measurements.
The package also contains the original split manifest and the canonical set of
464 strategy IDs; their split identifier remains `fw_f5faa4893dce3afb`.

New strategy input is a JSON list. Each record has `champion_id`, `arm`, `fold`,
`member.genome`, `member.train_traded_day_pnl`, and the replay settings in
`provenance`. Include `train_traded_days` when available. Genomes include entry,
exit and sizing trees and the original structure parameters. Training P&L must
match the reconstructed training trades.

`potofgold-court-prepare` uses the market paths in `CAT_RUNNER_SCALAR_DATA`
(indicators), or `CAT_RUNNER_DATA` and `CAT_RUNNER_TARGETS` (embeddings), and
`CAT_RUNNER_NFOLDS`. It writes one `exhibits.json` containing training decision
margins (`g5`), perturbations (`g6`), regime measurements (`g7`) and intraday
measurements. The court accepts this through `--exhibits`; separate exhibit files
remain accepted through `--g5`, `--g6`, `--g7` and `--intraday`.

Supply `CAT_RUNNER_VIX_STATE` for the regime exhibit: a CSV with `date`,
`vix_1030`, `vix_term_slope_1030` and `atm_iv_1030`, using the same 10:30 snapshots
as the strategy inputs. Its hash is included in preflight and resume checks.
The market parquet must include `NetGammaProxy` and the option IV grid; embedding
inputs also need all 13 `EMB_*` vector columns. Targets must follow the parquet's
row order and the recorded basis configuration. Replay uses the shared
`evolutionary_search` implementation: the `iron_condor` template with evolved
structures, lean readouts and optional ridge target residualization.

[generalization_case_file.py](../src/potofgold/multi_agent/strategy_profiles/generalization_case_file.py)
defines the fields visible to agents. The public
[diagnostic factory](../src/potofgold/multi_agent/input_tools/prepare.py)
verifies training replay and supplies the same cached measurements to the court
and single LLM. `--catalogue-runner-factory module:function` also accepts a custom
factory with that contract. Market input hashes bind the preflight receipt and
resume state. Optional frozen peer references use `--peer-reference` and
`--peer-reference-sha256`.

The evaluation command accepts new decision files and an `outcomes.npz` with the
same six arrays documented above. Strategy IDs must match both decision files.
The evaluator reads the held-out outcomes and labels the report as a new sample.

The historical v12 evaluation profiles and complete per-case input packets are
unavailable. The saved decisions and outcomes reproduce the report; rerunning
the assessments requires matching market inputs, encoder readouts, strategy
records and fold definitions.
