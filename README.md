# pot-of-gold

**Multi-paradigm representation learning + genetic programming for 0DTE SPX options** —
the research code accompanying a Doctor of Engineering (D.Eng) praxis in AI/ML.

> **Status:** research code, released incrementally. This repository is a *curated* public
> mirror of a larger private research codebase; artifacts are added only after review.
> It is shared for transparency and reproducibility of the methodology, **not** as investment
> advice or a turnkey trading system.

## What this is

A three-paradigm pipeline studied as a D.Eng praxis:

1. **Representation learning** — a self-supervised iTransformer encoder over minute-resolution
   0DTE options + underlying features (tokenizer ablation: linear / patch / convolutional).
2. **Signal discovery** — a strongly-typed genetic program that searches defined-risk options
   structures over the learned representation vs. hand-crafted priced features.
3. **Composition** — a validation/LLM layer over the discovered candidates.

The research questions concern *whether the learned representation carries predictive structure
that priced statistical features miss*, evaluated with adversarial nulls (row-shuffle floors,
day-permutation tests) rather than a single headline number.

## Repository scope

**Included (incrementally):** methodology source — encoder/training, GP grammar + evaluator,
validation harnesses, QC code-generation, and the unit tests for each.

**Deliberately excluded** (see `.gitignore`):

- **Market data and its derivatives** — the SPX / options data is licensed (via the data vendor
  / QuantConnect) and **cannot be redistributed**. This covers `raw_data/`, all `*.parquet` /
  `*.npz` feature bundles, and the encoder checkpoints derived from them.
- **Secrets** — no API keys / tokens / `.env` files are ever committed.
- Run outputs, logs, and internal working notes.

## Data availability

The underlying 0DTE SPX options and index data are obtained under license and are **not**
included here. To reproduce results, a user must independently license equivalent data and
regenerate the feature bundles with the collection/preprocessing scripts in this repo. See
[`docs/DATA.md`](docs/DATA.md) *(to be added)* for the schema, the collection procedure, and the
exact preprocessing steps.

## Reproducibility

<!-- TODO: environment (Python version, key deps), how to run the encoder training, the GP
     search, and the validation harness end-to-end once the corresponding scripts are added. -->

## Citation

<!-- TODO: add the dissertation citation / DOI once available. -->

## License

Released under the [MIT License](LICENSE).

## Disclaimer

This is academic research software. It is provided "as is", without warranty of any kind, and is
**not** financial or investment advice. Nothing here is a recommendation to trade any instrument.
