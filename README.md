# Pot of Gold

Research code for a modular 0DTE options trading pipeline.

## Modules

- **[Transformer](transformer/README.md)** — original encoder checkpoints,
  representation-learning experiments, saved results and report reproduction,
  plus separate steps to generate data, train, inspect embeddings and evaluate.
  See the module guide for setup and use.

Recomputing the original experimental measurements requires the original dataset
and the corresponding encoder checkpoints (included in this repo). The licensed data cannot be
redistributed and is not included. Saved results allow the report to be regenerated;
synthetic data demonstrates the workflow without reproducing those measurements.

### Coming soon
Additional modules:
1. Genetic Programming 
1. Multi-agent LLM validation workflow

[Citation](CITATION.cff) · [AI assistance](AI_DISCLOSURE.md) · [License](LICENSE)
