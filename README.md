# Pot of Gold

Research code for a modular 0DTE options trading pipeline.

## Modules

- **[Transformer](transformer/README.md)** — original encoder checkpoints,
  representation-learning experiments, saved results and report reproduction,
  plus separate steps to generate data, train, inspect embeddings and evaluate.
  See the module guide for setup and use.
- **[Evolutionary search](evolutionary_search/README.md)** — strongly typed genetic
  programming, original strategies and outcomes, reproducible metrics and reports,
  and separate steps to run searches on synthetic data.

The licensed market data cannot be redistributed and is not included. Recomputing
the transformer measurements or replaying the original trading strategies requires
the matching original inputs. The published [GP report](evolutionary_search/results/report.html)
can be regenerated from the released outcomes without market data. The transformer
report can be rendered from its saved measurements. Synthetic data demonstrates
the workflow and produces new results.

### Coming soon
Multi-agent LLM validation workflow.

[Citation](CITATION.cff) · [AI assistance](AI_DISCLOSURE.md) · [License](LICENSE)
