# Pot of Gold

**Dynamic Strategy Discovery and Optimization for Zero-Days-to-Expiration Options Trading Using a Multi-Agent Model**

Code and results accompanying the D.Eng. praxis.
The research combines transformer representations, genetic programming and
multi-agent strategy assessment.

## Modules

- **[Transformer](transformer/README.md)** compares linear, patch and
  convolutional tokenizers. Includes the original encoder checkpoints,
  measurements and report, with commands for training and exporting embeddings.
- **[Evolutionary search](evolutionary_search/README.md)** evolves trading rules
  using market indicators and embeddings. Includes the search code, original
  strategies, outcomes and report, with commands for new searches.
- **[Multi-agent](multi-agent/README.md)** compares portfolio risk under a court
  of LLM agents, a single LLM and two allocation baselines. Includes agents,
  prompts, training diagnostics, saved decisions and the H3 report, with commands
  for assessing new strategies. Strategy replay uses the evolutionary-search engine.

Each module's README gives setup, reproduction and new-data commands. All three
reports can be regenerated from saved measurements, decisions and outcomes.
Reproducing the H3 report requires no model calls.

Rerunning the original encoders, trading strategies or agent assessments requires
the matching market inputs. The licensed market dataset is not included.

[Citation](CITATION.cff) · [AI assistance](AI_DISCLOSURE.md) · [License](LICENSE)
