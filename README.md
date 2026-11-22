# MetaMo-Python

MetaMo-Python implements the MetaMo motivational framework in Python. It combines OpenPsi appraisal, MAGUS action selection, and state transition checks in a reusable core. The repository also includes a research assistant and a Gridworld application.

## Architecture

- `core/` defines motivation schemas, states, stimuli, actions, and configuration.
- `openpsi/` appraises stimuli; `magus/` evaluates candidate actions.
- `category/` composes appraisal and decision in the MetaMo pseudo-bimonad.
- `dynamics/` checks safety and coherence of state transitions.
- `applications/research_assistant/` provides an interactive assistant with optional paper context.
- `applications/gridworld/` applies MetaMo to a Gridworld environment and includes an evaluation runner.
- `s_expression/` serializes MetaMo data as S-expressions.

Motivational state consists of a goal vector `G` and a modulator vector `M`. Each application supplies a profile and schema for its goals, modulators, appraisal, and decision behavior. The default schema has eight goals and six modulators. See `core/config.py` for the current safety and transition parameters.

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux and macOS, then install the application dependencies:

```bash
python -m pip install -r requirements.txt
```

## Tests

Install `pytest` and run the complete test suite:

```bash
python -m pip install pytest
python -m pytest -q tests
```

The CI workflow tests Python 3.10, 3.11, and 3.12 with NumPy and pytest.

## Applications

### Research assistant

Copy [`.env.example`](.env.example) to `.env` at the repository root, then set `PROVIDER_NAME`, `MODEL_NAME`, and `API_KEY` to your provider's values. Supported provider names are `gemini`, `google`, and `snet`. The example also documents the optional `BASE_URL` and `REQUEST_TIMEOUT` settings. The `.env` file is ignored by Git.

Start an interactive session, optionally loading a text or PDF paper:

```bash
python -m applications.research_assistant
python -m applications.research_assistant path/to/paper.txt
```

PDF ingestion requires the `pdftotext` command from Poppler to be available on your path.

### Gridworld

Run a small evaluation from the repository root:

```bash
python -m applications.gridworld.evaluation.runner --train-episodes 5 --eval-episodes 5 --output-dir eval_results/gridworld
```

For the interactive simulation, run `python -m applications.gridworld.simulation.main`. See [the Gridworld README](applications/gridworld/README.md) for application details.

## References

1. *MetaMo: A Robust Motivational Framework for Open-Ended AGI* (AGI-25).
2. *Embodying Abstract Motivational Principles in Concrete AGI Systems: From MetaMo to Open-Ended OpenPsi*.

## License

See [LICENSE](LICENSE).
