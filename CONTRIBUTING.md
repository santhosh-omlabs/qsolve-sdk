# Contributing

Thanks for helping. This repository is the SDK only: the contract, the Qiskit bridge, the validation
harness and the command line. Quantum algorithms live in workflows, not here.

## Set up

```bash
uv sync --extra dev --extra ibm
uv run pytest
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy src
```

All four must pass before a pull request is ready. Python 3.12 or newer.

## Guidelines

- Keep the SDK free of algorithms and of workflow-specific logic.
- A behaviour change needs a test; a bug fix needs a test that fails without it.
- Public names are exported from `qsolve/__init__.py` and documented in `docs/`. Everything else
  (the executor protocol in `backends/remote.py`, the sandbox entry point) is internal and may change.
- Comments say why, not what. Keep lines within 100 characters.

## Reporting a security problem

See [SECURITY.md](SECURITY.md). Please do not open a public issue.
