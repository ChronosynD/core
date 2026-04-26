# chronosynd-py

Research reference implementation of ChronosynD. Home of the **Sediment** algorithm, the naive baseline used as the prior-work reference point, the poisoning-attack harness, and the experiment scripts that produce the paper's figures.

See the [top-level README](../README.md) for the research context. The code in this directory is the canonical reference for every algorithm. The Rust port at [`../chronosynd-rs/crates/baseline`](../chronosynd-rs/crates/baseline) is kept in bit-equivalent parity with it via the CI parity check.

## Usage

```bash
# Install with research and visualization extras
uv sync --extra dev --extra viz

# Lint, type-check, and test
uv run ruff check chronosynd_py tests
uv run mypy
uv run pytest

# Regenerate the test vectors the Rust parity check validates against
uv run python -m chronosynd_py.parity.emit

# Run a single experiment
uv run python -m chronosynd_py.evaluation.experiments.exp04_budget_sweep

# Render its figure
uv run python -m chronosynd_py.evaluation.figures.fig04_budget_sweep
```

The full pipeline (every experiment, every figure) is wired through [`../scripts/reproduce_all.sh`](../scripts/reproduce_all.sh) at the repository root.