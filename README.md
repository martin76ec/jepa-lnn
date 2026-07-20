# LeWM Liquid Predictors

Controlled experiment repository for evaluating CfC and LTC predictors against MLP and Transformer baselines in LEWorldModel.

## Environments

The repository has two intentionally separate execution profiles:

- `configs/local.yaml` is a deterministic smoke-test profile. It uses a small fraction of the training data and a single seed to validate the pipeline locally.
- `configs/h200.yaml` is the full experiment profile for the H200 server. It uses complete splits and the shared five-seed protocol.

Local runs are for correctness and iteration only; they are not experimental evidence.

## Setup

```bash
uv sync --group dev
source .venv/bin/activate
```

The PyTorch wheel must match the target accelerator. On the H200 server, install the cluster-approved CUDA-enabled PyTorch build before running `uv sync` or configure the relevant package index.

## Validation commands

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest
```

## Make Targets

```bash
make sync
make download-data
make inspect-data
make validate-local
make check
```

`make download-data` is resumable and writes the official PushT Lance archive
under ignored `data/raw/`. The current targets validate the pipeline and data
integration; train/evaluate targets will be added with their executable entry points.

## Layout

```text
configs/     Versioned local and H200 experiment profiles
src/         Library code: data, predictors, training, and evaluation
tests/       Fast unit and integration tests
scripts/     Thin, runnable entry points
data/        Ignored raw and processed datasets
runs/        Ignored run metadata, metrics, and artifacts
docs/        Experimental protocol and decisions
```

See `AGENTS.md` for the mandatory engineering and experimental guidelines.
