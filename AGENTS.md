# Engineering Guidelines

## Scope and experiment integrity

- This repository evaluates liquid predictors as a controlled replacement for LeWM's dynamics predictor. Keep the encoder, data split, loss, evaluation protocol, and seeds comparable across predictor variants unless a configuration explicitly records an architecture-adjusted experiment.
- Use `configs/local.yaml` only for inexpensive smoke tests on a deterministic fraction of the data. Use `configs/h200.yaml` for the full experimental protocol. Never treat local results as reported results.
- Do not commit datasets, checkpoints, runs, credentials, or machine-specific paths.
- Every train/evaluation run must save its resolved configuration, Git commit, seed, package versions, device metadata, and metrics under `runs/`.

## Python design

- Target Python 3.11 and use complete type annotations for public functions and methods.
- Prefer small, pure functions for transformations, metrics, configuration validation, and orchestration.
- Use classes for stateful or polymorphic components: datasets, predictors, trainers, evaluators, and configuration objects.
- Define explicit protocols/abstract base classes at extension points. The predictor API must support `init_state`, `step`, and `rollout`.
- Use dataclasses for immutable value/configuration objects; prefer `frozen=True` when mutation is not necessary.
- Keep control flow flat: use guard clauses and early returns instead of deeply nested `if`/`else` blocks.
- Avoid broad `try`/`except`. Catch only expected, specific exceptions where recovery is meaningful; otherwise let failures surface with context.
- Avoid comments that restate code. Add a short comment only for non-obvious constraints, scientific rationale, or intentionally surprising behavior. Use docstrings for public APIs.
- Keep functions focused. Extract named helpers rather than accumulating flags, mutable shared state, or long conditional chains.
- Do not introduce global mutable state. Pass dependencies and random generators explicitly.

## Quality gates

- Format and lint with `uv run ruff format .` and `uv run ruff check .`.
- Run `uv run mypy src` and targeted `uv run pytest` before considering a change complete.
- Add unit tests for predictor state reset, tensor shapes/devices/gradients, masking/variable sequence lengths, and repeated `step` equivalence to `rollout`.
