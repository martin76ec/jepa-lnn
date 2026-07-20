# TODO

> Keep the source of truth for supplied and outstanding experiment inputs in
> [`docs/EXPERIMENT_INPUTS.md`](docs/EXPERIMENT_INPUTS.md).

## 0. Reproduce the LeWM baseline

- [ ] Identify the upstream LeWM repository, commit, license, and environment.
- [ ] Document the encoder, latent representation, predictor, loss, datasets, training loop, and planning/control evaluation.
- [ ] Add the baseline as a pinned dependency or documented external checkout; do not copy unversioned source.
- [ ] Run one local smoke test on a deterministic data fraction.
- [ ] Reproduce the baseline on the H200 with the full protocol before changing the predictor.
- [ ] Record baseline metrics, wall time, peak memory, GPU utilization, package versions, and Git commit.

## 1. Pre-register the experiment

- [ ] Finalize environments, task versions, trajectory-level train/validation/test splits, and `dt`.
- [ ] Set the closed-loop primary horizon and normalized latent-error definition.
- [ ] Define secondary horizons, long-horizon evaluation, divergence threshold, and control metric.
- [ ] Freeze five shared seeds, parameter-matching tolerance, stopping rule, and hyperparameter budget.
- [ ] Fill the placeholders in `configs/h200.yaml` for batch size, epochs, and long horizon.
- [ ] Add configuration schema and validation.

## 2. Build the shared data and run infrastructure

- [ ] Implement a dataset adapter that preserves episode boundaries and variable sequence lengths.
- [ ] Add deterministic trajectory-level splitting and split manifests.
- [ ] Implement local fractional sampling without changing held-out test data.
- [ ] Add run provenance: resolved config, Git commit, seed, device, package versions, timing, and memory metrics.
- [ ] Add CLI entry points for train, evaluate, and reproduce a run.
- [ ] Add tests for splits, masking, reproducibility, and run metadata.

## 3. Implement predictors behind the common API

- [ ] Implement `PredictorMLP` as the stateless reference.
- [ ] Implement `PredictorTransformer` with causal context, episode-start policy, and autoregressive rollout.
- [ ] Select and pin a maintained CfC implementation; implement `PredictorCfC` with explicit state and `dt`.
- [ ] Select and pin a maintained LTC implementation; implement `PredictorLTC` with explicit state, integration, and `dt`.
- [ ] Add factory/configuration support for all predictor variants.
- [ ] Test shapes, devices, gradients, state reset, variable lengths, and `step`/`rollout` equivalence.
- [ ] Test that the Transformer cannot access future states.

## 4. Train and evaluate direct substitutions

- [ ] Implement the shared training loop with identical data, loss, optimizer, scheduler, and early stopping across models.
- [ ] Match capacity within the pre-registered tolerance and log exact parameter counts.
- [ ] Run local smoke tests for every predictor with `configs/local.yaml`.
- [ ] Run all direct-substitution experiments on the H200 for the five shared seeds.
- [ ] Measure one-step error separately from closed-loop rollout error.
- [ ] Report rollout error at every horizon, area under the rollout curve, divergence rate/time, control score, latency, and memory.

## 5. Robustness, ablations, and reporting

- [ ] Implement pre-defined observation, action, initial-condition, and dynamics perturbations.
- [ ] Run robustness curves only on the held-out test sets.
- [ ] Add latent diagnostics: linear probes, neighborhood preservation, temporal variation, and anomalous transitions.
- [ ] Run architecture-adjusted experiments with equal tuning budgets.
- [ ] Run focused ablations for state reset, `dt`/context, capacity, and multi-horizon loss if introduced.
- [ ] Compute paired seed-level comparisons, confidence intervals, and performance-cost frontier.
- [ ] Publish tables, curves, configs, commands, limitations, and negative results.
