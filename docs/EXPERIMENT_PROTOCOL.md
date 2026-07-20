# Experiment Protocol

This repository implements the plan in the parent LeWM workspace: evaluate stateful CfC and LTC predictors against MLP and causal Transformer predictors without changing the encoder, latent space, dataset split, objective, or evaluation procedure.

The primary outcome is normalized latent error in a closed-loop rollout at the pre-registered horizon. One-step teacher-forced error is diagnostic only. Secondary outcomes include rollout curves, divergence rate, control performance, robustness under pre-defined perturbations, latent-structure diagnostics, parameters, latency, throughput, and memory.

The local profile is restricted to a small data fraction and exists only to prove data flow, numerical correctness, logging, and test coverage. All reported metrics must use the full H200 profile with the shared seed set and pre-registered configuration.

The rollout horizons are `H = [1, 5, 10, 20, 50]` predictor steps, with `H=20` as the primary outcome. A rollout diverges when normalized latent error exceeds `10.0` or any predicted value is non-finite.

## Metrics

The primary metric is mean normalized latent mean-squared error at closed-loop horizon `H=20`, averaged over test trajectories and reported across five shared seeds. Report mean, standard deviation, 95% confidence interval, and paired difference versus MLP.

Secondary metrics are:

- normalized latent error at every horizon in `[1, 5, 10, 20, 50]`;
- area under the rollout-error curve;
- divergence rate and median time-to-divergence;
- teacher-forced one-step error as a diagnostic;
- PushT planning/control success and return under the same evaluation budget;
- robustness degradation under pre-defined observation, action, and initial-state perturbations;
- parameter count, training time, inference latency, rollout throughput, peak memory, and estimated FLOPs;
- optional latent probes and neighborhood/temporal-structure diagnostics, clearly labeled exploratory.

## Deliverables

- reproducible train/evaluate CLIs and pinned environment/configuration files;
- per-run provenance, checkpoints, metrics, and resolved configurations;
- tables comparing MLP, Transformer, CfC, and LTC under direct substitution and architecture-adjusted tuning;
- rollout-error, divergence, robustness, control, and performance-cost plots;
- a concise report covering methodology, results, limitations, negative results, and recommended next experiments;
- reproduction commands for both the local smoke profile and the full H200 protocol.

The predictor interface must represent recurrent state explicitly:

```python
state = predictor.init_state(batch_size, device)
z_next, state = predictor.step(z_t, action_t, state, dt)
z_pred, final_state = predictor.rollout(z_0, actions, state=state, dt=dt)
```

Before training candidates, finalize the environments, split manifests, rollout horizons, divergence threshold, capacity-matching bounds, compute budget, and stopping criterion.
